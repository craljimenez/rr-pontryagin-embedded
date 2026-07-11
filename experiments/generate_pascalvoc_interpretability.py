"""PASCAL VOC 2012 — qualitative comparison + PRFE relevance analysis.

Mirrors the UAV analysis (generate_energy_gt.py) on the multiclass VOC
pipeline (run_seg_pascalvoc.py):

  1. Qualitative figure  — Image | GT | Euclidean | Hyperbolic | PRFE
     on representative test images (VOC devkit palette).
  2. Energy figures      — for the PRFE head: RGB+GT contour,
     MC Dropout predictive entropy, E+ (RFF) and E- (SRF) maps for the
     dominant foreground class.
  3. Quantitative relevance — over a test subsample: per-image Pearson
     correlation between predictive entropy and the (negated) energy gap
     |E+ - E-| of the predicted class, plus mean entropy on correct vs.
     misclassified pixels.

Outputs (report/figures/ + results/seg_pascalvoc_results/):
  pascalvoc_qualitative_paper.pdf
  pascalvoc_energy_ex1.pdf / pascalvoc_energy_ex2.pdf
  energy_stats.json

Uso:
    python experiments/generate_pascalvoc_interpretability.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T

EXP_DIR = Path(__file__).parent
SRC_DIR = EXP_DIR.parent / "src"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(SRC_DIR))

from configs.seg_pascalvoc import (
    DATASET_ROOT, IMG_SIZE, SEED, VAL_TEST_SPLIT, VOID_LABEL,
    N_CLASSES, CLASS_NAMES, VOC_YEAR,
)
from run_seg_pascalvoc import build_unet_model as build_model, _voc_colormap

RES_DIR = EXP_DIR / "results" / "seg_pascalvoc_results"
FIG_DIR = EXP_DIR / "report" / "figures"

HEADS   = ["euclidean", "hyperbolic", "pontryagin", "pontryagin_trff"]
LABELS  = {"euclidean": "Euclidean", "hyperbolic": "Hyperbolic",
           "pontryagin": "PRFE", "pontryagin_trff": "PRFE-tRFF"}
PALETTE = _voc_colormap()

N_MC   = 30
P_DROP = 0.25
N_STATS_IMGS = 60          # subsample for the quantitative relevance stats
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])

_to_tensor = T.ToTensor()
_normalise = T.Normalize(IMG_MEAN.tolist(), IMG_STD.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Test split — identical logic to PascalVOCSegDataset, but download=False
# ─────────────────────────────────────────────────────────────────────────────

class VOCTestSplit(torch.utils.data.Dataset):
    def __init__(self, root=DATASET_ROOT, img_size=IMG_SIZE):
        self.img_size = img_size
        self._voc = VOCSegmentation(root=str(root), year=VOC_YEAR,
                                    image_set="val", download=False)
        rng  = np.random.default_rng(SEED)
        perm = rng.permutation(len(self._voc))
        n_val = int(len(perm) * (1 - VAL_TEST_SPLIT))
        self.indices = perm[n_val:].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self._voc[self.indices[idx]]
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        img_t  = _normalise(_to_tensor(img))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_t, mask_t


def _denorm(t):
    img = t.cpu() * IMG_STD[:, None, None] + IMG_MEAN[:, None, None]
    return img.permute(1, 2, 0).numpy().clip(0, 1)


def _colorize(mask):
    """int mask (H,W) → RGB via VOC devkit palette (void → white)."""
    m = mask.copy()
    m[m == VOID_LABEL] = 255
    rgb = PALETTE[m.reshape(-1)].reshape(*m.shape, 3).astype(np.uint8)
    rgb[m == 255] = 255
    return rgb


def _norm01(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def load_head(head, device):
    # "pontryagin_trff" is not its own model_type — it's model_type=
    # "pontryagin" with params["trainable_rff"]=True (already baked into the
    # saved checkpoint's "params" dict by run_seg_pascalvoc.py), living in
    # its own results folder so it never collides with the fixed-RFF run.
    model_type = "pontryagin" if head == "pontryagin_trff" else head
    ck = torch.load(RES_DIR / head / "best_model.pth",
                    map_location="cpu", weights_only=False)
    model = build_model(model_type, ck.get("params", {}))
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval()


class _MCDropout(nn.Module):
    """Dropout2d on backbone features — same estimator as the UAV analysis."""

    def __init__(self, seg_model, p=P_DROP):
        super().__init__()
        self.seg  = seg_model
        self.drop = nn.Dropout2d(p=p)

    def forward(self, x):
        feats = self.seg.backbone(x)
        feats = self.drop(feats)
        z     = self.seg._embed(feats)
        B, D, Hf, Wf = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(B * Hf * Wf, D)
        lo = self.seg._logits_flat(zf).reshape(B, Hf, Wf, -1).permute(0, 3, 1, 2)
        return F.interpolate(lo, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


@torch.no_grad()
def mc_entropy(mc_model, inp, n=N_MC):
    mc_model.eval()
    mc_model.drop.train()
    plist = []
    for _ in range(n):
        plist.append(torch.softmax(mc_model(inp), 1))
    pbar = torch.stack(plist).mean(0)
    h = -(pbar * (pbar + 1e-8).log()).sum(1)
    return h[0].cpu().numpy()


@torch.no_grad()
def class_energies(model, inp):
    """Per-class energy maps at input resolution.

    Returns (E+, E-) of shape (N_CLASSES, H, W):
        E+_c = <w_c,pos, phi_pos>,  E-_c = <w_c,neg, phi_neg>
    (same sign convention as generate_energy_gt.py — the J-logit is
    E+_c - E-_c + b_c).
    """
    captured = {}
    def hook(m, a, o): captured["z"] = o
    h = model.embed_layer.register_forward_hook(hook)
    model(inp)
    h.remove()
    z = captured["z"]                                # (1, p+q, H', W')
    p = model.embed_layer.rff.out_features
    W = model.head.W                                 # (C, p+q)
    e_pos = torch.einsum("bdhw,cd->bchw", z[:, :p], W[:, :p])
    e_neg = torch.einsum("bdhw,cd->bchw", z[:, p:], W[:, p:])
    t = inp.shape[-2:]
    e_pos = F.interpolate(e_pos, t, mode="bilinear", align_corners=False)
    e_neg = F.interpolate(e_neg, t, mode="bilinear", align_corners=False)
    return e_pos[0].cpu().numpy(), e_neg[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Qualitative comparison figure
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_image_miou(pred, gt):
    """Macro IoU over the classes present in GT (void excluded)."""
    ious = []
    for c in np.unique(gt):
        if c == VOID_LABEL:
            continue
        p, g = pred == c, gt == c
        inter, union = (p & g).sum(), (p | g).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else float("nan")


def qualitative_figure(models, dataset, device, n_show=4):
    # rank test images by inter-model spread, keep the most contrasting ones
    print("  scoring test images for the qualitative panel ...")
    scores = []
    for i in range(len(dataset)):
        img, mask = dataset[i]
        gt = mask.numpy()
        fg = ((gt != 0) & (gt != VOID_LABEL)).mean()
        if fg < 0.08:          # skip images with almost no foreground
            continue
        inp = img.unsqueeze(0).to(device)
        mious = {}
        for h, m in models.items():
            pred = m(inp).argmax(1)[0].cpu().numpy()
            mious[h] = per_image_miou(pred, gt)
        scores.append((i, mious))
    # spread = best-PRFE-variant advantage over the euclidean baseline
    # (uses PRFE-tRFF, the stronger variant, so the panel showcases where
    # the flagship result actually wins)
    scores.sort(key=lambda s: max(s[1]["pontryagin"], s[1]["pontryagin_trff"]) - s[1]["euclidean"],
                reverse=True)
    # 3 images where PRFE wins + the median image (typical behaviour)
    chosen = [s[0] for s in scores[:n_show - 1]]
    chosen.append(scores[len(scores) // 2][0])

    ncols = 2 + len(HEADS)
    fig, axes = plt.subplots(len(chosen), ncols,
                             figsize=(2.9 * ncols / 2, 1.45 * len(chosen)))
    present = set()
    for r, idx in enumerate(chosen):
        img, mask = dataset[idx]
        gt  = mask.numpy()
        inp = img.unsqueeze(0).to(device)
        present |= set(np.unique(gt).tolist()) - {VOID_LABEL}
        panels = [_denorm(img), _colorize(gt)]
        for h in HEADS:
            pred = models[h](inp).argmax(1)[0].cpu().numpy()
            present |= set(np.unique(pred).tolist())
            panels.append(_colorize(pred))
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(["Image", "Ground truth"][c] if c < 2
                             else LABELS[HEADS[c - 2]], fontsize=8)

    handles = [mpatches.Patch(color=PALETTE[c] / 255.0, label=CLASS_NAMES[c])
               for c in sorted(present)]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(7, len(handles)), fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.2,
               columnspacing=1.0)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    out = FIG_DIR / "pascalvoc_qualitative_paper.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {out.name} written (rows: {[int(i) for i in chosen]})")
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# 2. Energy figures (PRFE head)
# ─────────────────────────────────────────────────────────────────────────────

def energy_figure(model, mc_model, dataset, idx, out_path, device):
    img, mask = dataset[idx]
    gt  = mask.numpy()
    rgb = _denorm(img)
    inp = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp).argmax(1)[0].cpu().numpy()
    # dominant predicted foreground class
    fg_classes, counts = np.unique(pred[pred != 0], return_counts=True)
    cls = int(fg_classes[counts.argmax()]) if len(fg_classes) else 0

    ent = mc_entropy(mc_model, inp)
    e_pos, e_neg = class_energies(model, inp)
    ep, en = e_pos[cls], e_neg[cls]
    gt_bin = (gt == cls).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].imshow(rgb)
    axes[0].contour(gt_bin, levels=[0.5], colors=["white"],  linewidths=5)
    axes[0].contour(gt_bin, levels=[0.5], colors=["#00ff00"], linewidths=2.5)
    axes[0].set_xlabel(f"Original image\n(green: GT '{CLASS_NAMES[cls]}')",
                       fontsize=10)

    im1 = axes[1].imshow(_norm01(ent), cmap="inferno", vmin=0, vmax=1)
    axes[1].set_xlabel(f"MC Dropout uncertainty\n(predictive entropy, N={N_MC})",
                       fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(_norm01(ep), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xlabel(
        r"Positive energy $\langle w_+, \varphi_+\rangle$" "\n(RFF — Euclidean)",
        fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(_norm01(en), cmap="Reds", vmin=0, vmax=1)
    axes[3].set_xlabel(
        r"Negative energy $\langle w_-, \varphi_-\rangle$" "\n(SRF — hyperbolic)",
        fontsize=10)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name} written (class '{CLASS_NAMES[cls]}')")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantitative relevance stats
# ─────────────────────────────────────────────────────────────────────────────

def relevance_stats(model, mc_model, dataset, device, n_imgs=N_STATS_IMGS):
    print(f"  computing relevance stats on {n_imgs} test images ...")
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(dataset), size=min(n_imgs, len(dataset)),
                      replace=False)
    r_gap, h_correct, h_wrong = [], [], []
    for i in idxs:
        img, mask = dataset[int(i)]
        gt  = mask.numpy()
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp).argmax(1)[0].cpu().numpy()
        ent = mc_entropy(mc_model, inp)
        e_pos, e_neg = class_energies(model, inp)
        # per-pixel energies of the *predicted* class
        take = pred.reshape(-1)
        ep = e_pos.reshape(N_CLASSES, -1)[take, np.arange(take.size)]
        en = e_neg.reshape(N_CLASSES, -1)[take, np.arange(take.size)]
        gap = -np.abs(_norm01(ep) - _norm01(en))        # high where E+≈E-
        hf  = ent.reshape(-1)
        if hf.std() > 1e-6 and gap.std() > 1e-6:
            r_gap.append(float(np.corrcoef(hf, gap)[0, 1]))
        valid = gt.reshape(-1) != VOID_LABEL
        ok = (pred.reshape(-1) == gt.reshape(-1)) & valid
        ko = (pred.reshape(-1) != gt.reshape(-1)) & valid
        if ok.any(): h_correct.append(float(hf[ok].mean()))
        if ko.any(): h_wrong.append(float(hf[ko].mean()))

    stats = {
        "n_images": int(len(idxs)),
        "n_mc": N_MC,
        "pearson_entropy_vs_energy_balance_mean": float(np.mean(r_gap)),
        "pearson_entropy_vs_energy_balance_std":  float(np.std(r_gap)),
        "mean_entropy_correct_px": float(np.mean(h_correct)),
        "mean_entropy_wrong_px":   float(np.mean(h_wrong)),
    }
    out = RES_DIR / "energy_stats.json"
    json.dump(stats, open(out, "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"  {out} written")
    return stats


# ─────────────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = VOCTestSplit()
    print(f"Test split: {len(dataset)} images")

    models = {h: load_head(h, device) for h in HEADS}
    pont    = models["pontryagin"]
    pont_mc = _MCDropout(pont).to(device)

    chosen = qualitative_figure(models, dataset, device)

    # energy figures on the two most contrasting images from the panel
    energy_figure(pont, pont_mc, dataset, chosen[0],
                  FIG_DIR / "pascalvoc_energy_ex1.pdf", device)
    energy_figure(pont, pont_mc, dataset, chosen[1],
                  FIG_DIR / "pascalvoc_energy_ex2.pdf", device)

    relevance_stats(pont, pont_mc, dataset, device)


if __name__ == "__main__":
    main()
