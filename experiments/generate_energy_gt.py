"""Regenerate FCN / UNet Pontryagin energy-decomposition figures
with a green ground-truth contour overlaid on the original image panel.

Outputs (written to report/figures/):
  pontryagin_fcn_energy_ex1.pdf   ← DJI_45   FCN
  pontryagin_fcn_energy_ex2.pdf   ← DJI_27   FCN
  pontryagin_unet_energy_ex1.pdf  ← DJI_45   UNet
  pontryagin_unet_energy_ex2.pdf  ← DJI_27   UNet

Each PDF is a 1×4 panel:
  1. Original RGB  (+ green GT contour)
  2. MC Dropout predictive entropy  (Inferno)
  3. Positive energy  E+  (Blues, RFF)
  4. Negative energy  E-  (Reds, SRF)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── paths ─────────────────────────────────────────────────────────────────────
EXP_DIR  = Path(__file__).parent
SRC_DIR  = EXP_DIR.parent / "src"
OUT_DIR  = EXP_DIR / "report" / "figures"

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(SRC_DIR))

# ── FCN imports ───────────────────────────────────────────────────────────────
from configs.seg_uav import (
    DATASET_ROOT, DEVICE, IMG_SIZE,
)
from configs.seg_uav import RESULTS_DIR as FCN_RESULTS_DIR
from run_seg_uav import UAVSegDataset, build_model as fcn_build_model

# ── UNet imports ──────────────────────────────────────────────────────────────
from configs import seg_uav_unet as _unet_cfg
UNET_RESULTS_DIR = _unet_cfg.RESULTS_DIR
from run_seg_uav_unet import build_unet_model as unet_build_model

# ── constants ─────────────────────────────────────────────────────────────────
N_MC   = 30
P_DROP = 0.25
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])

TARGET_STEMS = ["DJI_36.rf", "DJI_27.rf"]   # ex1 (25% cane), ex2 (3% cane)


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm01(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def _denormalize(t):
    img = t.cpu().clone()
    img = img * IMG_STD[:, None, None] + IMG_MEAN[:, None, None]
    return img.permute(1, 2, 0).numpy().clip(0, 1)


# ── model wrappers ────────────────────────────────────────────────────────────

def load_fcn():
    md = FCN_RESULTS_DIR / "pontryagin"
    params = {}
    p = md / "hpo" / "best_params.json"
    if p.exists():
        params = json.load(open(p))
    model = fcn_build_model("pontryagin", params)
    ck = torch.load(md / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def load_unet():
    md = UNET_RESULTS_DIR / "pontryagin"
    params = {}
    p = md / "hpo" / "best_params.json"
    if p.exists():
        params = json.load(open(p))
    model = unet_build_model("pontryagin", params)
    ck = torch.load(md / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


class _MCDropout(nn.Module):
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
        return F.interpolate(lo, size=x.shape[-2:], mode="bilinear", align_corners=False)


def mc_uncertainty(mc_model, inp, n=N_MC):
    mc_model.eval()
    mc_model.drop.train()
    plist = []
    with torch.no_grad():
        for _ in range(n):
            plist.append(torch.softmax(mc_model(inp), 1).cpu())
    pbar = torch.stack(plist).mean(0)
    h = -(pbar * (pbar + 1e-8).log()).sum(1)
    return h[0].numpy()


def embedding_energies(model, inp, class_idx=1):
    captured = {}
    def hook(m, a, o): captured["z"] = o
    h = model.embed_layer.register_forward_hook(hook)
    with torch.no_grad():
        model(inp)
    h.remove()
    z   = captured["z"]                          # (1, p+q, H', W')
    p   = model.embed_layer.rff.out_features
    phi_pos = z[:, :p]
    phi_neg = z[:, p:]
    W = model.head.W
    w_pos, w_neg = W[class_idx, :p], W[class_idx, p:]
    e_pos = (phi_pos * w_pos[None, :, None, None]).sum(1, keepdim=True)
    e_neg = (phi_neg * w_neg[None, :, None, None]).sum(1, keepdim=True)
    t = inp.shape[-2:]
    e_pos = F.interpolate(e_pos, t, mode="bilinear", align_corners=False)
    e_neg = F.interpolate(e_neg, t, mode="bilinear", align_corners=False)
    return e_pos[0, 0].detach().cpu().numpy(), e_neg[0, 0].detach().cpu().numpy()


# ── figure builder ────────────────────────────────────────────────────────────

def save_figure(rgb, gt_mask, uncertainty, e_pos, e_neg, out_path, n_mc=N_MC):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    # Panel 0: original + GT contour (white halo + green line for visibility)
    axes[0].imshow(rgb)
    axes[0].contour(gt_mask, levels=[0.5], colors=["white"],  linewidths=5)
    axes[0].contour(gt_mask, levels=[0.5], colors=["#00ff00"], linewidths=2.5)
    axes[0].set_xlabel("Original image\n(green: ground-truth mask)", fontsize=10)

    # Panel 1: MC Dropout entropy
    im1 = axes[1].imshow(_norm01(uncertainty), cmap="inferno", vmin=0, vmax=1)
    axes[1].set_xlabel(
        f"MC Dropout uncertainty\n(predictive entropy, N={n_mc})", fontsize=10
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 2: positive energy
    im2 = axes[2].imshow(_norm01(e_pos), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xlabel(
        r"Positive energy $\langle w_+, \varphi_+\rangle$" "\n(RFF — Euclidean)",
        fontsize=10,
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 3: negative energy
    im3 = axes[3].imshow(_norm01(e_neg), cmap="Reds", vmin=0, vmax=1)
    axes[3].set_xlabel(
        r"Negative energy $\langle w_-, \varphi_-\rangle$" "\n(SRF — hyperbolic)",
        fontsize=10,
    )
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")   # no GPU needed for inference on 13 test images

    dataset = UAVSegDataset(
        root=DATASET_ROOT,
        split="test",
        img_size=IMG_SIZE,
        augment=False,
        binary_sugarcane=True,
        only_sugarcane=False,
    )
    stem_to_idx = {Path(s).stem: i for i, s in enumerate(dataset.samples)}

    print("Loading FCN Pontryagin …")
    fcn_model   = load_fcn().to(device)
    fcn_mc      = _MCDropout(fcn_model).to(device)

    print("Loading UNet Pontryagin …")
    unet_model  = load_unet().to(device)
    unet_mc     = _MCDropout(unet_model).to(device)

    suffix_map = {TARGET_STEMS[0]: "ex1", TARGET_STEMS[1]: "ex2"}

    for stem, suffix in suffix_map.items():
        idx = stem_to_idx.get(stem)
        if idx is None:
            print(f"  [WARN] {stem} not found in test split — skipping")
            continue

        img_t, mask_t = dataset[idx]
        gt_mask = mask_t.numpy().astype(float)           # (H, W) 0/1
        inp     = img_t.unsqueeze(0).to(device)
        rgb     = _denormalize(img_t)

        print(f"\n── {stem} ({suffix}) ──")

        # FCN
        unc_fcn = mc_uncertainty(fcn_mc, inp)
        ep_fcn, en_fcn = embedding_energies(fcn_model, inp)
        save_figure(rgb, gt_mask, unc_fcn, ep_fcn, en_fcn,
                    OUT_DIR / f"pontryagin_fcn_energy_{suffix}.pdf")

        # UNet
        unc_unet = mc_uncertainty(unet_mc, inp)
        ep_unet, en_unet = embedding_energies(unet_model, inp)
        save_figure(rgb, gt_mask, unc_unet, ep_unet, en_unet,
                    OUT_DIR / f"pontryagin_unet_energy_{suffix}.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
