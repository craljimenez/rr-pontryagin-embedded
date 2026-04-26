"""SegScoreCAM interpretability analysis for UNet segmentation models.

For each model (vanilla / euclidean / hyperbolic / pontryagin) and each test
image produces a 1×4 PDF:
  [original + background heatmap] [original + sugarcane heatmap]
  [predicted mask] [ground-truth mask]

Additionally, for the Pontryagin model only, also generates the embedding
energy + MC Dropout uncertainty figure (1×4 PDF) in the same output directory.

SegScoreCAM target layer: last decoder double-conv block of UNetBackbone
  → model.backbone.dec_blocks[-1]

Metrics saved per model to:
  results/seg_uav_unet/<model>/gradcam/metrics.json

Usage:
    python interpretability_unet.py --model all
    python interpretability_unet.py --model pontryagin
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.seg_uav_unet import (
    AVAILABLE_MODELS, DATASET_ROOT, DEVICE, IMG_SIZE, RESULTS_DIR,
)
from run_seg_uav_unet import UAVSegDataset, build_unet_model

# ── Reuse metric helpers from FCN interpretability ───────────────────────────
from interpretability_analysis import (
    _otsu_threshold, compute_cam_metrics, compute_confidence_gain,
)

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])
N_PASSES_MC = 30
P_DROP      = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_unet_model(model_type: str, results_dir=None):
    model_dir  = (Path(results_dir) if results_dir else RESULTS_DIR) / model_type
    ckpt_path  = model_dir / "best_model.pth"
    params_path = model_dir / "hpo" / "best_params.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}")
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
    model = build_unet_model(model_type, params)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError) as e:
        raise FileNotFoundError(
            f"Checkpoint for {model_type} is corrupt or incomplete "
            f"({ckpt_path}): {e}"
        ) from e
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, params


# ─────────────────────────────────────────────────────────────────────────────
# SegScoreCAM
# ─────────────────────────────────────────────────────────────────────────────

class SegScoreCAM:
    """ScoreCAM adapted for segmentation (pixel-level softmax score)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._feature_maps = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, *args):
        self._feature_maps = args[-1]

    def remove(self):
        self._handle.remove()

    def generate_cam(self, input_tensor: torch.Tensor,
                     class_idx: int) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        feature_maps = self._feature_maps.detach()   # (1, C, H', W')
        _, n_ch, fh, fw = feature_maps.shape

        scores = []
        for k in range(n_ch):
            fm = feature_maps[:, k:k+1, :, :]        # (1, 1, H', W')
            # Normalise to [0,1]
            lo, hi = fm.min(), fm.max()
            if hi - lo > 1e-8:
                fm = (fm - lo) / (hi - lo)
            mask_k = F.interpolate(fm, size=input_tensor.shape[-2:],
                                   mode="bilinear", align_corners=False)
            with torch.no_grad():
                out = self.model(input_tensor * mask_k)
                score = torch.softmax(out, dim=1)[:, class_idx].mean().item()
            scores.append(score)

        scores_t = torch.tensor(scores, dtype=torch.float32,
                                device=feature_maps.device)
        # Raw scores preserve class discrimination; F.softmax collapses them to
        # near-uniform weights for binary segmentation (P(bg)+P(sg)≈1 per channel).
        total = scores_t.sum()
        weights = scores_t / total if total > 1e-8 else torch.ones_like(scores_t) / n_ch

        cam = (feature_maps[0] * weights[:, None, None]).sum(dim=0)  # (H', W')
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode="bilinear", align_corners=False,
        )[0, 0].cpu().numpy()

        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        return cam


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout uncertainty (Pontryagin only)
# ─────────────────────────────────────────────────────────────────────────────

class _MCDropoutUNet(nn.Module):
    def __init__(self, seg_model, p=P_DROP):
        super().__init__()
        self.seg  = seg_model
        self.drop = nn.Dropout2d(p=p)

    def forward(self, x):
        feats = self.seg.backbone(x)
        feats = self.drop(feats)
        z     = self.seg._embed(feats)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        logits = self.seg._logits_flat(z_flat).reshape(B, H, W, -1)
        return logits.permute(0, 3, 1, 2)


def mc_uncertainty(mc_model, inp, n=N_PASSES_MC):
    mc_model.eval()
    mc_model.drop.train()
    probs = []
    with torch.no_grad():
        for _ in range(n):
            p = torch.softmax(mc_model(inp), dim=1).cpu()
            probs.append(p)
    p_bar = torch.stack(probs).mean(0)
    eps = 1e-8
    h = -(p_bar * (p_bar + eps).log()).sum(dim=1)
    return h[0].numpy()


def embedding_energies(model, inp, class_idx=1):
    captured = {}
    handle = model.embed_layer.register_forward_hook(
        lambda *a: captured.__setitem__("z", a[-1])
    )
    with torch.no_grad():
        model(inp)
    handle.remove()
    z   = captured["z"]
    p   = model.embed_layer.rff.out_features
    W   = model.head.W
    phi_pos = z[:, :p]
    phi_neg = z[:, p:]
    w_pos = W[class_idx, :p]
    w_neg = W[class_idx, p:]
    e_pos = (phi_pos * w_pos[None, :, None, None]).sum(1, keepdim=True)
    e_neg = (phi_neg * w_neg[None, :, None, None]).sum(1, keepdim=True)
    target = inp.shape[-2:]
    e_pos = F.interpolate(e_pos, size=target, mode="bilinear", align_corners=False)
    e_neg = F.interpolate(e_neg, size=target, mode="bilinear", align_corners=False)
    return e_pos[0, 0].detach().cpu().numpy(), e_neg[0, 0].detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm01(a):
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-8 else (a - lo) / (hi - lo)


def _denorm(t):
    img = (t.cpu() * IMG_STD[:, None, None] + IMG_MEAN[:, None, None])
    return img.permute(1, 2, 0).numpy().clip(0, 1)


def _overlay(rgb, cam, alpha=0.5, cmap="jet"):
    heatmap = plt.colormaps[cmap](cam)[..., :3]
    return (alpha * heatmap + (1 - alpha) * rgb).clip(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_model(model_type: str, device: torch.device,
                  data_root=None, results_dir=None):
    print(f"\n── {model_type.upper()} UNet ──────────────────────────────")
    _res = Path(results_dir) if results_dir else RESULTS_DIR
    model, _ = load_unet_model(model_type, results_dir=_res)
    model     = model.to(device)

    target_layer = model.backbone.dec_blocks[-1]
    ssc = SegScoreCAM(model, target_layer)

    dataset = UAVSegDataset(
        root=Path(data_root) if data_root else DATASET_ROOT,
        split="test", img_size=IMG_SIZE,
        augment=False, binary_sugarcane=True, only_sugarcane=False,
    )

    out_dir = _res / model_type / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    mc_model = None
    if model_type == "pontryagin":
        mc_model = _MCDropoutUNet(model, p=P_DROP).to(device)
        energy_dir = _res / model_type / "embedding_energy"
        energy_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    CLASS_NAMES_LOCAL = ["background", "Sugarcane"]

    for idx in range(len(dataset)):
        img_t, mask_t = dataset[idx]
        stem = dataset.samples[idx]
        inp  = img_t.unsqueeze(0).to(device)
        gt   = mask_t.numpy()

        with torch.no_grad():
            pred = model(inp).argmax(dim=1)[0].cpu().numpy()

        rgb = _denorm(img_t)

        cam_bg = ssc.generate_cam(inp, class_idx=0)
        cam_sg = ssc.generate_cam(inp, class_idx=1)

        # ── 1×4 SegScoreCAM figure ──────────────────────────────────────────
        gt_bin = (gt == 1).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        axes[0].imshow(_overlay(rgb, cam_bg))
        axes[0].set_xlabel("Background heatmap (SegScoreCAM)", fontsize=10)
        axes[1].imshow(_overlay(rgb, cam_sg))
        axes[1].set_xlabel("Sugarcane heatmap (SegScoreCAM)", fontsize=10)
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2].set_xlabel("Predicted mask", fontsize=10)
        axes[3].imshow(gt_bin, cmap="gray", vmin=0, vmax=1)
        axes[3].set_xlabel("Ground-truth mask", fontsize=10)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}.pdf", dpi=150, format="pdf",
                    bbox_inches="tight")
        plt.close(fig)

        # ── Pontryagin: energy + MC Dropout figure ──────────────────────────
        if mc_model is not None:
            unc = mc_uncertainty(mc_model, inp)
            e_pos, e_neg = embedding_energies(model, inp)

            fig2, axes2 = plt.subplots(1, 4, figsize=(17, 4.2))
            axes2[0].imshow(rgb)
            axes2[0].set_xlabel("Original image", fontsize=10)
            im1 = axes2[1].imshow(_norm01(unc), cmap="inferno", vmin=0, vmax=1)
            axes2[1].set_xlabel(f"MC Dropout uncertainty\n(predictive entropy, N={N_PASSES_MC})", fontsize=10)
            plt.colorbar(im1, ax=axes2[1], fraction=0.046, pad=0.04)
            im2 = axes2[2].imshow(_norm01(e_pos), cmap="Blues", vmin=0, vmax=1)
            axes2[2].set_xlabel(r"Positive energy $\langle w_+,\varphi_+\rangle$" "\n(RFF — Euclidean)", fontsize=10)
            plt.colorbar(im2, ax=axes2[2], fraction=0.046, pad=0.04)
            im3 = axes2[3].imshow(_norm01(e_neg), cmap="Reds", vmin=0, vmax=1)
            axes2[3].set_xlabel(r"Negative energy $\langle w_-,\varphi_-\rangle$" "\n(SRF — hyperbolic)", fontsize=10)
            plt.colorbar(im3, ax=axes2[3], fraction=0.046, pad=0.04)
            for ax in axes2:
                ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            fig2.savefig(energy_dir / f"{stem}.pdf", dpi=150, format="pdf",
                         bbox_inches="tight")
            plt.close(fig2)

        # ── Metrics ─────────────────────────────────────────────────────────
        m_sg = compute_cam_metrics(cam_sg, gt_mask=gt_bin)
        cg   = compute_confidence_gain(model, inp, cam_sg, class_idx=1,
                                        gt_mask=gt_bin)
        all_metrics[stem] = {
            "background": compute_cam_metrics(cam_bg),
            "sugarcane":  {**m_sg, "confidence_gain": cg},
        }
        print(f"  [{idx+1:2d}/{len(dataset)}] {stem}  "
              f"iou={m_sg.get('gt_iou', float('nan')):.3f}  "
              f"prec={m_sg.get('gt_precision', float('nan')):.3f}")

    ssc.remove()
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  metrics.json → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(AVAILABLE_MODELS) + ["all"],
                    default="all")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--data-root",   type=str, default=None,
                    help="Path to UAV_segmantation dataset root (overrides config).")
    ap.add_argument("--results-dir", type=str, default=None,
                    help="Output directory for results (overrides config).")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = list(AVAILABLE_MODELS) if args.model == "all" else [args.model]
    for mt in models:
        try:
            analyse_model(mt, device,
                          data_root=args.data_root,
                          results_dir=args.results_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {mt}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
