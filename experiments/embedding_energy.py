"""Embedding energy + MC Dropout uncertainty for the Pontryagin model.

For every test image produces a 1×4 PDF saved to
  results/seg_uav/pontryagin/embedding_energy/<stem>.pdf

Panels (left to right):
  1. Original RGB image
  2. MC Dropout predictive entropy  (N_MC forward passes, Dropout2d injected
     after backbone, model BatchNorm kept in eval mode)
  3. Positive-subspace energy  E_+(h,w) = ||φ_+(x_{hw})||²  (RFF — Euclidean)
  4. Negative-subspace energy  E_-(h,w) = ||φ_-(x_{hw})||²  (SRF — hyperbolic)

E_+ and E_- are normalised independently to [0,1] per image.
Predictive entropy is normalised to [0,1] per image for display.
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

from configs.seg_uav import (
    DATASET_ROOT, DEVICE, IMG_SIZE, RESULTS_DIR,
)
from run_seg_uav import UAVSegDataset, build_model

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_TYPE = "pontryagin"
N_MC       = 30     # MC Dropout passes
P_DROP     = 0.25   # Dropout2d probability injected after backbone
IMG_MEAN   = torch.tensor([0.485, 0.456, 0.406])
IMG_STD    = torch.tensor([0.229, 0.224, 0.225])

OUT_DIR = RESULTS_DIR / MODEL_TYPE / "embedding_energy"


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model(results_dir=None):
    model_dir  = (Path(results_dir) if results_dir else RESULTS_DIR) / MODEL_TYPE
    ckpt_path  = model_dir / "best_model.pth"
    params_path = model_dir / "hpo" / "best_params.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
    model = build_model(MODEL_TYPE, params)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── MC Dropout wrapper ────────────────────────────────────────────────────────

class _MCDropoutSeg(nn.Module):
    """Inserts Dropout2d after the backbone; BN stats are frozen (eval mode)."""

    def __init__(self, seg_model: nn.Module, p: float = P_DROP) -> None:
        super().__init__()
        self.seg   = seg_model
        self.drop  = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats      = self.seg.backbone(x)           # (B, C, H', W')
        feats      = self.drop(feats)               # stochastic channel zeroing
        z          = self.seg._embed(feats)         # (B, p+q, H', W')
        B, D, Hf, Wf = z.shape
        z_flat     = z.permute(0, 2, 3, 1).reshape(B * Hf * Wf, D)
        logits_low = self.seg._logits_flat(z_flat).reshape(B, Hf, Wf, -1)
        logits_low = logits_low.permute(0, 3, 1, 2)
        return F.interpolate(logits_low, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


def _mc_uncertainty(mc_model: nn.Module,
                    inp: torch.Tensor,
                    n: int = N_MC) -> np.ndarray:
    """Return predictive entropy map (H, W), unnormalised, in nats."""
    mc_model.eval()
    mc_model.drop.train()   # keep Dropout stochastic
    probs_list = []
    with torch.no_grad():
        for _ in range(n):
            logits = mc_model(inp)                   # (1, C, H, W)
            probs  = torch.softmax(logits, dim=1)    # (1, C, H, W)
            probs_list.append(probs.cpu())
    p_bar = torch.stack(probs_list, dim=0).mean(dim=0)   # (1, C, H, W)
    eps   = 1e-8
    h     = -(p_bar * (p_bar + eps).log()).sum(dim=1)    # (1, H, W)
    return h[0].numpy()                                  # (H, W)


# ── embedding energy extraction ───────────────────────────────────────────────

def _embedding_energies(model: nn.Module,
                        inp: torch.Tensor,
                        class_idx: int = 1):
    """Return (E_pos, E_neg) each of shape (H, W) as numpy arrays.

    Following the Krein-space convention (TG-NerveUTP):
      E_+(h,w) = Σ_{d<p}  W_{c,d} · φ_+(x_{hw})   — weighted RFF contribution
      E_-(h,w) = Σ_{d≥p}  W_{c,d} · φ_-(x_{hw})   — weighted SRF contribution

    Both are the per-pixel dot products of the classifier weight vector (for
    class `class_idx`) with the corresponding feature sub-vector, giving the
    spatial distribution of logit contributions from each subspace.
    """
    captured = {}

    def _hook(module, args, output):
        captured["z"] = output      # (B, p+q, H', W')

    handle = model.embed_layer.register_forward_hook(_hook)
    with torch.no_grad():
        model(inp)
    handle.remove()

    z   = captured["z"]                             # (1, p+q, H', W')
    p   = model.embed_layer.rff.out_features        # 2 * n_rff
    phi_pos = z[:, :p, :, :]                        # (1, p, H', W')
    phi_neg = z[:, p:, :, :]                        # (1, q, H', W')

    # Classifier weight vector for the target class
    W = model.head.W                                # (n_classes, p+q)
    w_pos = W[class_idx, :p]                        # (p,)
    w_neg = W[class_idx, p:]                        # (q,)

    # Weighted inner product per pixel
    e_pos_lr = (phi_pos * w_pos[None, :, None, None]).sum(dim=1, keepdim=True)  # (1,1,H',W')
    e_neg_lr = (phi_neg * w_neg[None, :, None, None]).sum(dim=1, keepdim=True)  # (1,1,H',W')

    # Upsample to input resolution
    target = inp.shape[-2:]
    e_pos = F.interpolate(e_pos_lr, size=target, mode="bilinear", align_corners=False)
    e_neg = F.interpolate(e_neg_lr, size=target, mode="bilinear", align_corners=False)
    return e_pos[0, 0].detach().cpu().numpy(), e_neg[0, 0].detach().cpu().numpy()


# ── visualisation ─────────────────────────────────────────────────────────────

def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _denormalize(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) → (H, W, 3) uint8 RGB."""
    img = t.cpu().clone()
    img = img * IMG_STD[:, None, None] + IMG_MEAN[:, None, None]
    img = img.permute(1, 2, 0).numpy().clip(0, 1)
    return (img * 255).astype(np.uint8)


def _save_figure(rgb: np.ndarray,
                 uncertainty: np.ndarray,
                 e_pos: np.ndarray,
                 e_neg: np.ndarray,
                 out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    axes[0].imshow(rgb)
    axes[0].set_xlabel("Original image", fontsize=10)

    im1 = axes[1].imshow(_norm01(uncertainty), cmap="inferno", vmin=0, vmax=1)
    axes[1].set_xlabel(
        f"MC Dropout uncertainty\n(predictive entropy, N={N_MC})", fontsize=10
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(_norm01(e_pos), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xlabel(
        r"Positive-subspace energy $\langle w_+, \varphi_+\rangle$"
        "\n(RFF — Euclidean logit contribution)",
        fontsize=10,
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(_norm01(e_neg), cmap="Reds", vmin=0, vmax=1)
    axes[3].set_xlabel(
        r"Negative-subspace energy $\langle w_-, \varphi_-\rangle$"
        "\n(SRF — hyperbolic logit contribution)",
        fontsize=10,
    )
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--data-root",   type=str, default=None,
                   help="Path to UAV_segmantation dataset root (overrides config).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for results (overrides config).")
    args = p.parse_args()

    data_root   = Path(args.data_root)   if args.data_root   else DATASET_ROOT
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    out_dir     = results_dir / MODEL_TYPE / "embedding_energy"

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print("Loading Pontryagin model …")
    model = _load_model(results_dir=results_dir).to(device)

    mc_model = _MCDropoutSeg(model, p=P_DROP).to(device)

    dataset = UAVSegDataset(
        root=data_root,
        split="test",
        img_size=IMG_SIZE,
        augment=False,
        binary_sugarcane=True,
        only_sugarcane=False,
    )
    print(f"Test images: {len(dataset)}")

    for idx in range(len(dataset)):
        img_t, _ = dataset[idx]
        stem      = Path(dataset.samples[idx]).stem
        inp       = img_t.unsqueeze(0).to(device)   # (1, 3, H, W)

        uncertainty = _mc_uncertainty(mc_model, inp)
        e_pos, e_neg = _embedding_energies(model, inp)
        rgb          = _denormalize(img_t)

        out_path = out_dir / f"{stem}.pdf"
        _save_figure(rgb, uncertainty, e_pos, e_neg, out_path)
        print(f"  [{idx+1:2d}/{len(dataset)}] {stem}.pdf")

    print(f"\nDone — PDFs saved to {out_dir}")


if __name__ == "__main__":
    main()
