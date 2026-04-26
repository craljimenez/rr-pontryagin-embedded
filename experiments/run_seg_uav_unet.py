"""U-Net backbone segmentation experiments — UAV sugarcane dataset.

Four models share the same UNetBackbone and training protocol:

    vanilla     — UNet + standard CE (no class weights)  [baseline]
    euclidean   — UNet + class-weighted CE               [mirrors FCN euclidean]
    hyperbolic  — UNet + Poincaré-ball MLR
    pontryagin  — UNet + PontryaginEmbedding + PontryaginMLR
                  (balance-on-weights loss, no cone penalty)

Usage:
    python run_seg_uav_unet.py --model pontryagin
    python run_seg_uav_unet.py --model all
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.seg_uav_unet import (
    AVAILABLE_MODELS, BACKBONE_OUT_CH, BATCH_SIZE, BINARY_ONLY_SUGARCANE,
    BINARY_SUGARCANE, CLASS_NAMES, CLASS_WEIGHTS, DATASET_ROOT, DEVICE,
    D_POLY, EARLY_STOP_PATIENCE, EPOCHS, HYPERBOLIC_C, IMG_SIZE, KAPPA,
    LAMBDA_BALANCE, LAMBDA_TOPO, LR, LR_MIN, N_CLASSES, N_RFF, NUM_WORKERS,
    RESULTS_DIR, RFF_MULTIPLIER, SIGMA, TOPO_KWARGS, UNET_BASE_CH,
    UNET_DEPTH, WEIGHT_DECAY,
)

# Reuse dataset + metrics + loops from the FCN experiment
from run_seg_uav import (
    UAVSegDataset, compute_metrics, evaluate, train_epoch,
    save_per_class_csv,
)

from prfe.layers.pontryagin import PontryaginEmbedding
from prfe.losses.mlr import PontryaginMLR
from prfe.losses.hyperbolic_mlr import HyperbolicMLR
from prfe.models.unet import UNetBackbone


# ─────────────────────────────────────────────────────────────────────────────
# Shared base class
# ─────────────────────────────────────────────────────────────────────────────

class _UNetSegBase(nn.Module):
    """Common interface for all four UNet models.

    forward(x)          → logits (B, N_CLASSES, H, W)  at input resolution
    compute_loss(x, y)  → scalar training loss
    """

    def __init__(self, unet_depth: int = UNET_DEPTH) -> None:
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=3,
            out_channels=BACKBONE_OUT_CH,
            base_ch=UNET_BASE_CH,
            depth=unet_depth,
        )

    def _embed(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _logits_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _loss_spatial(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)                           # (B, C, H, W)
        z     = self._embed(feats)                         # (B, D, H, W)
        B, D, H, W = z.shape
        z_flat     = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        logits     = self._logits_flat(z_flat).reshape(B, H, W, N_CLASSES)
        return logits.permute(0, 3, 1, 2)                  # (B, N_CLASSES, H, W)

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        z     = self._embed(feats)
        return self._loss_spatial(z, labels)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vanilla UNet  — standard CE, no class weights
# ─────────────────────────────────────────────────────────────────────────────

class VanillaUNet(_UNetSegBase):
    """Standard U-Net baseline: UNet backbone + 1×1 conv + CE."""

    def __init__(self, unet_depth: int = UNET_DEPTH) -> None:
        super().__init__(unet_depth=unet_depth)
        self.classifier = nn.Conv2d(BACKBONE_OUT_CH, N_CLASSES, kernel_size=1)

    def _embed(self, feats):
        return feats

    def _logits_flat(self, z_flat):
        # unused — forward is overridden below for efficiency
        B_HW, C = z_flat.shape
        return z_flat @ self.classifier.weight.view(N_CLASSES, C).T \
               + self.classifier.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Euclidean UNet  — class-weighted CE (mirrors FCN euclidean)
# ─────────────────────────────────────────────────────────────────────────────

class EuclideanUNet(_UNetSegBase):
    """UNet backbone + class-weighted linear head."""

    def __init__(self, unet_depth: int = UNET_DEPTH) -> None:
        super().__init__(unet_depth=unet_depth)
        self.classifier = nn.Conv2d(BACKBONE_OUT_CH, N_CLASSES, kernel_size=1)
        self.register_buffer(
            "class_weights", torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
        )

    def _embed(self, feats):
        return feats

    def _logits_flat(self, z_flat):
        B_HW, C = z_flat.shape
        return z_flat @ self.classifier.weight.view(N_CLASSES, C).T \
               + self.classifier.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.forward(x), labels, weight=self.class_weights)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperbolic UNet  — UNet + Poincaré-ball MLR
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicUNet(_UNetSegBase):
    """UNet backbone + Hyperbolic MLR head (Poincaré ball)."""

    def __init__(self, c: float = HYPERBOLIC_C, unet_depth: int = UNET_DEPTH) -> None:
        super().__init__(unet_depth=unet_depth)
        self.head = HyperbolicMLR(
            n_classes=N_CLASSES,
            in_features=BACKBONE_OUT_CH,
            c=c,
            class_weights=torch.tensor(CLASS_WEIGHTS),
        )

    def _embed(self, feats):
        return feats

    def _logits_flat(self, z_flat):
        return self.head.logits(z_flat)

    def _loss_spatial(self, z, labels):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        return self.head(z_flat, labels.reshape(B * H * W))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pontryagin UNet  — UNet + PontryaginEmbedding + PontryaginMLR
# ─────────────────────────────────────────────────────────────────────────────

class PontryaginUNet(_UNetSegBase):
    """UNet backbone + Pontryagin space embedding + J-hyperplane MLR.

    Uses the new balance-on-weights loss (no cone penalty) so that both
    the positive (RFF) and negative (SRF) subspaces receive non-zero gradients.
    """

    def __init__(
        self,
        kappa: int = KAPPA,
        d_poly: int = D_POLY,
        rff_multiplier: int = RFF_MULTIPLIER,
        sigma: float = SIGMA,
        lambda_topo: float = LAMBDA_TOPO,
        lambda_balance: float = LAMBDA_BALANCE,
        unet_depth: int = UNET_DEPTH,
    ) -> None:
        super().__init__(unet_depth=unet_depth)
        n_rff = int(rff_multiplier) * BACKBONE_OUT_CH
        n_srf = int(kappa)
        self.kappa = kappa
        self.d_poly = d_poly
        self.embed_layer = PontryaginEmbedding(
            BACKBONE_OUT_CH, n_rff, n_srf, int(kappa),
            d_poly=int(d_poly), sigma=sigma,
        )
        self.head = PontryaginMLR(
            N_CLASSES,
            self.embed_layer.out_channels,
            self.embed_layer.J,
            class_weights=torch.tensor(CLASS_WEIGHTS),
            lambda_topo=lambda_topo,
            lambda_balance=lambda_balance,
            topo_kwargs=TOPO_KWARGS,
        )

    def _embed(self, feats):
        return self.embed_layer(feats)

    def _logits_flat(self, z_flat):
        return self.head.logits(z_flat)

    def _loss_spatial(self, z, labels):
        return self.head.forward_spatial(z, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_unet_model(model_type: str, params: dict | None = None) -> _UNetSegBase:
    """Instantiate a UNet segmentation model.

    Recognised params keys:
        common      : unet_depth
        hyperbolic  : hyperbolic_c
        pontryagin  : kappa, d_poly, rff_multiplier, sigma,
                      lambda_topo, lambda_balance
    """
    params = params or {}
    unet_depth = params.get("unet_depth", UNET_DEPTH)
    if model_type == "vanilla":
        return VanillaUNet(unet_depth=unet_depth)
    if model_type == "euclidean":
        return EuclideanUNet(unet_depth=unet_depth)
    if model_type == "hyperbolic":
        return HyperbolicUNet(
            c=params.get("hyperbolic_c", HYPERBOLIC_C),
            unet_depth=unet_depth,
        )
    if model_type == "pontryagin":
        return PontryaginUNet(
            kappa=params.get("kappa", KAPPA),
            d_poly=params.get("d_poly", D_POLY),
            rff_multiplier=params.get("rff_multiplier", RFF_MULTIPLIER),
            sigma=params.get("sigma", SIGMA),
            lambda_topo=params.get("lambda_topo", LAMBDA_TOPO),
            lambda_balance=params.get("lambda_balance", LAMBDA_BALANCE),
            unet_depth=unet_depth,
        )
    raise ValueError(f"Unknown model type: {model_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(data_root=None):
    kw = dict(
        root=Path(data_root) if data_root else DATASET_ROOT,
        img_size=IMG_SIZE,
        binary_sugarcane=BINARY_SUGARCANE,
        only_sugarcane=BINARY_ONLY_SUGARCANE,
    )
    train_ds = UAVSegDataset(split="train", augment=True,  **kw)
    val_ds   = UAVSegDataset(split="valid", augment=False, **kw)
    test_ds  = UAVSegDataset(split="test",  augment=False, **kw)
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=True)
    return {
        "train_ds": train_ds,
        "val_ds":   val_ds,
        "test_ds":  test_ds,
        "train_dl": DataLoader(train_ds, shuffle=True,  **loader_kw),
        "val_dl":   DataLoader(val_ds,   shuffle=False, **loader_kw),
        "test_dl":  DataLoader(test_ds,  shuffle=False, **loader_kw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (same palette as FCN experiment)
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = np.array([
    [  0,   0,   0],
    [255, 165,   0],
], dtype=np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[mask.clip(0, len(_PALETTE) - 1)]


def _denormalise(t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = t.cpu().permute(1, 2, 0).numpy()
    return ((img * std + mean).clip(0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def save_validation_grid(model, dataset, device, out_path, n_samples=6):
    model.eval()
    n_samples = min(n_samples, len(dataset))
    indices   = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples),
                             squeeze=False)
    for ax, title in zip(axes[0], ["Image", "Ground Truth", "Prediction"]):
        ax.set_title(title, fontsize=13, fontweight="bold")
    for row, idx in enumerate(indices):
        img_t, mask_gt = dataset[idx]
        pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
        axes[row, 0].imshow(_denormalise(img_t))
        axes[row, 1].imshow(_mask_to_rgb(mask_gt.numpy()))
        axes[row, 2].imshow(_mask_to_rgb(pred))
        for ax in axes[row]: ax.axis("off")
    patches = [mpatches.Patch(color=_PALETTE[c] / 255, label=CLASS_NAMES[c])
               for c in range(N_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_model(
    model_type: str,
    params: dict | None = None,
    data: dict | None = None,
    device: torch.device | None = None,
    epochs: int = EPOCHS,
    out_dir: Path | None = None,
    verbose: bool = True,
    save_viz: bool = True,
) -> dict:
    """Train one UNet model variant and save results.

    Returns the best validation metrics dict.
    """
    params   = params or {}
    device   = device or torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    data     = data or build_dataloaders()
    out_dir  = out_dir or (RESULTS_DIR / model_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir  = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    model = build_unet_model(model_type, params).to(device)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[{model_type.upper()} UNet] trainable params: {n_params:,}")

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=params.get("lr", LR),
        weight_decay=params.get("weight_decay", WEIGHT_DECAY),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=LR_MIN
    )

    best_val_iou   = -1.0
    best_val_metrics = {}
    patience_cnt   = 0
    final_ckpt_path = out_dir / "best_model.pth"
    # Write checkpoints to local /tmp to avoid partial writes on Drive
    tmp_ckpt = Path(tempfile.mktemp(suffix=".pth"))

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, data["train_dl"], optimiser, device)
        val_m   = evaluate(model, data["val_dl"], device)
        val_iou = val_m["global"]["macro_iou"]
        scheduler.step()

        if val_iou > best_val_iou:
            best_val_iou     = val_iou
            best_val_metrics = val_m
            patience_cnt     = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "model_type": model_type,
                "val_metrics": val_m["global"],
                "params": params,
            }, tmp_ckpt)
        else:
            patience_cnt += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  ep {epoch:3d}  loss={tr_loss:.4f}  "
                  f"val_mIoU={val_iou:.4f}  best={best_val_iou:.4f}")

        if patience_cnt >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training time: {elapsed:.0f}s")

    # Copy completed checkpoint from /tmp to final destination (Drive)
    shutil.copy2(tmp_ckpt, final_ckpt_path)
    tmp_ckpt.unlink(missing_ok=True)

    # ── Test evaluation ──────────────────────────────────────────────────────
    ckpt = torch.load(final_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, data["test_dl"], device)

    # ── Save artefacts ───────────────────────────────────────────────────────
    with open(out_dir / "metrics_detailed.json", "w") as f:
        json.dump(test_m, f, indent=2)
    save_per_class_csv(test_m, out_dir / "metrics_per_class.csv")

    summary = {**test_m["global"], "model_type": model_type,
               "best_val_macro_iou": best_val_iou}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        g = test_m["global"]
        print(f"\n  Test  mIoU={g['macro_iou']:.4f}  "
              f"pixel_acc={g['pixel_acc']:.4f}  "
              f"macro_dice={g['macro_dice']:.4f}")

    if save_viz:
        save_validation_grid(
            model, data["test_ds"], device,
            out_path=viz_dir / "test_final.png", n_samples=8,
        )

    return test_m


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Train UNet segmentation models")
    p.add_argument("--model", choices=list(AVAILABLE_MODELS) + ["all"],
                   default="pontryagin")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--params", type=str, default=None,
                   help="JSON string or path to best_params.json")
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to UAV_segmantation dataset root (overrides config).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for results (overrides config).")
    return p.parse_args()


def main():
    args        = _parse()
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    data        = build_dataloaders(data_root=args.data_root)

    models = list(AVAILABLE_MODELS) if args.model == "all" else [args.model]

    for mt in models:
        params = {}
        if args.params:
            src = Path(args.params)
            if src.exists():
                with open(src) as f:
                    params = json.load(f)
            else:
                params = json.loads(args.params)
        else:
            hpo_path = results_dir / mt / "hpo" / "best_params.json"
            if hpo_path.exists():
                with open(hpo_path) as f:
                    params = json.load(f)

        train_one_model(
            mt, params=params, data=data, device=device,
            epochs=args.epochs, out_dir=results_dir / mt, verbose=True,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
