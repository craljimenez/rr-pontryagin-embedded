"""U-Net backbone segmentation experiments — PASCAL VOC 2012 (21 classes).

Public multiclass benchmark used to test PRFE generalisation beyond the
binary sugarcane-vs-background UAV task (see run_seg_uav_unet.py, the
"this work" result reported in the paper: U-Net+PRFE mIoU 0.878).

Four models share the same UNetBackbone and training protocol:

    vanilla     — UNet + standard CE (no class weights)  [baseline]
    euclidean   — UNet + class-weighted CE               [mirrors FCN euclidean]
    hyperbolic  — UNet + Poincaré-ball MLR
    pontryagin  — UNet + PontryaginEmbedding + PontryaginMLR
                  (balance-on-weights loss, no cone penalty)

VOID_LABEL=255 (unlabeled border pixels in the official VOC annotations) is
excluded from both the loss (remapped to -100, PyTorch's cross_entropy
default ignore_index) and the evaluation metrics (filtered out before
computing IoU/Dice so border predictions are never counted as false
positives).

Usage:
    python run_seg_pascalvoc.py --model pontryagin
    python run_seg_pascalvoc.py --model all
"""

import argparse
import csv
import json
import os
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.seg_pascalvoc import (
    AVAILABLE_MODELS, BACKBONE_OUT_CH, BATCH_SIZE, CLASS_NAMES,
    CLASS_WEIGHTS, DATASET_ROOT, DEVICE, D_POLY, EARLY_STOP_PATIENCE, EPOCHS,
    HYPERBOLIC_C, IMG_SIZE, KAPPA, LAMBDA_BALANCE, LAMBDA_TOPO, LR, LR_MIN,
    N_CLASSES, N_RFF, NUM_WORKERS, RESULTS_DIR, RFF_MULTIPLIER, SEED, SIGMA,
    TOPO_KWARGS, UNET_BASE_CH, UNET_DEPTH, VAL_TEST_SPLIT, VOC_YEAR,
    VOID_LABEL, WEIGHT_DECAY,
)

from prfe.layers.pontryagin import PontryaginEmbedding
from prfe.losses.mlr import PontryaginMLR
from prfe.losses.hyperbolic_mlr import HyperbolicMLR
from prfe.models.unet import UNetBackbone


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_to_tensor = T.ToTensor()
_normalise = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)


class PascalVOCSegDataset(Dataset):
    """PASCAL VOC 2012 semantic segmentation, resized to a fixed square.

    VOC 2012 ships a "train" (1464 imgs) and "val" (1449 imgs) split with
    public masks; the official test set's masks are withheld by the
    challenge server. We train on "train" and deterministically split
    "val" in half into val/test so both are drawn from the same
    distribution and neither ever touches training data.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        img_size: int = IMG_SIZE,
        augment: bool = False,
        year: str = VOC_YEAR,
        seed: int = SEED,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        self.img_size = img_size
        self.augment  = augment

        image_set = "train" if split == "train" else "val"
        self._voc = VOCSegmentation(
            root=str(root), year=year, image_set=image_set, download=True,
        )

        if split == "train":
            self.indices = list(range(len(self._voc)))
        else:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(self._voc))
            n_val = int(len(perm) * (1 - VAL_TEST_SPLIT))
            self.indices = (
                perm[:n_val].tolist() if split == "val" else perm[n_val:].tolist()
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, mask = self._voc[self.indices[idx]]
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img_t  = _normalise(_to_tensor(img))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        if self.augment and torch.rand(1) > 0.5:
            img_t  = img_t.flip(-1)
            mask_t = mask_t.flip(-1)

        return img_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Shared base class — same architecture family as run_seg_uav_unet.py
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


class VanillaUNet(_UNetSegBase):
    """Standard U-Net baseline: UNet backbone + 1×1 conv + CE."""

    def __init__(self, unet_depth: int = UNET_DEPTH) -> None:
        super().__init__(unet_depth=unet_depth)
        self.classifier = nn.Conv2d(BACKBONE_OUT_CH, N_CLASSES, kernel_size=1)

    def _embed(self, feats):
        return feats

    def _logits_flat(self, z_flat):
        B_HW, C = z_flat.shape
        return z_flat @ self.classifier.weight.view(N_CLASSES, C).T \
               + self.classifier.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.cross_entropy(logits, labels)


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


class PontryaginUNet(_UNetSegBase):
    """UNet backbone + Pontryagin space embedding + J-hyperplane MLR.

    Uses the balance-on-weights loss (no cone penalty) so that both the
    positive (RFF) and negative (SRF) subspaces receive non-zero gradients.
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
    root = Path(data_root) if data_root else DATASET_ROOT
    train_ds = PascalVOCSegDataset(root, "train", IMG_SIZE, augment=True)
    val_ds   = PascalVOCSegDataset(root, "val",   IMG_SIZE, augment=False)
    test_ds  = PascalVOCSegDataset(root, "test",  IMG_SIZE, augment=False)
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
# Metrics — per class (TP/FP/FN/IoU/Dice/Precision/Recall) + global summary.
# Identical convention to run_seg_uav.py's compute_metrics.
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return (a / b) if b > 0 else 0.0


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict:
    """Flat prediction / label tensors → full metrics bundle.

    Callers must have already filtered out VOID_LABEL pixels — this
    function assumes every remaining label lies in [0, n_classes).
    """
    total = int(labels.numel())
    per_class = []
    total_tp = total_fp = total_fn = 0

    for cls in range(n_classes):
        pred_cls = preds == cls
        lab_cls  = labels == cls
        tp = int((pred_cls & lab_cls).sum().item())
        fp = int((pred_cls & ~lab_cls).sum().item())
        fn = int((~pred_cls & lab_cls).sum().item())
        tn = total - tp - fp - fn
        support = tp + fn

        iou  = _safe_div(tp, tp + fp + fn)
        dice = _safe_div(2 * tp, 2 * tp + fp + fn)
        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)

        per_class.append({
            "class": CLASS_NAMES[cls],
            "support": support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": iou, "dice": dice,
            "precision": prec, "recall": rec,
            "specificity": spec,
        })
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_tn = (n_classes - 1) * total - total_fp - total_fn

    valid = [c for c in per_class if c["support"] > 0]
    n_valid = max(len(valid), 1)
    macro_iou  = sum(c["iou"]  for c in valid) / n_valid
    macro_dice = sum(c["dice"] for c in valid) / n_valid
    macro_prec = sum(c["precision"] for c in valid) / n_valid
    macro_rec  = sum(c["recall"]    for c in valid) / n_valid
    macro_spec = sum(c["specificity"] for c in valid) / n_valid

    micro_iou  = _safe_div(total_tp, total_tp + total_fp + total_fn)
    micro_dice = _safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn)

    pixel_acc = (preds == labels).float().mean().item()

    return {
        "per_class": per_class,
        "global": {
            "pixel_acc": pixel_acc,
            "macro_iou":  macro_iou,
            "macro_dice": macro_dice,
            "macro_precision": macro_prec,
            "macro_recall":    macro_rec,
            "macro_specificity": macro_spec,
            "micro_iou":  micro_iou,
            "micro_dice": micro_dice,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tn": total_tn,
            "n_valid_classes": len(valid),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loops
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(dim=1)
        preds_flat  = preds.reshape(-1)
        labels_flat = masks.reshape(-1)
        valid = labels_flat != VOID_LABEL
        all_preds.append(preds_flat[valid].cpu())
        all_labels.append(labels_flat[valid].cpu())
    return compute_metrics(
        torch.cat(all_preds), torch.cat(all_labels), N_CLASSES
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        # VOID_LABEL → -100, PyTorch's cross_entropy default ignore_index,
        # so border pixels contribute no gradient in any of the four heads.
        masks_for_loss = masks.masked_fill(masks == VOID_LABEL, -100)
        optimiser.zero_grad()
        loss = model.compute_loss(imgs, masks_for_loss)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — standard PASCAL VOC devkit colormap
# ─────────────────────────────────────────────────────────────────────────────

def _voc_colormap(n: int = 256) -> np.ndarray:
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


_PALETTE = _voc_colormap()
_PALETTE[VOID_LABEL] = [224, 224, 192]   # matches the official devkit's void colour


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[mask]


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
    present = set()
    for row, idx in enumerate(indices):
        img_t, mask_gt = dataset[idx]
        pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
        axes[row, 0].imshow(_denormalise(img_t))
        axes[row, 1].imshow(_mask_to_rgb(mask_gt.numpy()))
        axes[row, 2].imshow(_mask_to_rgb(pred))
        for ax in axes[row]:
            ax.axis("off")
        present |= set(np.unique(mask_gt.numpy())) | set(np.unique(pred))
    present -= {VOID_LABEL}
    present = sorted(present)
    patches = [mpatches.Patch(color=_PALETTE[c] / 255, label=CLASS_NAMES[c])
               for c in present if c < N_CLASSES]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Result persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_per_class_csv(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class", "support", "tp", "fp", "fn", "tn",
                  "iou", "dice", "precision", "recall", "specificity"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in metrics["per_class"]:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in row.items()})


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

    Returns the test metrics dict.
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
    _fd, _tmp_path = tempfile.mkstemp(suffix=".pth")
    os.close(_fd)
    tmp_ckpt = Path(_tmp_path)

    t0 = time.time()
    try:
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
    finally:
        if tmp_ckpt.exists():
            shutil.copy2(tmp_ckpt, final_ckpt_path)
            tmp_ckpt.unlink(missing_ok=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training time: {elapsed:.0f}s")

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
    p = argparse.ArgumentParser(description="Train UNet segmentation models on PASCAL VOC 2012")
    p.add_argument("--model", choices=list(AVAILABLE_MODELS) + ["all"],
                   default="pontryagin")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--params", type=str, default=None,
                   help="JSON string or path to best_params.json")
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to VOCdevkit root (downloads via torchvision if omitted).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for results (overrides config).")
    return p.parse_args()


def main():
    args        = _parse()
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    env_root  = os.environ.get("PASCALVOC_DATA_ROOT")
    data_root = args.data_root or env_root or None
    data      = build_dataloaders(data_root=data_root)
    print(f"Train: {len(data['train_ds'])} | Val: {len(data['val_ds'])} "
          f"| Test: {len(data['test_ds'])}")

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
                print(f"  Loaded HPO params for {mt}: {hpo_path}")

        train_one_model(
            mt, params=params, data=data, device=device,
            epochs=args.epochs, out_dir=results_dir / mt, verbose=True,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
