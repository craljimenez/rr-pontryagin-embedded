"""Semantic segmentation on the UAV dataset — three comparable heads.

    python experiments/run_seg_uav.py --model euclidean    # classical FCN
    python experiments/run_seg_uav.py --model hyperbolic   # Ganea/Atigh
    python experiments/run_seg_uav.py --model pontryagin   # this work

All three share the same deep FCN backbone, dataset, augmentations, optimiser,
cosine LR schedule, and early-stopping policy — only the segmentation head
changes. Backbone depth is also tunable as a common hyperparameter.
Results are saved under experiments/results/seg_uav/<model>/:

    metrics.csv             — per-epoch train loss + validation global metrics
    metrics_per_class.csv   — per-class IoU/Dice/TP/FP/FN at the best checkpoint
    metrics_detailed.json   — full per-class + global metrics (test set)
    best_model.pth          — weights with highest val macro mIoU
    viz/*.png               — image / GT / prediction panels
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ── make the prfe package importable without installing ───────────────────────
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from prfe.layers import PoincareEmbedding, PontryaginEmbedding
from prfe.losses import HyperbolicMLR, PontryaginMLR

from configs.seg_uav import (
    AVAILABLE_MODELS, BACKBONE_OUT_CH, BACKBONE_DEPTH, BATCH_SIZE,
    BINARY_ONLY_SUGARCANE, BINARY_SUGARCANE, CLASS_NAMES, CLASS_WEIGHTS,
    DATASET_ROOT, DEVICE, D_POLY, EARLY_STOP_PATIENCE, EPOCHS, HYPERBOLIC_C,
    IMG_SIZE, KAPPA, LAMBDA_BALANCE, LAMBDA_CONE_W, LAMBDA_TOPO, LR, LR_MIN,
    N_CLASSES, N_RFF, NUM_WORKERS, RESULTS_DIR, RFF_MULTIPLIER, SIGMA,
    TOPO_KWARGS, WEIGHT_DECAY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _rasterise_yolo_seg(label_path: Path, w: int, h: int) -> np.ndarray:
    """YOLO polygon label → integer mask (H, W), 0=background, 1..9=class.

    Pixels not covered by any polygon stay 0, representing the implicit
    background class — which the model learns and is evaluated on.
    """
    mask = np.zeros((h, w), dtype=np.int64)
    if not label_path.exists():
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:          # need at least 3 points
                continue
            cls = int(parts[0]) + 1     # YOLO 0-indexed → 1-indexed (0=bg)
            coords = list(map(float, parts[1:]))
            polygon = [
                (coords[i] * w, coords[i + 1] * h)
                for i in range(0, len(coords) - 1, 2)
            ]
            if len(polygon) < 3:
                continue
            tmp = Image.new("L", (w, h), 0)
            ImageDraw.Draw(tmp).polygon(polygon, fill=cls)
            arr = np.asarray(tmp)
            mask = np.where(arr > 0, arr, mask)
    return mask


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_to_tensor  = T.ToTensor()
_normalise  = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)


class UAVSegDataset(Dataset):
    """UAV crop/weed segmentation — YOLO polygon format."""

    def __init__(
        self,
        root: Path,
        split: str,
        img_size: int = 512,
        augment: bool = False,
        binary_sugarcane: bool = False,
        only_sugarcane: bool = False,
    ) -> None:
        self.img_dir  = root / split / "images"
        self.lbl_dir  = root / split / "labels"
        self.img_size = img_size
        self.augment  = augment
        self.binary_sugarcane = binary_sugarcane
        self.only_sugarcane = only_sugarcane
        self.samples  = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        if self.only_sugarcane and self.binary_sugarcane:
            self.samples = [s for s in self.samples if self._has_sugarcane(s)]
        if not self.samples:
            raise RuntimeError(f"No .jpg images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _has_sugarcane(self, stem: str) -> bool:
        label_path = self.lbl_dir / f"{stem}.txt"
        if not label_path.exists():
            return False
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls == 5:  # YOLO class 5 = Sugarcane
                    return True
        return False

    def __getitem__(self, idx: int):
        stem = self.samples[idx]
        img  = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = _rasterise_yolo_seg(
            self.lbl_dir / f"{stem}.txt", self.img_size, self.img_size
        )

        if self.binary_sugarcane:
            mask = (mask == 6).astype("int64")

        img_t  = _normalise(_to_tensor(img))
        mask_t = torch.from_numpy(mask).long()

        if self.augment:
            if torch.rand(1) > 0.5:
                img_t  = img_t.flip(-1)
                mask_t = mask_t.flip(-1)
            if torch.rand(1) > 0.5:
                img_t  = img_t.flip(-2)
                mask_t = mask_t.flip(-2)

        return img_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Shared backbone: DeepFCN → (B, BACKBONE_OUT_CH, H' , W')
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class DeepFCN(nn.Module):
    """Configurable deep FCN backbone with stacked conv blocks."""

    def __init__(self, out_channels: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.depth = max(1, depth)
        self.blocks = nn.ModuleList()
        in_c = 3
        stage_channels = [32, 64, 128, 256, 256]
        for i in range(self.depth):
            out_c = stage_channels[min(i, len(stage_channels) - 1)]
            self.blocks.append(_conv_block(in_c, out_c))
            in_c = out_c
        self.proj = nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Three seg models with a common interface
#   forward(x)           → logits at input resolution
#   compute_loss(x, y)   → scalar
# ─────────────────────────────────────────────────────────────────────────────

class _SegModelBase(nn.Module):
    """Common I/O behaviour; subclasses implement `_head_forward`."""

    def __init__(self, backbone_depth: int = BACKBONE_DEPTH) -> None:
        super().__init__()
        self.backbone = DeepFCN(out_channels=BACKBONE_OUT_CH, depth=backbone_depth)

    # ---- hooks to be implemented by subclasses --------------------------
    def _embed(self, feats: torch.Tensor) -> torch.Tensor:
        """Return features that will be fed to the head (same spatial shape)."""
        raise NotImplementedError

    def _logits_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        """(N, d) → (N, C) — delegated to each head."""
        raise NotImplementedError

    def _loss_spatial(self, z: torch.Tensor, labels_ds: torch.Tensor) -> torch.Tensor:
        """(B, d, H', W') + (B, H', W') → scalar loss."""
        raise NotImplementedError

    # ---- shared I/O ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats      = self.backbone(x)
        z          = self._embed(feats)
        B, D, Hf, Wf = z.shape
        z_flat     = z.permute(0, 2, 3, 1).reshape(B * Hf * Wf, D)
        logits_low = self._logits_flat(z_flat).reshape(B, Hf, Wf, N_CLASSES)
        logits_low = logits_low.permute(0, 3, 1, 2)
        return F.interpolate(
            logits_low, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats     = self.backbone(x)
        z         = self._embed(feats)
        labels_ds = F.interpolate(
            labels.float().unsqueeze(1),
            size=feats.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        return self._loss_spatial(z, labels_ds)


class EuclideanSeg(_SegModelBase):
    """Classical FCN — 1×1 conv classifier on the backbone features."""

    def __init__(self, backbone_depth: int = BACKBONE_DEPTH) -> None:
        super().__init__(backbone_depth=backbone_depth)
        self.classifier = nn.Conv2d(BACKBONE_OUT_CH, N_CLASSES, kernel_size=1)
        self.register_buffer("class_weights",
                             torch.tensor(CLASS_WEIGHTS, dtype=torch.float))

    def _embed(self, feats):
        return feats    # no transformation

    def _logits_flat(self, z_flat):
        w = self.classifier.weight.view(N_CLASSES, BACKBONE_OUT_CH)
        return z_flat @ w.T + self.classifier.bias

    def _loss_spatial(self, z, labels_ds):
        logits = self.classifier(z)                          # (B, C, H', W')
        return F.cross_entropy(logits, labels_ds, weight=self.class_weights)


class HyperbolicSeg(_SegModelBase):
    """Poincaré-ball MLR head (Ganea/Atigh)."""

    def __init__(self, c: float = HYPERBOLIC_C, backbone_depth: int = BACKBONE_DEPTH) -> None:
        super().__init__(backbone_depth=backbone_depth)
        self.c = c
        self.embed_layer = PoincareEmbedding(c=c)
        self.head = HyperbolicMLR(
            N_CLASSES,
            BACKBONE_OUT_CH,
            c=c,
            class_weights=torch.tensor(CLASS_WEIGHTS),
        )

    def _embed(self, feats):
        return self.embed_layer(feats)

    def _logits_flat(self, z_flat):
        return self.head.logits(z_flat)

    def _loss_spatial(self, z, labels_ds):
        return self.head.forward_spatial(z, labels_ds)


class PontryaginSeg(_SegModelBase):
    """Pontryagin RFF/SRF embedding + J-hyperplane MLR (this work).

    Signature convention (see tuning constraints in the config):
        n_srf = kappa                         (negative subspace = κ)
        n_rff = rff_multiplier × in_channels  (positive subspace width)
    """

    def __init__(
        self,
        kappa: int = KAPPA,
        d_poly: int = D_POLY,
        rff_multiplier: int = RFF_MULTIPLIER,
        sigma: float = SIGMA,
        lambda_cone_W: float = LAMBDA_CONE_W,   # kept for API compat; ignored
        lambda_topo: float = LAMBDA_TOPO,
        lambda_balance: float = LAMBDA_BALANCE,
        backbone_depth: int = BACKBONE_DEPTH,
    ) -> None:
        super().__init__(backbone_depth=backbone_depth)
        n_rff = int(rff_multiplier) * BACKBONE_OUT_CH
        n_srf = int(kappa)          # signature κ = número de dims negativas
        self.kappa = kappa
        self.d_poly = d_poly
        self.rff_multiplier = rff_multiplier
        self.sigma = sigma
        self.embed_layer = PontryaginEmbedding(
            BACKBONE_OUT_CH, n_rff, n_srf, int(kappa), d_poly=int(d_poly), sigma=sigma
        )
        self.head = PontryaginMLR(
            N_CLASSES,
            self.embed_layer.out_channels,
            self.embed_layer.J,
            class_weights=torch.tensor(CLASS_WEIGHTS),
            lambda_cone_W=lambda_cone_W,
            lambda_topo=lambda_topo,
            lambda_balance=lambda_balance,
            topo_kwargs=TOPO_KWARGS,
        )

    def _embed(self, feats):
        return self.embed_layer(feats)

    def _logits_flat(self, z_flat):
        return self.head.logits(z_flat)

    def _loss_spatial(self, z, labels_ds):
        return self.head.forward_spatial(z, labels_ds)


def build_model(model_type: str, params: dict | None = None) -> _SegModelBase:
    """Instantiate a seg model, optionally overriding head-specific knobs.

    Recognised keys in `params` (any unknown keys are ignored):
        common      : backbone_depth
        hyperbolic  : c
        pontryagin  : kappa, rff_multiplier, sigma, lambda_topo, lambda_balance
        euclidean   : (none)
    """
    params = params or {}
    backbone_depth = params.get("backbone_depth", BACKBONE_DEPTH)
    if model_type == "euclidean":
        return EuclideanSeg(backbone_depth=backbone_depth)
    if model_type == "hyperbolic":
        return HyperbolicSeg(
            c=params.get("hyperbolic_c", HYPERBOLIC_C),
            backbone_depth=backbone_depth,
        )
    if model_type == "pontryagin":
        return PontryaginSeg(
            kappa=params.get("kappa", KAPPA),
            d_poly=params.get("d_poly", D_POLY),
            rff_multiplier=params.get("rff_multiplier", RFF_MULTIPLIER),
            sigma=params.get("sigma", SIGMA),
            lambda_topo=params.get("lambda_topo", LAMBDA_TOPO),
            lambda_balance=params.get("lambda_balance", LAMBDA_BALANCE),
            backbone_depth=backbone_depth,
        )
    raise ValueError(f"Unknown model {model_type!r}. Choices: {AVAILABLE_MODELS}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — per class (TP/FP/FN/IoU/Dice/Precision/Recall) + global summary.
# Background is included; classes with zero support (never appear in labels)
# are reported but excluded from the macro averages.
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return (a / b) if b > 0 else 0.0


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict:
    """Flat prediction / label tensors → full metrics bundle.

    Per class we report the full confusion matrix entries (TP, FP, FN, TN)
    and the standard derivatives IoU, Dice, Precision, Recall, Specificity.
    Globally we report macro averages (over non-empty classes), micro averages
    (pooled across all pixels), and the summed confusion counts.
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
        tn = total - tp - fp - fn                       # implicit negatives
        support = tp + fn

        iou  = _safe_div(tp, tp + fp + fn)
        dice = _safe_div(2 * tp, 2 * tp + fp + fn)
        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)                   # specificity / TNR

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

    # Micro TN follows from the one-vs-rest formulation summed across classes.
    # With a single-label target each pixel contributes 1 TP to exactly one
    # class ⇒ Σ_y TN_y = (C−1)·total − Σ_y (FP_y + FN_y).
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
        all_preds.append(preds.cpu().reshape(-1))
        all_labels.append(masks.cpu().reshape(-1))
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
        optimiser.zero_grad()
        loss = model.compute_loss(imgs, masks)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = np.array([
    [  0,   0,   0],   # 0  background
    [255, 215,   0],   # 1  Banana
    [ 34, 139,  34],   # 2  Banana_tree
    [  0, 200, 100],   # 3  Banana_tree_
    [255,  99,  71],   # 4  Pepper
    [135, 206, 235],   # 5  Spinach
    [255, 165,   0],   # 6  Sugarcane
    [139,  69,  19],   # 7  Tree
    [148,   0, 211],   # 8  Tree_Plant
    [255, 255,   0],   # 9  Weed
], dtype=np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[mask.clip(0, len(_PALETTE) - 1)]


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def save_validation_grid(
    model: nn.Module,
    dataset,
    device: torch.device,
    out_path: Path,
    n_samples: int = 6,
) -> None:
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
        for ax in axes[row]:
            ax.axis("off")

    patches = [mpatches.Patch(color=_PALETTE[c] / 255, label=CLASS_NAMES[c])
               for c in range(N_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → visualisation saved: {out_path.relative_to(RESULTS_DIR)}")


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


def print_metrics_summary(metrics: dict, tag: str) -> None:
    g = metrics["global"]
    print(f"\n── {tag} ──")
    print(f"  pixel acc     : {g['pixel_acc']:.4f}")
    print(f"  macro mIoU    : {g['macro_iou']:.4f}   "
          f"(over {g['n_valid_classes']} non-empty classes)")
    print(f"  macro Dice    : {g['macro_dice']:.4f}")
    print(f"  micro IoU     : {g['micro_iou']:.4f}")
    print(f"  micro Dice    : {g['micro_dice']:.4f}")
    print(f"  TP/FP/FN/TN   : {g['total_tp']:,} / {g['total_fp']:,} / "
          f"{g['total_fn']:,} / {g['total_tn']:,}")
    print(f"  macro specif. : {g['macro_specificity']:.4f}")
    print("  per-class IoU / Dice:")
    for pc in metrics["per_class"]:
        if pc["support"] == 0:
            tag_empty = "(absent)"
            print(f"    {pc['class']:<14} IoU={pc['iou']:.4f}  Dice={pc['dice']:.4f}  {tag_empty}")
        else:
            print(f"    {pc['class']:<14} IoU={pc['iou']:.4f}  Dice={pc['dice']:.4f}  "
                  f"TP={pc['tp']:>8}  FP={pc['fp']:>8}  FN={pc['fn']:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(data_root=None):
    root = Path(data_root) if data_root else DATASET_ROOT
    train_ds = UAVSegDataset(
        root, "train", IMG_SIZE, augment=True,
        binary_sugarcane=BINARY_SUGARCANE,
        only_sugarcane=BINARY_ONLY_SUGARCANE,
    )
    valid_ds = UAVSegDataset(
        root, "valid", IMG_SIZE, augment=False,
        binary_sugarcane=BINARY_SUGARCANE,
        only_sugarcane=BINARY_ONLY_SUGARCANE,
    )
    test_ds  = UAVSegDataset(
        root, "test",  IMG_SIZE, augment=False,
        binary_sugarcane=BINARY_SUGARCANE,
        only_sugarcane=BINARY_ONLY_SUGARCANE,
    )

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    valid_dl = DataLoader(valid_ds, BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    return {
        "train_ds": train_ds, "valid_ds": valid_ds, "test_ds": test_ds,
        "train_dl": train_dl, "valid_dl": valid_dl, "test_dl":  test_dl,
    }


def train_one_model(
    model_type: str,
    params: dict | None,
    out_dir: Path,
    epochs: int,
    patience: int,
    device: torch.device,
    data: dict,
    save_viz: bool = True,
    verbose: bool = True,
) -> dict:
    """Train a single seg model and return its test metrics + best val mIoU.

    `params` can contain:
        lr, weight_decay (float)
        + any keys forwarded to build_model (kappa, rff_multiplier, …)
    """
    params = params or {}
    lr = float(params.get("lr", LR))
    weight_decay = float(params.get("weight_decay", WEIGHT_DECAY))

    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_type, params).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Trainable parameters: {n_params:,}  |  lr={lr:.2e}  wd={weight_decay:.2e}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=LR_MIN
    )

    csv_path   = out_dir / "metrics.csv"
    model_path = out_dir / "best_model.pth"
    viz_dir    = out_dir / "viz"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "lr",
            "val_pixel_acc", "val_macro_iou", "val_macro_dice",
            "val_micro_iou", "val_micro_dice",
        ])

    best_miou = -1.0
    best_epoch = 0
    patience_counter = 0

    if verbose:
        header = (f"{'Epoch':>6}  {'Loss':>8}  {'PixAcc':>8}  "
                  f"{'mIoU':>8}  {'Dice':>8}  {'LR':>10}")
        print("\n" + header)
        print("─" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, data["train_dl"], optimiser, device)
        val_metrics = evaluate(model, data["valid_dl"], device)
        scheduler.step()

        cur_lr = scheduler.get_last_lr()[0]
        g  = val_metrics["global"]

        if verbose:
            print(
                f"{epoch:6d}  {train_loss:8.4f}  {g['pixel_acc']:8.4f}  "
                f"{g['macro_iou']:8.4f}  {g['macro_dice']:8.4f}  "
                f"{cur_lr:10.2e}  ({time.time()-t0:.1f}s)"
            )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(train_loss, 5), round(cur_lr, 8),
                round(g["pixel_acc"], 5),
                round(g["macro_iou"], 5), round(g["macro_dice"], 5),
                round(g["micro_iou"], 5), round(g["micro_dice"], 5),
            ])

        if g["macro_iou"] > best_miou:
            best_miou = g["macro_iou"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "model_type": model_type,
                "params": params,
            }, model_path)
            save_per_class_csv(val_metrics, out_dir / "val_per_class_best.csv")
            if verbose:
                print(f"  ↑ best model saved (macro mIoU = {best_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping after {patience_counter} epochs without improvement.")
                break

        if save_viz and (epoch % 10 == 0 or epoch == epochs):
            save_validation_grid(model, data["valid_ds"], device,
                                 out_path=viz_dir / f"valid_epoch{epoch:03d}.png")

    # ── test evaluation with best checkpoint ──────────────────────────────────
    if verbose:
        print("\nLoading best model for test evaluation …")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, data["test_dl"], device)
    if verbose:
        print_metrics_summary(test_metrics, f"TEST ({model_type})")

    # Extract learned hyperparameters that the model has tuned by gradient
    # (currently: σ of the RFF, only for the Pontryagin head).
    learned = {}
    if model_type == "pontryagin":
        with torch.no_grad():
            learned["rff_sigma_learned"] = float(model.embed_layer.rff.sigma.item())
            learned["rff_log_sigma"] = float(model.embed_layer.rff.log_sigma.item())

    save_per_class_csv(test_metrics, out_dir / "metrics_per_class.csv")
    with open(out_dir / "metrics_detailed.json", "w") as f:
        json.dump({
            "model": model_type,
            "best_val_macro_iou": best_miou,
            "best_epoch": best_epoch,
            "test": test_metrics,
            "hyperparams": {
                "backbone_out_ch": BACKBONE_OUT_CH,
                "batch_size": BATCH_SIZE,
                "epochs": epochs,
                "lr": lr,
                "lr_min": LR_MIN,
                "weight_decay": weight_decay,
                "early_stop_patience": patience,
                "img_size": IMG_SIZE,
                **params,
            },
            "learned": learned,
        }, f, indent=2)

    if save_viz:
        save_validation_grid(model, data["test_ds"], device,
                             out_path=viz_dir / "test_final.png", n_samples=8)
    if verbose:
        print(f"\nAll outputs saved to {out_dir}")

    return {
        "best_val_macro_iou": best_miou,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=AVAILABLE_MODELS, required=True,
                   help="Segmentation head to train.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE,
                   help="Early-stop patience on val macro mIoU.")
    p.add_argument("--load-best-params", action="store_true",
                   help="Load tuned hyperparameters from results/seg_uav/<model>/hpo/best_params.json "
                        "(generated by tune_seg_uav.py) and override the config defaults.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to UAV_segmantation dataset root (overrides config).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for results (overrides config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    out_dir = results_dir / args.model
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    params: dict = {}
    if args.load_best_params:
        bp_path = results_dir / args.model / "hpo" / "best_params.json"
        if not bp_path.exists():
            raise FileNotFoundError(
                f"No best_params.json found at {bp_path}. "
                f"Run `python tune_seg_uav.py --model {args.model}` first."
            )
        with open(bp_path) as f:
            params = json.load(f)
        print(f"Loaded tuned params from {bp_path}:")
        for k, v in params.items():
            print(f"  {k} = {v}")

    print(f"Model : {args.model}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    data = build_dataloaders(data_root=args.data_root)
    print(f"Train: {len(data['train_ds'])} | Valid: {len(data['valid_ds'])} "
          f"| Test: {len(data['test_ds'])}")

    train_one_model(
        model_type=args.model,
        params=params,
        out_dir=out_dir,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        data=data,
        save_viz=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
