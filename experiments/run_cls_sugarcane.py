"""ConvNeXtV2-Tiny classification experiments — Sugarcane Leaf Disease Dataset.

Two models share the same ConvNeXtV2-Tiny backbone and training protocol:

    euclidean   — ConvNeXtV2-Tiny + fine-tuned linear head (CE + label smoothing)
    pontryagin  — ConvNeXtV2-Tiny + PontryaginEmbedding + PontryaginMLR

Kaggle dataset: nirmalsankalana/sugarcane-leaf-disease-dataset
Download requires KAGGLE_USERNAME and KAGGLE_KEY in .env (copy from .env.example).

Usage:
    python run_cls_sugarcane.py --model euclidean
    python run_cls_sugarcane.py --model pontryagin
    python run_cls_sugarcane.py --model all
    python run_cls_sugarcane.py --model all --data-root /path/to/dataset
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.cls_sugarcane import (
    AVAILABLE_MODELS, BACKBONE_NAME, BACKBONE_OUT_CH, BATCH_SIZE, D_POLY,
    DEVICE, EARLY_STOP_PATIENCE, EPOCHS, IMG_SIZE, KAPPA, LABEL_SMOOTHING,
    LAMBDA_BALANCE, LAMBDA_TOPO, LR, LR_BACKBONE, LR_MIN, N_RFF, NUM_WORKERS,
    PRETRAINED, RESULTS_DIR, RFF_MULTIPLIER, SEED, SIGMA, TEST_SPLIT,
    TOPO_KWARGS, VAL_SPLIT, WEIGHT_DECAY,
)
from prfe.layers.pontryagin import PontryaginEmbedding
from prfe.losses.mlr import PontryaginMLR

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for ConvNeXtV2-Tiny. Install with: pip install timm"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset download
# ─────────────────────────────────────────────────────────────────────────────

def _load_env():
    """Load .env from project root if python-dotenv is installed."""
    env_path = Path(__file__).parents[1] / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass


def download_dataset() -> Path:
    """Download Sugarcane Leaf Disease Dataset via kagglehub.

    Reads credentials from .env or existing ~/.kaggle/kaggle.json.
    Returns the local path to the downloaded dataset.
    """
    _load_env()
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required. Install with: pip install kagglehub"
        )
    print("Downloading sugarcane-leaf-disease-dataset from Kaggle…")
    path = kagglehub.dataset_download("nirmalsankalana/sugarcane-leaf-disease-dataset")
    print(f"  Dataset at: {path}")
    return Path(path)


def _find_dataset_root(base: Path) -> Path:
    """Find the root that contains class subfolders (images).

    Handles datasets that may be nested one or two levels deep.
    """
    # Direct class folders?
    if any(p.is_dir() for p in base.iterdir()):
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        # If subdirs contain images, this is the root
        if any(list(s.glob("*.jpg")) + list(s.glob("*.png")) + list(s.glob("*.jpeg"))
               for s in subdirs):
            return base
        # Otherwise go one level deeper
        for sub in subdirs:
            candidate = _find_dataset_root(sub)
            if candidate is not None:
                return candidate
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SugarcaneLeafDataset(Dataset):
    """Flat image-folder dataset for Sugarcane Leaf Disease classification.

    Expects root/ClassName/image.jpg structure (auto-detected via
    _find_dataset_root).  Supports train/val/test random splits (no
    pre-existing splits assumed).

    Args:
        root:      path returned by _find_dataset_root
        split:     'train' | 'val' | 'test'
        transform: torchvision transform (defaults per split)
        seed:      random seed for deterministic splits
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform=None,
        seed: int = SEED,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or (_TRAIN_TF if split == "train" else _EVAL_TF)

        # Collect class folders (sorted for reproducibility)
        class_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        self.classes   = [p.name for p in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all (path, label) pairs
        all_samples: list[tuple[Path, int]] = []
        for cls_dir in class_dirs:
            label = self.class_to_idx[cls_dir.name]
            for f in cls_dir.iterdir():
                if f.suffix.lower() in _IMG_EXTS:
                    all_samples.append((f, label))

        # Deterministic split
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_samples))
        n   = len(idx)
        n_test = int(n * TEST_SPLIT)
        n_val  = int(n * VAL_SPLIT)
        if split == "test":
            chosen = idx[:n_test]
        elif split == "val":
            chosen = idx[n_test:n_test + n_val]
        else:
            chosen = idx[n_test + n_val:]

        self.samples = [all_samples[i] for i in chosen]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class _ClsBase(nn.Module):
    """Common interface for both classification models.

    forward(x)         → logits (B, n_classes)
    compute_loss(x, y) → scalar training loss
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EuclideanConvNext(_ClsBase):
    """ConvNeXtV2-Tiny with fine-tuned linear classification head.

    Standard fine-tuning baseline matching the state-of-the-art approach on
    this dataset.  Uses discriminative learning rates: backbone at LR_BACKBONE,
    classifier head at LR.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=PRETRAINED,
            num_classes=n_classes,
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.forward(x), labels,
                               label_smoothing=LABEL_SMOOTHING)

    def param_groups(self, lr: float = LR, lr_backbone: float = LR_BACKBONE):
        """Return param groups with discriminative learning rates."""
        head_params = list(self.backbone.head.parameters())
        head_ids    = {id(p) for p in head_params}
        bb_params   = [p for p in self.backbone.parameters() if id(p) not in head_ids]
        return [
            {"params": bb_params,   "lr": lr_backbone},
            {"params": head_params, "lr": lr},
        ]


class PontryaginConvNext(_ClsBase):
    """ConvNeXtV2-Tiny + PontryaginEmbedding + PontryaginMLR for classification.

    Architecture:
        ConvNeXtV2-Tiny (forward_features) → (B, 768, H', W')
        PontryaginEmbedding               → (B, p+q, H', W')
        AdaptiveAvgPool2d(1) + flatten    → (B, p+q)
        PontryaginMLR.logits              → (B, n_classes)

    For training, the full Pontryagin loss (CE + balance + topo) is applied on
    the pooled (B, p+q) embeddings.
    """

    def __init__(
        self,
        n_classes: int,
        kappa: int = KAPPA,
        d_poly: int = D_POLY,
        rff_multiplier: int = RFF_MULTIPLIER,
        sigma: float = SIGMA,
        lambda_topo: float = LAMBDA_TOPO,
        lambda_balance: float = LAMBDA_BALANCE,
    ) -> None:
        super().__init__()
        # Backbone: spatial features only (no global pool, no head)
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=PRETRAINED,
            num_classes=0,
            global_pool="",
        )
        n_rff  = int(rff_multiplier) * BACKBONE_OUT_CH
        n_srf  = int(kappa)
        self.embed_layer = PontryaginEmbedding(
            BACKBONE_OUT_CH, n_rff, n_srf, int(kappa),
            d_poly=int(d_poly), sigma=sigma,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = PontryaginMLR(
            n_classes,
            self.embed_layer.out_channels,
            self.embed_layer.J,
            lambda_topo=lambda_topo,
            lambda_balance=lambda_balance,
            topo_kwargs=TOPO_KWARGS,
        )
        self.n_classes = n_classes

    def _embed_global(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, p+q) global Pontryagin embedding."""
        feats  = self.backbone.forward_features(x)   # (B, 768, H', W')
        z_map  = self.embed_layer(feats)              # (B, p+q, H', W')
        return self.pool(z_map).flatten(1)            # (B, p+q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.logits(self._embed_global(x))

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = self._embed_global(x)
        return self.head(z, labels)

    def param_groups(self, lr: float = LR, lr_backbone: float = LR_BACKBONE):
        """Discriminative learning rates: backbone vs. embedding+head."""
        bb_params   = list(self.backbone.parameters())
        bb_ids      = {id(p) for p in bb_params}
        head_params = [p for p in self.parameters() if id(p) not in bb_ids]
        return [
            {"params": bb_params,   "lr": lr_backbone},
            {"params": head_params, "lr": lr},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_cls_model(model_type: str, n_classes: int, params: dict | None = None) -> _ClsBase:
    params = params or {}
    if model_type == "euclidean":
        return EuclideanConvNext(n_classes=n_classes)
    if model_type == "pontryagin":
        return PontryaginConvNext(
            n_classes=n_classes,
            kappa=params.get("kappa", KAPPA),
            d_poly=params.get("d_poly", D_POLY),
            rff_multiplier=params.get("rff_multiplier", RFF_MULTIPLIER),
            sigma=params.get("sigma", SIGMA),
            lambda_topo=params.get("lambda_topo", LAMBDA_TOPO),
            lambda_balance=params.get("lambda_balance", LAMBDA_BALANCE),
        )
    raise ValueError(f"Unknown model type: {model_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(data_root: Path):
    kw = dict(root=data_root)
    train_ds = SugarcaneLeafDataset(**kw, split="train")
    val_ds   = SugarcaneLeafDataset(**kw, split="val")
    test_ds  = SugarcaneLeafDataset(**kw, split="test")

    assert train_ds.classes == val_ds.classes == test_ds.classes, (
        "Class mismatch across splits — check dataset root."
    )

    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    return {
        "train_ds": train_ds,
        "val_ds":   val_ds,
        "test_ds":  test_ds,
        "classes":  train_ds.classes,
        "n_classes": len(train_ds.classes),
        "train_dl": DataLoader(train_ds, shuffle=True,  **loader_kw),
        "val_dl":   DataLoader(val_ds,   shuffle=False, **loader_kw),
        "test_dl":  DataLoader(test_ds,  shuffle=False, **loader_kw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: list[int],
    all_labels: list[int],
    class_names: list[str],
) -> dict:
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    report = classification_report(labels, preds, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm     = confusion_matrix(labels, preds)
    return {
        "accuracy": float((preds == labels).mean()),
        "per_class": {
            c: {
                "precision": report[c]["precision"],
                "recall":    report[c]["recall"],
                "f1":        report[c]["f1-score"],
                "support":   report[c]["support"],
            }
            for c in class_names
        },
        "macro_f1":    report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimiser.zero_grad()
        loss = model.compute_loss(imgs, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return compute_metrics(all_preds, all_labels, class_names)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_IMG_MEAN = np.array([0.485, 0.456, 0.406])
_IMG_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).cpu().numpy()
    return ((img * _IMG_STD + _IMG_MEAN) * 255).clip(0, 255).astype(np.uint8)


def save_confusion_matrix(cm: list, class_names: list[str], out_path: Path):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm_arr[i, j], ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_sample_predictions(
    model, dataset, device, class_names, out_path: Path, n: int = 16
):
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, min(n, len(dataset)), dtype=int)
    cols    = 4
    rows    = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    for ax, idx in zip(axes, indices):
        img_t, true_lbl = dataset[idx]
        pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1).item()
        ax.imshow(_denorm(img_t))
        color = "green" if pred == true_lbl else "red"
        ax.set_title(
            f"GT: {class_names[true_lbl]}\nPred: {class_names[pred]}",
            color=color, fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(indices):]:
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_model(
    model_type: str,
    data: dict,
    params: dict | None = None,
    device: torch.device | None = None,
    epochs: int = EPOCHS,
    out_dir: Path | None = None,
    verbose: bool = True,
    save_viz: bool = True,
) -> dict:
    """Train one classification model and save results.

    Returns the test metrics dict.
    """
    params   = params or {}
    device   = device or torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    out_dir  = out_dir or (RESULTS_DIR / model_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir  = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    class_names = data["classes"]
    n_classes   = data["n_classes"]
    model = build_cls_model(model_type, n_classes=n_classes, params=params).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[{model_type.upper()}] trainable params: {n_params:,}")
        print(f"  classes ({n_classes}): {class_names}")

    param_groups = model.param_groups(
        lr=params.get("lr", LR),
        lr_backbone=params.get("lr_backbone", LR_BACKBONE),
    )
    optimiser = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=LR_MIN
    )

    best_val_acc     = -1.0
    best_val_metrics = {}
    patience_cnt     = 0
    final_ckpt       = out_dir / "best_model.pth"
    _fd, _tmp        = tempfile.mkstemp(suffix=".pth")
    os.close(_fd)
    tmp_ckpt = Path(_tmp)

    history = []
    t0 = time.time()
    try:
        for epoch in range(1, epochs + 1):
            tr_loss = train_epoch(model, data["train_dl"], optimiser, device)
            val_m   = evaluate(model, data["val_dl"], device, class_names)
            val_acc = val_m["accuracy"]
            scheduler.step()

            history.append({"epoch": epoch, "train_loss": tr_loss, **{
                k: val_m[k] for k in ("accuracy", "macro_f1", "weighted_f1")
            }})

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_val_metrics = val_m
                patience_cnt     = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch":       epoch,
                    "model_type":  model_type,
                    "val_metrics": val_m,
                    "params":      params,
                    "classes":     class_names,
                }, tmp_ckpt)
            else:
                patience_cnt += 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  ep {epoch:3d}  loss={tr_loss:.4f}  "
                      f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

            if patience_cnt >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"  Early stop at epoch {epoch}")
                break
    finally:
        if tmp_ckpt.exists():
            shutil.copy2(tmp_ckpt, final_ckpt)
            tmp_ckpt.unlink(missing_ok=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training time: {elapsed:.0f}s")

    # ── Test evaluation ──────────────────────────────────────────────────────
    ckpt = torch.load(final_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, data["test_dl"], device, class_names)

    # ── Save artefacts ───────────────────────────────────────────────────────
    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(out_dir / "metrics_val_best.json", "w") as f:
        json.dump(best_val_metrics, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if save_viz:
        save_confusion_matrix(
            test_m["confusion_matrix"], class_names,
            out_path=viz_dir / "confusion_matrix.png",
        )
        save_sample_predictions(
            model, data["test_ds"], device, class_names,
            out_path=viz_dir / "sample_predictions.png",
        )
        _save_training_curves(history, out_dir / "training_curves.png")

    if verbose:
        print(f"\n  Test  acc={test_m['accuracy']:.4f}  "
              f"macro_f1={test_m['macro_f1']:.4f}  "
              f"weighted_f1={test_m['weighted_f1']:.4f}")

    return test_m


def _save_training_curves(history: list[dict], out_path: Path):
    epochs    = [h["epoch"] for h in history]
    losses    = [h["train_loss"] for h in history]
    accs      = [h["accuracy"] for h in history]
    macro_f1s = [h["macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, losses, marker="o", ms=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, accs,      marker="o", ms=3, label="Val accuracy")
    ax2.plot(epochs, macro_f1s, marker="s", ms=3, label="Val macro-F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.set_title("Validation metrics")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Train ConvNeXtV2-Tiny on Sugarcane Leaf Disease")
    p.add_argument("--model", choices=list(AVAILABLE_MODELS) + ["all"],
                   default="pontryagin")
    p.add_argument("--epochs",   type=int, default=EPOCHS)
    p.add_argument("--device",   default=DEVICE)
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to sugarcane dataset root. Downloads if omitted.")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory (overrides config).")
    p.add_argument("--params", type=str, default=None,
                   help="JSON string or path to best_params.json.")
    return p.parse_args()


def main():
    args        = _parse()
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    # Resolve dataset root
    if args.data_root:
        data_root = _find_dataset_root(Path(args.data_root))
    else:
        env_root = os.environ.get("SUGARCANE_DATA_ROOT")
        if env_root:
            data_root = _find_dataset_root(Path(env_root))
        else:
            data_root = _find_dataset_root(download_dataset())

    print(f"Dataset root: {data_root}")
    data = build_dataloaders(data_root)
    print(f"Classes: {data['classes']}")
    print(f"Train: {len(data['train_ds'])}  Val: {len(data['val_ds'])}  "
          f"Test: {len(data['test_ds'])}")

    models = list(AVAILABLE_MODELS) if args.model == "all" else [args.model]
    for mt in models:
        params = {}
        if args.params:
            src = Path(args.params)
            params = json.load(open(src)) if src.exists() else json.loads(args.params)
        else:
            hpo_path = results_dir / mt / "hpo" / "best_params.json"
            if hpo_path.exists():
                with open(hpo_path) as f:
                    params = json.load(f)
                print(f"  Loaded HPO params for {mt}: {hpo_path}")

        train_one_model(
            mt, data=data, params=params, device=device,
            epochs=args.epochs, out_dir=results_dir / mt, verbose=True,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
