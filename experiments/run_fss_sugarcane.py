"""1-way K-shot few-shot segmentation — UAV Sugarcane dataset.

Compares:
    euclidean   — PANet backbone + cosine similarity
    pontryagin  — PANet backbone + Pontryagin J-inner product (this work)

Both models share the same backbone, episodic dataset, augmentations,
optimiser, cosine LR schedule, and early-stopping policy.
Only the embedding layer and similarity function differ.

Usage:
    python run_fss_sugarcane.py --model euclidean --k-shot 1
    python run_fss_sugarcane.py --model pontryagin --k-shot 5
    python run_fss_sugarcane.py --model all --k-shot 1

Results per run (saved to experiments/results/fss_sugarcane/<model>_<k>shot/):
    metrics.csv             — per-epoch train loss + val IoU/Dice/F1
    metrics_test.json       — final test metrics
    best_model.pth          — weights at best val IoU
    viz/*.png               — support / query / GT / prediction panels
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from configs.fss_sugarcane import (
    AVAILABLE_MODELS, BASE_CH, BATCH_EPISODES, BCE_WEIGHT,
    CONE_EPSILON, DATASET_ROOT, DEPTH, DEVICE, EARLY_STOP_PATIENCE, EPOCHS,
    IMG_SIZE, K_SHOT, LAMBDA_CONE, LAMBDA_ORTH, LR, LR_MIN,
    N_EPISODES_TEST, N_EPISODES_TRAIN, N_EPISODES_VAL, N_RFF, N_SRF,
    NUM_WORKERS, POS_WEIGHT, RESULTS_DIR, SIGMA, TARGET_CLS, TRAINABLE_RFF,
    WEIGHT_DECAY,
)
from prfe.data.fss_dataset import EpisodicUAVDataset
from prfe.losses.fss import EuclideanFSSLoss, PontryaginFSSLoss
from prfe.models.fss import EuclideanFewShotSeg, PontryaginFewShotSeg


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute IoU, Dice, precision, recall for binary masks.

    Args:
        pred:   (B, H, W) binary float
        target: (B, H, W) binary float
    Returns:
        dict with mean values over the batch
    """
    p = pred.reshape(pred.shape[0], -1).bool()
    t = target.reshape(target.shape[0], -1).bool()

    tp = (p & t).float().sum(dim=1)
    fp = (p & ~t).float().sum(dim=1)
    fn = (~p & t).float().sum(dim=1)

    iou   = (tp / (tp + fp + fn + 1e-6)).mean().item()
    dice  = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    prec  = (tp / (tp + fp + 1e-6)).mean().item()
    rec   = (tp / (tp + fn + 1e-6)).mean().item()

    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor → (H, W, 3) uint8."""
    img = t.permute(1, 2, 0).cpu().numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_viz(
    batch: dict,
    pred: torch.Tensor,
    out_path: Path,
    n_rows: int = 4,
) -> None:
    """Save a panel: support image(s) | query | GT mask | prediction."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    B   = min(n_rows, pred.shape[0])
    K   = batch["support_imgs"].shape[1]
    n_cols = K + 3   # K support + query + GT + pred

    fig, axes = plt.subplots(B, n_cols, figsize=(3 * n_cols, 3 * B))
    if B == 1:
        axes = axes[None]

    for b in range(B):
        for k in range(K):
            axes[b, k].imshow(_denorm(batch["support_imgs"][b, k]))
            if k == 0:
                axes[b, k].set_ylabel(f"ep {b}", fontsize=8)
            axes[b, k].set_title(f"sup {k+1}", fontsize=8)
            axes[b, k].axis("off")

        axes[b, K].imshow(_denorm(batch["query_img"][b]))
        axes[b, K].set_title("query", fontsize=8)
        axes[b, K].axis("off")

        axes[b, K + 1].imshow(
            batch["query_mask"][b].cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[b, K + 1].set_title("GT", fontsize=8)
        axes[b, K + 1].axis("off")

        axes[b, K + 2].imshow(
            pred[b].cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[b, K + 2].set_title("pred", fontsize=8)
        axes[b, K + 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model + loss factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    k_shot: int,
    device: torch.device,
    params: dict | None = None,
    trainable_rff: bool = TRAINABLE_RFF,
):
    """Instantiate model + loss.

    params overrides config defaults for HPO trials.  Accepted keys:
        Common    : bce_weight, pos_weight
        Pontryagin: rff_multiplier (int, n_rff = mult × N_RFF),
                    srf_multiplier (int, n_srf = mult × N_SRF; also sets kappa),
                    sigma, lambda_cone, lambda_orth
    trainable_rff: if True, W and b of the RFF layer are learned parameters.
    """
    p = params or {}

    if model_name == "euclidean":
        model = EuclideanFewShotSeg(
            base_ch=BASE_CH,
            depth=DEPTH,
        )
        loss_fn = EuclideanFSSLoss(
            bce_weight=p.get("bce_weight", BCE_WEIGHT),
            pos_weight=p.get("pos_weight", POS_WEIGHT),
        )
    elif model_name == "pontryagin":
        n_rff_eff = int(p.get("rff_multiplier", 1)) * N_RFF
        n_srf_eff = int(p.get("srf_multiplier", 1)) * N_SRF
        model = PontryaginFewShotSeg(
            base_ch=BASE_CH,
            depth=DEPTH,
            n_rff=n_rff_eff,
            n_srf=n_srf_eff,
            kappa=n_srf_eff,   # kappa must equal n_srf (negative subspace dim)
            sigma=p.get("sigma", SIGMA),
            trainable_rff=trainable_rff,
        )
        loss_fn = PontryaginFSSLoss(
            J=model.J,
            bce_weight=p.get("bce_weight", BCE_WEIGHT),
            pos_weight=p.get("pos_weight", POS_WEIGHT),
            lambda_cone=p.get("lambda_cone", LAMBDA_CONE),
            lambda_orth=p.get("lambda_orth", LAMBDA_ORTH),
            cone_epsilon=CONE_EPSILON,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    model   = model.to(device)
    loss_fn = loss_fn.to(device)
    return model, loss_fn


def train_one_model(
    model_name: str,
    data_root: Path,
    device: torch.device,
    params: dict,
    k_shot: int = K_SHOT,
    epochs: int = EPOCHS,
    out_dir: Path | None = None,
    verbose: bool = True,
    save_viz: bool = False,
    n_episodes_train: int = N_EPISODES_TRAIN,
    n_episodes_val: int = N_EPISODES_VAL,
    trainable_rff: bool = TRAINABLE_RFF,
) -> dict:
    """Single training run used by the HPO loop.

    Returns a dict with at least ``val_iou`` (best over all epochs).
    """
    lr           = float(params.get("lr",           LR))
    weight_decay = float(params.get("weight_decay", WEIGHT_DECAY))

    train_ds = EpisodicUAVDataset(
        data_root, "train",
        k_shot=k_shot, n_episodes=n_episodes_train,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=True,
    )
    val_ds = EpisodicUAVDataset(
        data_root, "valid",
        k_shot=k_shot, n_episodes=n_episodes_val,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=False, seed=0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_EPISODES, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_EPISODES, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model, loss_fn = build_model(model_name, k_shot, device, params=params,
                                 trainable_rff=trainable_rff)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=LR_MIN
    )

    best_val_iou = -1.0
    ckpt_path    = (out_dir / "ckpt.pth") if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimiser, device)
        val_m      = eval_epoch(model, val_loader, device)
        scheduler.step()

        if verbose:
            print(
                f"  ep {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                f"val_iou={val_m['iou']:.4f}  val_dice={val_m['dice']:.4f}"
            )

        if val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)

    return {"val_iou": best_val_iou}


# ─────────────────────────────────────────────────────────────────────────────
# Training + evaluation loops
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_epoch(model, loader, loss_fn, optimiser, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        optimiser.zero_grad()
        loss = model.compute_loss(
            batch["support_imgs"],
            batch["support_masks"],
            batch["query_img"],
            batch["query_mask"],
            loss_fn,
        )
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_iou, all_dice, all_prec, all_rec = [], [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        pred  = model.predict(
            batch["support_imgs"],
            batch["support_masks"],
            batch["query_img"],
        )
        m = binary_metrics(pred, batch["query_mask"])
        all_iou.append(m["iou"])
        all_dice.append(m["dice"])
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])
    return {
        "iou":       float(np.mean(all_iou)),
        "dice":      float(np.mean(all_dice)),
        "precision": float(np.mean(all_prec)),
        "recall":    float(np.mean(all_rec)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def run(
    model_name: str,
    k_shot: int,
    data_root: Path | None = None,
    results_dir: Path | None = None,
    best_params: dict | None = None,
    epochs: int = EPOCHS,
    trainable_rff: bool = TRAINABLE_RFF,
) -> None:
    """Train and evaluate one model variant.

    Args:
        model_name:    'euclidean' or 'pontryagin'
        k_shot:        number of support images per episode
        data_root:     path to UAV_segmantation root; overrides DATASET_ROOT
        results_dir:   output directory; overrides RESULTS_DIR
        best_params:   HPO-found hyperparameters; overrides config defaults
        epochs:        training epochs; overrides EPOCHS from config
        trainable_rff: if True appends '_trainable' to the output folder name
    """
    _data_root   = Path(data_root)   if data_root   is not None else DATASET_ROOT
    _results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    p = best_params or {}

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  K-shot: {k_shot}  |  Device: {device}"
          + ("  |  trainable_rff=True" if trainable_rff else ""))
    print(f"{'='*60}")

    suffix  = "_trainable" if trainable_rff else ""
    run_dir = _results_dir / f"{model_name}_{k_shot}shot{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = EpisodicUAVDataset(
        _data_root, "train",
        k_shot=k_shot, n_episodes=N_EPISODES_TRAIN,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=True,
    )
    val_ds = EpisodicUAVDataset(
        _data_root, "valid",
        k_shot=k_shot, n_episodes=N_EPISODES_VAL,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=False, seed=0,
    )
    test_ds = EpisodicUAVDataset(
        _data_root, "test",
        k_shot=k_shot, n_episodes=N_EPISODES_TEST,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=False, seed=42,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_EPISODES, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_EPISODES, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_EPISODES, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(
        f"Dataset — train: {len(train_ds)} ep  "
        f"val: {len(val_ds)} ep  test: {len(test_ds)} ep"
    )
    print(
        f"  Eligible images: {len(train_ds.samples)} train  "
        f"{len(val_ds.samples)} val  {len(test_ds.samples)} test"
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model, loss_fn = build_model(model_name, k_shot, device, params=best_params,
                                 trainable_rff=trainable_rff)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Trainable params: {n_params:,}")
    if best_params:
        print(f"Using HPO params: { {k: v for k, v in best_params.items()} }")

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(p.get("lr", LR)),
        weight_decay=float(p.get("weight_decay", WEIGHT_DECAY)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=LR_MIN
    )

    # ── training loop ─────────────────────────────────────────────────────────
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_iou", "val_dice",
                         "val_precision", "val_recall", "lr"])

    best_val_iou   = -1.0
    patience_count = 0
    best_ckpt      = run_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        t0        = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimiser, device)
        val_m      = eval_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr_now = optimiser.param_groups[0]["lr"]
        print(
            f"Ep {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
            f"val_iou={val_m['iou']:.4f}  val_dice={val_m['dice']:.4f}  "
            f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss,
                val_m["iou"], val_m["dice"],
                val_m["precision"], val_m["recall"],
                lr_now,
            ])

        if val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    # ── test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val IoU={best_val_iou:.4f}) …")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_m = eval_epoch(model, test_loader, device)

    print(
        f"TEST  iou={test_m['iou']:.4f}  dice={test_m['dice']:.4f}  "
        f"prec={test_m['precision']:.4f}  rec={test_m['recall']:.4f}"
    )

    results = {
        "model": model_name,
        "k_shot": k_shot,
        "best_val_iou": best_val_iou,
        **{f"test_{k}": v for k, v in test_m.items()},
    }
    with open(run_dir / "metrics_test.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── visualisation on test episodes ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 3:
                break
            batch_dev = _to_device(batch, device)
            pred = model.predict(
                batch_dev["support_imgs"],
                batch_dev["support_masks"],
                batch_dev["query_img"],
            )
            save_viz(batch, pred, viz_dir / f"test_batch_{i:02d}.png")

    print(f"Visualisations saved to {viz_dir}")
    print(f"Results saved to {run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Few-shot segmentation — UAV Sugarcane dataset"
    )
    parser.add_argument(
        "--model", default="all",
        choices=list(AVAILABLE_MODELS) + ["all"],
        help="Model to train (default: all)",
    )
    parser.add_argument(
        "--k-shot", type=int, default=K_SHOT,
        help=f"Number of support images per episode (default: {K_SHOT})",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override DATASET_ROOT from config",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Output directory; overrides RESULTS_DIR from config",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--trainable-rff", action="store_true", default=False,
        help="Make RFF frequencies and phases learnable parameters; appends '_trainable' to output folder",
    )
    args = parser.parse_args()

    _results_dir = args.results_dir or RESULTS_DIR

    models = list(AVAILABLE_MODELS) if args.model == "all" else [args.model]
    for m in models:
        # Auto-load HPO best params if they exist
        hpo_json = _results_dir / f"{m}_{args.k_shot}shot" / "hpo" / "best_params.json"
        best_params = None
        if hpo_json.exists():
            best_params = {
                k: v for k, v in json.loads(hpo_json.read_text()).items()
                if not k.startswith("_")
            }
            print(f"Loaded HPO best params from {hpo_json}")
        run(m, args.k_shot,
            data_root=args.data_root,
            results_dir=args.results_dir,
            best_params=best_params,
            epochs=args.epochs,
            trainable_rff=args.trainable_rff)


if __name__ == "__main__":
    main()
