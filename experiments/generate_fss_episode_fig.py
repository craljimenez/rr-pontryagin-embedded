"""generate_fss_episode_fig.py

Regenera Figure 4.10: panel de episodio FSS en layout 2×4 (Euclidean,
Hyperbolic y PRFE, en el mismo orden usado en el resto del paper) con
títulos normalizados (PRFE en lugar de Pontryagin) y 300 DPI PDF.

Layout 2×4:
  Row 0: Support 1  | FSS-Euclidean ScoreCAM  | FSS-Hyperbolic ScoreCAM  | FSS-PRFE ScoreCAM
  Row 1: Query      | FSS-Euclidean Pred.      | FSS-Hyperbolic Pred.     | FSS-PRFE Pred.

Usage:
    python experiments/generate_fss_episode_fig.py [--episode 1] [--k-shot 1]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent / "src"))

from configs.fss_sugarcane import (
    DATASET_ROOT, DEVICE, IMG_SIZE, N_EPISODES_TEST,
    RESULTS_DIR, TARGET_CLS,
)
from prfe.data.fss_dataset import EpisodicUAVDataset
from run_fss_sugarcane import build_model

# Reuse helpers from interpretability_fss
from interpretability_fss import (
    FSSScoreCAM, _denorm, _norm01, _overlay, _gt_contour, _axoff, _iou,
    _load_model,
)

OUT_DIR = EXP_DIR / "report" / "figures"

# (column label, key suffix) — Euclidean / Hyperbolic / PRFE, matching the
# ordering used throughout the paper (Table 4, VOC qualitative figure, etc.)
_COLUMNS = [
    ("euclidean",  "FSS-Euclidean"),
    ("hyperbolic", "FSS-Hyperbolic"),
    ("pontryagin", "FSS-PRFE"),
]


# ─────────────────────────────────────────────────────────────────────────────

def make_2x4_figure(
    sup_rgb, q_rgb, gt,
    heats, preds, ious,
    episode_idx,
    out_path,
):
    """
    Row 0: Support 1  | FSS-Euclidean ScoreCAM  | FSS-Hyperbolic ScoreCAM  | FSS-PRFE ScoreCAM
    Row 1: Query      | FSS-Euclidean Pred.      | FSS-Hyperbolic Pred.     | FSS-PRFE Pred.

    heats, preds, ious: dicts keyed by model_name ("euclidean", "hyperbolic", "pontryagin").
    """
    n_cols = 1 + len(_COLUMNS)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(4.3 * n_cols, 8),
        gridspec_kw={"wspace": 0.06, "hspace": 0.18},
    )

    # ── Row 0 ────────────────────────────────────────────────────────────────
    axes[0, 0].imshow(sup_rgb)
    _gt_contour(axes[0, 0], gt, color="lime")
    axes[0, 0].set_title("Support 1", fontsize=11, fontweight="bold")
    _axoff(axes[0, 0])

    for col, (model_name, label) in enumerate(_COLUMNS, start=1):
        axes[0, col].imshow(_overlay(q_rgb, heats[model_name]))
        _gt_contour(axes[0, col], gt)
        axes[0, col].set_title(f"{label}\nScoreCAM", fontsize=11, fontweight="bold")
        _axoff(axes[0, col])

    # ── Row 1 ────────────────────────────────────────────────────────────────
    axes[1, 0].imshow(q_rgb)
    _gt_contour(axes[1, 0], gt)
    axes[1, 0].set_title("Query", fontsize=11, fontweight="bold")
    _axoff(axes[1, 0])

    for col, (model_name, label) in enumerate(_COLUMNS, start=1):
        axes[1, col].imshow(preds[model_name], cmap="gray", vmin=0, vmax=1)
        _gt_contour(axes[1, col], gt)
        axes[1, col].set_title(
            f"{label} Prediction\nIoU = {ious[model_name]:.3f}",
            fontsize=11, fontweight="bold",
        )
        _axoff(axes[1, col])

    fig.suptitle(
        f"Episode {episode_idx:03d} — ScoreCAM comparison  (green contour = GT)",
        fontsize=11, y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1,
                    help="0-based episode index (default 1, matches IoU 0.911/0.876)")
    ap.add_argument("--k-shot",  type=int, default=1)
    ap.add_argument("--device",  type=str, default=DEVICE)
    ap.add_argument("--out", type=str, default=None,
                    help="Output PDF path (default: report/figures/fss_ep_example.pdf)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models …")
    models = {
        model_name: _load_model(model_name, args.k_shot, RESULTS_DIR, device, False)
        for model_name, _ in _COLUMNS
    }
    print("OK")

    test_ds = EpisodicUAVDataset(
        DATASET_ROOT, "test",
        k_shot=args.k_shot, n_episodes=N_EPISODES_TEST,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=False, seed=42,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    for ep_idx, batch in enumerate(loader):
        if ep_idx < args.episode:
            continue

        sup_imgs  = batch["support_imgs"].to(device)
        sup_masks = batch["support_masks"].to(device)
        q_img     = batch["query_img"].to(device)
        q_mask    = batch["query_mask"].to(device)

        gt    = q_mask[0].cpu().numpy()
        q_rgb = _denorm(q_img[0])
        sup_rgb = _denorm(sup_imgs[0, 0])   # first support image

        print(f"Episode {ep_idx}: computing predictions …")
        preds, ious, heats = {}, {}, {}
        with torch.no_grad():
            for model_name, model in models.items():
                pred = (model.forward(sup_imgs, sup_masks, q_img)[0] > 0).float().cpu().numpy()
                preds[model_name] = pred
                ious[model_name]  = _iou(pred, gt)
        print("  IoU —", ", ".join(f"{n}: {ious[n]:.3f}" for n in models))

        print("  Computing ScoreCAM …")
        for model_name, model in models.items():
            cam = FSSScoreCAM(model)
            heats[model_name] = cam.compute(sup_imgs, sup_masks, q_img)
            cam.remove()

        out_path = Path(args.out) if args.out else (OUT_DIR / "fss_ep_example.pdf")
        print("  Building 2×4 figure …")
        make_2x4_figure(
            sup_rgb, q_rgb, gt,
            heats, preds, ious,
            ep_idx, out_path,
        )
        break   # only one episode

    print("Done.")


if __name__ == "__main__":
    main()
