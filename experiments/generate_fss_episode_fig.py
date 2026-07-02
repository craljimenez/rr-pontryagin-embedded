"""generate_fss_episode_fig.py

Regenera Figure 4.10: panel de episodio FSS en layout 2×3 con títulos
normalizados (RFPE en lugar de Pontryagin) y 300 DPI PDF.

Layout 2×3:
  Row 0: Support 1  | FSS-Euclidean ScoreCAM      | FSS-RFPE ScoreCAM
  Row 1: Query      | FSS-Euclidean Prediction     | FSS-RFPE Prediction

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


# ─────────────────────────────────────────────────────────────────────────────

def make_2x3_figure(
    sup_rgb, q_rgb, gt,
    heat_e, pred_e, iou_e,
    heat_p, pred_p, iou_p,
    episode_idx,
    out_path,
):
    """
    Row 0: Support 1  | FSS-Euclidean ScoreCAM  | FSS-RFPE ScoreCAM
    Row 1: Query      | FSS-Euclidean Prediction | FSS-RFPE Prediction
    """
    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 8),
        gridspec_kw={"wspace": 0.06, "hspace": 0.18},
    )

    # ── Row 0 ────────────────────────────────────────────────────────────────
    axes[0, 0].imshow(sup_rgb)
    _gt_contour(axes[0, 0], gt, color="lime")
    axes[0, 0].set_title("Support 1", fontsize=11, fontweight="bold")
    _axoff(axes[0, 0])

    axes[0, 1].imshow(_overlay(q_rgb, heat_e))
    _gt_contour(axes[0, 1], gt)
    axes[0, 1].set_title("FSS-Euclidean\nScoreCAM", fontsize=11, fontweight="bold")
    _axoff(axes[0, 1])

    axes[0, 2].imshow(_overlay(q_rgb, heat_p))
    _gt_contour(axes[0, 2], gt)
    axes[0, 2].set_title("FSS-RFPE\nScoreCAM", fontsize=11, fontweight="bold")
    _axoff(axes[0, 2])

    # ── Row 1 ────────────────────────────────────────────────────────────────
    axes[1, 0].imshow(q_rgb)
    _gt_contour(axes[1, 0], gt)
    axes[1, 0].set_title("Query", fontsize=11, fontweight="bold")
    _axoff(axes[1, 0])

    axes[1, 1].imshow(pred_e, cmap="gray", vmin=0, vmax=1)
    _gt_contour(axes[1, 1], gt)
    axes[1, 1].set_title(
        f"FSS-Euclidean Prediction\nIoU = {iou_e:.3f}",
        fontsize=11, fontweight="bold",
    )
    _axoff(axes[1, 1])

    axes[1, 2].imshow(pred_p, cmap="gray", vmin=0, vmax=1)
    _gt_contour(axes[1, 2], gt)
    axes[1, 2].set_title(
        f"FSS-RFPE Prediction\nIoU = {iou_p:.3f}",
        fontsize=11, fontweight="bold",
    )
    _axoff(axes[1, 2])

    fig.suptitle(
        f"Episode {episode_idx:03d} — ScoreCAM comparison  (green contour = GT)",
        fontsize=11, y=1.01,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models …")
    model_euc  = _load_model("euclidean",  args.k_shot, RESULTS_DIR, device, False)
    model_pont = _load_model("pontryagin", args.k_shot, RESULTS_DIR, device, False)
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
        with torch.no_grad():
            pred_e = (model_euc.forward(sup_imgs, sup_masks, q_img)[0] > 0).float().cpu().numpy()
            pred_p = (model_pont.forward(sup_imgs, sup_masks, q_img)[0] > 0).float().cpu().numpy()

        iou_e = _iou(pred_e, gt)
        iou_p = _iou(pred_p, gt)
        print(f"  IoU — Euclidean: {iou_e:.3f}   RFPE: {iou_p:.3f}")

        print("  Computing ScoreCAM …")
        cam_euc  = FSSScoreCAM(model_euc)
        cam_pont = FSSScoreCAM(model_pont)
        heat_e = cam_euc.compute(sup_imgs, sup_masks, q_img)
        heat_p = cam_pont.compute(sup_imgs, sup_masks, q_img)
        cam_euc.remove()
        cam_pont.remove()

        out_path = OUT_DIR / f"fss_ep_example.pdf"
        print("  Building 2×3 figure …")
        make_2x3_figure(
            sup_rgb, q_rgb, gt,
            heat_e, pred_e, iou_e,
            heat_p, pred_p, iou_p,
            ep_idx, out_path,
        )
        break   # only one episode

    print("Done.")


if __name__ == "__main__":
    main()
