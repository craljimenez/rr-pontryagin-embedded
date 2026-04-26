"""Compare segmentation and interpretability metrics across 4 UNet models.

Loads:
  results/seg_uav_unet/<model>/metrics.json          → segmentation metrics
  results/seg_uav_unet/<model>/gradcam/metrics.json  → SegScoreCAM metrics

Produces:
  results/seg_uav_unet/seg_comparison.csv    — segmentation metrics table
  results/seg_uav_unet/seg_comparison.pdf    — bar chart
  results/seg_uav_unet/gradcam_comparison.csv
  results/seg_uav_unet/gradcam_comparison.pdf

Usage:
    python compare_unet.py
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from configs.seg_uav_unet import AVAILABLE_MODELS, RESULTS_DIR as _DEFAULT_RESULTS_DIR

COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation comparison
# ─────────────────────────────────────────────────────────────────────────────

SEG_METRICS = ["macro_iou", "macro_dice", "macro_precision",
               "macro_recall", "pixel_acc"]
SEG_LABELS  = {
    "macro_iou":       "Macro IoU ↑",
    "macro_dice":      "Macro Dice ↑",
    "macro_precision": "Macro Precision ↑",
    "macro_recall":    "Macro Recall ↑",
    "pixel_acc":       "Pixel Accuracy ↑",
}


def load_seg_metrics(results_dir):
    rows = []
    for mt in AVAILABLE_MODELS:
        path = results_dir / mt / "metrics.json"
        if not path.exists():
            print(f"  [skip] {mt}: no metrics.json")
            continue
        with open(path) as f:
            d = json.load(f)
        rows.append({"model": mt, **{k: d.get(k, float("nan"))
                                      for k in SEG_METRICS}})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Interpretability comparison
# ─────────────────────────────────────────────────────────────────────────────

CAM_METRICS = ["gt_iou", "gt_recall", "gt_precision", "gt_pearson",
               "mean_activation", "mean_conf_change"]
CAM_LABELS  = {
    "gt_iou":           "GT IoU ↑",
    "gt_recall":        "GT Recall ↑",
    "gt_precision":     "GT Precision ↑",
    "gt_pearson":       "GT Pearson ↑",
    "mean_activation":  "Mean Activation ↓",
    "mean_conf_change": "Confidence Change % (→ 0)",
}
CAM_LOWER_IS_BETTER  = {"mean_activation"}
CAM_CLOSER_TO_ZERO   = {"mean_conf_change"}   # best = |value| closest to 0


def load_cam_metrics(results_dir):
    rows = []
    for mt in AVAILABLE_MODELS:
        path = results_dir / mt / "gradcam" / "metrics.json"
        if not path.exists():
            print(f"  [skip] {mt}: no gradcam/metrics.json")
            continue
        with open(path) as f:
            data = json.load(f)
        for img_name, classes in data.items():
            sg = classes.get("sugarcane", {})
            cg = sg.get("confidence_gain", {}) or {}
            rows.append({
                "model":            mt,
                "image":            img_name,
                "gt_iou":           sg.get("gt_iou",           float("nan")),
                "gt_recall":        sg.get("gt_recall",         float("nan")),
                "gt_precision":     sg.get("gt_precision",      float("nan")),
                "gt_pearson":       sg.get("gt_pearson",        float("nan")),
                "mean_activation":  sg.get("mean_activation",   float("nan")),
                "mean_conf_change": cg.get("mean_pct_confidence_change",
                                           float("nan")),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _single_metric_chart(metric: str, label: str,
                         means: pd.DataFrame, stds: pd.DataFrame,
                         out_path: Path) -> None:
    """One PDF per metric — 4 bars (one per model) with ±1 std error bars."""
    present = [m for m in AVAILABLE_MODELS if m in means.index]
    x       = np.arange(len(present))
    vals    = means.loc[present, metric].values.astype(float)
    errs    = stds.loc[present, metric].values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, vals, 0.55,
           color=[COLOURS[i % len(COLOURS)] for i in range(len(present))],
           alpha=0.85,
           yerr=errs, capsize=5,
           error_kw={"elinewidth": 1.4, "ecolor": "black", "alpha": 0.75})

    # Zero reference line (visible when values span negative range)
    lo = (vals - errs).min()
    hi = (vals + errs).max()
    if lo < 0 < hi or metric in CAM_CLOSER_TO_ZERO:
        ax.axhline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"{label}  (mean ± std, test set)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _bar_chart(means: pd.DataFrame, metrics: list, labels: dict,
               title: str, out_path: Path,
               stds: pd.DataFrame | None = None) -> None:
    present = [m for m in AVAILABLE_MODELS if m in means.index]
    n_m = len(metrics)
    n_mdl = len(present)
    x = np.arange(n_m)
    width = 0.8 / n_mdl

    fig, ax = plt.subplots(figsize=(max(10, n_m * 1.8), 5))
    for i, mt in enumerate(present):
        vals = means.loc[mt, metrics].values.astype(float)
        yerr = None
        if stds is not None and mt in stds.index:
            yerr = stds.loc[mt, metrics].values.astype(float)
        ax.bar(x + i * width, vals, width, label=mt,
               color=COLOURS[i % len(COLOURS)], alpha=0.85,
               yerr=yerr, capsize=4,
               error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7})

    ax.set_xticks(x + width * (n_mdl - 1) / 2)
    ax.set_xticklabels([labels[m] for m in metrics], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=None,
                    help="Output directory for results (overrides config).")
    args = ap.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else _DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Segmentation ─────────────────────────────────────────────────────────
    print("=== Segmentation metrics ===")
    seg_df = load_seg_metrics(results_dir)
    if not seg_df.empty:
        seg_df.to_csv(results_dir / "seg_comparison.csv", index=False)
        print(seg_df.set_index("model")[SEG_METRICS].round(4).to_string())
        print()

        seg_means = seg_df.set_index("model")[SEG_METRICS]
        print("Best model per segmentation metric:")
        for m in SEG_METRICS:
            winner = seg_means[m].idxmax()
            print(f"  {SEG_LABELS[m]:<22} → {winner}")

        _bar_chart(seg_means, SEG_METRICS, SEG_LABELS,
                   "UNet segmentation comparison",
                   results_dir / "seg_comparison.pdf")

    # ── Interpretability ─────────────────────────────────────────────────────
    print("\n=== SegScoreCAM interpretability metrics ===")
    cam_df = load_cam_metrics(results_dir)
    if not cam_df.empty:
        cam_df.to_csv(results_dir / "gradcam_comparison.csv", index=False)

        agg = cam_df.groupby("model")[CAM_METRICS].agg(["mean", "std"]).round(4)
        print(agg.to_string())
        print()

        cam_means = cam_df.groupby("model")[CAM_METRICS].mean()
        cam_stds  = cam_df.groupby("model")[CAM_METRICS].std(ddof=1).fillna(0)
        print("Best model per CAM metric:")
        for m in CAM_METRICS:
            if m in CAM_LOWER_IS_BETTER:
                winner = cam_means[m].idxmin()
                note = "(lower is better)"
            elif m in CAM_CLOSER_TO_ZERO:
                winner = cam_means[m].abs().idxmin()
                note = "(closer to 0 is better)"
            else:
                winner = cam_means[m].idxmax()
                note = "(higher is better)"
            print(f"  {CAM_LABELS[m]:<22} → {winner}  {note}")

        # Combined overview chart
        _bar_chart(cam_means, CAM_METRICS, CAM_LABELS,
                   "UNet SegScoreCAM interpretability comparison (mean ± std across test images)",
                   results_dir / "gradcam_comparison.pdf",
                   stds=cam_stds)

        # Individual per-metric charts for detailed inspection
        gradcam_dir = results_dir / "gradcam_per_metric"
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        for m in CAM_METRICS:
            _single_metric_chart(
                metric=m, label=CAM_LABELS[m],
                means=cam_means, stds=cam_stds,
                out_path=gradcam_dir / f"{m}.pdf",
            )


if __name__ == "__main__":
    main()
