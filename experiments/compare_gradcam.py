"""Compare SegScoreCAM interpretability metrics across the three models.

Loads gradcam/metrics.json from each model directory, aggregates per-image
scores, and produces:
  - results/seg_uav/gradcam_comparison.csv   — full per-image table
  - results/seg_uav/gradcam_comparison.pdf   — bar chart (mean ± std)
  - printed summary table to stdout

Metrics (sugarcane class, where GT alignment and confidence gain are defined):
  gt_iou            — IoU between Otsu-thresholded CAM and GT mask   (↑ better)
  gt_recall         — fraction of GT pixels covered by the CAM        (↑ better)
  gt_precision      — fraction of CAM pixels that fall on GT          (↑ better)
  gt_pearson        — Pearson r between continuous CAM and binary GT  (↑ better)
  mean_activation   — mean spatial activation                         (↓ = focused)
  mean_conf_change  — mean % confidence change on sugarcane pixels    (↑ better)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from configs.seg_uav import AVAILABLE_MODELS, RESULTS_DIR

METRICS = [
    "gt_iou", "gt_recall", "gt_precision", "gt_pearson",
    "mean_activation", "mean_conf_change",
]
LABELS = {
    "gt_iou":           "GT IoU ↑",
    "gt_recall":        "GT Recall ↑",
    "gt_precision":     "GT Precision ↑",
    "gt_pearson":       "GT Pearson ↑",
    "mean_activation":  "Mean activation ↓",
    "mean_conf_change": "Conf. change % ↑",
}
# metrics where lower is better
LOWER_IS_BETTER = {"mean_activation"}


def load_metrics(model_type: str) -> pd.DataFrame:
    path = RESULTS_DIR / model_type / "gradcam" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"metrics.json not found for {model_type}: {path}")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for image_name, classes in data.items():
        sg = classes.get("sugarcane", {})
        cg = sg.get("confidence_gain", {}) or {}
        rows.append({
            "model":            model_type,
            "image":            image_name,
            "gt_iou":           sg.get("gt_iou",           float("nan")),
            "gt_recall":        sg.get("gt_recall",         float("nan")),
            "gt_precision":     sg.get("gt_precision",      float("nan")),
            "gt_pearson":       sg.get("gt_pearson",        float("nan")),
            "mean_activation":  sg.get("mean_activation",   float("nan")),
            "mean_conf_change": cg.get("mean_pct_confidence_change", float("nan")),
        })
    return pd.DataFrame(rows)


def main():
    frames = []
    for model_type in AVAILABLE_MODELS:
        try:
            frames.append(load_metrics(model_type))
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    if not frames:
        print("No metrics found. Run interpretability_analysis.py first.")
        return

    df = pd.concat(frames, ignore_index=True)

    # --- per-image CSV ---
    csv_path = RESULTS_DIR / "gradcam_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-image table → {csv_path}\n")

    # --- aggregate stats ---
    agg = (
        df.groupby("model")[METRICS]
        .agg(["mean", "std"])
        .round(4)
    )
    print("=== SegScoreCAM interpretability — mean ± std across test images ===")
    print(agg.to_string())
    print()

    # Winner per metric
    means = df.groupby("model")[METRICS].mean()
    print("=== Best model per metric ===")
    for m in METRICS:
        if m in LOWER_IS_BETTER:
            winner = means[m].idxmin()
            note = "(lower is better)"
        else:
            winner = means[m].idxmax()
            note = "(higher is better)"
        print(f"  {LABELS[m]:<22} → {winner}  {note}")
    print()

    # --- bar chart ---
    n_metrics = len(METRICS)
    n_models  = len(AVAILABLE_MODELS)
    x = np.arange(n_metrics)
    width = 0.22
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_type in enumerate(AVAILABLE_MODELS):
        if model_type not in means.index:
            continue
        mu  = means.loc[model_type, METRICS].values
        std = df[df["model"] == model_type][METRICS].std().values
        ax.bar(
            x + i * width,
            mu,
            width,
            yerr=std,
            capsize=4,
            label=model_type,
            color=colours[i % len(colours)],
            alpha=0.85,
        )

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([LABELS[m] for m in METRICS], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("SegScoreCAM — interpretability comparison", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    pdf_path = RESULTS_DIR / "gradcam_comparison.pdf"
    fig.savefig(pdf_path, dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart → {pdf_path}")


if __name__ == "__main__":
    main()
