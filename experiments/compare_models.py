"""Aggregate test-set metrics of the three seg heads into a comparison table.

Reads experiments/results/seg_uav/<model>/metrics_detailed.json for every
model found, prints side-by-side tables (global + per-class IoU/Dice), and
saves a consolidated `comparison.csv` plus a `comparison.png` bar chart.

    python experiments/compare_models.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from configs.seg_uav import AVAILABLE_MODELS, CLASS_NAMES, RESULTS_DIR


GLOBAL_KEYS = [
    "pixel_acc", "macro_iou", "macro_dice",
    "macro_precision", "macro_recall",
    "micro_iou", "micro_dice",
]


def load_results() -> dict:
    results = {}
    for model in AVAILABLE_MODELS:
        path = RESULTS_DIR / model / "metrics_detailed.json"
        if path.exists():
            with open(path) as f:
                results[model] = json.load(f)
        else:
            print(f"  ⚠  missing: {path}")
    return results


def print_global_table(results: dict) -> None:
    models = list(results.keys())
    if not models:
        return
    col_w = 14
    print(f"\n{'Metric':<20}" + "".join(f"{m:>{col_w}}" for m in models))
    print("─" * (20 + col_w * len(models)))
    for key in GLOBAL_KEYS:
        row = f"{key:<20}"
        for m in models:
            v = results[m]["test"]["global"].get(key, 0.0)
            row += f"{v:>{col_w}.4f}"
        print(row)


def print_per_class_table(results: dict, metric: str = "iou") -> None:
    models = list(results.keys())
    if not models:
        return
    col_w = 14
    print(f"\nPer-class {metric.upper()}")
    print(f"{'Class':<18}" + "".join(f"{m:>{col_w}}" for m in models))
    print("─" * (18 + col_w * len(models)))
    for idx, name in enumerate(CLASS_NAMES):
        row = f"{name:<18}"
        for m in models:
            entry = results[m]["test"]["per_class"][idx]
            if entry["support"] == 0:
                row += f"{'—':>{col_w}}"
            else:
                row += f"{entry[metric]:>{col_w}.4f}"
        print(row)


def save_comparison_csv(results: dict, path: Path) -> None:
    models = list(results.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key"] + models)
        # Global metrics
        for key in GLOBAL_KEYS:
            row = ["global", key]
            for m in models:
                row.append(round(results[m]["test"]["global"][key], 6))
            w.writerow(row)
        # Per-class IoU and Dice
        for idx, name in enumerate(CLASS_NAMES):
            for metric in ("iou", "dice"):
                row = [f"class_{metric}", name]
                for m in models:
                    entry = results[m]["test"]["per_class"][idx]
                    row.append("" if entry["support"] == 0
                               else round(entry[metric], 6))
                w.writerow(row)


def save_comparison_plot(results: dict, path: Path) -> None:
    models = list(results.keys())
    if not models:
        return
    present_classes = [
        (i, CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))
        if any(results[m]["test"]["per_class"][i]["support"] > 0 for m in models)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Global macro metrics bar chart
    metrics = ["macro_iou", "macro_dice", "micro_iou", "micro_dice", "pixel_acc"]
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [results[m]["test"]["global"][k] for k in metrics]
        axes[0].bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=20, ha="right")
    axes[0].set_ylabel("score")
    axes[0].set_title("Global metrics")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    # Per-class IoU
    x = np.arange(len(present_classes))
    for i, m in enumerate(models):
        vals = [results[m]["test"]["per_class"][idx]["iou"] for idx, _ in present_classes]
        axes[1].bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([name for _, name in present_classes], rotation=30, ha="right")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Per-class IoU (test)")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_results()
    if not results:
        print("No results found. Train the models first.")
        return

    print_global_table(results)
    print_per_class_table(results, metric="iou")
    print_per_class_table(results, metric="dice")

    out_csv = RESULTS_DIR / "comparison.csv"
    out_png = RESULTS_DIR / "comparison.png"
    save_comparison_csv(results, out_csv)
    save_comparison_plot(results, out_png)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
