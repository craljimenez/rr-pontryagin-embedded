"""Combined FCN vs UNet comparison figures for the proposal.

Loads results from:
  results/seg_uav/<model>/          → FCN segmentation  (metrics_detailed.json)
  results/seg_uav/<model>/gradcam/  → FCN SegScoreCAM   (metrics.json)
  results/seg_uav_unet/<model>/     → UNet segmentation (metrics_per_class.csv + metrics.json)
  results/seg_uav_unet/<model>/gradcam/ → UNet SegScoreCAM (metrics.json)

Ignores 'vanilla' (no FCN vanilla exists; excluded by request).

Outputs (results/combined/):
  seg_comparison.pdf          — FCN vs UNet segmentation, grouped by model
  cam_<metric>.pdf            — one PDF per CAM metric, FCN vs UNet
"""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent / "results"
OUT  = BASE / "combined"
OUT.mkdir(parents=True, exist_ok=True)

MODELS  = ["euclidean", "hyperbolic", "pontryagin"]
LABELS  = {"euclidean": "Euclidean", "hyperbolic": "Hyperbolic", "pontryagin": "Pontryagin"}

FCN_COLOR  = "#4C72B0"
UNET_COLOR = "#DD8452"

CONF_CLIP = 100.0   # |confidence change| > this is clipped and marked


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_fcn_seg(model: str) -> dict | None:
    path = BASE / "seg_uav" / model / "metrics_detailed.json"
    if not path.exists():
        print(f"  [skip] FCN {model}: no metrics_detailed.json")
        return None
    d = json.loads(path.read_text())["test"]
    per = d["per_class"]
    return {
        "global": d["global"],
        "per_class_iou":  [c["iou"]  for c in per],
        "per_class_dice": [c["dice"] for c in per],
        "per_class_prec": [c["precision"] for c in per],
        "per_class_rec":  [c["recall"]    for c in per],
    }


def load_unet_seg(model: str) -> dict | None:
    g_path = BASE / "seg_uav_unet" / model / "metrics.json"
    c_path = BASE / "seg_uav_unet" / model / "metrics_per_class.csv"
    if not g_path.exists():
        print(f"  [skip] UNet {model}: no metrics.json")
        return None
    g = json.loads(g_path.read_text())
    per = list(csv.DictReader(c_path.open()))
    return {
        "global": g,
        "per_class_iou":  [float(r["iou"])       for r in per],
        "per_class_dice": [float(r["dice"])       for r in per],
        "per_class_prec": [float(r["precision"])  for r in per],
        "per_class_rec":  [float(r["recall"])     for r in per],
    }


def load_cam(root: Path, model: str) -> list[dict]:
    path = root / model / "gradcam" / "metrics.json"
    if not path.exists():
        print(f"  [skip] {root.name}/{model}: no gradcam/metrics.json")
        return []
    data = json.loads(path.read_text())
    rows = []
    for img, cls in data.items():
        sg = cls.get("sugarcane", {})
        cg = (sg.get("confidence_gain") or {})
        raw_cc = cg.get("mean_pct_confidence_change")
        clipped = False
        if raw_cc is not None and abs(raw_cc) > CONF_CLIP:
            raw_cc = float(np.sign(raw_cc) * CONF_CLIP)
            clipped = True
        rows.append({
            "gt_iou":          sg.get("gt_iou",          np.nan),
            "gt_recall":       sg.get("gt_recall",        np.nan),
            "gt_precision":    sg.get("gt_precision",     np.nan),
            "gt_pearson":      sg.get("gt_pearson",       np.nan),
            "mean_activation": sg.get("mean_activation",  np.nan),
            "conf_change":     raw_cc if raw_cc is not None else np.nan,
            "conf_clipped":    clipped,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nanstd(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _nanmean(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation figure
# ─────────────────────────────────────────────────────────────────────────────

SEG_KEYS = [
    ("macro_iou",       "Macro IoU ↑"),
    ("macro_dice",      "Macro Dice ↑"),
    ("macro_precision", "Macro Precision ↑"),
    ("macro_recall",    "Macro Recall ↑"),
    ("pixel_acc",       "Pixel Accuracy ↑"),
]
SEG_CLASS_KEY = {
    "macro_iou":       "per_class_iou",
    "macro_dice":      "per_class_dice",
    "macro_precision": "per_class_prec",
    "macro_recall":    "per_class_rec",
    "pixel_acc":       None,
}


def plot_seg_comparison():
    print("Loading segmentation data …")
    fcn_data  = {m: load_fcn_seg(m)  for m in MODELS}
    unet_data = {m: load_unet_seg(m) for m in MODELS}

    n_metrics = len(SEG_KEYS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5), sharey=False)

    bar_w = 0.32
    x     = np.array([0, 1, 2])

    for ax, (key, label) in zip(axes, SEG_KEYS):
        cls_key = SEG_CLASS_KEY[key]

        fcn_vals, fcn_errs   = [], []
        unet_vals, unet_errs = [], []

        for m in MODELS:
            fd = fcn_data[m]
            ud = unet_data[m]

            if fd:
                fcn_vals.append(fd["global"][key])
                fcn_errs.append(_nanstd(fd[cls_key]) if cls_key else 0.0)
            else:
                fcn_vals.append(np.nan); fcn_errs.append(0.0)

            if ud:
                unet_vals.append(ud["global"][key])
                unet_errs.append(_nanstd(ud[cls_key]) if cls_key else 0.0)
            else:
                unet_vals.append(np.nan); unet_errs.append(0.0)

        ax.bar(x - bar_w / 2, fcn_vals,  bar_w, label="FCN",  color=FCN_COLOR,
               alpha=0.85, yerr=fcn_errs,  capsize=4,
               error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7})
        ax.bar(x + bar_w / 2, unet_vals, bar_w, label="UNet", color=UNET_COLOR,
               alpha=0.85, yerr=unet_errs, capsize=4,
               error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7},
               hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=9, pad=4)
        ax.set_ylim(bottom=0, top=1.08)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        if ax == axes[0]:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("FCN vs UNet — Segmentation comparison (test set, error bars = std over classes)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = OUT / "seg_comparison.pdf"
    fig.savefig(out, dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CAM figures (one PDF per metric)
# ─────────────────────────────────────────────────────────────────────────────

CAM_KEYS = [
    ("gt_iou",          "GT IoU ↑",                      False),
    ("gt_recall",       "GT Recall ↑",                   False),
    ("gt_precision",    "GT Precision ↑",                False),
    ("gt_pearson",      "GT Pearson correlation ↑",      False),
    ("mean_activation", "Mean CAM Activation ↓",         False),
    ("conf_change",     f"Confidence Change % (→ 0, clipped ±{CONF_CLIP:.0f}%)", True),
]


def plot_cam_comparison():
    print("Loading CAM data …")
    fcn_rows  = {m: load_cam(BASE / "seg_uav",      m) for m in MODELS}
    unet_rows = {m: load_cam(BASE / "seg_uav_unet", m) for m in MODELS}

    cam_dir = OUT / "cam_per_metric"
    cam_dir.mkdir(exist_ok=True)

    x     = np.array([0, 1, 2])
    bar_w = 0.32

    for key, label, is_conf in CAM_KEYS:
        fig, ax = plt.subplots(figsize=(6, 4))

        fcn_means, fcn_stds   = [], []
        unet_means, unet_stds = [], []
        any_clipped = False

        for m in MODELS:
            fv = [r[key] for r in fcn_rows[m]  if not np.isnan(r[key])]
            uv = [r[key] for r in unet_rows[m] if not np.isnan(r[key])]

            if is_conf:
                fc = any(r["conf_clipped"] for r in fcn_rows[m])
                uc = any(r["conf_clipped"] for r in unet_rows[m])
                if fc or uc:
                    any_clipped = True

            fcn_means.append(_nanmean(fv));  fcn_stds.append(_nanstd(fv))
            unet_means.append(_nanmean(uv)); unet_stds.append(_nanstd(uv))

        ax.bar(x - bar_w / 2, fcn_means,  bar_w, label="FCN",  color=FCN_COLOR,
               alpha=0.85, yerr=fcn_stds,  capsize=5,
               error_kw={"elinewidth": 1.3, "ecolor": "black", "alpha": 0.75})
        ax.bar(x + bar_w / 2, unet_means, bar_w, label="UNet", color=UNET_COLOR,
               alpha=0.85, yerr=unet_stds, capsize=5,
               error_kw={"elinewidth": 1.3, "ecolor": "black", "alpha": 0.75},
               hatch="//")

        # Reference line at 0 for metrics that span negative values
        all_vals = [v for v in fcn_means + unet_means if not np.isnan(v)]
        if all_vals and (min(all_vals) < 0 or is_conf):
            ax.axhline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=11)
        ax.set_ylabel(label, fontsize=10)

        title = f"FCN vs UNet — {label}\n(mean ± std over test images)"
        if is_conf and any_clipped:
            title += f"\n† values outside ±{CONF_CLIP:.0f}% were clipped"
        ax.set_title(title, fontsize=9, pad=6)
        ax.legend(fontsize=9)
        plt.tight_layout()

        out = cam_dir / f"{key}.pdf"
        fig.savefig(out, dpi=150, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=None)
    args = ap.parse_args()
    if args.results_dir:
        global BASE, OUT
        BASE = Path(args.results_dir)
        OUT  = BASE / "combined"
        OUT.mkdir(parents=True, exist_ok=True)

    print("=== Segmentation comparison ===")
    plot_seg_comparison()

    print("\n=== CAM comparison ===")
    plot_cam_comparison()

    print(f"\nAll outputs → {OUT}")


if __name__ == "__main__":
    main()
