"""generate_fss_cam_metrics.py

Regenera Figure 4.12: ScoreCAM quality metrics (CAM-GT IoU, Recall,
Precision, Pearson) en layout 2×2 con terminología RFPE y 300 DPI PDF.

Lee el CSV ya calculado en:
  results/fss_sugarcane/interpretability_1shot/metrics.csv

Output: report/figures/fss_cam_metrics.pdf
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR  = Path(__file__).parent
OUT_DIR  = EXP_DIR / "report" / "figures"
CSV_PATH = EXP_DIR / "results" / "fss_sugarcane" / "interpretability_1shot" / "metrics.csv"

K_SHOT = 1

CAM_PAIRS = [
    ("cam_euc_gt_iou",       "cam_pont_gt_iou",       "GT IoU"),
    ("cam_euc_gt_recall",    "cam_pont_gt_recall",    "GT Recall"),
    ("cam_euc_gt_precision", "cam_pont_gt_precision", "GT Precision"),
    ("cam_euc_gt_pearson",   "cam_pont_gt_pearson",   "GT Pearson r"),
]


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})

    print(f"Loaded {len(rows)} episodes from {CSV_PATH.name}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9),
                             gridspec_kw={"wspace": 0.32, "hspace": 0.42})
    axes_flat = axes.flatten()

    for ax, (key_e, key_p, label) in zip(axes_flat, CAM_PAIRS):
        vals_e = np.array([r[key_e] for r in rows], dtype=float)
        vals_p = np.array([r[key_p] for r in rows], dtype=float)
        valid  = ~(np.isnan(vals_e) | np.isnan(vals_p))

        if valid.any():
            bp = ax.boxplot(
                [vals_e[valid], vals_p[valid]],
                labels=["Euclidean", "RFPE"],
                patch_artist=True,
                medianprops=dict(color="red", linewidth=2),
            )
            bp["boxes"][0].set_facecolor("lightsteelblue")
            bp["boxes"][1].set_facecolor("peachpuff")
            m_e = vals_e[valid].mean()
            m_p = vals_p[valid].mean()
            ax.set_title(
                f"{label}\nEuc={m_e:.3f}  RFPE={m_p:.3f}",
                fontsize=10,
            )
        ax.set_ylabel(label, fontsize=10)

    fig.suptitle(
        f"ScoreCAM quality vs GT — {K_SHOT}-shot FSS",
        fontsize=12, fontweight="bold",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fss_cam_metrics.pdf"
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
