"""generate_fss_iou.py

Regenera Figure 4.11: IoU per test episode — 1-shot FSS.
Lee el CSV ya calculado y corrige el legend/título (RFPE en lugar de Pontryagin).

Output: report/figures/fss_iou_comparison.pdf  (300 DPI)
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
W      = 0.35


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}"); sys.exit(1)

    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})

    ep    = np.array([r["episode"]        for r in rows])
    iou_e = np.array([r["iou_euclidean"]  for r in rows])
    iou_p = np.array([r["iou_pontryagin"] for r in rows])

    fig, ax = plt.subplots(figsize=(max(8, len(ep) * 0.4 + 2), 4))

    ax.bar(ep - W / 2, iou_e, width=W, label="Euclidean", alpha=0.8, color="steelblue")
    ax.bar(ep + W / 2, iou_p, width=W, label="RFPE",      alpha=0.8, color="darkorange")
    ax.axhline(iou_e.mean(), color="steelblue",  linestyle="--", linewidth=1,
               label=f"Euc mean={iou_e.mean():.3f}")
    ax.axhline(iou_p.mean(), color="darkorange", linestyle="--", linewidth=1,
               label=f"RFPE mean={iou_p.mean():.3f}")

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("IoU",     fontsize=10)
    ax.set_title(f"IoU per test episode — {K_SHOT}-shot FSS", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xticks(ep)

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fss_iou_comparison.pdf"
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
