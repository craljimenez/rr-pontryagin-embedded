"""generate_fss_mc_variance.py

Regenera Figure 4.13: MC Dropout uncertainty per episode — 1-shot FSS.
Corrige legend (RFPE en lugar de Pontryagin). 300 DPI PDF.

Output: report/figures/fss_mc_variance.pdf
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
N_MC   = 50
W      = 0.35


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}"); sys.exit(1)

    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})

    ep    = np.array([r["episode"]           for r in rows])
    var_e = np.array([r["mc_var_euc_mean"]   for r in rows])
    var_p = np.array([r["mc_var_pont_mean"]  for r in rows])

    fig, ax = plt.subplots(figsize=(max(8, len(ep) * 0.4 + 2), 4))

    ax.bar(ep - W / 2, var_e, width=W, label="Euclidean", alpha=0.8, color="steelblue")
    ax.bar(ep + W / 2, var_p, width=W, label="RFPE",      alpha=0.8, color="darkorange")

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Mean predictive variance", fontsize=10)
    ax.set_title(
        f"MC Dropout uncertainty per episode — {K_SHOT}-shot FSS (N={N_MC})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_xticks(ep)

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fss_mc_variance.pdf"
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
