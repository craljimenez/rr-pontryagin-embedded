"""Tabulate PASCAL VOC 2012 segmentation results and generate the
per-class comparison figure, formatted for the cas-dc paper.

Reads results/seg_pascalvoc_results/{euclidean,hyperbolic,pontryagin}/
    metrics.json            → global test metrics
    metrics_per_class.csv   → per-class TP/FP/FN + IoU/Dice/Prec/Rec

Outputs:
    results/seg_pascalvoc_results/summary_overall.csv
    results/seg_pascalvoc_results/summary_per_class.csv
    report/tables/pascalvoc_seg_table.tex        (tabla global, estilo tab:seg)
    report/tables/pascalvoc_perclass_table.tex   (tabla IoU por clase)
    report/figures/pascalvoc_perclass_paper.pdf  (dot plot IoU por clase)

Uso:
    python experiments/generate_pascalvoc_tables_figs.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR  = Path(__file__).parent
RES_DIR  = EXP_DIR / "results" / "seg_pascalvoc_results"
FIG_DIR  = EXP_DIR / "report" / "figures"
TAB_DIR  = EXP_DIR / "report" / "tables"

HEADS  = ["euclidean", "hyperbolic", "pontryagin", "pontryagin_trff"]
LABELS = {"euclidean": "Euclidean", "hyperbolic": "Hyperbolic",
          "pontryagin": "PRFE", "pontryagin_trff": "PRFE-tRFF"}
# LaTeX-only labels with dagger/double-dagger, matching tab:cls's convention
# for fixed-RFF vs trainable-RFF Pontryagin variants.
LATEX_LABELS = {"euclidean": "Euclidean", "hyperbolic": "Hyperbolic",
                "pontryagin": r"PRFE\textsuperscript{\dag}",
                "pontryagin_trff": r"PRFE-tRFF\textsuperscript{\ddag}"}
# Paleta categórica validada (dataviz): azul / aqua / naranja / violeta
COLORS = {"euclidean": "#2a78d6", "hyperbolic": "#1baf7a",
          "pontryagin": "#eb6834", "pontryagin_trff": "#4a3aa7"}


def load_results():
    glob, per = {}, {}
    for h in HEADS:
        glob[h] = json.load(open(RES_DIR / h / "metrics.json"))
        with open(RES_DIR / h / "metrics_per_class.csv") as f:
            per[h] = list(csv.DictReader(f))
    return glob, per


# ─────────────────────────────────────────────────────────────────────────────
# CSV summaries
# ─────────────────────────────────────────────────────────────────────────────

def write_summaries(glob, per):
    keys = ["macro_iou", "macro_dice", "macro_precision", "macro_recall",
            "micro_iou", "pixel_acc", "best_val_macro_iou"]
    with open(RES_DIR / "summary_overall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head"] + keys)
        for h in HEADS:
            w.writerow([h] + [f"{glob[h][k]:.4f}" for k in keys])

    classes = [r["class"] for r in per[HEADS[0]]]
    with open(RES_DIR / "summary_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support"]
                   + [f"iou_{h}" for h in HEADS]
                   + [f"dice_{h}" for h in HEADS])
        for i, c in enumerate(classes):
            w.writerow([c, per[HEADS[0]][i]["support"]]
                       + [f"{float(per[h][i]['iou']):.4f}" for h in HEADS]
                       + [f"{float(per[h][i]['dice']):.4f}" for h in HEADS])
    print("  summary_overall.csv / summary_per_class.csv written")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX tables (misma convención tipográfica que tab:seg del paper)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_row(vals, best_mask):
    out = []
    for v, b in zip(vals, best_mask):
        s = f"{v:.3f}"
        out.append(rf"$\mathbf{{{s}}}$" if b else f"${s}$")
    return out


def write_global_table(glob):
    metric_keys = ["macro_iou", "macro_dice", "macro_precision",
                   "macro_recall", "micro_iou", "pixel_acc"]
    vals = {h: [glob[h][k] for k in metric_keys] for h in HEADS}
    best = {k: max(HEADS, key=lambda h: glob[h][k]) for k in metric_keys}

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Multiclass semantic segmentation on PASCAL VOC 2012"
        r" (21 classes, 725 held-out test images). All heads share the same"
        r" ImageNet-pretrained ResNet-34 U-Net backbone; hyperparameters"
        r" tuned per head with Bayesian optimisation (20--30 trials,"
        r" scaled with search-space dimensionality; Section~\ref{sec:setup:data})."
        r" \textsuperscript{\dag}~Fixed RFF. \textsuperscript{\ddag}~Trainable RFF (tRFF)."
        r" Best values in \textbf{bold}.}",
        r"\label{tab:voc}",
        r"\centering",
        r"\small",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}",
        r"\toprule",
        r"Head & mIoU$\uparrow$ & Dice$\uparrow$ & Prec.$\uparrow$"
        r" & Rec.$\uparrow$ & $\text{IoU}_{\mu}$$\uparrow$ & Acc.$\uparrow$ \\",
        r"\midrule",
    ]
    for h in HEADS:
        mask = [best[k] == h for k in metric_keys]
        cells = _fmt_row(vals[h], mask)
        lines.append(f"{LATEX_LABELS[h]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    (TAB_DIR / "pascalvoc_seg_table.tex").write_text("\n".join(lines) + "\n")
    print("  pascalvoc_seg_table.tex written")


# VOC-literature convention: heads as rows, the 21 classes as columns
# (rotated headers), IoU in %, best per class in bold, final mIoU column.
_SHORT = {
    "background": "bg", "aeroplane": "aero", "bicycle": "bike",
    "diningtable": "table", "motorbike": "mbike", "pottedplant": "plant",
    "tvmonitor": "tv",
}


def _fmt_support(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{round(n / 1e3):.0f}k"


def write_perclass_table(per):
    classes = [r["class"] for r in per[HEADS[0]]]
    support = [int(r["support"]) for r in per[HEADS[0]]]
    iou = {h: [100 * float(r["iou"]) for r in per[h]] for h in HEADS}
    miou = {h: sum(iou[h]) / len(classes) for h in HEADS}
    best = [max(HEADS, key=lambda h: iou[h][i]) for i in range(len(classes))]
    best_miou = max(HEADS, key=lambda h: miou[h])

    heads_rot = " & ".join(
        rf"\rotatebox{{90}}{{{_SHORT.get(c, c)} ({_fmt_support(s)})}}"
        for c, s in zip(classes, support)
    )
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Per-class test IoU (\%) on PASCAL VOC 2012, computed by"
        r" pooling TP/FP/FN across all 725 test images per class (the"
        r" standard per-class protocol; not a per-image average, since most"
        r" classes are present in only a small subset of the 725 images —"
        r" e.g.\ \emph{sheep} appears in 25, \emph{person} in 221)."
        r" In parentheses, the ground-truth pixel support of each class."
        r" \textsuperscript{\dag}~Fixed RFF. \textsuperscript{\ddag}~Trainable RFF (tRFF)."
        r" Best per class in \textbf{bold}.}",
        r"\label{tab:voc_perclass}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l"
        + "c" * len(classes) + r"|c@{}}",
        r"\toprule",
        rf"Head & {heads_rot} & \rotatebox{{90}}{{mIoU}} \\",
        r"\midrule",
    ]
    for h in HEADS:
        cells = []
        for i in range(len(classes)):
            s = f"{iou[h][i]:.1f}"
            cells.append(rf"\textbf{{{s}}}" if best[i] == h else s)
        m = f"{miou[h]:.1f}"
        m = rf"\textbf{{{m}}}" if best_miou == h else m
        lines.append(f"{LATEX_LABELS[h]} & " + " & ".join(cells) + f" & {m}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    (TAB_DIR / "pascalvoc_perclass_table.tex").write_text("\n".join(lines) + "\n")
    print("  pascalvoc_perclass_table.tex written")


# ─────────────────────────────────────────────────────────────────────────────
# Per-class dot plot (Cleveland) — single-column paper figure
# ─────────────────────────────────────────────────────────────────────────────

def perclass_figure(per):
    classes = [r["class"] for r in per[HEADS[0]]]
    iou = {h: np.array([float(r["iou"]) for r in per[h]]) for h in HEADS}

    order = np.argsort(iou["pontryagin"])          # difícil abajo → fácil arriba
    ypos  = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(4.2, 5.6))
    for j, k in enumerate(order):
        vals = [iou[h][k] for h in HEADS]
        ax.plot([min(vals), max(vals)], [j, j],
                color="#c3c2b7", lw=1, zorder=1)
    for h in HEADS:
        ax.scatter(iou[h][order], ypos, s=26, color=COLORS[h],
                   label=LABELS[h], zorder=2,
                   edgecolors="white", linewidths=0.5)

    ax.set_yticks(ypos)
    ax.set_yticklabels([classes[k] for k in order], fontsize=8)
    ax.set_xlabel("Test IoU", fontsize=10)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.legend(fontsize=8, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="#e1e0d9")

    fig.tight_layout()
    out = FIG_DIR / "pascalvoc_perclass_paper.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name} written")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    glob, per = load_results()
    write_summaries(glob, per)
    write_global_table(glob)
    write_perclass_table(per)
    perclass_figure(per)

    # resumen rápido en consola
    print("\n  head              mIoU    Dice    Prec    Rec     Acc")
    for h in HEADS:
        g = glob[h]
        print(f"  {h:<17} {g['macro_iou']:.4f}  {g['macro_dice']:.4f}"
              f"  {g['macro_precision']:.4f}  {g['macro_recall']:.4f}"
              f"  {g['pixel_acc']:.4f}")


if __name__ == "__main__":
    main()
