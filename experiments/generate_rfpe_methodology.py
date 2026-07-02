"""Generate a clean RFPE general methodology diagram (v3 — doubled fonts).

Output: report/figures/rfpe_methodology.pdf

Font sizing rationale:
  figure width  = 15 in
  thesis \linewidth ≈ 6.3 in  →  scale = 6.3/15 = 0.42
  to get 9 pt body text: need 9/0.42 ≈ 21 pt in matplotlib
  All font sizes here are ≈2× the visual sizes of v2.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent / "report" / "figures" / "rfpe_methodology.pdf"

# Match thesis font (mathptmx = Times Roman). STIX is the closest substitute.
plt.rcParams.update({
    "font.family":      "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size":        18,
})

# ── colours ────────────────────────────────────────────────────────────────────
C_BACKBONE = "#4a6fa5"
C_RFF      = "#1565C0"
C_SRF      = "#C62828"
C_PONT     = "#6A1B9A"
C_JHEAD    = "#2E7D32"
C_ARROW    = "#444444"
C_SEG      = "#0277BD"
C_CLS      = "#558B2F"
C_FSS      = "#6A1B9A"

# ── figure  15×11 in — y-coords scaled ×1.222 vs v2 ──────────────────────────
fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlim(0, 15)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("white")


# ── helpers ────────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, color, alpha=0.15, radius=0.2, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=2.2, edgecolor=color, facecolor=color,
        alpha=alpha, zorder=zorder))


def border(ax, x, y, w, h, color, radius=0.2, lw=2.5, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=color, facecolor="none", zorder=zorder))


def lbl(ax, x, y, txt, color="black", size=18, weight="normal",
        ha="center", va="center", zorder=5):
    ax.text(x, y, txt, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, zorder=zorder)


def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=2.0,
        style="-|>", ms=16, zorder=4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=ms),
                zorder=zorder)


# ══════════════════════════════════════════════════════════════════════════════
# OVERALL TITLE
# ══════════════════════════════════════════════════════════════════════════════
lbl(ax, 7.5, 10.65,
    "Pontryagin Random Feature Embeddings (PRFE) — General Methodology",
    "#111111", 26, "bold")

# ══════════════════════════════════════════════════════════════════════════════
# 1. INPUT block   (x=0.25, y=6.23, w=1.8, h=2.57)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.25, 6.23, 1.8, 2.57, C_BACKBONE, alpha=0.13)
border(ax, 0.25, 6.23, 1.8, 2.57, C_BACKBONE)
lbl(ax, 1.15, 8.43, "Input",                                         C_BACKBONE, 22, "bold")
lbl(ax, 1.15, 7.94, r"$\mathbf{x}\in\mathbb{R}^{3\times H\times W}$", C_BACKBONE, 18)
lbl(ax, 1.15, 7.45, "Image / tensor",                                 C_BACKBONE, 17)

# ══════════════════════════════════════════════════════════════════════════════
# 2. BACKBONE block   (x=2.35, y=5.44, w=2.3, h=3.36)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 2.35, 5.44, 2.3, 3.36, C_BACKBONE, alpha=0.13)
border(ax, 2.35, 5.44, 2.3, 3.36, C_BACKBONE)
lbl(ax, 3.5, 8.43, "Backbone",            C_BACKBONE, 22, "bold")
lbl(ax, 3.5, 7.97, "FCN / UNet /",        C_BACKBONE, 18)
lbl(ax, 3.5, 7.55, "ConvNeXtV2 …",        C_BACKBONE, 18)
lbl(ax, 3.5, 7.00, r"$\mathbf{f}\in\mathbb{R}^{C\times H\times W}$", C_BACKBONE, 18)
lbl(ax, 3.5, 6.48, "Dense feature map",   C_BACKBONE, 16)

# Input → Backbone arrow
arr(ax, 2.05, 7.35, 2.35, 7.35)
lbl(ax, 2.2, 7.62, r"$\mathbf{x}$", C_BACKBONE, 20)

# ══════════════════════════════════════════════════════════════════════════════
# 3. RFPE MODULE outer box   (x=4.85, y=2.63, w=5.9, h=7.45)  top=10.08
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 4.85, 2.63, 5.9, 7.45, C_PONT, alpha=0.05, radius=0.3)
border(ax, 4.85, 2.63, 5.9, 7.45, C_PONT, radius=0.3, lw=3.2)
lbl(ax, 7.8, 9.75,
    "PRFE Module  —  Pontryagin Random Feature Embeddings",
    C_PONT, 19, "bold")

# Backbone → RFPE entry arrow  (at backbone centre-y ≈ 7.12)
arr(ax, 4.65, 7.12, 4.85, 7.12, lw=2.5)
lbl(ax, 4.75, 7.42, r"$\mathbf{f}$", C_BACKBONE, 20)

# Vertical splitter at x=4.85 from SRF-centre(4.52) to RFF-centre(8.00)
ax.plot([4.85, 4.85], [4.52, 8.00], color=C_ARROW, lw=2.0, zorder=4)

# ── 3a. RFF sub-box   (x=5.15, y=6.54, w=2.8, h=2.93)  centre-y=8.00 ────────
box(ax, 5.15, 6.54, 2.8, 2.93, C_RFF, alpha=0.16, radius=0.14)
lbl(ax, 6.55, 9.14,
    r"$\boldsymbol{\varphi}_+(\mathbf{f})$ — RFF (positive)",  C_RFF, 20, "bold")
lbl(ax, 6.55, 8.61,
    r"$K_+=\exp\!\left(-\dfrac{\|\mathbf{x}-\mathbf{y}\|^2}{2\sigma^2}\right)$",
    C_RFF, 18)
lbl(ax, 6.55, 8.07,
    r"$\omega_i\!\sim\!\mathcal{N}(0,\sigma^{-2}I),\;b_i\!\sim\!\mathcal{U}[0,2\pi]$",
    C_RFF, 16)
lbl(ax, 6.55, 7.58,
    r"$\varphi_+(\mathbf{f})=\frac{1}{\sqrt{n}}[\cos(\omega_i^\top\mathbf{f}+b_i)]$",
    C_RFF, 18)
lbl(ax, 6.55, 7.10,
    r"$\varphi_+(\mathbf{f})\in\mathbb{R}^{p}$",
    C_RFF, 16)

# ── 3b. SRF sub-box   (x=5.15, y=3.18, w=2.8, h=2.69)  centre-y=4.52 ────────
box(ax, 5.15, 3.18, 2.8, 2.69, C_SRF, alpha=0.14, radius=0.14)
lbl(ax, 6.55, 5.60,
    r"$\boldsymbol{\varphi}_-(\mathbf{f})$ — SRF (negative)",  C_SRF, 20, "bold")
lbl(ax, 6.55, 5.08,
    r"$K_-(\mathbf{x},\mathbf{y})=(\mathbf{x}^\top\mathbf{y})^{d}$",
    C_SRF, 18)
lbl(ax, 6.55, 4.58,
    r"$\omega_{j\ell}\!\sim\!\mathcal{U}(S^{d-1})$",
    C_SRF, 16)
lbl(ax, 6.55, 4.10,
    r"$\varphi_-(\mathbf{f})_j=\frac{1}{\sqrt{\kappa}}\prod_\ell(\omega_{j\ell}^\top\mathbf{f})$",
    C_SRF, 18)
lbl(ax, 6.55, 3.60,
    r"$\varphi_-(\mathbf{f})\in\mathbb{R}^{\kappa}$",
    C_SRF, 16)

# Splitter → sub-box arrows
arr(ax, 4.85, 8.00, 5.15, 8.00)   # → RFF centre
arr(ax, 4.85, 4.52, 5.15, 4.52)   # → SRF centre

# ── 3c. Pontryagin sub-box   (x=8.2, y=4.09, w=2.3, h=3.91)  centre-y=6.05 ──
box(ax, 8.2, 4.09, 2.3, 3.91, C_PONT, alpha=0.20, radius=0.15)
lbl(ax, 9.35, 7.67, "Pontryagin space",                                  C_PONT, 20, "bold")
lbl(ax, 9.35, 7.21, r"$\mathbf{z}=[\boldsymbol{\varphi}_+\!\parallel\!\boldsymbol{\varphi}_-]$", C_PONT, 18)
lbl(ax, 9.35, 6.72, r"$\mathbf{z}\in\Pi_\kappa=\mathbb{R}^{p+\kappa}$",     C_PONT, 18)
lbl(ax, 9.35, 6.23, r"$\langle\mathbf{u},\mathbf{v}\rangle_J=\mathbf{u}^\top J\mathbf{v}$", C_PONT, 18)
lbl(ax, 9.35, 5.74, r"$J=\mathrm{diag}(+1^{p},\,-1^{\kappa})$",           C_PONT, 18)
lbl(ax, 9.35, 5.16, r"Learnable: $\log\sigma$",                            C_PONT, 17)
lbl(ax, 9.35, 4.68, r"(RFF bandwidth)",                                    C_PONT, 16)

# RFF → Pont  and  SRF → Pont
arr(ax, 7.95, 8.00, 8.2, 7.3)
arr(ax, 7.95, 4.52, 8.2, 5.05)

# ══════════════════════════════════════════════════════════════════════════════
# 4. J-HYPERPLANE   (x=10.8, y=4.09, w=3.15, h=3.91)  centre-y=6.05
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 10.8, 4.09, 3.15, 3.91, C_JHEAD, alpha=0.15, radius=0.2)
lbl(ax, 12.37, 7.67, "J-Hyperplane Classifier",                         C_JHEAD, 20, "bold")
lbl(ax, 12.37, 7.19, r"$\zeta_y(\mathbf{z})=\langle\mathbf{W}_y,\mathbf{z}\rangle_J+b_y$", C_JHEAD, 18)
lbl(ax, 12.37, 6.67, r"$= E_+^y - E_-^y$",                             C_JHEAD, 20, "bold")
lbl(ax, 12.37, 6.17, r"$E_+^y=\langle\mathbf{w}_{+,y},\boldsymbol{\varphi}_+\rangle$",     C_JHEAD, 18)
lbl(ax, 12.37, 5.68, r"$E_-^y=\langle\mathbf{w}_{-,y},\boldsymbol{\varphi}_-\rangle$",     C_JHEAD, 18)
lbl(ax, 12.37, 5.16, r"$\hat{y}=\arg\max_y\;\zeta_y(\mathbf{z})$",     C_JHEAD, 18)
lbl(ax, 12.37, 4.62, r"$\mathbf{W}_y\in\mathbb{R}^{p+\kappa}$ (learnable)", C_JHEAD, 16)

# Pont → J-head  (Pont right edge = 10.5, centre-y = 6.05)
arr(ax, 10.5, 6.05, 10.8, 6.05)

# ══════════════════════════════════════════════════════════════════════════════
# 5. APPLICATIONS strip   (y=0.38 to y=2.18, h=1.80)
# ══════════════════════════════════════════════════════════════════════════════
ax.plot([0.2, 14.8], [2.33, 2.33], color="#bbbbbb", lw=1.5, ls="--", zorder=3)
lbl(ax, 7.5, 2.47, "Applications of PRFE", "#333333", 20, "bold")

apps = [
    (2.5,   C_SEG, "Dense Segmentation",    "FCN / UNet backbone"),
    (7.5,   C_CLS, "Image Classification",  "ConvNeXtV2 backbone"),
    (12.5,  C_FSS, "Few-Shot Segmentation", "J-prototype matching"),
]
for xc, col, title, sub in apps:
    box(ax, xc - 1.95, 0.38, 3.9, 1.80, col, alpha=0.15, radius=0.14)
    lbl(ax, xc, 1.52, title, col, 20, "bold")
    lbl(ax, xc, 0.90, sub,   col, 17)

for xd in [5.45, 9.55]:
    ax.plot([xd, xd], [0.38, 2.18], color="#cccccc", lw=1.5, ls="--", zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND — below the axes (included by bbox_inches='tight')
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=C_BACKBONE, alpha=0.6, label="Backbone (any architecture)"),
    mpatches.Patch(facecolor=C_RFF,      alpha=0.6, label=r"Positive subspace — RFF ($K_+$, RBF)"),
    mpatches.Patch(facecolor=C_SRF,      alpha=0.6, label=r"Negative subspace — SRF ($K_-$, polynomial)"),
    mpatches.Patch(facecolor=C_PONT,     alpha=0.6, label=r"Pontryagin space $\Pi_\kappa$ (J-metric)"),
    mpatches.Patch(facecolor=C_JHEAD,    alpha=0.6, label="J-hyperplane classifier"),
]
ax.legend(handles=legend_items,
          loc="upper center",
          bbox_to_anchor=(0.5, -0.03),
          fontsize=18,
          framealpha=0.97,
          ncol=3,
          columnspacing=1.2,
          handlelength=1.6,
          handletextpad=0.7,
          edgecolor="#cccccc")

plt.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=300, format="pdf", bbox_inches="tight")
print(f"Saved → {OUT}")
