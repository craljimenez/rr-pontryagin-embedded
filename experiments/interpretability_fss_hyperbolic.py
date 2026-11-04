"""Extend the FSS interpretability comparison with the Hyperbolic baseline.

`interpretability_fss.py` only compares Euclidean vs Pontryagin. This script
computes the same per-episode scalar metrics (IoU, ScoreCAM GT-alignment, MC
Dropout uncertainty) for FSS-Hyperbolic on the identical 20 test episodes
(same dataset construction, seed=42, order), merges them into a copy of the
existing metrics.csv, and regenerates the three aggregate comparison figures
as 3-way (Euclidean / Hyperbolic / PRFE) plots with corrected "PRFE" labelling
(the original figures still said "RFPE").

Usage:
    python interpretability_fss_hyperbolic.py --k-shot 1 --n-viz 20 --n-mc 30
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.fss_sugarcane import (
    DATASET_ROOT, DEVICE, IMG_SIZE, K_SHOT, N_EPISODES_TEST,
    RESULTS_DIR, TARGET_CLS, TRAINABLE_RFF,
)
from prfe.data.fss_dataset import EpisodicUAVDataset
from interpretability_analysis import compute_cam_metrics
from interpretability_fss import (
    _load_model, FSSScoreCAM, mc_uncertainty_fss, _iou, N_PASSES_MC,
)

REPORT_FIGS = Path(__file__).parent / "report" / "figures"


def _safe_mean(arr, mask):
    return float(arr[mask].mean()) if mask.any() else float("nan")


def compute_hyp_rows(k_shot, data_root, results_dir, device, n_viz, n_mc,
                      trainable_rff):
    model_hyp = _load_model("hyperbolic", k_shot, results_dir, device,
                             trainable_rff)

    test_ds = EpisodicUAVDataset(
        data_root, "test",
        k_shot=k_shot, n_episodes=N_EPISODES_TEST,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=False, seed=42,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=0, pin_memory=False)

    n_analyse = min(n_viz, len(test_ds))
    print(f"Computing FSS-Hyperbolic metrics on {n_analyse} episodes …")

    rows = []
    for ep_idx, batch in enumerate(loader):
        if ep_idx >= n_analyse:
            break
        print(f"  Episode {ep_idx + 1:3d}/{n_analyse}", end="\r")

        sup_imgs  = batch["support_imgs"].to(device)
        sup_masks = batch["support_masks"].to(device)
        q_img     = batch["query_img"].to(device)
        q_mask    = batch["query_mask"].to(device)
        gt        = q_mask[0].cpu().numpy()

        with torch.no_grad():
            pred_h = (model_hyp.forward(sup_imgs, sup_masks, q_img)[0] > 0) \
                .float().cpu().numpy()
        iou_h = _iou(pred_h, gt)

        cam_hyp = FSSScoreCAM(model_hyp)
        heat_h  = cam_hyp.compute(sup_imgs, sup_masks, q_img)
        cam_hyp.remove()

        unc_h = mc_uncertainty_fss(model_hyp, sup_imgs, sup_masks, q_img,
                                    n_passes=n_mc)
        cam_m_h = compute_cam_metrics(heat_h, gt_mask=gt)

        fg_mask = gt > 0.5
        bg_mask = ~fg_mask

        rows.append({
            "episode":                 ep_idx,
            "iou_hyperbolic":          iou_h,
            "mc_var_hyp_mean":         float(unc_h.mean()),
            "mc_var_hyp_fg":           _safe_mean(unc_h, fg_mask),
            "mc_var_hyp_bg":           _safe_mean(unc_h, bg_mask),
            "cam_hyp_mean_activation": cam_m_h.get("mean_activation", float("nan")),
            "cam_hyp_coverage_top10":  cam_m_h.get("coverage_top10pct", float("nan")),
            "cam_hyp_gt_iou":          cam_m_h.get("gt_iou",       float("nan")),
            "cam_hyp_gt_recall":       cam_m_h.get("gt_recall",    float("nan")),
            "cam_hyp_gt_precision":    cam_m_h.get("gt_precision", float("nan")),
            "cam_hyp_gt_pearson":      cam_m_h.get("gt_pearson",   float("nan")),
            "cam_hyp_otsu_thresh":     cam_m_h.get("otsu_threshold", float("nan")),
        })
    print()
    return rows


def merge_csv(base_csv: Path, hyp_rows: list[dict], out_csv: Path):
    with open(base_csv, newline="") as f:
        base_rows = list(csv.DictReader(f))

    assert len(base_rows) == len(hyp_rows), (
        f"Episode count mismatch: base={len(base_rows)} hyp={len(hyp_rows)}"
    )

    merged = []
    for br, hr in zip(base_rows, hyp_rows):
        assert int(br["episode"]) == hr["episode"], "Episode index mismatch"
        merged_row = dict(br)
        merged_row.update({k: v for k, v in hr.items() if k != "episode"})
        merged.append(merged_row)

    fieldnames = list(merged[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    print(f"Merged CSV written to {out_csv}")
    return merged


def plot_iou_3way(rows: list[dict], k_shot: int, out_path: Path):
    ep    = np.array([float(r["episode"])         for r in rows])
    iou_e = np.array([float(r["iou_euclidean"])   for r in rows])
    iou_h = np.array([float(r["iou_hyperbolic"])  for r in rows])
    iou_p = np.array([float(r["iou_pontryagin"])  for r in rows])

    w = 0.27
    fig, ax = plt.subplots(figsize=(max(9, len(ep) * 0.45 + 2), 4))
    ax.bar(ep - w, iou_e, width=w, label="Euclidean",  alpha=0.85, color="steelblue")
    ax.bar(ep,     iou_h, width=w, label="Hyperbolic", alpha=0.85, color="seagreen")
    ax.bar(ep + w, iou_p, width=w, label="PRFE",       alpha=0.85, color="darkorange")
    ax.axhline(iou_e.mean(), color="steelblue",  linestyle="--", linewidth=1,
               label=f"Euc mean={iou_e.mean():.3f}")
    ax.axhline(iou_h.mean(), color="seagreen",   linestyle="--", linewidth=1,
               label=f"Hyp mean={iou_h.mean():.3f}")
    ax.axhline(iou_p.mean(), color="darkorange", linestyle="--", linewidth=1,
               label=f"PRFE mean={iou_p.mean():.3f}")
    ax.set_xlabel("Episode"); ax.set_ylabel("IoU")
    ax.set_title(f"IoU per test episode — {k_shot}-shot FSS")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cam_metrics_3way(rows: list[dict], k_shot: int, out_path: Path):
    pairs = [
        ("cam_euc_gt_iou",     "cam_hyp_gt_iou",     "cam_pont_gt_iou",     "GT IoU"),
        ("cam_euc_gt_recall",  "cam_hyp_gt_recall",  "cam_pont_gt_recall",  "GT Recall"),
        ("cam_euc_gt_precision", "cam_hyp_gt_precision", "cam_pont_gt_precision", "GT Precision"),
        ("cam_euc_gt_pearson", "cam_hyp_gt_pearson", "cam_pont_gt_pearson", "GT Pearson r"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for ax, (ke, kh, kp, label) in zip(axes, pairs):
        ve = np.array([float(r[ke]) for r in rows])
        vh = np.array([float(r[kh]) for r in rows])
        vp = np.array([float(r[kp]) for r in rows])
        valid = ~(np.isnan(ve) | np.isnan(vh) | np.isnan(vp))
        bp = ax.boxplot(
            [ve[valid], vh[valid], vp[valid]],
            labels=["Euclidean", "Hyperbolic", "PRFE"],
            patch_artist=True, medianprops=dict(color="red", linewidth=2),
        )
        for box, c in zip(bp["boxes"], ["lightsteelblue", "lightgreen", "peachpuff"]):
            box.set_facecolor(c)
        ax.set_title(
            f"{label}\nEuc={ve[valid].mean():.3f}  Hyp={vh[valid].mean():.3f}  "
            f"PRFE={vp[valid].mean():.3f}", fontsize=8.5
        )
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    plt.suptitle(f"ScoreCAM quality vs GT — {k_shot}-shot FSS", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mc_variance_3way(rows: list[dict], k_shot: int, n_mc: int, out_path: Path):
    ep    = np.array([float(r["episode"])         for r in rows])
    var_e = np.array([float(r["mc_var_euc_mean"])  for r in rows])
    var_h = np.array([float(r["mc_var_hyp_mean"])  for r in rows])
    var_p = np.array([float(r["mc_var_pont_mean"]) for r in rows])

    w = 0.27
    fig, ax = plt.subplots(figsize=(max(9, len(ep) * 0.45 + 2), 4))
    ax.bar(ep - w, var_e, width=w, label="Euclidean",  alpha=0.85, color="steelblue")
    ax.bar(ep,     var_h, width=w, label="Hyperbolic", alpha=0.85, color="seagreen")
    ax.bar(ep + w, var_p, width=w, label="PRFE",       alpha=0.85, color="darkorange")
    ax.set_xlabel("Episode"); ax.set_ylabel("Mean predictive variance")
    ax.set_title(f"MC Dropout uncertainty per episode — {k_shot}-shot FSS (N={n_mc})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-shot",  type=int, default=K_SHOT)
    ap.add_argument("--n-viz",   type=int, default=20)
    ap.add_argument("--n-mc",    type=int, default=N_PASSES_MC)
    ap.add_argument("--device",  default=DEVICE)
    ap.add_argument("--trainable-rff", action="store_true", default=TRAINABLE_RFF)
    return ap.parse_args()


def main():
    args   = _parse()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    suffix   = "_trainable" if args.trainable_rff else ""
    interp_dir = RESULTS_DIR / f"interpretability_{args.k_shot}shot{suffix}"
    base_csv = interp_dir / "metrics.csv"
    if not base_csv.exists():
        raise FileNotFoundError(
            f"{base_csv} not found — run interpretability_fss.py first."
        )

    hyp_rows = compute_hyp_rows(
        args.k_shot, DATASET_ROOT, RESULTS_DIR, device,
        args.n_viz, args.n_mc, args.trainable_rff,
    )

    out_csv = interp_dir / "metrics_with_hyperbolic.csv"
    merged  = merge_csv(base_csv, hyp_rows, out_csv)

    REPORT_FIGS.mkdir(parents=True, exist_ok=True)
    plot_iou_3way(merged, args.k_shot, REPORT_FIGS / "fss_iou_comparison.pdf")
    plot_cam_metrics_3way(merged, args.k_shot, REPORT_FIGS / "fss_cam_metrics.pdf")
    plot_mc_variance_3way(merged, args.k_shot, args.n_mc,
                           REPORT_FIGS / "fss_mc_variance.pdf")

    # ── Summary for LaTeX tables ────────────────────────────────────────────
    def col(key):
        return np.array([float(r[key]) for r in merged])

    print("\n" + "─" * 70)
    print("Summary (mean ± std over", len(merged), "episodes) — for LaTeX tables")
    print("─" * 70)
    for label, iou_k, pear_k in [
        ("Euclidean",  "cam_euc_gt_iou",  "cam_euc_gt_pearson"),
        ("Hyperbolic", "cam_hyp_gt_iou",  "cam_hyp_gt_pearson"),
        ("PRFE",       "cam_pont_gt_iou", "cam_pont_gt_pearson"),
    ]:
        iou_v, pear_v = col(iou_k), col(pear_k)
        print(f"  CAM-GT IoU     {label:<10}: {iou_v.mean():.3f} ± {iou_v.std():.3f}")
        print(f"  CAM-GT Pearson {label:<10}: {pear_v.mean():.3f} ± {pear_v.std():.3f}")
    print()
    for label, fg_k, bg_k in [
        ("Euclidean",  "mc_var_euc_fg",  "mc_var_euc_bg"),
        ("Hyperbolic", "mc_var_hyp_fg",  "mc_var_hyp_bg"),
        ("PRFE",       "mc_var_pont_fg", "mc_var_pont_bg"),
    ]:
        fg_v, bg_v = col(fg_k), col(bg_k)
        print(f"  MC Var FG/BG   {label:<10}: {fg_v.mean():.3e} / {bg_v.mean():.3e}")
    print("─" * 70)


if __name__ == "__main__":
    main()
