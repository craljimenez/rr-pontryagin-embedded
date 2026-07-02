"""generate_energy_cls.py

Generate a multi-row composite embedding-energy figure for the RFPE-tRFF
classification model.  Selects N correctly classified test images (one per
class where possible) and renders a rows×3 panel figure:
  col 0 — original image + GT/Pred caption
  col 1 — positive energy E+  (RFF — Euclidean, Blues)
  col 2 — negative energy E-  (SRF, Reds)

Output: report/figures/embedding_energy_correct.pdf  (300 DPI)

Usage:
    python experiments/generate_energy_cls.py [--n-show 3] [--data-root PATH]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent / "src"))

from configs.cls_sugarcane import DEVICE, RESULTS_DIR
from run_cls_sugarcane import (
    SugarcaneLeafDataset, _effective_name, _find_dataset_root,
    build_cls_model, download_dataset, _denorm,
)

OUT_DIR   = EXP_DIR / "report" / "figures"
MODEL_DIR = RESULTS_DIR / "pontryagin_trff"
METRICS_PATH = MODEL_DIR / "embedding_energy" / "metrics.json"

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def _load_model(device):
    ckpt_path = MODEL_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    params  = ckpt.get("params", {})
    tr      = ckpt.get("trainable_rff", True)
    model   = build_cls_model("pontryagin", n_classes=len(classes),
                               params=params, trainable_rff=tr)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes


def _embedding_energies(model, inp):
    """Return (e_pos, e_neg, pred, probs) for a single image tensor."""
    captured = {}
    handle = model.embed_layer.register_forward_hook(
        lambda *a: captured.__setitem__("z", a[-1])
    )
    with torch.no_grad():
        logits = model(inp)
        pred   = logits.argmax(1).item()
        probs  = torch.softmax(logits, 1)[0].cpu().numpy()
    handle.remove()

    z_map = captured["z"]                      # (1, p+q, H', W')
    p     = model.embed_layer.rff.out_features
    W     = model.head.W                        # (n_classes, p+q)

    phi_pos = z_map[:, :p]
    phi_neg = z_map[:, p:]
    w_pos   = W[pred, :p]
    w_neg   = W[pred, p:]

    e_pos_lr = (phi_pos * w_pos[None, :, None, None]).sum(1, keepdim=True)
    e_neg_lr = (phi_neg * w_neg[None, :, None, None]).sum(1, keepdim=True)

    target = inp.shape[-2:]
    e_pos = F.interpolate(e_pos_lr, target, mode="bilinear", align_corners=False)
    e_neg = F.interpolate(e_neg_lr, target, mode="bilinear", align_corners=False)
    return (e_pos[0, 0].detach().cpu().numpy(),
            e_neg[0, 0].detach().cpu().numpy(),
            pred, probs)


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(rows_data, classes, out_path, paper_mode=False):
    """rows_data: list of dicts with keys rgb, e_pos, e_neg, true, pred, probs,
    mean_ep, mean_en, rho, delta.
    paper_mode: compact layout for double-column journal figures (~8.5 x 3.2*n in)."""
    n = len(rows_data)
    if paper_mode:
        figsize = (8.5, 3.0 * n)
        fs_label = 8
        fs_header = 8
        hspace = 0.55
    else:
        figsize = (13, 4.5 * n)
        fs_label = 9
        fs_header = 10
        hspace = 0.40

    fig, axes = plt.subplots(
        n, 3,
        figsize=figsize,
        gridspec_kw={"wspace": 0.06, "hspace": hspace},
    )
    if n == 1:
        axes = [axes]

    for row, d in enumerate(rows_data):
        header = (
            f"✓ Correct  |  {classes[d['true']]}  —  "
            f"$\\bar{{e}}_+={d['mean_ep']:.4f}$  "
            f"$\\bar{{e}}_-={d['mean_en']:.3f}$  "
            f"$\\rho={d['rho']:.4f}$  "
            f"$\\Delta={d['delta']:.3f}$"
        )
        axes[row][1].set_title(header, fontsize=fs_header, fontweight="bold",
                               color="#1a6e1a", pad=4)

        # Panel 0: image
        axes[row][0].imshow(d["rgb"])
        axes[row][0].set_xlabel(
            f"GT: {classes[d['true']]}\nPred: {classes[d['pred']]} "
            f"({d['probs'][d['pred']]:.2f})",
            fontsize=fs_label,
        )

        # Panel 1: positive energy
        im1 = axes[row][1].imshow(_norm01(d["e_pos"]), cmap="Blues", vmin=0, vmax=1)
        axes[row][1].set_xlabel(
            r"$E_+^{ij}$  (RFF, Gaussian)",
            fontsize=fs_label,
        )
        plt.colorbar(im1, ax=axes[row][1], fraction=0.046, pad=0.04)

        # Panel 2: negative energy
        im2 = axes[row][2].imshow(_norm01(d["e_neg"]), cmap="Reds", vmin=0, vmax=1)
        axes[row][2].set_xlabel(
            r"$E_-^{ij}$  (SRF, polynomial)",
            fontsize=fs_label,
        )
        plt.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-show",    type=int, default=3,
                    help="Number of correctly classified examples to show.")
    ap.add_argument("--paper", action="store_true",
                    help="Compact layout for journal submission (8.5×3n in, 200 dpi).")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--device",    type=str, default=DEVICE)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    if args.data_root:
        data_root = _find_dataset_root(Path(args.data_root))
    else:
        env = os.environ.get("SUGARCANE_DATA_ROOT")
        data_root = _find_dataset_root(Path(env) if env else download_dataset())
    dataset = SugarcaneLeafDataset(root=data_root, split="test")

    # Model
    print("Loading RFPE-tRFF model…")
    model, classes = _load_model(device)
    print(f"Classes: {classes}")

    # Load precomputed metrics to pick correctly classified examples
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    # Select one example per class (first correct), up to n_show total
    selected_by_class: dict = {}
    for idx_str, m in metrics.items():
        if not m.get("correct", False):
            continue
        cls = m["true_class"]
        if cls not in selected_by_class:
            selected_by_class[cls] = int(idx_str)
        if len(selected_by_class) >= len(classes):
            break

    # Fallback: fill to n_show from any correct example
    chosen = list(selected_by_class.values())[: args.n_show]
    if len(chosen) < args.n_show:
        for idx_str, m in metrics.items():
            if m.get("correct", False) and int(idx_str) not in chosen:
                chosen.append(int(idx_str))
            if len(chosen) >= args.n_show:
                break
    chosen.sort()
    print(f"Selected indices: {chosen}")

    # Build rows
    rows_data = []
    for idx in chosen:
        img_t, true_lbl = dataset[idx]
        inp = img_t.unsqueeze(0).to(device)
        e_pos, e_neg, pred, probs = _embedding_energies(model, inp)
        rgb = _denorm(img_t)

        mean_ep = float(e_pos.mean())
        mean_en = float(e_neg.mean())
        rho     = mean_ep / (mean_ep + abs(mean_en) + 1e-8)
        delta   = mean_ep - mean_en

        rows_data.append({
            "rgb": rgb, "e_pos": e_pos, "e_neg": e_neg,
            "true": int(true_lbl), "pred": pred, "probs": probs,
            "mean_ep": mean_ep, "mean_en": mean_en,
            "rho": rho, "delta": delta,
        })
        print(f"  idx={idx}  {classes[int(true_lbl)]:>10} → {classes[pred]}  "
              f"ē+={mean_ep:.4f}  ē-={mean_en:.4f}  ρ={rho:.4f}  Δ={delta:.3f}")

    if args.paper:
        out_path = OUT_DIR / "embedding_energy_paper.pdf"
    else:
        out_path = OUT_DIR / "embedding_energy_correct.pdf"
    print("\nGenerating figure…")
    make_figure(rows_data, classes, out_path, paper_mode=args.paper)
    print("Done.")


if __name__ == "__main__":
    main()
