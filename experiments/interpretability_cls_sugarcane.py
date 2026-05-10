"""Interpretability analysis for ConvNeXtV2-Tiny classification models.

For each model and each test image produces:
  1. ScoreCAM figure (1×n_classes PDF) — per-class heatmaps overlaid on image.
  2. LIME figure (1×3 PDF) — superpixel attribution for predicted and true class.

Additionally, for any Pontryagin-family model (has embed_layer):
  3. Embedding energy figure (1×3 PDF) — positive / negative subspace energy maps.

ScoreCAM target layer: backbone.stages[-1] for ALL models.

Outputs per model (folder = effective model name, e.g. pontryagin_trff):
  results/cls_sugarcane/<run_name>/scorecam/
  results/cls_sugarcane/<run_name>/lime/
  results/cls_sugarcane/<run_name>/embedding_energy/  (Pontryagin only)

Usage:
    python interpretability_cls_sugarcane.py --model all
    python interpretability_cls_sugarcane.py --model pontryagin --trainable-rff
    python interpretability_cls_sugarcane.py --model pontryagin --n-samples 20
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.cls_sugarcane import (
    AVAILABLE_MODELS, DEVICE, IMG_SIZE, RESULTS_DIR, TRAINABLE_RFF,
)
from run_cls_sugarcane import (
    SugarcaneLeafDataset, _effective_name, _find_dataset_root, _denorm,
    build_cls_model, download_dataset,
)

import os

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    _LIME_OK = True
except ImportError:
    _LIME_OK = False
    print("WARNING: lime or scikit-image not found — LIME analysis disabled.")

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])
N_PASSES_MC = 30
P_DROP      = 0.25

_EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cls_model(
    model_type: str,
    results_dir: Path,
    trainable_rff: bool = TRAINABLE_RFF,
):
    """Load a saved checkpoint.

    trainable_rff is read from the checkpoint if present (saved by train_one_model),
    falling back to the CLI flag so old checkpoints still load correctly.
    """
    run_name  = _effective_name(model_type, trainable_rff)
    model_dir = results_dir / run_name
    ckpt_path = model_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError) as e:
        raise FileNotFoundError(
            f"Checkpoint for {run_name} is corrupt ({ckpt_path}): {e}"
        ) from e
    class_names   = ckpt["classes"]
    n_classes     = len(class_names)
    params        = ckpt.get("params", {})
    trainable_rff = ckpt.get("trainable_rff", trainable_rff)  # prefer checkpoint value
    model = build_cls_model(
        model_type, n_classes=n_classes, params=params, trainable_rff=trainable_rff,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, run_name


# ─────────────────────────────────────────────────────────────────────────────
# ScoreCAM for classification
# ─────────────────────────────────────────────────────────────────────────────

class ClassScoreCAM:
    """ScoreCAM for image classification.

    Hooks into target_layer, normalises each feature-map channel to [0,1],
    upsamples to input size, masks the input, runs a forward pass, and
    computes softmax-score-weighted sum of feature maps for the target class.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model          = model
        self._feature_maps  = None
        self._handle        = target_layer.register_forward_hook(self._hook)

    def _hook(self, *args):
        self._feature_maps = args[-1].detach()

    def remove(self):
        self._handle.remove()

    @torch.no_grad()
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        _ = self.model(input_tensor)                       # populate hook

        fmaps = self._feature_maps                         # (1, C, H', W')
        _, n_ch, fh, fw = fmaps.shape

        scores = []
        for k in range(n_ch):
            fm = fmaps[:, k:k+1]                           # (1, 1, H', W')
            lo, hi = fm.min(), fm.max()
            fm_norm = (fm - lo) / (hi - lo + 1e-8)
            mask_k  = F.interpolate(fm_norm, size=input_tensor.shape[-2:],
                                    mode="bilinear", align_corners=False)
            out    = self.model(input_tensor * mask_k)
            score  = torch.softmax(out, dim=1)[0, class_idx].item()
            scores.append(score)

        weights = torch.tensor(scores, dtype=torch.float32)
        total   = weights.sum()
        if total > 1e-8:
            weights = weights / total
        else:
            weights = torch.ones(n_ch) / n_ch

        cam = (fmaps[0] * weights[:, None, None]).sum(0)   # (H', W')
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode="bilinear", align_corners=False,
        )[0, 0].cpu().numpy()

        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        return cam


def _overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap = plt.colormaps["jet"](cam)[..., :3]
    return (alpha * heatmap + (1 - alpha) * rgb).clip(0, 1)


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-8 else (a - lo) / (hi - lo)


def scorecam_analysis(
    model_type: str,
    model,
    dataset,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
    n_samples: int | None = None,
):
    """Run ScoreCAM on all (or n_samples) test images. Save PDFs + metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)

    # Target layer: last ConvNeXt stage for all models.
    # With the corrected architecture (GAP → embed_1d) the spatial feature maps
    # live in the backbone; the Pontryagin embedding operates on the pooled vector
    # and has no spatial extent to hook into.
    target_layer = model.backbone.stages[-1]

    cam = ClassScoreCAM(model.to(device), target_layer)

    indices   = range(len(dataset)) if n_samples is None else range(min(n_samples, len(dataset)))
    all_metrics: dict = {}

    for idx in indices:
        img_t, true_lbl = dataset[idx]
        inp  = img_t.unsqueeze(0).to(device)
        rgb  = _denorm(img_t)
        rgb_f = rgb.astype(np.float32) / 255.0

        with torch.no_grad():
            logits = model(inp)
            pred   = logits.argmax(dim=1).item()
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # ScoreCAM for every class
        cams = [cam.generate_cam(inp, c) for c in range(n_classes)]

        # Figure: original + one heatmap per class
        n_cols = n_classes + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
        axes[0].imshow(rgb)
        axes[0].set_xlabel(
            f"GT: {class_names[true_lbl]}\nPred: {class_names[pred]} "
            f"({probs[pred]:.2f})",
            fontsize=9,
        )
        for c, (ax, hm) in enumerate(zip(axes[1:], cams)):
            ax.imshow(_overlay(rgb_f, hm))
            ax.set_xlabel(
                f"{class_names[c]}\n(p={probs[c]:.2f})", fontsize=9
            )
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(
            f"ScoreCAM — {model_type}  [{idx}]", fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        fig.savefig(out_dir / f"{idx:04d}.pdf", dpi=150, format="pdf",
                    bbox_inches="tight")
        plt.close(fig)

        # ── Interpretability metrics ────────────────────────────────────────
        def _cam_entropy(c):
            h = cams[c]
            total = h.sum() + 1e-8
            p = h / total
            return float(-np.where(p > 0, p * np.log(p + 1e-12), 0).sum())

        def _confidence_gain(c):
            cam_mask = torch.tensor(cams[c], dtype=torch.float32,
                                    device=device)[None, None]
            cam_mask = cam_mask.expand_as(inp)
            with torch.no_grad():
                p_mask = torch.softmax(model(inp * cam_mask), dim=1)[0, c].item()
            return float(p_mask - probs[c])

        metrics_entry = {
            "true_class": class_names[true_lbl],
            "pred_class": class_names[pred],
            "correct":    pred == true_lbl,
            "probs":      probs.tolist(),
        }
        for c, cn in enumerate(class_names):
            cam_c = cams[c]
            metrics_entry[cn] = {
                "mean_activation": float(cam_c.mean()),
                "activation_entropy": _cam_entropy(c),
                "confidence_gain":    _confidence_gain(c),
            }
        all_metrics[str(idx)] = metrics_entry
        print(f"  [{idx+1:3d}] {class_names[true_lbl]:>10} → {class_names[pred]}  "
              f"{'✓' if pred == true_lbl else '✗'}  "
              f"Δp_pred={metrics_entry[class_names[pred]]['confidence_gain']:+.3f}")

    cam.remove()
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ScoreCAM metrics → {out_dir / 'metrics.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# LIME analysis
# ─────────────────────────────────────────────────────────────────────────────

def _make_predict_fn(model, device: torch.device):
    """Return a LIME-compatible predict function.

    Accepts (N, H, W, 3) uint8 numpy arrays, returns (N, n_classes) probabilities.
    """
    def predict_fn(images: np.ndarray) -> np.ndarray:
        tensors = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            t   = _EVAL_TF(pil)
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        return probs
    return predict_fn


def lime_analysis(
    model,
    dataset,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
    n_samples: int | None = None,
    num_lime_samples: int = 1000,
):
    """Run LIME on all (or n_samples) test images. Save PDFs + metrics."""
    if not _LIME_OK:
        print("  LIME analysis skipped (lime not installed).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer  = lime_image.LimeImageExplainer()
    predict_fn = _make_predict_fn(model, device)

    indices    = range(len(dataset)) if n_samples is None else range(min(n_samples, len(dataset)))
    all_metrics: dict = {}

    for idx in indices:
        img_t, true_lbl = dataset[idx]
        # LIME works on denormalised uint8 images
        img_np = _denorm(img_t)   # (H, W, 3) uint8

        with torch.no_grad():
            logits = model(img_t.unsqueeze(0).to(device))
            pred   = logits.argmax(dim=1).item()
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels=len(class_names),
            num_samples=num_lime_samples,
            random_seed=42,
        )

        # Figure: image | LIME for predicted class | LIME for true class
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

        axes[0].imshow(img_np)
        axes[0].set_xlabel(
            f"GT: {class_names[true_lbl]}\nPred: {class_names[pred]} ({probs[pred]:.2f})",
            fontsize=9,
        )

        # Predicted-class LIME
        temp_pred, mask_pred = explanation.get_image_and_mask(
            pred, positive_only=False, num_features=10, hide_rest=False
        )
        axes[1].imshow(mark_boundaries(temp_pred / 255.0, mask_pred))
        axes[1].set_xlabel(f"LIME — predicted: {class_names[pred]}", fontsize=9)

        # True-class LIME (same as predicted if correct)
        temp_true, mask_true = explanation.get_image_and_mask(
            true_lbl, positive_only=False, num_features=10, hide_rest=False
        )
        axes[2].imshow(mark_boundaries(temp_true / 255.0, mask_true))
        axes[2].set_xlabel(f"LIME — true: {class_names[true_lbl]}", fontsize=9)

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(
            f"LIME — {idx}  {'✓' if pred == true_lbl else '✗'}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(out_dir / f"{idx:04d}.pdf", dpi=150, format="pdf",
                    bbox_inches="tight")
        plt.close(fig)

        # ── LIME interpretability metrics ────────────────────────────────
        importances = explanation.local_exp.get(pred, [])
        top5_mean_abs = (
            float(np.mean([abs(w) for _, w in importances[:5]]))
            if len(importances) >= 1 else 0.0
        )
        # R² fidelity of the local linear model (sklearn score attribute)
        local_r2 = float(getattr(explanation, "score", float("nan")))

        all_metrics[str(idx)] = {
            "true_class":           class_names[true_lbl],
            "pred_class":           class_names[pred],
            "correct":              pred == true_lbl,
            "local_fidelity_r2":    local_r2,
            "top5_mean_abs_weight": top5_mean_abs,
        }
        print(f"  [{idx+1:3d}] {class_names[true_lbl]:>10} → {class_names[pred]}  "
              f"{'✓' if pred == true_lbl else '✗'}  "
              f"R²={local_r2:.3f}  |w|₅={top5_mean_abs:.4f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  LIME metrics → {out_dir / 'metrics.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Pontryagin embedding energy (positive vs. negative subspace)
# ─────────────────────────────────────────────────────────────────────────────

def embedding_energy_analysis(
    model,
    dataset,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
    n_samples: int | None = None,
):
    """Visualise positive / negative subspace energy maps for Pontryagin model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    indices    = range(len(dataset)) if n_samples is None else range(min(n_samples, len(dataset)))
    all_metrics: dict = {}

    for idx in indices:
        img_t, true_lbl = dataset[idx]
        inp = img_t.unsqueeze(0).to(device)
        rgb = _denorm(img_t).astype(np.float32) / 255.0

        # Capture the spatial Pontryagin map (B, p+q, H', W')
        captured: dict = {}
        handle = model.embed_layer.register_forward_hook(
            lambda *a: captured.__setitem__("z", a[-1])
        )
        with torch.no_grad():
            logits = model(inp)
            pred   = logits.argmax(dim=1).item()
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        handle.remove()

        z_map = captured["z"]                            # (1, p+q, H', W')
        p     = model.embed_layer.rff.out_features
        W     = model.head.W                             # (n_classes, p+q)

        phi_pos_map = z_map[:, :p]                       # (1, p, H', W')
        phi_neg_map = z_map[:, p:]                       # (1, q, H', W')
        w_pos = W[pred, :p]
        w_neg = W[pred, p:]

        # Spatial energy maps: weighted sum over subspace channels
        e_pos = (phi_pos_map * w_pos[None, :, None, None]).sum(1)  # (1, H', W')
        e_neg = (phi_neg_map * w_neg[None, :, None, None]).sum(1)

        target = inp.shape[-2:]
        e_pos = F.interpolate(e_pos.unsqueeze(1), size=target,
                              mode="bilinear", align_corners=False)[0, 0]
        e_neg = F.interpolate(e_neg.unsqueeze(1), size=target,
                              mode="bilinear", align_corners=False)[0, 0]
        e_pos = e_pos.detach().cpu().numpy()
        e_neg = e_neg.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        axes[0].imshow(rgb)
        axes[0].set_xlabel(
            f"GT: {class_names[true_lbl]}\nPred: {class_names[pred]} ({probs[pred]:.2f})",
            fontsize=9,
        )
        im1 = axes[1].imshow(_norm01(e_pos), cmap="Blues", vmin=0, vmax=1)
        axes[1].set_xlabel(
            r"Positive energy $\langle w_+,\varphi_+\rangle$" "\n(RFF — Euclidean)",
            fontsize=9,
        )
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(_norm01(e_neg), cmap="Reds", vmin=0, vmax=1)
        axes[2].set_xlabel(
            r"Negative energy $\langle w_-,\varphi_-\rangle$" "\n(SRF — hyperbolic)",
            fontsize=9,
        )
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(
            f"Pontryagin embedding energy — {idx}", fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        fig.savefig(out_dir / f"{idx:04d}.pdf", dpi=150, format="pdf",
                    bbox_inches="tight")
        plt.close(fig)

        # ── Embedding energy metrics ──────────────────────────────────────
        mean_ep = float(e_pos.mean())
        mean_en = float(e_neg.mean())
        rho     = mean_ep / (mean_ep + abs(mean_en) + 1e-8)   # energy balance ρ
        delta   = mean_ep - mean_en                            # energy contrast Δ
        all_metrics[str(idx)] = {
            "true_class":            class_names[true_lbl],
            "pred_class":            class_names[pred],
            "correct":               pred == true_lbl,
            "mean_e_pos":            mean_ep,
            "mean_e_neg":            mean_en,
            "energy_balance_rho":    rho,
            "energy_contrast_delta": delta,
        }
        print(f"  [{idx+1:3d}] {class_names[true_lbl]:>10} → {class_names[pred]}  "
              f"E+={mean_ep:.4f}  E-={mean_en:.4f}  ρ={rho:.3f}  Δ={delta:.4f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Embedding energy metrics → {out_dir / 'metrics.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-model orchestration
# ─────────────────────────────────────────────────────────────────────────────

def analyse_model(
    model_type: str,
    device: torch.device,
    data_root: Path,
    results_dir: Path,
    n_samples: int | None,
    num_lime_samples: int,
    trainable_rff: bool = TRAINABLE_RFF,
):
    print(f"\n── {_effective_name(model_type, trainable_rff).upper()} ConvNeXtV2 ──────────────────────────")
    model, class_names, run_name = load_cls_model(
        model_type, results_dir, trainable_rff=trainable_rff,
    )
    model = model.to(device)
    out_base = results_dir / run_name

    dataset = SugarcaneLeafDataset(root=data_root, split="test")

    print("  Running ScoreCAM…")
    scorecam_analysis(
        model_type=run_name,
        model=model,
        dataset=dataset,
        class_names=class_names,
        device=device,
        out_dir=out_base / "scorecam",
        n_samples=n_samples,
    )

    print("  Running LIME…")
    lime_analysis(
        model=model,
        dataset=dataset,
        class_names=class_names,
        device=device,
        out_dir=out_base / "lime",
        n_samples=n_samples,
        num_lime_samples=num_lime_samples,
    )

    # Embedding energy: any model that has a Pontryagin embed_layer
    if hasattr(model, "embed_layer"):
        print("  Running embedding energy…")
        embedding_energy_analysis(
            model=model,
            dataset=dataset,
            class_names=class_names,
            device=device,
            out_dir=out_base / "embedding_energy",
            n_samples=n_samples,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="ScoreCAM + LIME interpretability for Sugarcane Leaf Disease classifiers"
    )
    ap.add_argument("--model", choices=list(AVAILABLE_MODELS) + ["all"], default="all")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--data-root", type=str, default=None,
                    help="Path to dataset root (downloads if omitted).")
    ap.add_argument("--results-dir", type=str, default=None,
                    help="Results directory (overrides config).")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Limit number of test images to analyse (all if omitted).")
    ap.add_argument("--lime-samples", type=int, default=1000,
                    help="Number of LIME perturbation samples (default 1000).")
    ap.add_argument("--trainable-rff", action="store_true", default=False,
                    help="Load the _trff variant of the Pontryagin model. "
                         "Ignored for euclidean. Auto-detected from checkpoint "
                         "when trainable_rff was saved there.")
    args = ap.parse_args()

    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    if args.data_root:
        data_root = _find_dataset_root(Path(args.data_root))
    else:
        env_root = os.environ.get("SUGARCANE_DATA_ROOT")
        if env_root:
            data_root = _find_dataset_root(Path(env_root))
        else:
            data_root = _find_dataset_root(download_dataset())

    models = list(AVAILABLE_MODELS) if args.model == "all" else [args.model]
    for mt in models:
        try:
            analyse_model(
                model_type=mt,
                device=device,
                data_root=data_root,
                results_dir=results_dir,
                n_samples=args.n_samples,
                num_lime_samples=args.lime_samples,
                trainable_rff=args.trainable_rff,
            )
        except FileNotFoundError as e:
            print(f"  Skipping {mt}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
