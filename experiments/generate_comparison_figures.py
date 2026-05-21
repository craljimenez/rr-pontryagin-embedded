"""
generate_comparison_figures.py

Carga Euclidean y Pontryagin-tRFF sobre las MISMAS imágenes del test set,
encuentra casos donde Euclidean falla y Pont-tRFF acierta, y genera
figuras comparativas ScoreCAM y LIME side-by-side.

Uso:
    python experiments/generate_comparison_figures.py --n-samples 150 --n-show 4
"""
import argparse, os, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.cls_sugarcane import DEVICE
from run_cls_sugarcane import (
    _effective_name, _find_dataset_root, build_dataloaders,
    download_dataset, build_cls_model,
)
from interpretability_cls_sugarcane import ClassScoreCAM, _denorm, _make_predict_fn

RESULTS_DIR = Path(__file__).parent / "results" / "cls_sugarcane"
OUT_DIR     = Path(__file__).parent / "report" / "figures"


# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_type, trainable_rff, device):
    run_name  = _effective_name(model_type, trainable_rff)
    ckpt_path = RESULTS_DIR / run_name / "best_model.pth"
    ckpt      = torch.load(ckpt_path, map_location=device)
    classes   = ckpt["classes"]
    params    = ckpt.get("params", {})
    tr        = ckpt.get("trainable_rff", trainable_rff)
    model     = build_cls_model(model_type, n_classes=len(classes),
                                params=params, trainable_rff=tr)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes


def get_scorecam(model, img_t, class_idx, device):
    target_layer = model.backbone.stages[-1]
    cam_gen = ClassScoreCAM(model, target_layer)
    result  = cam_gen.generate_cam(img_t.unsqueeze(0).to(device), class_idx)
    cam_gen.remove()
    return result   # H×W numpy float [0,1]


def get_lime_mask(model, img_np, class_idx, classes, device,
                  n_segments=50, n_samples=300):
    from lime import lime_image
    from skimage.segmentation import slic

    predict_fn = _make_predict_fn(model, device)
    explainer  = lime_image.LimeImageExplainer()
    expl = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=len(classes),
        hide_color=0,
        num_samples=n_samples,
        segmentation_fn=lambda x: slic(x, n_segments=n_segments,
                                       compactness=10, sigma=1,
                                       start_label=0),
    )
    _, mask = expl.get_image_and_mask(
        class_idx, positive_only=False, num_features=10, hide_rest=False,
    )
    return mask   # +1 support, -1 contradict, 0 neutral


def overlay_lime(img_np, mask):
    out = img_np.copy().astype(float)
    alpha = 0.55
    out[mask ==  1] = (1 - alpha) * out[mask ==  1] + alpha * np.array([30, 200, 80])
    out[mask == -1] = (1 - alpha) * out[mask == -1] + alpha * np.array([220, 30, 30])
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
def find_cases(dataset, euc_model, ptr_model, device, n_scan, n_cases):
    cases = []
    for i in range(min(n_scan, len(dataset))):
        img_t, true_lbl = dataset[i]
        inp = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            euc_pred = euc_model(inp).argmax(1).item()
            ptr_pred = ptr_model(inp).argmax(1).item()
        if euc_pred != true_lbl and ptr_pred == true_lbl:
            cases.append({
                "img_np":   _denorm(img_t),
                "img_t":    img_t,
                "true":     int(true_lbl),
                "euc_pred": euc_pred,
                "ptr_pred": ptr_pred,
            })
            print(f"  Case {len(cases)}: img#{i}  "
                  f"true={true_lbl}  euc={euc_pred}  ptr={ptr_pred}")
        if len(cases) >= n_cases:
            break
    return cases


# ─────────────────────────────────────────────────────────────────────────────
def make_scorecam_figure(cases, classes, device, out_path):
    n = len(cases)
    fig, axes = plt.subplots(
        n, 3, figsize=(14, 4.2 * n),
        gridspec_kw={"wspace": 0.04, "hspace": 0.38},
    )
    if n == 1:
        axes = [axes]

    for row, c in enumerate(cases):
        img_np  = c["img_np"]
        img_t   = c["img_t"]
        true_c  = classes[c["true"]]
        euc_c   = classes[c["euc_pred"]]
        ptr_c   = classes[c["ptr_pred"]]

        print(f"  ScoreCAM row {row+1}/{n}...")
        cam_euc = get_scorecam(c["euc_model"], img_t, c["true"], device)
        cam_ptr = get_scorecam(c["ptr_model"], img_t, c["true"], device)

        for col, (title, cam, correct) in enumerate([
            (f"Original — True: {true_c}", None, None),
            (f"Euclidean\n✗ Predicted: {euc_c}", cam_euc, False),
            (f"Pontryagin-tRFF\n✓ Correct: {true_c}", cam_ptr, True),
        ]):
            ax = axes[row][col]
            ax.imshow(img_np)
            if cam is not None:
                ax.imshow(cam, cmap="jet", alpha=0.55,
                          vmin=0, vmax=cam.max() + 1e-8)
            ax.axis("off")
            if correct is not None:
                ec = "#27ae60" if correct else "#c0392b"
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(4)
            color = "#1a6e1a" if correct else ("#8b0000" if correct is False else "#2c3e50")
            ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=4)

    fig.legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="#27ae60", linewidth=3.5,
                           label="Correct — Pontryagin-tRFF"),
            mpatches.Patch(facecolor="white", edgecolor="#c0392b", linewidth=3.5,
                           label="Wrong — Euclidean"),
        ],
        loc="lower center", ncol=2, fontsize=10,
        frameon=True, bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_lime_figure(cases, classes, device, out_path):
    n = len(cases)
    fig, axes = plt.subplots(
        n, 3, figsize=(14, 4.2 * n),
        gridspec_kw={"wspace": 0.04, "hspace": 0.38},
    )
    if n == 1:
        axes = [axes]

    for row, c in enumerate(cases):
        img_np  = c["img_np"]
        img_t   = c["img_t"]
        true_c  = classes[c["true"]]
        euc_c   = classes[c["euc_pred"]]

        print(f"  LIME row {row+1}/{n}...")
        mask_euc = get_lime_mask(c["euc_model"], img_np, c["euc_pred"], classes, device)
        mask_ptr = get_lime_mask(c["ptr_model"], img_np, c["ptr_pred"], classes, device)

        for col, (title, overlay, correct) in enumerate([
            (f"Original — True: {true_c}", img_np, None),
            (f"Euclidean\n✗ Predicted: {euc_c}", overlay_lime(img_np, mask_euc), False),
            (f"Pontryagin-tRFF\n✓ Correct: {true_c}", overlay_lime(img_np, mask_ptr), True),
        ]):
            ax = axes[row][col]
            ax.imshow(overlay)
            ax.axis("off")
            if correct is not None:
                ec = "#27ae60" if correct else "#c0392b"
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(4)
            color = "#1a6e1a" if correct else ("#8b0000" if correct is False else "#2c3e50")
            ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=4)

    fig.legend(
        handles=[
            mpatches.Patch(color=(30/255, 200/255, 80/255),
                           label="Supports prediction"),
            mpatches.Patch(color=(220/255, 30/255, 30/255),
                           label="Contradicts prediction"),
        ],
        loc="lower center", ncol=2, fontsize=10,
        frameon=True, bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=150)
    ap.add_argument("--n-show",    type=int, default=4)
    ap.add_argument("--device",    default=DEVICE)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    if args.data_root:
        data_root = _find_dataset_root(Path(args.data_root))
    else:
        env = os.environ.get("SUGARCANE_DATA_ROOT")
        data_root = _find_dataset_root(Path(env) if env else download_dataset())
    data    = build_dataloaders(data_root)
    classes = data["classes"]
    print(f"Classes: {classes}")

    # Modelos
    print("Loading models...")
    euc_model, _ = load_model("euclidean",  False, device)
    ptr_model, _ = load_model("pontryagin", True,  device)   # trainable_rff → pontryagin_trff
    print("OK")

    # Encuentra casos
    print(f"\nScanning {args.n_samples} test images for Euc-wrong / Pont-right cases...")
    cases = find_cases(data["test_ds"], euc_model, ptr_model,
                       device, args.n_samples, args.n_show)
    print(f"Found {len(cases)} cases.")
    if not cases:
        print("No cases found. Try increasing --n-samples.")
        return

    # Añade referencias a los modelos en cada caso
    for c in cases:
        c["euc_model"] = euc_model
        c["ptr_model"] = ptr_model

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating ScoreCAM figure...")
    make_scorecam_figure(cases, classes, device,
                         OUT_DIR / "comparison_scorecam_matched.pdf")

    print("\nGenerating LIME figure...")
    make_lime_figure(cases, classes, device,
                     OUT_DIR / "comparison_lime_matched.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
