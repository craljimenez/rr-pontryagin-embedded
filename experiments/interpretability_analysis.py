"""SegScoreCAM interpretability analysis for UAV semantic segmentation models.

ScoreCAM adapted for pixel-wise segmentation: the score for each feature-map
channel is the mean softmax probability of the target class over all spatial
positions of the segmentation output, replacing the scalar logit used in the
original classification version.

Outputs per model (results/seg_uav/<model>/gradcam/):
  - <image_name>_scorecam.pdf  — 1×4 figure per image
  - metrics.json               — per-image activation, GT-alignment and
                                 confidence-gain metrics
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from configs.seg_uav import AVAILABLE_MODELS, RESULTS_DIR, DATASET_ROOT, IMG_SIZE
from run_seg_uav import UAVSegDataset, build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0.0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def _otsu_threshold(cam: np.ndarray) -> float:
    """Otsu's method on a [0,1] CAM — returns the optimal binarisation threshold."""
    hist, edges = np.histogram(cam, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.5
    best_thresh, best_var = 0.5, -1.0
    w0, sum0 = 0, 0.0
    total_sum = float((hist * centers).sum())
    for i in range(len(hist)):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * centers[i]
        mu0 = sum0 / w0
        mu1 = (total_sum - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = centers[i]
    return float(best_thresh)


def compute_cam_metrics(cam: np.ndarray, gt_mask: np.ndarray | None = None) -> dict:
    """Activation statistics and GT-alignment metrics for one CAM.

    Binarisation uses Otsu's threshold (adaptive per image) instead of a
    fixed 0.5, producing more meaningful IoU/precision/recall comparisons
    across models.
    """
    metrics: dict = {
        "mean_activation": float(cam.mean()),
        "coverage_top10pct": float((cam > np.percentile(cam, 90)).mean()),
    }

    if gt_mask is not None:
        thresh = _otsu_threshold(cam)
        metrics["otsu_threshold"] = thresh

        cam_bin = (cam >= thresh).astype(np.uint8)
        gt_bin  = (gt_mask > 0).astype(np.uint8)

        intersection = int((cam_bin & gt_bin).sum())
        union        = int((cam_bin | gt_bin).sum())
        cam_pos      = int(cam_bin.sum())
        gt_pos       = int(gt_bin.sum())

        metrics["gt_iou"]       = intersection / union     if union    > 0 else 0.0
        metrics["gt_recall"]    = intersection / gt_pos    if gt_pos   > 0 else 0.0
        metrics["gt_precision"] = intersection / cam_pos   if cam_pos  > 0 else 0.0

        # Pearson correlation between continuous CAM and binary GT mask
        cam_flat = cam.ravel()
        gt_flat  = gt_bin.ravel().astype(float)
        if cam_flat.std() > 0 and gt_flat.std() > 0:
            metrics["gt_pearson"] = float(np.corrcoef(cam_flat, gt_flat)[0, 1])
        else:
            metrics["gt_pearson"] = 0.0

    return metrics


def compute_confidence_gain(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    cam: np.ndarray,
    class_idx: int,
    gt_mask: np.ndarray,
) -> dict:
    """Pixel-level confidence change when the input is masked by the CAM.

    The explanatory image  x_exp = x * CAM  is fed through the model.  For
    every pixel that belongs to `class_idx` in the GT mask we compute:

        pct_change(i,j) = (P_exp(i,j) - P_orig(i,j)) / P_orig(i,j) * 100

    where P_orig and P_exp are the softmax probabilities for `class_idx`.
    Mean and std of pct_change are returned, summarising whether the
    explanation increases or decreases model confidence on the relevant pixels.
    """
    cam_t = torch.from_numpy(cam).float().to(input_tensor.device)
    cam_t = cam_t.unsqueeze(0).unsqueeze(0)          # [1, 1, H, W]

    with torch.no_grad():
        p_orig = torch.softmax(model(input_tensor), dim=1)[0, class_idx]        # [H, W]
        p_exp  = torch.softmax(model(input_tensor * cam_t), dim=1)[0, class_idx]

    p_orig_np = p_orig.cpu().numpy()
    p_exp_np  = p_exp.cpu().numpy()

    class_pixels = (gt_mask == class_idx)
    n_pixels = int(class_pixels.sum())

    if n_pixels == 0:
        return {
            "mean_pct_confidence_change": None,
            "std_pct_confidence_change":  None,
            "n_pixels": 0,
        }

    delta  = p_exp_np[class_pixels] - p_orig_np[class_pixels]
    pct    = delta / (p_orig_np[class_pixels] + 1e-8) * 100.0

    return {
        "mean_pct_confidence_change": float(pct.mean()),
        "std_pct_confidence_change":  float(pct.std()),
        "n_pixels": n_pixels,
    }


# ---------------------------------------------------------------------------
# SegScoreCAM
# ---------------------------------------------------------------------------

class SegScoreCAM:
    """ScoreCAM adapted for pixel-wise segmentation models.

    For each feature-map channel the score is the mean softmax probability of
    the target class across all spatial positions of the segmentation output.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._feature_maps: torch.Tensor | None = None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, *args):
        self._feature_maps = args[-1]

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self._feature_maps = None
        with torch.no_grad():
            self.model(input_tensor)
        if self._feature_maps is None:
            raise RuntimeError("Forward hook did not capture feature maps.")
        feature_maps = self._feature_maps.detach()

        upsampled = F.interpolate(
            feature_maps,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).detach()

        n_ch = upsampled.shape[1]
        scores: list[float] = []

        with torch.no_grad():
            for k in range(n_ch):
                fmap = upsampled[:, k : k + 1]
                lo, hi = fmap.min(), fmap.max()
                mask = (fmap - lo) / (hi - lo) if hi > lo else torch.zeros_like(fmap)
                output = self.model(input_tensor * mask)
                score = torch.softmax(output, dim=1)[:, class_idx, :, :].mean()
                scores.append(score.item())

        weights = torch.tensor(scores, device=feature_maps.device)
        weights = torch.abs(weights)
        total = weights.sum()
        weights = weights / total if total > 0 else torch.ones_like(weights) / n_ch

        _, n_ch, fh, fw = feature_maps.shape
        cam = torch.zeros((fh, fw), device=feature_maps.device)
        for k in range(n_ch):
            cam += weights[k] * feature_maps[0, k]
        cam = F.relu(cam)
        cam = (
            F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return _normalize_cam(cam)


# ---------------------------------------------------------------------------
# Model / layer utilities
# ---------------------------------------------------------------------------

def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Conv2d | None:
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def load_best_model(model_type: str, results_dir=None):
    model_dir = (Path(results_dir) if results_dir else RESULTS_DIR) / model_type
    params_path = model_dir / "hpo" / "best_params.json"
    checkpoint_path = model_dir / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)

    model = build_model(model_type, params)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
    except (EOFError, RuntimeError) as e:
        raise FileNotFoundError(
            f"Checkpoint for {model_type} is corrupt or incomplete "
            f"({checkpoint_path}): {e}"
        ) from e
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, params


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_grid_image(
    out_path: Path,
    image: np.ndarray,
    cam_bg: np.ndarray,
    cam_sg: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
):
    """1×4 figure: heatmap bg | heatmap sugarcane | predicted mask | ground truth."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    def _overlay(ax, img, cam, title):
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _overlay(axes[0], image, cam_bg, "Background heatmap")
    _overlay(axes[1], image, cam_sg, "Sugarcane heatmap")

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Ground truth mask", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--data-root",   type=str, default=None,
                   help="Path to UAV_segmantation dataset root (overrides config).")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for results (overrides config).")
    args = p.parse_args()
    data_root   = Path(args.data_root)   if args.data_root   else DATASET_ROOT
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    print(f"Using dataset root: {data_root}")
    dataset = UAVSegDataset(
        root=data_root,
        split="test",
        img_size=IMG_SIZE,
        augment=False,
        binary_sugarcane=True,
        only_sugarcane=False,
    )
    print(f"Test split: {len(dataset)} images")

    for model_type in AVAILABLE_MODELS:
        try:
            model, params = load_best_model(model_type, results_dir=results_dir)
        except FileNotFoundError as exc:
            print(f"Skipping {model_type}: {exc}")
            continue

        print(f"Processing {model_type} — params: {params}")
        output_dir = results_dir / model_type / "gradcam"
        output_dir.mkdir(parents=True, exist_ok=True)

        target_layer = find_last_conv_layer(model.backbone)
        if target_layer is None:
            raise RuntimeError(f"No Conv2d found in backbone of {model_type}")

        scorecam = SegScoreCAM(model, target_layer)
        model_metrics: dict = {}

        for idx in range(len(dataset)):
            image_tensor, mask_tensor = dataset[idx]
            image_name = dataset.samples[idx]
            input_tensor = image_tensor.unsqueeze(0)
            image_vis = denormalize_image(image_tensor).permute(1, 2, 0).cpu().numpy()
            image_vis = np.clip(image_vis, 0.0, 1.0)
            gt_mask = mask_tensor.cpu().numpy()      # [H, W] values in {0, 1}

            with torch.no_grad():
                logits = model(input_tensor)
                pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            cam_bg = scorecam.generate_cam(input_tensor, 0)
            cam_sg = scorecam.generate_cam(input_tensor, 1)

            model_metrics[image_name] = {
                "background": {
                    **compute_cam_metrics(cam_bg),
                    "confidence_gain": compute_confidence_gain(
                        model, input_tensor, cam_bg, class_idx=0, gt_mask=gt_mask
                    ),
                },
                "sugarcane": {
                    **compute_cam_metrics(cam_sg, gt_mask),
                    "confidence_gain": compute_confidence_gain(
                        model, input_tensor, cam_sg, class_idx=1, gt_mask=gt_mask
                    ),
                },
            }

            save_grid_image(
                out_path=output_dir / f"{image_name}_scorecam.pdf",
                image=image_vis,
                cam_bg=cam_bg,
                cam_sg=cam_sg,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )

        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(model_metrics, f, indent=2)

        print(f"  Saved CAM figures  → {output_dir}")
        print(f"  Saved metrics      → {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    main()
