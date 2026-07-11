"""Interpretability analysis for Few-Shot Segmentation (FSS) models.

Compares Euclidean vs Pontryagin on UAV Sugarcane test episodes:

  1. ScoreCAM on query image — which backbone regions drive prototype similarity.
     Gradient-free: each activation channel is used as a mask on the query; the
     resulting model score (mean logit, prototype fixed) weights that channel.
     Top-K channels by activation magnitude are evaluated for efficiency.
     Same target layer (last decoder block) for both models → fair comparison.

  2. Pontryagin embedding energy per pixel:
       E_+(pixel) = || φ_+(pixel) ||²   (RFF / positive-definite channels)
       E_-(pixel) = || φ_-(pixel) ||²   (SRF / negative-definite channels)
     and their signed contributions to the J-inner product with the prototype:
       c_+(pixel) = ⟨φ_+(pixel), proto_fg+⟩
       c_-(pixel) = −⟨φ_-(pixel), proto_fg-⟩  (negative subspace contribution)

  3. Monte Carlo Dropout uncertainty — T stochastic passes with Dropout2d
     injected after the backbone.  Prototype is deterministic; only query
     encoding is randomised.  Output: per-pixel predictive variance.

  4. Cross-episode scatter — mean MC variance vs mean embedding energy
     (positive and negative) per episode, coloured by Pontryagin IoU.

  5. CAM quality metrics (Otsu binarisation of each ScoreCAM):
       gt_iou, gt_recall, gt_precision, gt_pearson,
       mean_activation, coverage_top10pct

  6. Confidence-gain analysis (FSS adaptation):
     The query is multiplied pixel-wise by the ScoreCAM and re-fed into the
     model (prototype fixed).  On GT-foreground pixels we compute:
       pct_change(i,j) = (P_masked − P_orig) / P_orig × 100
     A positive mean = the explanation highlights the features that drive
     foreground confidence.

Outputs saved to  results/fss_sugarcane/interpretability_<k>shot[_trainable]/:
  ep_NNN_comparison.pdf        Euclidean vs Pontryagin ScoreCAM + prediction
  ep_NNN_energy.pdf            energy maps + MC Dropout
  scatter_unc_vs_energy.pdf    aggregate scatter per episode
  iou_comparison.pdf           per-episode IoU bar chart
  mc_variance_comparison.pdf   per-episode MC variance bar chart
  conf_gain_comparison.pdf     confidence-gain box plots
  cam_metrics_comparison.pdf   gt_iou / gt_pearson bar charts
  metrics.csv                  all per-episode scalar metrics

Usage:
    python interpretability_fss.py --k-shot 1
    python interpretability_fss.py --k-shot 1 --n-viz 20 --n-mc 30 --device cuda
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.fss_sugarcane import (
    DATASET_ROOT, DEVICE, IMG_SIZE, K_SHOT, N_EPISODES_TEST,
    RESULTS_DIR, TARGET_CLS, TRAINABLE_RFF,
)
from prfe.data.fss_dataset import EpisodicUAVDataset
from prfe.models.fss import _poincare_dist
from run_fss_sugarcane import build_model

# Reuse CAM metric helpers from the FCN interpretability script
from interpretability_analysis import _otsu_threshold, compute_cam_metrics

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

N_PASSES_MC = 30
P_DROP      = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _denorm(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor → (H, W, 3) float [0, 1]."""
    img = t.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-8 else (a - lo) / (hi - lo)


def _overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5,
             cmap: str = "jet") -> np.ndarray:
    heat = plt.colormaps[cmap](cam)[..., :3]
    return (alpha * heat + (1 - alpha) * rgb).clip(0, 1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _iou(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(bool); t = target.astype(bool)
    return float((p & t).sum() / ((p | t).sum() + 1e-6))


def _axoff(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _gt_contour(ax, gt_mask: np.ndarray,
                color: str = "lime", linewidth: float = 1.5) -> None:
    """Overlay GT boundary as a green contour — no extra subplot needed."""
    if gt_mask.max() > 0.5:
        ax.contour(gt_mask, levels=[0.5], colors=[color], linewidths=[linewidth])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(
    model_name:   str,
    k_shot:       int,
    results_dir:  Path,
    device:       torch.device,
    trainable_rff: bool = False,
) -> nn.Module:
    suffix  = "_trainable" if trainable_rff else ""
    run_dir = results_dir / f"{model_name}_{k_shot}shot{suffix}"
    ckpt    = run_dir / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    params = {}
    hpo_json = run_dir / "hpo" / "best_params.json"
    if hpo_json.exists():
        params = {k: v for k, v in json.loads(hpo_json.read_text()).items()
                  if not k.startswith("_")}

    model, _ = build_model(model_name, k_shot, device, params=params,
                           trainable_rff=trainable_rff)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ScoreCAM for FSS
# ─────────────────────────────────────────────────────────────────────────────

class FSSScoreCAM:
    """ScoreCAM applied to the query image of a few-shot episode.

    Gradient-free: each channel of the target activation map is upsampled and
    normalised to [0, 1], used as a spatial mask on the query image, and fed
    through the model with the prototype fixed.  The resulting mean logit score
    is the channel weight.  Softmax of scores → weighted sum of activations →
    ReLU → upsample → normalise.

    Only the top-max_channels channels (by mean |activation|) are scored for
    efficiency.  Target: last decoder block of the shared UNet backbone.
    """

    def __init__(self, model: nn.Module, max_channels: int = 64) -> None:
        self.model        = model
        self.max_channels = max_channels
        self._features    = None
        target = model.backbone.dec_blocks[-1]
        self._fwd = target.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._features = out.detach()

    def remove(self):
        self._fwd.remove()

    def compute(
        self,
        support_imgs:  torch.Tensor,  # (1, K, 3, H, W)
        support_masks: torch.Tensor,  # (1, K, H, W)
        query_img:     torch.Tensor,  # (1, 3, H, W)
    ) -> np.ndarray:
        """Return normalised [0, 1] ScoreCAM heatmap on the query image."""
        H, W = query_img.shape[-2:]
        self.model.eval()
        self._features = None

        with torch.no_grad():
            proto_fg, _ = self.model._prototypes(support_imgs, support_masks)
            # Capture query activations (hook fires here, overwrites support ones)
            _ = self.model.backbone(query_img)

        if self._features is None:
            return np.zeros((H, W))

        features = self._features.clone()   # (1, C, H', W') — freeze before loop
        C = features.shape[1]

        # ── Step 1: build predicted-FG mask using the model itself ───────────
        # Mean-over-all-pixels is dominated by background (≥90% of image).
        # Both channel selection and scoring are restricted to predicted-FG pixels
        # so background channels cannot win by majority vote.
        with torch.no_grad():
            orig_logits = _logits_fixed_proto(self.model, proto_fg, query_img)  # (1,H,W)
        pred_fg = (orig_logits[0] > 0)          # (H, W) bool — model's own FG mask

        # ── Step 2: select top-K channels by mean activation INSIDE pred-FG ──
        feats_up   = F.interpolate(features, size=(H, W),
                                   mode="bilinear", align_corners=False)[0]  # (C,H,W)
        if pred_fg.any():
            importance = feats_up[:, pred_fg].mean(dim=1)   # (C,) — mean inside pred FG
        else:
            importance = features[0].abs().mean(dim=(-2, -1))
        top_k   = min(self.max_channels, C)
        top_idx = importance.topk(top_k).indices             # (top_k,)

        # ── Step 3: baseline score restricted to pred-FG pixels ──────────────
        with torch.no_grad():
            base_logits = _logits_fixed_proto(
                self.model, proto_fg, torch.zeros_like(query_img)
            )[0]   # (H, W)
        baseline_score = (base_logits[pred_fg].mean() if pred_fg.any()
                          else base_logits.mean())

        # ── Step 4: score each channel (FG-pixel delta from baseline) ─────────
        scores = torch.zeros(top_k, device=features.device)
        with torch.no_grad():
            for i, c_idx in enumerate(top_idx):
                act_up   = feats_up[c_idx]                   # already at (H, W)
                lo, hi   = act_up.min(), act_up.max()
                if hi - lo < 1e-8:
                    continue
                mask_ch  = (act_up - lo) / (hi - lo)
                masked_q = query_img * mask_ch.unsqueeze(0).unsqueeze(0)
                logits   = _logits_fixed_proto(self.model, proto_fg, masked_q)[0]  # (H,W)
                fg_score = (logits[pred_fg].mean() if pred_fg.any() else logits.mean())
                scores[i] = fg_score - baseline_score

        # ── Step 5: ReLU weights — only FG-boosting channels contribute ───────
        weights = F.relu(scores)
        w_sum   = weights.sum()
        if w_sum < 1e-8:                         # fallback: uniform weights
            weights = torch.ones_like(weights) / top_k
        else:
            weights = weights / w_sum

        top_acts = feats_up[top_idx]                         # (top_k, H, W)
        cam      = (weights[:, None, None] * top_acts).sum(dim=0)   # (H, W)
        cam      = F.relu(cam)
        return _norm01(cam.cpu().numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Pontryagin embedding energy decomposition
# ─────────────────────────────────────────────────────────────────────────────

def embedding_energy_fss(
    model,                          # PontryaginFewShotSeg
    support_imgs:  torch.Tensor,    # (1, K, 3, H, W)
    support_masks: torch.Tensor,    # (1, K, H, W)
    query_img:     torch.Tensor,    # (1, 3, H, W)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel Pontryagin embedding energy decomposition.

    Consistent with interpretability_cls_sugarcane and interpretability_unet:
    energy = prototype-weighted projection ⟨proto, φ⟩, not raw squared norm.

    Returns (H, W) maps:
        e_pos  : ⟨proto_fg+, φ_+(pixel)⟩   — RFF subspace contribution
        e_neg  : ⟨proto_fg-, φ_-(pixel)⟩   — SRF subspace contribution
    """
    p = model.embed.rff.out_features          # 2 * n_rff

    with torch.no_grad():
        proto_fg, _ = model._prototypes(support_imgs, support_masks)
        q_feat = model.backbone(query_img)
        q_emb  = model.embed(q_feat)          # (1, D, H, W)

    phi_pos = q_emb[:, :p]                   # (1, p, H, W)
    phi_neg = q_emb[:, p:]                   # (1, q, H, W)

    e_pos = (phi_pos * proto_fg[:, :p, None, None]).sum(dim=1)[0].cpu().numpy()
    e_neg = (phi_neg * proto_fg[:, p:, None, None]).sum(dim=1)[0].cpu().numpy()

    return e_pos, e_neg


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo Dropout uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def mc_uncertainty_fss(
    model,
    support_imgs:  torch.Tensor,
    support_masks: torch.Tensor,
    query_img:     torch.Tensor,
    n_passes:      int   = N_PASSES_MC,
    p_drop:        float = P_DROP,
) -> np.ndarray:
    """Per-pixel predictive variance via Monte Carlo Dropout.

    Dropout2d(p) is injected between the backbone output and the embedding /
    similarity head.  The support prototype is computed once (deterministic);
    only the query encoding is stochastic.  This isolates query-prediction
    uncertainty from support-prototype uncertainty.

    Returns:
        variance map (H, W) — higher = model is more uncertain about that pixel.
    """
    dev  = next(model.parameters()).device
    drop = nn.Dropout2d(p=p_drop).to(dev)
    drop.train()

    is_pont = hasattr(model, "embed")

    with torch.no_grad():
        proto_fg, _ = model._prototypes(support_imgs, support_masks)

    probs_list = []
    with torch.no_grad():
        for _ in range(n_passes):
            q_feat = drop(model.backbone(query_img))

            if is_pont:
                q_emb  = model.embed(q_feat)
                J      = model.J[None, :, None, None]
                logits = (q_emb * proto_fg[:, :, None, None] * J).sum(dim=1)
            else:
                q_norm = F.normalize(q_feat, dim=1)
                p_norm = F.normalize(proto_fg, dim=1)[:, :, None, None]
                logits = model.alpha * (q_norm * p_norm).sum(dim=1)

            probs_list.append(torch.sigmoid(logits)[0].cpu())  # (H, W)

    return torch.stack(probs_list).var(dim=0).numpy()          # (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# CAM quality metrics (FSS adaptation)
# ─────────────────────────────────────────────────────────────────────────────

def _logits_fixed_proto(
    model,
    proto_fg:  torch.Tensor,  # (1, D) precomputed prototype
    query_img: torch.Tensor,  # (1, 3, H, W)
) -> torch.Tensor:
    """Forward pass reusing a precomputed prototype — avoids re-encoding support."""
    if hasattr(model, "delta"):          # Hyperbolic (unique attribute)
        q_emb = model.embed(model.backbone(query_img))     # (1, C, H, W)
        q_pts = q_emb.permute(0, 2, 3, 1)                  # (1, H, W, C)
        p_pts = proto_fg[:, None, None, :]                 # (1, 1, 1, C)
        dist  = _poincare_dist(q_pts, p_pts, model.c, model.eps)
        return model.alpha * (model.delta - dist)
    elif hasattr(model, "embed"):        # Pontryagin
        q_feat = model.backbone(query_img)
        q_emb  = model.embed(q_feat)
        J      = model.J[None, :, None, None]
        return (q_emb * proto_fg[:, :, None, None] * J).sum(dim=1)
    else:                                # Euclidean
        q_feat = model.backbone(query_img)
        q_norm = F.normalize(q_feat, dim=1)
        p_norm = F.normalize(proto_fg, dim=1)[:, :, None, None]
        return model.alpha * (q_norm * p_norm).sum(dim=1)


def compute_confidence_gain_fss(
    model,
    support_imgs:  torch.Tensor,   # (1, K, 3, H, W)
    support_masks: torch.Tensor,   # (1, K, H, W)
    query_img:     torch.Tensor,   # (1, 3, H, W)
    cam:           np.ndarray,     # (H, W) normalised [0, 1]
    gt_mask:       np.ndarray,     # (H, W) binary float
) -> dict:
    """% confidence change when the query is masked by the ScoreCAM.

    Prototype is computed once from support images (deterministic).
    The masked query  x_exp = x_query * CAM  is fed into the model using the
    same prototype.  On GT-foreground pixels:

        pct_change = (P_masked − P_orig) / P_orig × 100

    Positive mean → the CAM highlights the features that the model actually
    uses to decide "foreground here".
    """
    cam_t = torch.from_numpy(cam).float().to(query_img.device)
    cam_t = cam_t.unsqueeze(0).unsqueeze(0)           # (1, 1, H, W)

    with torch.no_grad():
        proto_fg, _ = model._prototypes(support_imgs, support_masks)
        logits_orig   = _logits_fixed_proto(model, proto_fg, query_img)
        logits_masked = _logits_fixed_proto(model, proto_fg, query_img * cam_t)

    p_orig   = torch.sigmoid(logits_orig[0]).cpu().numpy()    # (H, W)
    p_masked = torch.sigmoid(logits_masked[0]).cpu().numpy()  # (H, W)

    fg = gt_mask > 0.5
    n_fg = int(fg.sum())

    if n_fg == 0:
        return {
            "mean_pct_conf_gain": float("nan"),
            "std_pct_conf_gain":  float("nan"),
            "n_fg_pixels":        0,
        }

    delta = p_masked[fg] - p_orig[fg]
    pct   = delta / (p_orig[fg] + 1e-8) * 100.0
    return {
        "mean_pct_conf_gain": float(pct.mean()),
        "std_pct_conf_gain":  float(pct.std()),
        "n_fg_pixels":        n_fg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_episode(
    episode_idx:  int,
    batch:        dict,
    model_euc:    nn.Module,
    model_pont:   nn.Module,
    device:       torch.device,
    out_dir:      Path,
    n_mc:         int = N_PASSES_MC,
) -> dict:
    """Produce all figures for one episode; return scalar metrics."""
    K = batch["support_imgs"].shape[1]

    sup_imgs  = batch["support_imgs"].to(device)    # (1, K, 3, H, W)
    sup_masks = batch["support_masks"].to(device)   # (1, K, H, W)
    q_img     = batch["query_img"].to(device)        # (1, 3, H, W)
    q_mask    = batch["query_mask"].to(device)       # (1, H, W)

    gt     = q_mask[0].cpu().numpy()
    q_rgb  = _denorm(q_img[0])
    sup_rgb  = [_denorm(sup_imgs[0, k]) for k in range(K)]
    sup_msk  = [sup_masks[0, k].cpu().numpy() for k in range(K)]

    # ── Predictions ──────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_e = (model_euc.forward(sup_imgs, sup_masks, q_img)[0] > 0).float().cpu().numpy()
        pred_p = (model_pont.forward(sup_imgs, sup_masks, q_img)[0] > 0).float().cpu().numpy()

    iou_e = _iou(pred_e, gt)
    iou_p = _iou(pred_p, gt)

    # ── ScoreCAM ─────────────────────────────────────────────────────────────
    cam_euc  = FSSScoreCAM(model_euc)
    cam_pont = FSSScoreCAM(model_pont)
    heat_e = cam_euc.compute(sup_imgs, sup_masks, q_img)
    heat_p = cam_pont.compute(sup_imgs, sup_masks, q_img)
    cam_euc.remove()
    cam_pont.remove()

    # ── Pontryagin energy ─────────────────────────────────────────────────────
    e_pos, e_neg = embedding_energy_fss(
        model_pont, sup_imgs, sup_masks, q_img
    )

    # ── MC Dropout uncertainty ────────────────────────────────────────────────
    unc_e = mc_uncertainty_fss(model_euc,  sup_imgs, sup_masks, q_img, n_passes=n_mc)
    unc_p = mc_uncertainty_fss(model_pont, sup_imgs, sup_masks, q_img, n_passes=n_mc)

    # ── CAM quality metrics (Otsu binarisation vs GT) ─────────────────────────
    cam_m_e = compute_cam_metrics(heat_e, gt_mask=gt)
    cam_m_p = compute_cam_metrics(heat_p, gt_mask=gt)

    # ── Confidence-gain analysis ──────────────────────────────────────────────
    cg_e = compute_confidence_gain_fss(model_euc,  sup_imgs, sup_masks, q_img, heat_e, gt)
    cg_p = compute_confidence_gain_fss(model_pont, sup_imgs, sup_masks, q_img, heat_p, gt)

    # ────────────────────────────────────────────────────────────────────────
    # Figure 1: ScoreCAM comparison — GT shown as green contour on every panel
    # Layout: [sup1..K (support mask contour)] [query] [cam_E] [pred_E] [cam_P] [pred_P]
    # ────────────────────────────────────────────────────────────────────────
    n_cols = K + 1 + 2 + 2   # sup + query + (cam_E, pred_E) + (cam_P, pred_P)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5))

    col = 0
    for k in range(K):
        axes[col].imshow(sup_rgb[k])
        _gt_contour(axes[col], sup_msk[k], color="lime")  # support mask contour
        axes[col].set_title(f"Support {k+1}", fontsize=9)
        _axoff(axes[col]); col += 1

    axes[col].imshow(q_rgb)
    _gt_contour(axes[col], gt)
    axes[col].set_title("Query", fontsize=9); _axoff(axes[col]); col += 1

    axes[col].imshow(_overlay(q_rgb, heat_e))
    _gt_contour(axes[col], gt)
    axes[col].set_title("Euclidean\nScoreCAM", fontsize=9); _axoff(axes[col]); col += 1

    axes[col].imshow(pred_e, cmap="gray", vmin=0, vmax=1)
    _gt_contour(axes[col], gt)
    axes[col].set_title(f"Euclidean pred\nIoU={iou_e:.3f}", fontsize=9)
    _axoff(axes[col]); col += 1

    axes[col].imshow(_overlay(q_rgb, heat_p))
    _gt_contour(axes[col], gt)
    axes[col].set_title("Pontryagin\nScoreCAM", fontsize=9); _axoff(axes[col]); col += 1

    axes[col].imshow(pred_p, cmap="gray", vmin=0, vmax=1)
    _gt_contour(axes[col], gt)
    axes[col].set_title(f"Pontryagin pred\nIoU={iou_p:.3f}", fontsize=9)
    _axoff(axes[col])

    plt.suptitle(
        f"Episode {episode_idx:03d} — ScoreCAM comparison  (green contour = GT)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"ep_{episode_idx:03d}_comparison.pdf",
                dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ────────────────────────────────────────────────────────────────────────
    # Figure 2a: MC Dropout — Euclidean  (1 × 3)
    # query | prediction | MC variance   — GT contour on all panels
    # ────────────────────────────────────────────────────────────────────────
    fig_euc, axes_euc = plt.subplots(1, 3, figsize=(12, 4))

    axes_euc[0].imshow(q_rgb)
    _gt_contour(axes_euc[0], gt)
    axes_euc[0].set_title("Query", fontsize=9)

    axes_euc[1].imshow(pred_e, cmap="gray", vmin=0, vmax=1)
    _gt_contour(axes_euc[1], gt)
    axes_euc[1].set_title(f"Prediction  IoU={iou_e:.3f}", fontsize=9)

    im_ue = axes_euc[2].imshow(_norm01(unc_e), cmap="inferno", vmin=0, vmax=1)
    _gt_contour(axes_euc[2], gt)
    axes_euc[2].set_title(
        f"MC Dropout variance\n(N={n_mc}, p={P_DROP})", fontsize=9
    )
    plt.colorbar(im_ue, ax=axes_euc[2], fraction=0.046, pad=0.04)

    for ax in axes_euc:
        _axoff(ax)

    plt.suptitle(
        f"Episode {episode_idx:03d} — Euclidean MC Dropout  (green contour = GT)",
        fontsize=10,
    )
    plt.tight_layout()
    fig_euc.savefig(out_dir / f"ep_{episode_idx:03d}_mc_euclidean.pdf",
                    dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig_euc)

    # ────────────────────────────────────────────────────────────────────────
    # Figure 2b: MC Dropout + Embedding energy — Pontryagin  (1 × 5)
    # query | prediction | MC variance | E_pos | E_neg — GT contour on all
    # ────────────────────────────────────────────────────────────────────────
    fig_pont, axes_pont = plt.subplots(1, 5, figsize=(20, 4))

    axes_pont[0].imshow(q_rgb)
    _gt_contour(axes_pont[0], gt)
    axes_pont[0].set_title("Query", fontsize=9)

    axes_pont[1].imshow(pred_p, cmap="gray", vmin=0, vmax=1)
    _gt_contour(axes_pont[1], gt)
    axes_pont[1].set_title(f"Prediction  IoU={iou_p:.3f}", fontsize=9)

    im_up = axes_pont[2].imshow(_norm01(unc_p), cmap="inferno", vmin=0, vmax=1)
    _gt_contour(axes_pont[2], gt)
    axes_pont[2].set_title(
        f"MC Dropout variance\n(N={n_mc}, p={P_DROP})", fontsize=9
    )
    plt.colorbar(im_up, ax=axes_pont[2], fraction=0.046, pad=0.04)

    im_ep = axes_pont[3].imshow(_norm01(e_pos), cmap="Blues", vmin=0, vmax=1)
    _gt_contour(axes_pont[3], gt)
    axes_pont[3].set_title(r"$\langle\mathrm{proto}_+,\varphi_+\rangle$  (RFF energy)", fontsize=9)
    plt.colorbar(im_ep, ax=axes_pont[3], fraction=0.046, pad=0.04)

    im_en = axes_pont[4].imshow(_norm01(e_neg), cmap="Reds", vmin=0, vmax=1)
    _gt_contour(axes_pont[4], gt)
    axes_pont[4].set_title(r"$\langle\mathrm{proto}_-,\varphi_-\rangle$  (SRF energy)", fontsize=9)
    plt.colorbar(im_en, ax=axes_pont[4], fraction=0.046, pad=0.04)

    for ax in axes_pont:
        _axoff(ax)

    plt.suptitle(
        f"Episode {episode_idx:03d} — Pontryagin MC Dropout & embedding energy"
        "  (green contour = GT)",
        fontsize=10,
    )
    plt.tight_layout()
    fig_pont.savefig(out_dir / f"ep_{episode_idx:03d}_mc_pontryagin.pdf",
                     dpi=150, format="pdf", bbox_inches="tight")
    plt.close(fig_pont)

    # ── Per-episode scalar metrics ─────────────────────────────────────────────
    fg_mask = gt > 0.5
    bg_mask = ~fg_mask

    def _safe_mean(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    return {
        # ── Prediction quality
        "episode":                  episode_idx,
        "iou_euclidean":            iou_e,
        "iou_pontryagin":           iou_p,
        # ── MC Dropout uncertainty
        "mc_var_euc_mean":          float(unc_e.mean()),
        "mc_var_pont_mean":         float(unc_p.mean()),
        "mc_var_euc_fg":            _safe_mean(unc_e, fg_mask),
        "mc_var_pont_fg":           _safe_mean(unc_p, fg_mask),
        "mc_var_euc_bg":            _safe_mean(unc_e, bg_mask),
        "mc_var_pont_bg":           _safe_mean(unc_p, bg_mask),
        # ── Embedding energy (Pontryagin)
        "energy_pos_mean":          float(e_pos.mean()),
        "energy_neg_mean":          float(e_neg.mean()),
        "energy_pos_fg":            _safe_mean(e_pos, fg_mask),
        "energy_neg_fg":            _safe_mean(e_neg, fg_mask),
        # ── ScoreCAM quality — Euclidean
        "cam_euc_mean_activation":  cam_m_e.get("mean_activation", float("nan")),
        "cam_euc_coverage_top10":   cam_m_e.get("coverage_top10pct", float("nan")),
        "cam_euc_gt_iou":           cam_m_e.get("gt_iou",       float("nan")),
        "cam_euc_gt_recall":        cam_m_e.get("gt_recall",    float("nan")),
        "cam_euc_gt_precision":     cam_m_e.get("gt_precision", float("nan")),
        "cam_euc_gt_pearson":       cam_m_e.get("gt_pearson",   float("nan")),
        "cam_euc_otsu_thresh":      cam_m_e.get("otsu_threshold", float("nan")),
        # ── ScoreCAM quality — Pontryagin
        "cam_pont_mean_activation": cam_m_p.get("mean_activation", float("nan")),
        "cam_pont_coverage_top10":  cam_m_p.get("coverage_top10pct", float("nan")),
        "cam_pont_gt_iou":          cam_m_p.get("gt_iou",       float("nan")),
        "cam_pont_gt_recall":       cam_m_p.get("gt_recall",    float("nan")),
        "cam_pont_gt_precision":    cam_m_p.get("gt_precision", float("nan")),
        "cam_pont_gt_pearson":      cam_m_p.get("gt_pearson",   float("nan")),
        "cam_pont_otsu_thresh":     cam_m_p.get("otsu_threshold", float("nan")),
        # ── Confidence-gain (% change on FG pixels when masked by CAM)
        "conf_gain_euc_mean":       cg_e["mean_pct_conf_gain"],
        "conf_gain_euc_std":        cg_e["std_pct_conf_gain"],
        "conf_gain_pont_mean":      cg_p["mean_pct_conf_gain"],
        "conf_gain_pont_std":       cg_p["std_pct_conf_gain"],
        # ── Cross-model correlation
        "pearson_unc_epos":         _pearson(unc_p, e_pos),
        "pearson_unc_eneg":         _pearson(unc_p, e_neg),
        "pearson_cam_e_cam_p":      _pearson(heat_e, heat_p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate figures
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(rows: list[dict], out_dir: Path, k_shot: int, n_mc: int):
    if not rows:
        return

    ep      = np.array([r["episode"]           for r in rows])
    iou_e   = np.array([r["iou_euclidean"]      for r in rows])
    iou_p   = np.array([r["iou_pontryagin"]     for r in rows])
    var_e   = np.array([r["mc_var_euc_mean"]    for r in rows])
    var_p   = np.array([r["mc_var_pont_mean"]   for r in rows])
    e_pos   = np.array([r["energy_pos_mean"]    for r in rows])
    e_neg   = np.array([r["energy_neg_mean"]    for r in rows])

    w = 0.35

    # ── IoU per episode ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(ep) * 0.4 + 2), 4))
    ax.bar(ep - w / 2, iou_e, width=w, label="Euclidean",  alpha=0.8, color="steelblue")
    ax.bar(ep + w / 2, iou_p, width=w, label="Pontryagin", alpha=0.8, color="darkorange")
    ax.axhline(iou_e.mean(), color="steelblue",  linestyle="--", linewidth=1,
               label=f"Euc mean={iou_e.mean():.3f}")
    ax.axhline(iou_p.mean(), color="darkorange", linestyle="--", linewidth=1,
               label=f"Pont mean={iou_p.mean():.3f}")
    ax.set_xlabel("Episode"); ax.set_ylabel("IoU")
    ax.set_title(f"IoU per test episode — {k_shot}-shot FSS")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "iou_comparison.pdf", dpi=150, format="pdf",
                bbox_inches="tight")
    plt.close(fig)

    # ── MC variance per episode ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(ep) * 0.4 + 2), 4))
    ax.bar(ep - w / 2, var_e, width=w, label="Euclidean",  alpha=0.8, color="steelblue")
    ax.bar(ep + w / 2, var_p, width=w, label="Pontryagin", alpha=0.8, color="darkorange")
    ax.set_xlabel("Episode"); ax.set_ylabel("Mean predictive variance")
    ax.set_title(f"MC Dropout uncertainty per episode — {k_shot}-shot FSS (N={n_mc})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "mc_variance_comparison.pdf", dpi=150, format="pdf",
                bbox_inches="tight")
    plt.close(fig)

    # ── Scatter: MC variance vs embedding energy ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc0 = axes[0].scatter(e_pos, var_p, c=iou_p, cmap="RdYlGn",
                          alpha=0.75, s=45, edgecolors="k", linewidths=0.4)
    axes[0].set_xlabel(r"Mean $\langle\mathrm{proto}_+,\varphi_+\rangle$ per episode (RFF)", fontsize=10)
    axes[0].set_ylabel(f"Mean MC Dropout variance (Pontryagin, N={n_mc})", fontsize=10)
    axes[0].set_title("Uncertainty vs Positive Energy")
    r0 = _pearson(e_pos, var_p)
    axes[0].annotate(f"r = {r0:.3f}", xy=(0.97, 0.05), xycoords="axes fraction",
                     ha="right", fontsize=10)
    plt.colorbar(sc0, ax=axes[0], label="Pontryagin IoU")

    sc1 = axes[1].scatter(e_neg, var_p, c=iou_p, cmap="RdYlGn",
                          alpha=0.75, s=45, edgecolors="k", linewidths=0.4)
    axes[1].set_xlabel(r"Mean $\langle\mathrm{proto}_-,\varphi_-\rangle$ per episode (SRF)", fontsize=10)
    axes[1].set_ylabel(f"Mean MC Dropout variance (Pontryagin, N={n_mc})", fontsize=10)
    axes[1].set_title("Uncertainty vs Negative Energy")
    r1 = _pearson(e_neg, var_p)
    axes[1].annotate(f"r = {r1:.3f}", xy=(0.97, 0.05), xycoords="axes fraction",
                     ha="right", fontsize=10)
    plt.colorbar(sc1, ax=axes[1], label="Pontryagin IoU")

    plt.suptitle(f"FSS {k_shot}-shot — MC Dropout uncertainty vs Pontryagin embedding energy",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_unc_vs_energy.pdf", dpi=150, format="pdf",
                bbox_inches="tight")
    plt.close(fig)

    # ── CAM quality metrics comparison (4 metrics, box plots) ────────────────
    cam_metrics_pairs = [
        ("cam_euc_gt_iou",       "cam_pont_gt_iou",       "GT IoU"),
        ("cam_euc_gt_recall",    "cam_pont_gt_recall",    "GT Recall"),
        ("cam_euc_gt_precision", "cam_pont_gt_precision", "GT Precision"),
        ("cam_euc_gt_pearson",   "cam_pont_gt_pearson",   "GT Pearson r"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (key_e, key_p, label) in zip(axes, cam_metrics_pairs):
        vals_e = np.array([r[key_e] for r in rows], dtype=float)
        vals_p = np.array([r[key_p] for r in rows], dtype=float)
        valid  = ~(np.isnan(vals_e) | np.isnan(vals_p))
        if valid.any():
            bp = ax.boxplot(
                [vals_e[valid], vals_p[valid]],
                labels=["Euclidean", "Pontryagin"],
                patch_artist=True,
                medianprops=dict(color="red", linewidth=2),
            )
            bp["boxes"][0].set_facecolor("lightsteelblue")
            bp["boxes"][1].set_facecolor("peachpuff")
            m_e = vals_e[valid].mean(); m_p = vals_p[valid].mean()
            ax.set_title(f"{label}\nEuc={m_e:.3f}  Pont={m_p:.3f}", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
    plt.suptitle(f"ScoreCAM quality vs GT — {k_shot}-shot FSS", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "cam_metrics_comparison.pdf", dpi=150, format="pdf",
                bbox_inches="tight")
    plt.close(fig)

    # ── Confidence-gain comparison (box plot + bar) ───────────────────────────
    cg_e_arr = np.array([r["conf_gain_euc_mean"]  for r in rows], dtype=float)
    cg_p_arr = np.array([r["conf_gain_pont_mean"] for r in rows], dtype=float)
    valid_cg  = ~(np.isnan(cg_e_arr) | np.isnan(cg_p_arr))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Box plot
    if valid_cg.any():
        bp = axes[0].boxplot(
            [cg_e_arr[valid_cg], cg_p_arr[valid_cg]],
            labels=["Euclidean", "Pontryagin"],
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("lightsteelblue")
        bp["boxes"][1].set_facecolor("peachpuff")
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean % confidence change on FG pixels", fontsize=9)
    axes[0].set_title(
        f"Confidence gain — CAM masking\n"
        f"Euc={cg_e_arr[valid_cg].mean():.1f}%  "
        f"Pont={cg_p_arr[valid_cg].mean():.1f}%", fontsize=9
    )

    # Per-episode bar chart
    ep_arr = np.array([r["episode"] for r in rows])
    axes[1].bar(ep_arr - w / 2, cg_e_arr, width=w, label="Euclidean",
                alpha=0.8, color="steelblue")
    axes[1].bar(ep_arr + w / 2, cg_p_arr, width=w, label="Pontryagin",
                alpha=0.8, color="darkorange")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Mean % confidence gain")
    axes[1].set_title("Per-episode confidence gain", fontsize=9)
    axes[1].legend(fontsize=8)

    plt.suptitle(
        f"CAM confidence-gain analysis — {k_shot}-shot FSS\n"
        "(positive = CAM highlights the features the model uses for FG prediction)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "conf_gain_comparison.pdf", dpi=150, format="pdf",
                bbox_inches="tight")
    plt.close(fig)

    # ── Print summary ─────────────────────────────────────────────────────────
    def _nanmean(a): return float(np.nanmean(a)) if not np.all(np.isnan(a)) else float("nan")
    def _nanstd(a):  return float(np.nanstd(a))  if not np.all(np.isnan(a)) else float("nan")

    cam_eiou = np.array([r["cam_euc_gt_iou"]  for r in rows], dtype=float)
    cam_piou = np.array([r["cam_pont_gt_iou"] for r in rows], dtype=float)
    cam_epr  = np.array([r["cam_euc_gt_pearson"]  for r in rows], dtype=float)
    cam_ppr  = np.array([r["cam_pont_gt_pearson"] for r in rows], dtype=float)

    print(f"\n{'─' * 60}")
    print(f"Summary over {len(rows)} episodes ({k_shot}-shot FSS):")
    print(f"  Prediction IoU   — Euc: {iou_e.mean():.4f}±{iou_e.std():.4f}  "
          f"Pont: {iou_p.mean():.4f}±{iou_p.std():.4f}")
    print(f"  MC variance      — Euc: {var_e.mean():.4f}  Pont: {var_p.mean():.4f}")
    print(f"  ScoreCAM GT-IoU  — Euc: {_nanmean(cam_eiou):.4f}  Pont: {_nanmean(cam_piou):.4f}")
    print(f"  ScoreCAM Pearson — Euc: {_nanmean(cam_epr):.4f}   Pont: {_nanmean(cam_ppr):.4f}")
    print(f"  Conf gain (mean) — Euc: {_nanmean(cg_e_arr):.2f}%  "
          f"Pont: {_nanmean(cg_p_arr):.2f}%")
    print(f"  Pearson unc/E+   : {r0:.4f}")
    print(f"  Pearson unc/E-   : {r1:.4f}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis routine
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    k_shot:        int,
    data_root:     Path,
    results_dir:   Path,
    device:        torch.device,
    n_viz:         int   = 20,
    n_mc:          int   = N_PASSES_MC,
    trainable_rff: bool  = False,
):
    print(f"\nLoading models (k_shot={k_shot}, trainable_rff={trainable_rff}) …")

    try:
        model_euc  = _load_model("euclidean",   k_shot, results_dir, device, trainable_rff)
    except FileNotFoundError as exc:
        print(f"  Euclidean: {exc} — skipping.")
        return
    try:
        model_pont = _load_model("pontryagin",  k_shot, results_dir, device, trainable_rff)
    except FileNotFoundError as exc:
        print(f"  Pontryagin: {exc} — skipping.")
        return

    suffix  = "_trainable" if trainable_rff else ""
    out_dir = results_dir / f"interpretability_{k_shot}shot{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ds = EpisodicUAVDataset(
        data_root, "test",
        k_shot=k_shot, n_episodes=N_EPISODES_TEST,
        img_size=IMG_SIZE, target_cls=TARGET_CLS, augment=False, seed=42,
    )
    loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    n_analyse = min(n_viz, len(test_ds))
    print(f"Analysing {n_analyse}/{len(test_ds)} test episodes …")

    rows = []
    for ep_idx, batch in enumerate(loader):
        if ep_idx >= n_analyse:
            break
        print(f"  Episode {ep_idx + 1:3d}/{n_analyse}", end="\r")
        metrics = visualise_episode(
            ep_idx, batch, model_euc, model_pont, device, out_dir, n_mc=n_mc
        )
        rows.append(metrics)

    print()
    _write_csv(rows, out_dir / "metrics.csv")
    plot_aggregate(rows, out_dir, k_shot, n_mc)
    print(f"Results saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    ap = argparse.ArgumentParser(
        description="Interpretability for FSS models (Euclidean vs Pontryagin)"
    )
    ap.add_argument("--k-shot",        type=int,  default=K_SHOT)
    ap.add_argument("--n-viz",         type=int,  default=20,
                    help="Number of test episodes to visualise (default 20)")
    ap.add_argument("--n-mc",          type=int,  default=N_PASSES_MC,
                    help=f"MC Dropout passes per episode (default {N_PASSES_MC})")
    ap.add_argument("--device",        default=DEVICE)
    ap.add_argument("--data-root",     type=str,  default=None,
                    help="Override DATASET_ROOT from config")
    ap.add_argument("--results-dir",   type=str,  default=None,
                    help="Override RESULTS_DIR from config")
    ap.add_argument("--trainable-rff", action="store_true", default=False,
                    help="Load checkpoints from the _trainable run")
    return ap.parse_args()


def main():
    args        = _parse()
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root   = Path(args.data_root)   if args.data_root   else DATASET_ROOT
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    analyse(
        k_shot=args.k_shot,
        data_root=data_root,
        results_dir=results_dir,
        device=device,
        n_viz=args.n_viz,
        n_mc=args.n_mc,
        trainable_rff=args.trainable_rff,
    )


if __name__ == "__main__":
    main()
