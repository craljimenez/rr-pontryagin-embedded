"""Few-shot segmentation models: Euclidean vs Hyperbolic vs Pontryagin.

All follow the PANet protocol (Wang et al. 2019):
  1. A shared UNetBackbone extracts (B, C, H, W) feature maps.
  2. Support features are masked-average-pooled → one prototype per episode.
  3. Each query pixel is scored against the prototype.

The only architectural difference is the embedding and similarity function:
  - EuclideanFewShotSeg : cosine similarity with learnable temperature α
  - HyperbolicFewShotSeg: PoincareEmbedding (expmap at the origin) +
                          negative Poincaré distance to the prototype,
                          with learnable temperature α and radius offset δ
  - PontryaginFewShotSeg: PontryaginEmbedding + J-inner product ⟨z, c⟩_J

All models expose the same forward / compute_loss / predict interface so the
training loop is identical across variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prfe.layers import PoincareEmbedding, PontryaginEmbedding
from prfe.models.unet import UNetBackbone


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────

def masked_avg_pool(
    features: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compute masked average pool over K support images → prototype.

    Args:
        features: (B, K, D, H, W)
        masks:    (B, K, H, W)   float 0/1
    Returns:
        (B, D)   — one prototype per episode in the batch
    """
    m     = masks.unsqueeze(2)                          # (B, K, 1, H, W)
    num   = (features * m).sum(dim=(1, 3, 4))           # (B, D)
    denom = m.sum(dim=(1, 3, 4)).clamp(min=1.0)         # (B, 1)
    return num / denom


# ─────────────────────────────────────────────────────────────────────────────
# Euclidean few-shot segmentation
# ─────────────────────────────────────────────────────────────────────────────

class EuclideanFewShotSeg(nn.Module):
    """PANet-style few-shot segmentation with cosine similarity.

    Prototype: masked average of backbone features over support images.
    Similarity: cosine(pixel_feat, prototype) × learnable temperature α.

    Args:
        in_channels: input image channels (default 3)
        base_ch:     first-block channel count for UNetBackbone
        depth:       number of encoder/decoder stages (3–5)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            out_channels=base_ch,
            base_ch=base_ch,
            depth=depth,
        )
        self.feat_dim = base_ch
        self.alpha    = nn.Parameter(torch.tensor(10.0))   # temperature

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) → (N, C, H, W)"""
        return self.backbone(x)

    def _prototypes(
        self,
        support_imgs: torch.Tensor,   # (B, K, 3, H, W)
        support_masks: torch.Tensor,  # (B, K, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute foreground and background prototypes.

        Returns:
            proto_fg: (B, C)
            proto_bg: (B, C)
        """
        B, K = support_imgs.shape[:2]
        H, W = support_imgs.shape[3:]

        sup_flat = support_imgs.reshape(B * K, *support_imgs.shape[2:])
        sup_feat = self._encode(sup_flat).reshape(B, K, self.feat_dim, H, W)

        proto_fg = masked_avg_pool(sup_feat, support_masks)           # (B, C)
        proto_bg = masked_avg_pool(sup_feat, 1.0 - support_masks)    # (B, C)
        return proto_fg, proto_bg

    def forward(
        self,
        support_imgs:  torch.Tensor,   # (B, K, 3, H, W)
        support_masks: torch.Tensor,   # (B, K, H, W)
        query_imgs:    torch.Tensor,   # (B, 3, H, W)
    ) -> torch.Tensor:
        """Return foreground logit map (B, H, W)."""
        proto_fg, _ = self._prototypes(support_imgs, support_masks)

        q_feat = self._encode(query_imgs)                            # (B, C, H, W)
        q_norm = F.normalize(q_feat, dim=1)
        p_norm = F.normalize(proto_fg, dim=1)[:, :, None, None]     # (B, C, 1, 1)

        return self.alpha * (q_norm * p_norm).sum(dim=1)            # (B, H, W)

    def compute_loss(
        self,
        support_imgs:  torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs:    torch.Tensor,
        query_masks:   torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Training step — returns scalar loss."""
        logits = self.forward(support_imgs, support_masks, query_imgs)
        return loss_fn(logits, query_masks)

    @torch.no_grad()
    def predict(
        self,
        support_imgs:  torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs:    torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Binary prediction map (B, H, W) — 1=foreground."""
        logits = self.forward(support_imgs, support_masks, query_imgs)
        return (logits > threshold).float()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperbolic few-shot segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float,
                eps: float = 1e-5) -> torch.Tensor:
    """Möbius addition x ⊕_c y along the last dimension.

    Mirrors prfe.losses.hyperbolic_mlr._mobius_add (duplicated here so the
    models module does not depend on the losses module).
    """
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / denom.clamp(min=eps)


def _poincare_dist(x: torch.Tensor, y: torch.Tensor, c: float,
                   eps: float = 1e-5) -> torch.Tensor:
    """Geodesic distance on the Poincaré ball 𝔹^d_c, last dim = d.

    d_c(x, y) = (2/√c) · artanh(√c · ‖(−x) ⊕_c y‖)
    """
    sqrt_c = c ** 0.5
    diff_norm = _mobius_add(-x, y, c, eps).norm(dim=-1)
    # artanh argument must stay strictly below 1 for finite distances
    arg = (sqrt_c * diff_norm).clamp(max=1.0 - eps)
    return (2.0 / sqrt_c) * torch.atanh(arg)


class HyperbolicFewShotSeg(nn.Module):
    """PANet-style few-shot segmentation on the Poincaré ball.

    Prototype: masked average of Poincaré-ball embeddings over support
    images. The ball is convex, so a convex combination of interior points
    remains interior — mirroring how the Euclidean and Pontryagin variants
    pool directly in their respective embedding spaces; a numerical shrink
    keeps the prototype strictly inside the ball.

    Similarity: α · (δ − d_c(pixel_embed, prototype)) — the standard
    distance-based classifier of hyperbolic prototype networks adapted to a
    dense binary logit map. α is a learnable temperature (initialised at
    10.0, mirroring EuclideanFewShotSeg) and δ a learnable decision radius,
    so the zero-crossing used by predict(threshold=0.0) is learned end to
    end rather than fixed.

    Args:
        in_channels: input image channels (default 3)
        base_ch:     UNetBackbone first-block channels
        depth:       UNetBackbone depth
        c:           curvature of the Poincaré ball
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        depth: int = 3,
        c: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            out_channels=base_ch,
            base_ch=base_ch,
            depth=depth,
        )
        self.embed    = PoincareEmbedding(c=c)
        self.c        = float(c)
        self.eps      = 1e-5
        self.feat_dim = base_ch
        self.alpha    = nn.Parameter(torch.tensor(10.0))   # temperature
        self.delta    = nn.Parameter(torch.tensor(1.0))    # decision radius

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) → (N, C, H, W) — backbone + expmap into the ball."""
        return self.embed(self.backbone(x))

    def _shrink_into_ball(self, p: torch.Tensor) -> torch.Tensor:
        """Numerically guarantee ‖p‖ < 1/√c (same guard as PoincareEmbedding)."""
        max_norm = (1.0 - self.eps) / (self.c ** 0.5)
        p_norm = p.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return p * torch.clamp(max_norm / p_norm, max=1.0)

    def _prototypes(
        self,
        support_imgs: torch.Tensor,   # (B, K, 3, H, W)
        support_masks: torch.Tensor,  # (B, K, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Foreground / background prototypes on the ball. Returns (B, C) each."""
        B, K = support_imgs.shape[:2]
        H, W = support_imgs.shape[3:]

        sup_flat = support_imgs.reshape(B * K, *support_imgs.shape[2:])
        sup_emb  = self._encode(sup_flat).reshape(B, K, self.feat_dim, H, W)

        proto_fg = self._shrink_into_ball(masked_avg_pool(sup_emb, support_masks))
        proto_bg = self._shrink_into_ball(masked_avg_pool(sup_emb, 1.0 - support_masks))
        return proto_fg, proto_bg

    def forward(
        self,
        support_imgs:  torch.Tensor,   # (B, K, 3, H, W)
        support_masks: torch.Tensor,   # (B, K, H, W)
        query_imgs:    torch.Tensor,   # (B, 3, H, W)
    ) -> torch.Tensor:
        """Return foreground logit map (B, H, W) = α·(δ − d_c(z, proto_fg))."""
        proto_fg, _ = self._prototypes(support_imgs, support_masks)

        q_emb = self._encode(query_imgs)                     # (B, C, H, W)
        q_pts = q_emb.permute(0, 2, 3, 1)                    # (B, H, W, C)
        p_pts = proto_fg[:, None, None, :]                   # (B, 1, 1, C)

        dist = _poincare_dist(q_pts, p_pts, self.c, self.eps)  # (B, H, W)
        return self.alpha * (self.delta - dist)

    def compute_loss(
        self,
        support_imgs:  torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs:    torch.Tensor,
        query_masks:   torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Training step — returns scalar loss."""
        logits = self.forward(support_imgs, support_masks, query_imgs)
        return loss_fn(logits, query_masks)

    @torch.no_grad()
    def predict(
        self,
        support_imgs:  torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs:    torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Binary prediction map (B, H, W) — 1=foreground."""
        logits = self.forward(support_imgs, support_masks, query_imgs)
        return (logits > threshold).float()


# ─────────────────────────────────────────────────────────────────────────────
# Pontryagin few-shot segmentation
# ─────────────────────────────────────────────────────────────────────────────

class PontryaginFewShotSeg(nn.Module):
    """PANet-style few-shot segmentation with the Pontryagin J-inner product.

    Prototype: masked average of Pontryagin embeddings over support images.
    Similarity: ⟨pixel_embed, prototype⟩_J ≈ K_Pontryagin(pixel, prototype).

    Args:
        in_channels:   input image channels
        base_ch:       UNetBackbone first-block channels
        depth:         UNetBackbone depth
        n_rff:         RFF half-dimension  (RFF output dim = 2*n_rff)
        n_srf:         SRF output dimension = kappa
        kappa:         Pontryagin signature κ (negative subspace width)
        sigma:         RBF bandwidth for K_+
        trainable_rff: expose RFF/SRF weights as learnable parameters
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        depth: int = 3,
        n_rff: int = 16,
        n_srf: int = 4,
        kappa: int = 2,
        sigma: float = 1.0,
        trainable_rff: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            out_channels=base_ch,
            base_ch=base_ch,
            depth=depth,
        )
        self.embed = PontryaginEmbedding(
            in_channels=base_ch,
            n_rff=n_rff,
            n_srf=n_srf,
            kappa=kappa,
            sigma=sigma,
            trainable=trainable_rff,
        )
        self.feat_dim = self.embed.out_channels

    @property
    def J(self) -> torch.Tensor:
        """Pontryagin metric vector — required by PontryaginFSSLoss."""
        return self.embed.J

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) → (N, D, H, W) — backbone + Pontryagin embedding."""
        return self.embed(self.backbone(x))

    def _prototypes(
        self,
        support_imgs: torch.Tensor,   # (B, K, 3, H, W)
        support_masks: torch.Tensor,  # (B, K, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute foreground and background Pontryagin prototypes.

        Returns:
            proto_fg: (B, D)
            proto_bg: (B, D)
        """
        B, K = support_imgs.shape[:2]
        H, W = support_imgs.shape[3:]

        sup_flat = support_imgs.reshape(B * K, *support_imgs.shape[2:])
        sup_emb  = self._encode(sup_flat).reshape(B, K, self.feat_dim, H, W)

        proto_fg = masked_avg_pool(sup_emb, support_masks)           # (B, D)
        proto_bg = masked_avg_pool(sup_emb, 1.0 - support_masks)    # (B, D)
        return proto_fg, proto_bg

    def forward(
        self,
        support_imgs:  torch.Tensor,   # (B, K, 3, H, W)
        support_masks: torch.Tensor,   # (B, K, H, W)
        query_imgs:    torch.Tensor,   # (B, 3, H, W)
    ) -> torch.Tensor:
        """Return foreground logit map (B, H, W) = ⟨pixel_embed, proto_fg⟩_J."""
        proto_fg, _ = self._prototypes(support_imgs, support_masks)

        q_emb = self._encode(query_imgs)                             # (B, D, H, W)
        J     = self.J[None, :, None, None]                          # (1, D, 1, 1)

        return (q_emb * proto_fg[:, :, None, None] * J).sum(dim=1)  # (B, H, W)

    def compute_loss(
        self,
        support_imgs:  torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs:    torch.Tensor,
        query_masks:   torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Training step — returns scalar loss, passes both prototypes to loss."""
        B, K = support_imgs.shape[:2]
        H, W = support_imgs.shape[3:]

        sup_flat = support_imgs.reshape(B * K, *support_imgs.shape[2:])
        sup_emb  = self._encode(sup_flat).reshape(B, K, self.feat_dim, H, W)

        proto_fg = masked_avg_pool(sup_emb, support_masks)
        proto_bg = masked_avg_pool(sup_emb, 1.0 - support_masks)

        q_emb  = self._encode(query_imgs)
        J      = self.J[None, :, None, None]
        logits = (q_emb * proto_fg[:, :, None, None] * J).sum(dim=1)

        return loss_fn(logits, query_masks, proto_fg=proto_fg, proto_bg=proto_bg)

    @torch.no_grad()
    def predict(
        self,
        support_imgs:  torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs:    torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Binary prediction map (B, H, W) — 1=foreground."""
        logits = self.forward(support_imgs, support_masks, query_imgs)
        return (logits > threshold).float()
