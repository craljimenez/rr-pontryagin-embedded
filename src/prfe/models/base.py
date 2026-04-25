"""Concrete model classes built on PontryaginEmbedding.

PontryaginSegNet
    Encoder-decoder backbone  →  PontryaginEmbedding  →  PontryaginMLR head.
    Designed for semantic segmentation; operates at pixel level.

PontryaginFewShotNet
    Backbone  →  PontryaginEmbedding  →  global-average-pool.
    Produces per-image Pontryagin embeddings for episodic training.
    The PontryaginPrototypical loss is applied externally by the trainer.
"""

import torch
import torch.nn as nn

from prfe.layers import PontryaginEmbedding
from prfe.losses import PontryaginMLR, IndefiniteTopographicPenalty


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation model
# ──────────────────────────────────────────────────────────────────────────────

class PontryaginSegNet(nn.Module):
    """Semantic segmentation in Pontryagin space.

    Args:
        backbone:       nn.Module that maps (B, 3, H, W) → (B, C, H', W').
                        Any encoder-decoder (DeepLab, UNet, …) works; the
                        output spatial resolution H', W' is preserved.
        in_channels:    C — number of backbone output channels
        n_classes:      number of segmentation classes
        n_rff:          RFF half-dimension (RFF output = 2*n_rff)
        n_srf:          SRF output dimension
        kappa:          Pontryagin signature κ (= polynomial degree for SRF)
        sigma:          RBF bandwidth for K_+
        lambda_cone_W:  weight for MLR weight-vector cone penalty
        lambda_topo:    weight for topographic regularisation on embeddings
        lambda_balance: weight for subspace balance penalty
        topo_kwargs:    extra kwargs for IndefiniteTopographicPenalty
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int,
        n_classes: int,
        n_rff: int,
        n_srf: int,
        kappa: int,
        sigma: float = 1.0,
        lambda_cone_W: float = 0.1,
        lambda_topo: float = 0.05,
        lambda_balance: float = 0.01,
        topo_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed = PontryaginEmbedding(
            in_channels, n_rff, n_srf, kappa, sigma=sigma
        )
        self.head = PontryaginMLR(
            n_classes,
            self.embed.out_channels,
            self.embed.J,
            lambda_cone_W=lambda_cone_W,
            lambda_topo=lambda_topo,
            lambda_balance=lambda_balance,
            topo_kwargs=topo_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: returns logit map (B, n_classes, H', W')."""
        features = self.backbone(x)                     # (B, C, H', W')
        z = self.embed(features)                        # (B, p+q, H', W')
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        logits_flat = self.head.logits(z_flat)          # (B*H*W, n_classes)
        return logits_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Training: returns scalar loss (CE + penalties).

        Args:
            x:      (B, 3, H, W)
            labels: (B, H', W') integer class map at backbone output resolution
        """
        features = self.backbone(x)
        z = self.embed(features)
        return self.head.forward_spatial(z, labels)


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot model
# ──────────────────────────────────────────────────────────────────────────────

class PontryaginFewShotNet(nn.Module):
    """Episodic few-shot backbone producing per-image Pontryagin embeddings.

    The network outputs a single (p+q)-dimensional vector per image.
    Episodic training is driven externally by PontryaginPrototypical.

    Args:
        backbone:    nn.Module: (B, 3, H, W) → (B, C, H', W')
        in_channels: C
        n_rff:       RFF half-dimension
        n_srf:       SRF output dimension
        kappa:       signature κ
        sigma:       RBF bandwidth
        pool:        spatial pooling strategy — 'avg' (default) or 'max'
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int,
        n_rff: int,
        n_srf: int,
        kappa: int,
        sigma: float = 1.0,
        pool: str = "avg",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed = PontryaginEmbedding(
            in_channels, n_rff, n_srf, kappa, sigma=sigma
        )
        if pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pool type: {pool!r}")

    @property
    def out_features(self) -> int:
        return self.embed.out_channels

    @property
    def J(self) -> torch.Tensor:
        """Pontryagin metric vector — pass to PontryaginPrototypical."""
        return self.embed.J

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) or any input accepted by the backbone
        Returns:
            (B, p+q) — per-image Pontryagin embedding
        """
        features = self.backbone(x)             # (B, C, H', W')
        z_map = self.embed(features)            # (B, p+q, H', W')
        return self.pool(z_map).flatten(1)      # (B, p+q)
