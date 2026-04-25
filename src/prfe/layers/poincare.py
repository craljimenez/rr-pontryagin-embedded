"""PoincareEmbedding: projects Euclidean features onto the Poincaré ball
𝔹^d_c = { x ∈ ℝ^d : c‖x‖² < 1 } via the exponential map at the origin.

Used as the lifting layer for the Hyperbolic Image Segmentation baseline
(Ganea et al. 2018; Atigh et al. CVPR 2022), so it can be compared head-to-head
against PontryaginEmbedding.

Exponential map at origin (Ganea 2018, eq. 2):

    expmap₀(v) = tanh(√c · ‖v‖) · v / (√c · ‖v‖)

The result is always strictly inside the ball of radius 1/√c.
"""

import torch
import torch.nn as nn


class PoincareEmbedding(nn.Module):
    """Lift a (B, C, H, W) Euclidean feature map to the Poincaré ball.

    Args:
        c:   curvature of the ball (radius = 1/√c). Default 1.0.
        eps: numerical floor for the norm to avoid division by zero.
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5) -> None:
        super().__init__()
        self.c = float(c)
        self.eps = eps

    @property
    def out_channels(self) -> int:
        # expmap preserves the feature dimension
        raise NotImplementedError("Use the input channel count; expmap keeps the dim.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel-wise expmap at the origin.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) with each C-vector inside the Poincaré ball.
        """
        sqrt_c = self.c ** 0.5
        # Channel-wise L2 norm: (B, 1, H, W)
        norm = x.norm(dim=1, keepdim=True).clamp(min=self.eps)
        scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
        y = x * scale
        # Project strictly inside the ball for numerical safety
        max_norm = (1.0 - self.eps) / sqrt_c
        y_norm = y.norm(dim=1, keepdim=True).clamp(min=self.eps)
        shrink = torch.clamp(max_norm / y_norm, max=1.0)
        return y * shrink
