"""Hyperbolic Multinomial Logistic Regression on the Poincaré ball.

Full reproduction of the MLR of Ganea et al. (NeurIPS 2018) as used by Atigh
et al. (CVPR 2022) for hyperbolic semantic segmentation. Serves as the
*hyperbolic* baseline head, to be compared against the flat Pontryagin head.

For each class y, two learnable parameters:

    a_y ∈ ℝ^d        — tangent (normal) vector of the gyroplane
    p_y ∈ 𝔹^d_c      — offset on the ball; stored as raw tangent `p_tangent`
                       and recovered by p_y = expmap₀(p_tangent)

Class logit (Ganea 2018, eq. 27):

    ζ_y(x) = (λ_{p_y} ‖a_y‖ / √c) ·
             sinh⁻¹( 2√c ⟨-p_y ⊕_c x, a_y⟩ /
                     ((1 − c‖-p_y ⊕_c x‖²) ‖a_y‖) )

with:

    λ_p  = 2 / (1 − c‖p‖²)                             (conformal factor)
    a ⊕_c b = [(1 + 2c⟨a,b⟩ + c‖b‖²) a + (1 − c‖a‖²) b]
              / [1 + 2c⟨a,b⟩ + c²‖a‖²‖b‖²]             (Möbius addition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """Möbius addition x ⊕_c y along the last dimension.

    Both tensors must be broadcastable. Returns the same last-dim shape.
    """
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / denom.clamp(min=eps)


def _expmap0(v: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at the origin: ℝ^d → 𝔹^d_c."""
    sqrt_c = c ** 0.5
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.tanh(sqrt_c * norm) / (sqrt_c * norm) * v


class HyperbolicMLR(nn.Module):
    """Full Ganea-style MLR on 𝔹^d_c.

    Args:
        n_classes:     C — number of output classes
        in_features:   d — feature dimension on the ball
        c:             curvature of the ball (default 1.0)
        class_weights: optional (C,) weight tensor for cross-entropy
        eps:           numerical floor
    """

    def __init__(
        self,
        n_classes: int,
        in_features: int,
        c: float = 1.0,
        class_weights: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.in_features = in_features
        self.c = float(c)
        self.eps = eps

        # Raw parameters — p_y lives on the ball (via expmap₀(p_tangent))
        self.p_tangent = nn.Parameter(torch.zeros(n_classes, in_features))
        self.a = nn.Parameter(torch.randn(n_classes, in_features) * 0.02)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic class logits.

        Args:
            x: (N, d) — points on 𝔹^d_c
        Returns:
            (N, C)
        """
        c = self.c
        sqrt_c = c ** 0.5

        # Class-wise ball offsets
        p = _expmap0(self.p_tangent, c, self.eps)         # (C, d)

        # Broadcast for pairwise Möbius addition: we need -p_y ⊕_c x_n
        # Shapes: x_exp (N, 1, d), p_exp (1, C, d)
        x_exp = x.unsqueeze(1)                            # (N, 1, d)
        mp_exp = -p.unsqueeze(0)                          # (1, C, d)

        # Broadcast to (N, C, d) explicitly so Möbius addition stays clean
        N = x.shape[0]
        C = self.n_classes
        u = _mobius_add(
            mp_exp.expand(N, C, -1),
            x_exp.expand(N, C, -1),
            c,
            self.eps,
        )                                                 # (N, C, d)

        # Per-class tangent norms and conformal factors
        a_norm = self.a.norm(dim=-1).clamp(min=self.eps)  # (C,)
        p_norm_sq = (p * p).sum(dim=-1)                   # (C,)
        lam_p = 2.0 / (1.0 - c * p_norm_sq).clamp(min=self.eps)   # (C,)

        # ⟨u, a⟩ and ‖u‖² per (n, c)
        inner = (u * self.a.unsqueeze(0)).sum(dim=-1)     # (N, C)
        u_norm_sq = (u * u).sum(dim=-1)                   # (N, C)

        # sinh⁻¹ argument — keep denominator strictly positive
        denom = (1.0 - c * u_norm_sq).clamp(min=self.eps) * a_norm.unsqueeze(0)
        arg = 2.0 * sqrt_c * inner / denom

        coeff = (lam_p * a_norm / sqrt_c).unsqueeze(0)    # (1, C)
        return coeff * torch.asinh(arg)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x).argmax(dim=-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss.

        Args:
            z:      (N, d) — features already on the ball
            labels: (N,)
        """
        return F.cross_entropy(self.logits(z), labels, weight=self.class_weights)

    # ------------------------------------------------------------------
    # Spatial (pixel-level) wrapper
    # ------------------------------------------------------------------

    def forward_spatial(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args: z (B, d, H, W), labels (B, H, W) → scalar loss."""
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        labels_flat = labels.reshape(B * H * W)
        return self.forward(z_flat, labels_flat)

    def predict_spatial(self, z: torch.Tensor) -> torch.Tensor:
        """Args: z (B, d, H, W) → (B, H, W)."""
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        return self.predict(z_flat).reshape(B, H, W)
