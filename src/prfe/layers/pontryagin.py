"""PontryaginEmbedding: maps a (B, C, H, W) tensor into a Pontryagin space
with signature κ via the decomposition K_Pontryagin = K_+ − K_-.

The output tensor has shape (B, p + q, H, W) where:
    • first p channels ↔ φ_+(x) from RFF  — positive definite component
    • last  q channels ↔ φ_-(x) from SRF  — negative definite component

The indefinite inner product (Pontryagin metric) is:
    ⟨u, v⟩_κ = Σ_{i<p} u_i v_i  −  Σ_{i≥p} u_i v_i

The metric matrix J = diag(+1,...,+1, −1,...,−1) is stored as a buffer so it
can be used by downstream layers / losses without recomputing.
"""

import torch
import torch.nn as nn

from prfe.layers.rff import RandomFourierFeatures
from prfe.layers.srf import SphericalRandomFeatures


class PontryaginEmbedding(nn.Module):
    """
    Args:
        in_channels:    C — number of input channels
        n_rff:          p/2 random Fourier frequencies; RFF output dim = 2*n_rff
        n_srf:          q — number of SRF components = Pontryagin signature κ
        kappa:          κ — Pontryagin signature (number of negative dims in J = n_srf)
        d_poly:         polynomial degree of K_-(x,y) = (xᵀy)^{d_poly}, independent of κ
        sigma:          RBF bandwidth for K_+
        trainable:      expose RFF/SRF weights as learnable parameters
        normalize_input: L2-normalise channel vectors before feeding SRF
                         (SRF is designed for inputs on the unit sphere)
    """

    def __init__(
        self,
        in_channels: int,
        n_rff: int,
        n_srf: int,
        kappa: int,
        d_poly: int = 2,
        sigma: float = 1.0,
        trainable: bool = False,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kappa = kappa
        self.d_poly = d_poly
        self.normalize_input = normalize_input

        self.rff = RandomFourierFeatures(in_channels, n_rff, sigma=sigma, trainable=trainable)
        self.srf = SphericalRandomFeatures(in_channels, n_srf, degree=d_poly, trainable=trainable)

        p = self.rff.out_features   # 2 * n_rff
        q = self.srf.out_features   # n_srf

        # Pontryagin metric: J = diag(+1 × p, -1 × q)
        J = torch.cat([torch.ones(p), -torch.ones(q)])
        self.register_buffer("J", J)    # (p+q,)

    @property
    def out_channels(self) -> int:
        return self.rff.out_features + self.srf.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, p+q, H, W)  — features in the Pontryagin space
        """
        B, C, H, W = x.shape

        # Flatten spatial dims: treat each pixel independently
        xf = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (N, C)

        phi_pos = self.rff(xf)                              # (N, p)

        if self.normalize_input:
            xf_norm = torch.nn.functional.normalize(xf, p=2, dim=-1)
        else:
            xf_norm = xf
        phi_neg = self.srf(xf_norm)                         # (N, q)

        z = torch.cat([phi_pos, phi_neg], dim=-1)           # (N, p+q)

        # Restore spatial structure
        return z.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)  # (B, p+q, H, W)

    def pontryagin_inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the indefinite Pontryagin inner product ⟨u, v⟩_J channel-wise.

        Args:
            u, v: (B, p+q, H, W)
        Returns:
            scalar map: (B, H, W)
        """
        return (u * v * self.J[None, :, None, None]).sum(dim=1)
