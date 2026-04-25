"""Spherical Random Features for Polynomial Kernels — approximates K_-.

Reference: Pennington, Yu & Kumar,
"Spherical Random Features for Polynomial Kernels", NeurIPS 2015.
https://proceedings.neurips.cc/paper_files/paper/2015/file/f7f580e11d00a75814d2ded41fe8e8fe-Paper.pdf

For a degree-κ polynomial kernel on the unit sphere:
    K_-(x, y) = (xᵀy)^κ

Random feature construction (one sample j):
    z_j(x) = ∏_{l=1}^{κ}  (ω_{jl}ᵀ x),   ω_{jl} ~ Uniform(S^{d-1})

Approximation:
    K_-(x,y) ≈ φ_-(x)ᵀ φ_-(y),   φ_-(x)_j = z_j(x) / √n_components

The degree κ equals the Pontryagin signature — it controls how many "negative"
directions the indefinite inner product will have once combined with K_+.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalRandomFeatures(nn.Module):
    """
    Args:
        in_features:  input dimension d (number of channels C)
        n_components: number of random features q (output dimension = Pontryagin signature κ)
        degree:       polynomial degree d_poly of K_-(x,y) = (xᵀy)^{d_poly}
                      independent of the signature / number of components
        trainable:    if True, random directions are learnable parameters
    """

    def __init__(
        self,
        in_features: int,
        n_components: int,
        degree: int,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_components = n_components
        self.degree = degree

        # W[j, l, :] is the l-th random direction for the j-th feature, on S^{d-1}
        W = torch.randn(n_components, degree, in_features)
        W = F.normalize(W, p=2, dim=-1)

        if trainable:
            self.W = nn.Parameter(W)
        else:
            self.register_buffer("W", W)

    @property
    def out_features(self) -> int:
        return self.n_components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d) — inputs should live on or near S^{d-1} (normalise upstream
               if needed; for image channels this is application-dependent)
        Returns:
            φ_-(x): (N, n_components)
        """
        # projections[n, j, l] = ω_{jl}ᵀ x_n
        projections = torch.einsum("nd,jld->njl", x, self.W)   # (N, q, d_poly)
        z = projections.prod(dim=-1)                             # (N, q) — ∏ over d_poly
        return z / math.sqrt(self.n_components)
