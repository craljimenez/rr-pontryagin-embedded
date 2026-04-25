"""Random Fourier Features — approximates a shift-invariant PD kernel K_+.

Reference: Rahimi & Recht, "Random Features for Large-Scale Kernel Machines",
NeurIPS 2007.

Approximation:
    K_+(x, y) ≈ φ_+(x)ᵀ φ_+(y)
    φ_+(x) = (1/√p) [cos(ω₁ᵀx+b₁), sin(ω₁ᵀx+b₁), …, cos(ω_{p/2}ᵀx+b_{p/2}), sin(…)]
    ωᵢ ~ N(0, σ⁻²I),  bᵢ ~ Uniform[0, 2π]

Re-parameterisation used here
─────────────────────────────
Instead of sampling `W ~ N(0, σ⁻²I)` directly, we store a *unit-variance*
sample `W ~ N(0, I)` and expose the bandwidth as `log_sigma`, a scalar
learnable parameter.  The effective frequency is then  W / σ  with
σ = exp(log_sigma).  This lets the model tune σ by gradient (removing it
from the HPO space) while preserving the Bochner identity at initialisation
(`log_sigma = log σ₀` ⇒ exact same distribution as the original sampler).

Output dimension: 2 * n_components  (cos and sin paired).
"""

import math
import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    """
    Args:
        in_features:   input dimension d (number of channels C)
        n_components:  p/2 random frequencies; output dim = 2 * n_components
        sigma:         initial RBF lengthscale (tuned further via gradient)
        trainable:     if True, the random frequencies W and phases b become
                       learnable parameters (breaks the kernel interpretation
                       of W).  σ is always learnable — that is the whole point
                       of this re-parameterisation.
    """

    def __init__(
        self,
        in_features: int,
        n_components: int,
        sigma: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_components = n_components

        # Unit-variance frequencies; scale is carried by log_sigma.
        W = torch.randn(n_components, in_features)
        b = torch.rand(n_components) * 2 * math.pi

        if trainable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W)
            self.register_buffer("b", b)

        # σ > 0 enforced by the exp parameterisation
        self.log_sigma = nn.Parameter(torch.tensor(math.log(float(sigma))))

    @property
    def out_features(self) -> int:
        return 2 * self.n_components

    @property
    def sigma(self) -> torch.Tensor:
        """Current bandwidth σ = exp(log_sigma)."""
        return torch.exp(self.log_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d)
        Returns:
            φ_+(x): (N, 2*n_components)
        """
        # Effective frequencies W_eff = W / σ follow N(0, σ⁻²I) at init.
        z = (x @ self.W.T) / self.sigma + self.b
        scale = math.sqrt(self.n_components)
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1) / scale
