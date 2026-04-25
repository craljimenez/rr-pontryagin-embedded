"""Pontryagin Multinomial Logistic Regression head for pixel-level classification.

Adapts the hyperbolic MLR of Ganea et al. (2018) and the pixel-level
reformulation of Atigh et al. (CVPR 2022) to a flat Pontryagin space with
signature κ.

Key difference from the hyperbolic case
─────────────────────────────────────────
  Hyperbolic (κ=1):   logit_y(z) = λ ‖w_y‖ sinh⁻¹( 2⟨−p_y ⊕_c z, w_y⟩ /
                                              ((1 − c‖−p_y ⊕_c z‖²) ‖w_y‖) )
  Pontryagin (κ≥1):   logit_y(z) = ⟨W_y, z⟩_J + b_y
                                  = Σ_d J_d W_{y,d} z_d  +  b_y

The Möbius addition is absent because our feature space is flat.  The
sinh⁻¹ linearises to the identity near the origin, recovering the Euclidean
limit.

Two additional penalty terms beyond CE
───────────────────────────────────────
  L_balance_W  Force the classifier to actually use both subspaces by
               penalising the squared difference between normalised weight
               norms (RMS per dim) in each subspace:
                 L_balance_W = (rms(W_+) − rms(W_−))²
               where rms(W_±) = √(‖W_±‖²_F / dim±).
               Unlike the old cone penalty, this has no saddle point at
               w_neg=0; gradient is always non-zero when one subspace
               dominates the other.

  Topo         Optional IndefiniteTopographicPenalty on the embeddings
               (light-cone + causal-consistency + signature-balance).

Total loss = CE(logits, y) + λ_balance · L_balance_W + λ_topo · Topo(z, y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prfe.losses.topographic import IndefiniteTopographicPenalty


class PontryaginMLR(nn.Module):
    """J-hyperplane multinomial logistic regression in Pontryagin space.

    Args:
        n_classes:      C — number of output classes
        in_features:    p+q — Pontryagin feature dimension
        J:              metric vector (p+q,) with values ±1
        class_weights:  optional (C,) tensor passed to F.cross_entropy as
                        `weight`; useful for imbalanced datasets.
        lambda_cone_W:  weight for the weight-vector cone penalty (default 0.1)
        lambda_topo:    weight for the topographic penalty        (default 0.05)
        lambda_balance: weight for the subspace balance penalty    (default 0.01)
        cone_epsilon:   null-cone margin ε                        (default 0.1)
        topo_kwargs:    kwargs forwarded to IndefiniteTopographicPenalty;
                        pass None to disable the topo penalty.
    """

    def __init__(
        self,
        n_classes: int,
        in_features: int,
        J: torch.Tensor,
        class_weights: torch.Tensor | None = None,
        lambda_cone_W: float = 0.0,   # kept for API compatibility; ignored
        lambda_topo: float = 0.05,
        lambda_balance: float = 0.1,
        cone_epsilon: float = 0.1,    # kept for API compatibility; ignored
        topo_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.lambda_topo = lambda_topo
        self.lambda_balance = lambda_balance

        self.W = nn.Parameter(torch.randn(n_classes, in_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_classes))
        self.register_buffer("J", J.float())
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        if lambda_topo > 0:
            kwargs = topo_kwargs or {}
            self.topo = IndefiniteTopographicPenalty(J, **kwargs)
        else:
            self.topo = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        """Compute class logits ζ_y(z) = ⟨W_y, z⟩_J + b_y.

        Args:
            z: (N, p+q)
        Returns:
            (N, C)
        """
        # (W * J[None,:]) has shape (C, p+q); z @ (...).T → (N, C)
        return z @ (self.W * self.J.unsqueeze(0)).T + self.b

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Argmax prediction.  Args: (N, p+q)  Returns: (N,)"""
        return self.logits(z).argmax(dim=-1)

    # ------------------------------------------------------------------
    # Penalty on classifier weights
    # ------------------------------------------------------------------

    def balance_penalty_W(self) -> torch.Tensor:
        """L_balance_W = (rms(W_+) − rms(W_−))²

        Forces both subspaces to contribute equally (per dimension) to the
        classifier.  Unlike the cone penalty this has no trivial minimum at
        w_neg=0; its gradient is always non-zero when one subspace dominates.
        """
        pos_mask = self.J > 0                                    # (p+q,)
        neg_mask = self.J < 0
        p_dim = pos_mask.sum().float()
        q_dim = neg_mask.sum().float()
        rms_pos = (self.W[:, pos_mask].pow(2).sum() / p_dim).sqrt()
        rms_neg = (self.W[:, neg_mask].pow(2).sum() / q_dim).sqrt()
        return (rms_pos - rms_neg).pow(2)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the full training loss.

        Args:
            z:      (N, p+q) — Pontryagin embeddings (flat, e.g. B*H*W pixels)
            labels: (N,)     — integer class indices

        Returns:
            scalar loss
        """
        loss = F.cross_entropy(self.logits(z), labels, weight=self.class_weights)

        if self.lambda_balance > 0:
            loss = loss + self.lambda_balance * self.balance_penalty_W()

        if self.topo is not None and self.lambda_topo > 0:
            loss = loss + self.lambda_topo * self.topo(z, labels)

        return loss

    # ------------------------------------------------------------------
    # Convenience: pixel-level interface accepting (B, p+q, H, W)
    # ------------------------------------------------------------------

    def forward_spatial(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Pixel-level wrapper.

        Args:
            z:      (B, p+q, H, W)
            labels: (B, H, W) integer class map

        Returns:
            scalar loss
        """
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        labels_flat = labels.reshape(B * H * W)
        return self.forward(z_flat, labels_flat)

    def predict_spatial(self, z: torch.Tensor) -> torch.Tensor:
        """Args: (B, p+q, H, W)  Returns: (B, H, W)"""
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
        return self.predict(z_flat).reshape(B, H, W)
