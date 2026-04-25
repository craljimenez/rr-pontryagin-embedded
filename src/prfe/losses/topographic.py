"""Topographic regularization for the indefinite Pontryagin feature space.

The J-quadratic form q_J(z) = ‖z_+‖² − ‖z_-‖² partitions R^{p+q} into:
    • Timelike  (causal cone interior):  q_J(z) < 0
    • Lightlike (null cone):             q_J(z) = 0   ← degenerate
    • Spacelike (causal cone exterior):  q_J(z) > 0

Three penalties control the geometry of the learned embedding:

    L_lc  — Light-cone penalty
            Pushes embeddings away from the null cone |q_J(z)| < ε.
            Prevents classifier collapse at degenerate directions.

    L_cc  — Causal consistency
            Within each class, q_J(z) should have low variance.
            Encourages each class to occupy a coherent causal sector.

    L_sb  — Signature balance
            Penalises the squared batch-mean of q_J(z).
            Prevents the embedding from collapsing entirely to one sector.

Total penalty: λ_lc·L_lc + λ_cc·L_cc + λ_sb·L_sb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndefiniteTopographicPenalty(nn.Module):
    """Standalone regulariser for the Pontryagin feature space.

    Args:
        J:          metric vector (p+q,) with values ±1; J[d] = +1 (positive
                    subspace) or −1 (negative subspace).  Usually taken
                    directly from PontryaginEmbedding.J.
        lambda_lc:  weight for the light-cone penalty        (default 1.0)
        lambda_cc:  weight for the causal-consistency term   (default 1.0)
        lambda_sb:  weight for the signature-balance term    (default 1.0)
        lc_epsilon: margin around the null cone              (default 0.1)
    """

    def __init__(
        self,
        J: torch.Tensor,
        lambda_lc: float = 1.0,
        lambda_cc: float = 1.0,
        lambda_sb: float = 1.0,
        lc_epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("J", J.float())
        self.lambda_lc = lambda_lc
        self.lambda_cc = lambda_cc
        self.lambda_sb = lambda_sb
        self.lc_epsilon = lc_epsilon

    # ------------------------------------------------------------------
    # Core quantity
    # ------------------------------------------------------------------

    def q_J(self, z: torch.Tensor) -> torch.Tensor:
        """J-quadratic form q_J(z) = ⟨z, z⟩_J = Σ_d J_d z_d².

        Args:
            z: (..., p+q)
        Returns:
            (...,)
        """
        return (z.pow(2) * self.J).sum(dim=-1)

    # ------------------------------------------------------------------
    # Individual penalty components
    # ------------------------------------------------------------------

    def light_cone_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """L_lc = mean( relu(ε − |q_J(z)|) ).

        Activates for embeddings inside the ε-neighbourhood of the null cone.
        """
        return F.relu(self.lc_epsilon - self.q_J(z).abs()).mean()

    def causal_consistency(
        self, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """L_cc = (1/C) Σ_c Var_{i: y_i=c}[ q_J(z_i) ].

        Vectorised via one-hot scatter — O(N·C) memory, no Python loops.

        Args:
            z:      (N, p+q)
            labels: (N,) integer class indices in [0, C)
        """
        qz = self.q_J(z)                                        # (N,)
        n_classes = int(labels.max().item()) + 1
        one_hot = F.one_hot(labels, n_classes).float()          # (N, C)
        count = one_hot.sum(0).clamp(min=1.0)                   # (C,)
        cls_mean = (one_hot * qz.unsqueeze(1)).sum(0) / count   # (C,)
        # Var = E[(q - mean)²]
        sq_dev = (qz.unsqueeze(1) - cls_mean.unsqueeze(0)).pow(2)   # (N, C)
        cls_var = (one_hot * sq_dev).sum(0) / count             # (C,)
        return cls_var.mean()

    def signature_balance(self, z: torch.Tensor) -> torch.Tensor:
        """L_sb = (mean_i q_J(z_i))².

        A nonzero batch mean indicates that most embeddings share the same
        causal type; this term encourages a balanced mix.
        """
        return self.q_J(z).mean().pow(2)

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:      (N, p+q) — Pontryagin feature vectors (flat batch)
            labels: (N,)     — integer class indices; required for L_cc
        Returns:
            scalar loss
        """
        loss = z.new_zeros(1).squeeze()

        if self.lambda_lc > 0:
            loss = loss + self.lambda_lc * self.light_cone_penalty(z)

        if self.lambda_cc > 0:
            if labels is None:
                raise ValueError("labels required for causal_consistency (lambda_cc > 0)")
            loss = loss + self.lambda_cc * self.causal_consistency(z, labels)

        if self.lambda_sb > 0:
            loss = loss + self.lambda_sb * self.signature_balance(z)

        return loss
