"""Pontryagin Margin Classification Loss.

Designed for supervised classification with global embeddings
(i.e. after GlobalAveragePooling — one vector per image, not spatial).

Replaces the topographic penalty of PontryaginMLR with two terms that
are more natural for global classification:

  L_margin_proto
      Batch-prototype margin: for each mini-batch, compute the mean
      embedding μ_y for each class present and penalise whenever two
      prototype J-pseudo-distances are below a margin m:

          L_margin = (1 / N_pairs) Σ_{y≠y'} relu(m − d_J(μ_y, μ_y'))

      where d_J(a, b) = sqrt(|q_J(a − b)|) is the J-pseudo-distance and
      N_pairs = N_way*(N_way−1)/2.  This directly encourages classes to
      occupy distinct causal sectors of Π_κ.

  L_orth_W
      J-orthogonality on the classifier weight matrix W:

          L_orth_W = mean_{y≠y'} |⟨W_y, W_{y'}⟩_J|² /
                                  max(|q_J(W_y)| · |q_J(W_{y'})|, δ)

      Prevents the logit directions W_y from collapsing to the same
      J-direction; analogous to the prototype orthogonality in
      PontryaginPrototypical.

  L_balance_W
      Same as in PontryaginMLR: (rms(W_+) − rms(W_−))².  Ensures both
      subspaces are actively used by the classifier.

Total loss:
    L = CE(⟨W_y, z⟩_J + b_y, y)
      + λ_balance · L_balance_W
      + λ_margin  · L_margin_proto
      + λ_orth_W  · L_orth_W
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PontryaginMarginCLS(nn.Module):
    """J-hyperplane classifier with batch-prototype margin regularisation.

    Architecture is identical to PontryaginMLR (same W, b, logits formula).
    The regularisation replaces the spatial topographic penalty with a
    margin between batch prototypes and a J-orthogonality penalty on W.

    Args:
        n_classes:      C — number of output classes
        in_features:    p+q — Pontryagin feature dimension
        J:              metric vector (p+q,) with values ±1
        class_weights:  optional (C,) tensor for F.cross_entropy `weight`
        lambda_balance: weight for L_balance_W              (default 0.1)
        lambda_margin:  weight for L_margin_proto           (default 0.5)
        lambda_orth_W:  weight for L_orth_W                 (default 0.1)
        margin:         J-pseudo-distance margin m          (default 1.0)
        orth_delta:     safe denominator in L_orth_W        (default 1e-8)
    """

    def __init__(
        self,
        n_classes: int,
        in_features: int,
        J: torch.Tensor,
        class_weights: torch.Tensor | None = None,
        lambda_balance: float = 0.1,
        lambda_margin: float = 0.5,
        lambda_orth_W: float = 0.1,
        margin: float = 1.0,
        orth_delta: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_classes      = n_classes
        self.lambda_balance = lambda_balance
        self.lambda_margin  = lambda_margin
        self.lambda_orth_W  = lambda_orth_W
        self.margin         = margin
        self.orth_delta     = orth_delta

        self.W = nn.Parameter(torch.randn(n_classes, in_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_classes))
        self.register_buffer("J", J.float())
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    # ------------------------------------------------------------------
    # Core geometry
    # ------------------------------------------------------------------

    def q_J(self, z: torch.Tensor) -> torch.Tensor:
        """J-quadratic form q_J(z) = Σ_d J_d z_d².  Args: (..., p+q) → (...)"""
        return (z.pow(2) * self.J).sum(dim=-1)

    def j_dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """J-pseudo-distance d_J(a,b) = sqrt(|q_J(a−b)|).

        Not a true metric (triangle ineq. may fail), but the natural
        geometric quantity in Pontryagin space.

        Args:
            a, b: (..., p+q)
        Returns:
            (...)
        """
        return self.q_J(a - b).abs().clamp(min=1e-12).sqrt()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        """ζ_y(z) = ⟨W_y, z⟩_J + b_y.  Args: (N, p+q) → (N, C)"""
        return z @ (self.W * self.J.unsqueeze(0)).T + self.b

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Argmax. Args: (N, p+q) → (N,)"""
        return self.logits(z).argmax(dim=-1)

    # ------------------------------------------------------------------
    # Penalty terms
    # ------------------------------------------------------------------

    def balance_penalty_W(self) -> torch.Tensor:
        """L_balance_W = (rms(W_+) − rms(W_−))²."""
        pos_mask = self.J > 0
        neg_mask = self.J < 0
        p_dim    = pos_mask.sum().float()
        q_dim    = neg_mask.sum().float()
        rms_pos  = (self.W[:, pos_mask].pow(2).sum() / p_dim).sqrt()
        rms_neg  = (self.W[:, neg_mask].pow(2).sum() / q_dim).sqrt()
        return (rms_pos - rms_neg).pow(2)

    def margin_proto_penalty(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """L_margin = mean_{y≠y'} relu(m − d_J(μ_y, μ_y')).

        Batch prototypes μ_y = mean(z[labels==y]).  Pairs with only one
        class represented in the batch contribute zero (handled by the
        loop guard).  If fewer than two classes are present, returns 0.

        Args:
            z:      (N, p+q)
            labels: (N,)
        Returns:
            scalar
        """
        classes = labels.unique()
        if classes.numel() < 2:
            return z.new_zeros(())

        protos = torch.stack([z[labels == c].mean(0) for c in classes])  # (K, p+q)

        total, count = z.new_zeros(()), 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                d      = self.j_dist(protos[i], protos[j])
                total  = total + F.relu(self.margin - d)
                count += 1

        return total / count

    def orth_W_penalty(self) -> torch.Tensor:
        """L_orth_W = mean_{y≠y'} |⟨W_y,W_{y'}⟩_J|² / max(|q_J(W_y)||q_J(W_{y'})|, δ).

        Encourages classifier weight vectors to be J-orthogonal, preventing
        logit directions from collapsing.
        """
        if self.n_classes < 2:
            return self.W.new_zeros(())

        # Gram matrix G[y,y'] = ⟨W_y, W_{y'}⟩_J
        G   = self.W @ (self.W * self.J.unsqueeze(0)).T          # (C, C)
        q   = self.q_J(self.W).abs()                             # (C,)
        den = (q.unsqueeze(0) * q.unsqueeze(1)).clamp(min=self.orth_delta)
        G_n = G.pow(2) / den                                     # (C, C)

        mask = ~torch.eye(self.n_classes, dtype=torch.bool, device=G.device)
        return G_n[mask].mean()

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Full training loss.

        Args:
            z:      (N, p+q)
            labels: (N,) integer class indices
        Returns:
            scalar loss
        """
        loss = F.cross_entropy(self.logits(z), labels, weight=self.class_weights)

        if self.lambda_balance > 0:
            loss = loss + self.lambda_balance * self.balance_penalty_W()

        if self.lambda_margin > 0:
            loss = loss + self.lambda_margin * self.margin_proto_penalty(z, labels)

        if self.lambda_orth_W > 0:
            loss = loss + self.lambda_orth_W * self.orth_W_penalty()

        return loss
