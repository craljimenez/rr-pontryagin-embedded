"""Pontryagin Prototypical loss for few-shot learning.

The J-inner product ⟨φ(z), c_y⟩_J directly approximates K_Pontryagin(z, c_y),
so this loss is the natural episodic loss when the embedding is a Pontryagin
Random Feature map.

Three penalty terms beyond CE
──────────────────────────────
  L_cone-proto  Prototypes must not lie on the null cone.  A lightlike
                prototype has no well-defined Pontryagin inner product
                interpretation.
                L_cone = (1/N_way) Σ_y relu(ε − |q_J(c_y)|)

  L_orth        Different class prototypes should be J-orthogonal.
                Normalised to remove scale dependency:
                L_orth = mean_{y≠y'} |⟨c_y, c_{y'}⟩_J|² /
                                      max(|q_J(c_y)| · |q_J(c_{y'})|, δ)

  Topo          Optional IndefiniteTopographicPenalty applied to the
                query embeddings (light-cone + causal-consistency +
                signature-balance).

Total = CE + λ_cone·L_cone + λ_orth·L_orth + λ_topo·Topo(z_query, y_query)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prfe.losses.topographic import IndefiniteTopographicPenalty


class PontryaginPrototypical(nn.Module):
    """Episodic few-shot loss in Pontryagin feature space.

    Args:
        J:              metric vector (p+q,) with values ±1
        lambda_cone:    weight for prototype cone penalty  (default 0.1)
        lambda_orth:    weight for J-orthogonality penalty (default 0.1)
        lambda_topo:    weight for topographic penalty     (default 0.05)
        cone_epsilon:   null-cone margin ε                 (default 0.1)
        orth_delta:     safe denominator for normalised
                        orthogonality                      (default 1e-8)
        topo_kwargs:    kwargs forwarded to IndefiniteTopographicPenalty;
                        pass None to disable.
    """

    def __init__(
        self,
        J: torch.Tensor,
        lambda_cone: float = 0.1,
        lambda_orth: float = 0.1,
        lambda_topo: float = 0.05,
        cone_epsilon: float = 0.1,
        orth_delta: float = 1e-8,
        topo_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("J", J.float())
        self.lambda_cone = lambda_cone
        self.lambda_orth = lambda_orth
        self.lambda_topo = lambda_topo
        self.cone_epsilon = cone_epsilon
        self.orth_delta = orth_delta

        if lambda_topo > 0:
            kwargs = topo_kwargs or {}
            self.topo = IndefiniteTopographicPenalty(J, **kwargs)
        else:
            self.topo = None

    # ------------------------------------------------------------------
    # Core utility
    # ------------------------------------------------------------------

    def j_inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """⟨u, v⟩_J = Σ_d J_d u_d v_d  (broadcast-safe).

        Works for any leading dimensions as long as the last dim is p+q.
        """
        return (u * v * self.J).sum(dim=-1)

    def q_J(self, z: torch.Tensor) -> torch.Tensor:
        """q_J(z) = ⟨z, z⟩_J = Σ_d J_d z_d²."""
        return (z.pow(2) * self.J).sum(dim=-1)

    # ------------------------------------------------------------------
    # Prototype construction
    # ------------------------------------------------------------------

    def compute_prototypes(
        self,
        support_z: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-class mean prototype in Pontryagin space.

        Args:
            support_z:      (N_way * K_shot, p+q)
            support_labels: (N_way * K_shot,) integer labels

        Returns:
            prototypes: (N_way, p+q) — one per unique class
            classes:    (N_way,)     — sorted unique class indices
        """
        classes = support_labels.unique(sorted=True)
        prototypes = torch.stack(
            [support_z[support_labels == c].mean(0) for c in classes]
        )
        return prototypes, classes

    # ------------------------------------------------------------------
    # Penalty terms
    # ------------------------------------------------------------------

    def cone_penalty(self, vecs: torch.Tensor) -> torch.Tensor:
        """relu(ε − |q_J(v)|) mean — generic for prototypes or weight vectors.

        Args:
            vecs: (K, p+q)
        Returns:
            scalar
        """
        return F.relu(self.cone_epsilon - self.q_J(vecs).abs()).mean()

    def j_orthogonality(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Normalised off-diagonal squared J-inner product.

        L_orth = mean_{y≠y'} |⟨c_y, c_{y'}⟩_J|² /
                              max(|q_J(c_y)| · |q_J(c_{y'})|, δ)

        Args:
            prototypes: (N_way, p+q)
        Returns:
            scalar
        """
        # Gram matrix G[y, y'] = ⟨c_y, c_{y'}⟩_J
        G = prototypes @ (prototypes * self.J.unsqueeze(0)).T      # (N_way, N_way)
        q = self.q_J(prototypes).abs()                             # (N_way,)
        denom = (q.unsqueeze(0) * q.unsqueeze(1)).clamp(min=self.orth_delta)
        G_norm = G.pow(2) / denom                                  # (N_way, N_way)

        # Mask diagonal
        mask = ~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
        return G_norm[mask].mean()

    # ------------------------------------------------------------------
    # Similarity and classification
    # ------------------------------------------------------------------

    def similarity_matrix(
        self,
        query_z: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """S[i, y] = ⟨φ(z_i), c_y⟩_J ≈ K_Pontryagin(z_i, c_y).

        Args:
            query_z:    (N_q, p+q)
            prototypes: (N_way, p+q)
        Returns:
            (N_q, N_way)
        """
        return query_z @ (prototypes * self.J.unsqueeze(0)).T

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        support_z: torch.Tensor,
        support_labels: torch.Tensor,
        query_z: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Episodic training loss.

        Args:
            support_z:      (N_way * K_shot, p+q)
            support_labels: (N_way * K_shot,) — original integer labels
            query_z:        (N_q, p+q)
            query_labels:   (N_q,)             — original integer labels

        Returns:
            scalar loss
        """
        prototypes, classes = self.compute_prototypes(support_z, support_labels)

        # Map original labels → local episode indices 0 … N_way-1
        # searchsorted requires sorted classes (guaranteed by unique(sorted=True))
        local_labels = torch.searchsorted(classes.contiguous(), query_labels.contiguous())

        S = self.similarity_matrix(query_z, prototypes)            # (N_q, N_way)
        loss = F.cross_entropy(S, local_labels)

        if self.lambda_cone > 0:
            loss = loss + self.lambda_cone * self.cone_penalty(prototypes)

        if self.lambda_orth > 0 and prototypes.shape[0] > 1:
            loss = loss + self.lambda_orth * self.j_orthogonality(prototypes)

        if self.topo is not None and self.lambda_topo > 0:
            loss = loss + self.lambda_topo * self.topo(query_z, local_labels)

        return loss

    @torch.no_grad()
    def predict(
        self,
        support_z: torch.Tensor,
        support_labels: torch.Tensor,
        query_z: torch.Tensor,
    ) -> torch.Tensor:
        """Nearest-prototype prediction.

        Returns predicted original class labels (N_q,).
        """
        prototypes, classes = self.compute_prototypes(support_z, support_labels)
        local_pred = self.similarity_matrix(query_z, prototypes).argmax(dim=-1)
        return classes[local_pred]
