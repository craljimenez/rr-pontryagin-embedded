"""Few-shot segmentation losses.

EuclideanFSSLoss
    BCE + Dice on the foreground logit map.
    Handles the class imbalance (few sugarcane pixels) via pos_weight.

PontryaginFSSLoss
    BCE + Dice  + L_cone (prototypes off the null cone)
                + L_orth (fg/bg prototypes J-orthogonal).
    Both models receive the same base segmentation loss; the Pontryagin
    variant adds geometry-aware penalties using the fg/bg prototypes.

Both losses accept a foreground logit map (B, H, W) and a binary target
(B, H, W).  PontryaginFSSLoss additionally requires proto_fg and proto_bg.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Dice loss (binary, soft)
# ─────────────────────────────────────────────────────────────────────────────

def _dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss averaged over the batch.

    Args:
        prob:   (B, H, W) — sigmoid probabilities in [0,1]
        target: (B, H, W) — binary ground-truth in {0,1}
    """
    p = prob.reshape(prob.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    return (1.0 - (2 * inter + eps) / (union + eps)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Euclidean loss
# ─────────────────────────────────────────────────────────────────────────────

class EuclideanFSSLoss(nn.Module):
    """BCE + Dice for Euclidean few-shot segmentation.

    Args:
        bce_weight: weight on BCE term; Dice gets (1 - bce_weight)
        pos_weight: scalar weight for the positive (foreground) class in BCE;
                    compensates for the severe background/foreground imbalance
                    in the UAV sugarcane dataset
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        pos_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = 1.0 - bce_weight
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(
        self,
        logits: torch.Tensor,   # (B, H, W)
        target: torch.Tensor,   # (B, H, W) float {0,1}
        **_kwargs,              # absorb unused proto_fg / proto_bg
    ) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        dice = _dice_loss(torch.sigmoid(logits), target)
        return self.bce_w * bce + self.dice_w * dice


# ─────────────────────────────────────────────────────────────────────────────
# Pontryagin loss
# ─────────────────────────────────────────────────────────────────────────────

class PontryaginFSSLoss(nn.Module):
    """BCE + Dice + L_cone + L_orth for Pontryagin few-shot segmentation.

    Penalty terms (beyond the base BCE+Dice):
        L_cone  — prototypes must not lie on the J-null cone;
                  a lightlike prototype has no stable distance interpretation.
                  L_cone = mean relu(ε - |q_J(proto)|)  over {fg, bg}

        L_orth  — fg and bg prototypes should be J-orthogonal so they span
                  distinct directions in Pontryagin space.
                  L_orth = |⟨proto_fg, proto_bg⟩_J|² /
                            max(|q_J(fg)| · |q_J(bg)|, δ)

    Args:
        J:            Pontryagin metric vector (D,) with values ±1
        bce_weight:   weight on BCE term
        pos_weight:   positive class weight for BCE
        lambda_cone:  weight for cone penalty
        lambda_orth:  weight for orthogonality penalty
        cone_epsilon: null-cone margin ε
    """

    def __init__(
        self,
        J: torch.Tensor,
        bce_weight: float = 0.5,
        pos_weight: float = 5.0,
        lambda_cone: float = 0.1,
        lambda_orth: float = 0.05,
        cone_epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("J", J.float())
        self.bce_w       = bce_weight
        self.dice_w      = 1.0 - bce_weight
        self.lambda_cone = lambda_cone
        self.lambda_orth = lambda_orth
        self.cone_eps    = cone_epsilon
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    # ------------------------------------------------------------------
    # Pontryagin geometry
    # ------------------------------------------------------------------

    def _q_J(self, z: torch.Tensor) -> torch.Tensor:
        """q_J(z) = ⟨z,z⟩_J, last dim = D."""
        return (z.pow(2) * self.J).sum(dim=-1)

    def _j_inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """⟨u,v⟩_J, last dim = D."""
        return (u * v * self.J).sum(dim=-1)

    def _cone_penalty(self, *protos: torch.Tensor) -> torch.Tensor:
        """Mean relu(ε - |q_J(p)|) over all given prototypes."""
        terms = [
            F.relu(self.cone_eps - self._q_J(p).abs())
            for p in protos
        ]
        return torch.stack(terms).mean()

    def _orth_penalty(
        self, proto_fg: torch.Tensor, proto_bg: torch.Tensor
    ) -> torch.Tensor:
        """Normalised squared J-inner product between fg and bg prototypes."""
        ip    = self._j_inner(proto_fg, proto_bg).abs()              # (B,)
        denom = (
            self._q_J(proto_fg).abs() * self._q_J(proto_bg).abs()
        ).clamp(min=1e-8)                                            # (B,)
        return (ip / denom).mean()

    # ------------------------------------------------------------------

    def forward(
        self,
        logits:    torch.Tensor,              # (B, H, W)
        target:    torch.Tensor,              # (B, H, W) float {0,1}
        proto_fg:  torch.Tensor | None = None,  # (B, D)
        proto_bg:  torch.Tensor | None = None,  # (B, D)
        **_kwargs,
    ) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        dice = _dice_loss(torch.sigmoid(logits), target)
        loss = self.bce_w * bce + self.dice_w * dice

        if proto_fg is not None and self.lambda_cone > 0:
            protos = [proto_fg] + ([proto_bg] if proto_bg is not None else [])
            loss   = loss + self.lambda_cone * self._cone_penalty(*protos)

        if (
            proto_fg is not None
            and proto_bg is not None
            and self.lambda_orth > 0
        ):
            loss = loss + self.lambda_orth * self._orth_penalty(proto_fg, proto_bg)

        return loss
