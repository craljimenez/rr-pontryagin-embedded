"""Tests for PoincareEmbedding and HyperbolicMLR (Ganea/Atigh baseline)."""

import pytest
import torch

from prfe.layers import PoincareEmbedding
from prfe.losses import HyperbolicMLR
from prfe.losses.hyperbolic_mlr import _expmap0, _mobius_add


# ── Möbius addition / expmap primitives ──────────────────────────────────────

class TestMobius:
    def test_add_with_zero_is_identity(self):
        x = torch.randn(4, 3) * 0.1
        zero = torch.zeros_like(x)
        out = _mobius_add(zero, x, c=1.0)
        assert torch.allclose(out, x, atol=1e-5)

    def test_add_is_left_inverse(self):
        """(-x) ⊕ x = 0 in the Poincaré ball."""
        x = torch.randn(4, 3) * 0.1
        out = _mobius_add(-x, x, c=1.0)
        assert torch.allclose(out, torch.zeros_like(x), atol=1e-5)

    def test_expmap0_on_ball(self):
        """expmap₀ output stays inside the unit ball (c = 1)."""
        torch.manual_seed(0)
        v = torch.randn(10, 5) * 3.0         # large norms
        y = _expmap0(v, c=1.0)
        # Due to fp precision tanh(large) can round to 1.0 exactly; allow the boundary
        assert (y.norm(dim=-1) <= 1.0).all()


# ── PoincareEmbedding ────────────────────────────────────────────────────────

class TestPoincareEmbedding:
    def test_output_shape(self):
        emb = PoincareEmbedding(c=1.0)
        x = torch.randn(2, 8, 4, 4)
        y = emb(x)
        assert y.shape == x.shape

    def test_output_inside_ball(self):
        emb = PoincareEmbedding(c=1.0)
        x = torch.randn(2, 8, 4, 4) * 5.0
        y = emb(x)
        norms = y.norm(dim=1)                 # (B, H, W)
        assert (norms < 1.0).all()


# ── HyperbolicMLR ────────────────────────────────────────────────────────────

@pytest.fixture
def setup():
    torch.manual_seed(0)
    N, d, C = 32, 8, 5
    emb = PoincareEmbedding(c=1.0)
    x = torch.randn(N, d) * 0.3
    # Project flat features through expmap by treating them as (N, d, 1, 1)
    z = emb(x.reshape(N, d, 1, 1)).reshape(N, d)
    labels = torch.randint(0, C, (N,))
    return z, labels, d, C


class TestHyperbolicMLR:
    def test_logits_shape(self, setup):
        z, labels, d, C = setup
        mlr = HyperbolicMLR(C, d, c=1.0)
        assert mlr.logits(z).shape == (z.shape[0], C)

    def test_predict_shape(self, setup):
        z, _, d, C = setup
        mlr = HyperbolicMLR(C, d, c=1.0)
        assert mlr.predict(z).shape == (z.shape[0],)

    def test_loss_finite(self, setup):
        z, labels, d, C = setup
        mlr = HyperbolicMLR(C, d, c=1.0)
        loss = mlr(z, labels)
        assert torch.isfinite(loss)

    def test_backward(self, setup):
        z, labels, d, C = setup
        z = z.detach().requires_grad_(True)
        mlr = HyperbolicMLR(C, d, c=1.0)
        loss = mlr(z, labels)
        loss.backward()
        # gradients for class params
        assert mlr.a.grad is not None
        assert mlr.p_tangent.grad is not None
        assert torch.isfinite(mlr.a.grad).all()
        assert torch.isfinite(mlr.p_tangent.grad).all()

    def test_spatial_forward(self):
        torch.manual_seed(1)
        B, d, H, W, C = 2, 6, 4, 4, 3
        emb = PoincareEmbedding(c=1.0)
        z = emb(torch.randn(B, d, H, W))
        labels = torch.randint(0, C, (B, H, W))
        mlr = HyperbolicMLR(C, d, c=1.0)
        loss = mlr.forward_spatial(z, labels)
        preds = mlr.predict_spatial(z)
        assert torch.isfinite(loss)
        assert preds.shape == (B, H, W)

    def test_class_weights(self, setup):
        z, labels, d, C = setup
        w = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])     # ignore class 1
        mlr = HyperbolicMLR(C, d, c=1.0, class_weights=w)
        loss = mlr(z, labels)
        assert torch.isfinite(loss)
