"""Tests for UNetBackbone and the four UNet segmentation model variants.

Coverage:
  UNetBackbone
    - output shape preserved across depth 3/4/5 and different base_ch
    - skip-connection alignment: decoder output matches input H×W
    - invalid depth raises ValueError
    - gradients flow from output to all encoder/decoder blocks

  VanillaUNet / EuclideanUNet / HyperbolicUNet / PontryaginUNet
    - forward produces (B, N_CLASSES, H, W) at full input resolution
    - compute_loss returns a finite scalar
    - backward populates gradients in all leaf parameters
    - batch size B=1 and B=4 both work

  build_unet_model factory
    - correct type returned for each model key
    - params dict overrides defaults (unet_depth, hyperbolic_c, kappa, …)
    - unknown key raises ValueError
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from prfe.models.unet import UNetBackbone
from run_seg_uav_unet import (
    EuclideanUNet,
    HyperbolicUNet,
    PontryaginUNet,
    VanillaUNet,
    build_unet_model,
)

# ── constants matching the experiment config ──────────────────────────────────
B, C, H, W = 2, 3, 64, 64   # small spatial size to keep tests fast
N_CLASSES   = 2
OUT_CH      = 64


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def img():
    return torch.randn(B, C, H, W)


@pytest.fixture
def labels():
    return torch.randint(0, N_CLASSES, (B, H, W))


# ─────────────────────────────────────────────────────────────────────────────
# UNetBackbone
# ─────────────────────────────────────────────────────────────────────────────

class TestUNetBackbone:

    @pytest.mark.parametrize("depth", [3, 4, 5])
    def test_output_shape_various_depths(self, img, depth):
        model = UNetBackbone(in_channels=C, out_channels=OUT_CH,
                             base_ch=32, depth=depth)
        out = model(img)
        assert out.shape == (B, OUT_CH, H, W), (
            f"depth={depth}: expected {(B, OUT_CH, H, W)}, got {out.shape}"
        )

    @pytest.mark.parametrize("base_ch", [16, 32, 64])
    def test_output_shape_various_base_ch(self, img, base_ch):
        model = UNetBackbone(in_channels=C, out_channels=OUT_CH,
                             base_ch=base_ch, depth=3)
        out = model(img)
        assert out.shape == (B, OUT_CH, H, W)

    def test_spatial_resolution_preserved(self, img):
        """Output H×W must match input H×W regardless of depth."""
        model = UNetBackbone(in_channels=C, out_channels=OUT_CH, depth=4)
        out = model(img)
        assert out.shape[-2:] == img.shape[-2:]

    def test_non_square_input(self):
        """Backbone must handle non-square inputs (e.g. 96×128)."""
        x = torch.randn(1, C, 96, 128)
        model = UNetBackbone(in_channels=C, out_channels=OUT_CH, depth=3)
        out = model(x)
        assert out.shape == (1, OUT_CH, 96, 128)

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError, match="depth must be"):
            UNetBackbone(depth=2)
        with pytest.raises(ValueError, match="depth must be"):
            UNetBackbone(depth=6)

    def test_gradient_flow_through_all_blocks(self, img):
        """Gradients must reach both the first enc block and the proj layer."""
        model = UNetBackbone(in_channels=C, out_channels=OUT_CH, depth=4)
        out = model(img)
        out.sum().backward()
        # Projection layer
        assert model.proj.weight.grad is not None
        assert model.proj.weight.grad.abs().sum() > 0
        # First encoder block (deepest in the compute graph)
        first_enc_conv = model.enc_blocks[0][0]
        assert first_enc_conv.weight.grad is not None
        assert first_enc_conv.weight.grad.abs().sum() > 0

    def test_output_finite(self, img):
        model = UNetBackbone(in_channels=C, out_channels=OUT_CH, depth=4)
        out = model(img)
        assert torch.isfinite(out).all()

    def test_batch_size_one(self):
        x = torch.randn(1, C, H, W)
        model = UNetBackbone(in_channels=C, out_channels=OUT_CH, depth=4)
        out = model(x)
        assert out.shape == (1, OUT_CH, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for segmentation model tests
# ─────────────────────────────────────────────────────────────────────────────

def _check_forward(model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(img)
    assert logits.shape == (img.shape[0], N_CLASSES, H, W), (
        f"{type(model).__name__}: expected {(img.shape[0], N_CLASSES, H, W)}, "
        f"got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), f"{type(model).__name__}: non-finite logits"
    return logits


def _check_loss_and_backward(model: nn.Module,
                              img: torch.Tensor,
                              labels: torch.Tensor) -> None:
    model.train()
    loss = model.compute_loss(img, labels)
    assert loss.ndim == 0,        f"{type(model).__name__}: loss must be scalar"
    assert torch.isfinite(loss),  f"{type(model).__name__}: loss is not finite"
    loss.backward()
    # At least one leaf parameter must have received a gradient
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), \
        f"{type(model).__name__}: no parameter received a gradient"
    assert any(g is not None and g.abs().sum() > 0 for g in grads), \
        f"{type(model).__name__}: all gradients are zero"


# ─────────────────────────────────────────────────────────────────────────────
# VanillaUNet
# ─────────────────────────────────────────────────────────────────────────────

class TestVanillaUNet:

    @pytest.fixture
    def model(self):
        return VanillaUNet(unet_depth=3)

    def test_forward_shape(self, model, img):
        _check_forward(model, img)

    def test_loss_and_backward(self, model, img, labels):
        _check_loss_and_backward(model, img, labels)

    @pytest.mark.parametrize("batch", [1, 4])
    def test_various_batch_sizes(self, batch):
        m = VanillaUNet(unet_depth=3)
        x = torch.randn(batch, C, H, W)
        y = torch.randint(0, N_CLASSES, (batch, H, W))
        _check_forward(m, x)
        _check_loss_and_backward(m, x, y)

    def test_no_class_weights(self, model):
        """VanillaUNet must NOT have a class_weights buffer."""
        buffers = dict(model.named_buffers())
        assert "class_weights" not in buffers


# ─────────────────────────────────────────────────────────────────────────────
# EuclideanUNet
# ─────────────────────────────────────────────────────────────────────────────

class TestEuclideanUNet:

    @pytest.fixture
    def model(self):
        return EuclideanUNet(unet_depth=3)

    def test_forward_shape(self, model, img):
        _check_forward(model, img)

    def test_loss_and_backward(self, model, img, labels):
        _check_loss_and_backward(model, img, labels)

    def test_class_weights_buffer(self, model):
        """EuclideanUNet must carry the class_weights buffer."""
        buffers = dict(model.named_buffers())
        assert "class_weights" in buffers
        assert buffers["class_weights"].shape == (N_CLASSES,)

    @pytest.mark.parametrize("batch", [1, 4])
    def test_various_batch_sizes(self, batch):
        m = EuclideanUNet(unet_depth=3)
        x = torch.randn(batch, C, H, W)
        y = torch.randint(0, N_CLASSES, (batch, H, W))
        _check_forward(m, x)
        _check_loss_and_backward(m, x, y)


# ─────────────────────────────────────────────────────────────────────────────
# HyperbolicUNet
# ─────────────────────────────────────────────────────────────────────────────

class TestHyperbolicUNet:

    @pytest.fixture
    def model(self):
        return HyperbolicUNet(c=1.0, unet_depth=3)

    def test_forward_shape(self, model, img):
        _check_forward(model, img)

    def test_loss_and_backward(self, model, img, labels):
        _check_loss_and_backward(model, img, labels)

    @pytest.mark.parametrize("c", [0.1, 1.0, 2.0])
    def test_various_curvatures(self, img, labels, c):
        m = HyperbolicUNet(c=c, unet_depth=3)
        _check_forward(m, img)
        _check_loss_and_backward(m, img, labels)

    @pytest.mark.parametrize("batch", [1, 4])
    def test_various_batch_sizes(self, batch):
        m = HyperbolicUNet(c=1.0, unet_depth=3)
        x = torch.randn(batch, C, H, W)
        y = torch.randint(0, N_CLASSES, (batch, H, W))
        _check_forward(m, x)
        _check_loss_and_backward(m, x, y)


# ─────────────────────────────────────────────────────────────────────────────
# PontryaginUNet
# ─────────────────────────────────────────────────────────────────────────────

class TestPontryaginUNet:

    @pytest.fixture
    def model(self):
        return PontryaginUNet(kappa=4, d_poly=2, rff_multiplier=1,
                              sigma=1.0, lambda_topo=0.05,
                              lambda_balance=0.1, unet_depth=3)

    def test_forward_shape(self, model, img):
        _check_forward(model, img)

    def test_loss_and_backward(self, model, img, labels):
        _check_loss_and_backward(model, img, labels)

    def test_embed_layer_output_channels(self, model):
        """embed_layer.out_channels == 2*n_rff + kappa."""
        expected = 2 * (1 * OUT_CH) + 4   # rff_multiplier=1, kappa=4
        assert model.embed_layer.out_channels == expected

    def test_J_signature(self, model):
        """J metric tensor: +1 for RFF dims, -1 for SRF dims."""
        p = model.embed_layer.rff.out_features     # 2 * n_rff
        J = model.embed_layer.J
        assert (J[:p] == 1).all()
        assert (J[p:] == -1).all()

    @pytest.mark.parametrize("kappa,d_poly,rff_mult", [
        (2, 2, 1),
        (8, 3, 2),
        (12, 4, 1),
    ])
    def test_various_hyperparams(self, img, labels, kappa, d_poly, rff_mult):
        m = PontryaginUNet(kappa=kappa, d_poly=d_poly,
                           rff_multiplier=rff_mult, unet_depth=3)
        _check_forward(m, img)
        _check_loss_and_backward(m, img, labels)

    @pytest.mark.parametrize("batch", [1, 4])
    def test_various_batch_sizes(self, batch):
        m = PontryaginUNet(kappa=4, d_poly=2, rff_multiplier=1, unet_depth=3)
        x = torch.randn(batch, C, H, W)
        y = torch.randint(0, N_CLASSES, (batch, H, W))
        _check_forward(m, x)
        _check_loss_and_backward(m, x, y)

    def test_balance_loss_activates(self, model, img, labels):
        """lambda_balance > 0 should add a non-zero balance term to the loss."""
        model_no_bal = PontryaginUNet(kappa=4, d_poly=2, rff_multiplier=1,
                                      lambda_balance=0.0, unet_depth=3)
        torch.manual_seed(0)
        loss_with    = model.compute_loss(img, labels).item()
        torch.manual_seed(0)
        loss_without = model_no_bal.compute_loss(img, labels).item()
        assert loss_with != loss_without, \
            "lambda_balance>0 should affect the total loss"


# ─────────────────────────────────────────────────────────────────────────────
# build_unet_model factory
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildUnetModel:

    @pytest.mark.parametrize("key,cls", [
        ("vanilla",     VanillaUNet),
        ("euclidean",   EuclideanUNet),
        ("hyperbolic",  HyperbolicUNet),
        ("pontryagin",  PontryaginUNet),
    ])
    def test_correct_type(self, key, cls):
        m = build_unet_model(key)
        assert isinstance(m, cls)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            build_unet_model("unknown_model")

    def test_unet_depth_override(self):
        m = build_unet_model("vanilla", params={"unet_depth": 3})
        assert m.backbone.depth == 3
        m5 = build_unet_model("vanilla", params={"unet_depth": 5})
        assert m5.backbone.depth == 5

    def test_hyperbolic_c_override(self, img, labels):
        m = build_unet_model("hyperbolic", params={"hyperbolic_c": 0.5,
                                                    "unet_depth": 3})
        assert isinstance(m, HyperbolicUNet)
        _check_forward(m, img)

    def test_pontryagin_params_override(self, img, labels):
        params = {"kappa": 6, "d_poly": 3, "rff_multiplier": 2,
                  "lambda_topo": 0.01, "lambda_balance": 0.05,
                  "unet_depth": 3}
        m = build_unet_model("pontryagin", params=params)
        assert isinstance(m, PontryaginUNet)
        assert m.kappa == 6
        assert m.d_poly == 3
        _check_forward(m, img)

    def test_empty_params_uses_defaults(self):
        m = build_unet_model("pontryagin", params={})
        assert isinstance(m, PontryaginUNet)

    @pytest.mark.parametrize("key", ["vanilla", "euclidean", "hyperbolic", "pontryagin"])
    def test_all_models_forward(self, key, img):
        m = build_unet_model(key, params={"unet_depth": 3})
        _check_forward(m, img)
