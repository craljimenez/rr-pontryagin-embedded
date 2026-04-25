import math
import torch
import pytest
from prfe.layers import PontryaginEmbedding, RandomFourierFeatures, SphericalRandomFeatures


@pytest.fixture
def rff():
    return RandomFourierFeatures(in_features=8, n_components=64, sigma=1.0)


@pytest.fixture
def srf():
    return SphericalRandomFeatures(in_features=8, n_components=32, degree=2)


@pytest.fixture
def embed():
    return PontryaginEmbedding(in_channels=8, n_rff=64, n_srf=32, kappa=2)


class TestRFF:
    def test_output_shape(self, rff):
        x = torch.randn(10, 8)
        assert rff(x).shape == (10, 128)

    def test_kernel_approximation(self, rff):
        """RFF Gram matrix should be close to RBF Gram matrix."""
        torch.manual_seed(0)
        rff_ = RandomFourierFeatures(8, 2048, sigma=1.0)
        x, y = torch.randn(5, 8), torch.randn(5, 8)
        phi_x, phi_y = rff_(x), rff_(y)
        K_approx = phi_x @ phi_y.T
        diff = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
        K_exact = torch.exp(-diff / 2)
        assert (K_approx - K_exact).abs().mean() < 0.05

    def test_log_sigma_is_parameter(self, rff):
        """σ is always tuned by gradient, even when W/b are fixed."""
        assert isinstance(rff.log_sigma, torch.nn.Parameter)
        assert rff.log_sigma.requires_grad
        # By default W and b are fixed (not parameters)
        param_names = {name for name, _ in rff.named_parameters()}
        assert param_names == {"log_sigma"}

    def test_log_sigma_receives_gradient(self):
        """Gradients flow into log_sigma under a simple objective.

        Note: ||φ(x)||² = ½ regardless of σ (cos²+sin²=1), so an L2 loss would
        give a zero gradient by construction. We use .sum() instead, which
        does depend on σ through cos(Wx/σ+b) + sin(Wx/σ+b).
        """
        torch.manual_seed(0)
        rff = RandomFourierFeatures(8, 64, sigma=1.0)
        x = torch.randn(4, 8)
        loss = rff(x).sum()
        loss.backward()
        assert rff.log_sigma.grad is not None
        assert torch.isfinite(rff.log_sigma.grad).all()
        assert rff.log_sigma.grad.abs().item() > 0

    def test_sigma_initial_value(self):
        """log_sigma stores log(σ₀) at construction."""
        rff = RandomFourierFeatures(4, 32, sigma=2.5)
        assert math.isclose(rff.sigma.item(), 2.5, rel_tol=1e-6)
        assert math.isclose(rff.log_sigma.item(), math.log(2.5), rel_tol=1e-6)


class TestSRF:
    def test_output_shape(self, srf):
        x = torch.randn(10, 8)
        assert srf(x).shape == (10, 32)

    def test_kernel_approximation(self, srf):
        """SRF Gram matrix should approximate the polynomial kernel."""
        torch.manual_seed(0)
        srf_ = SphericalRandomFeatures(8, 4096, degree=2)
        x = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        y = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        K_approx = srf_(x) @ srf_(y).T
        K_exact = (x @ y.T).pow(2)
        assert (K_approx - K_exact).abs().mean() < 0.1


class TestPontryaginEmbedding:
    def test_output_shape(self, embed):
        x = torch.randn(2, 8, 4, 4)
        out = embed(x)
        assert out.shape == (2, embed.out_channels, 4, 4)

    def test_metric_buffer(self, embed):
        assert embed.J.shape == (embed.out_channels,)
        p = embed.rff.out_features
        assert (embed.J[:p] == 1).all()
        assert (embed.J[p:] == -1).all()

    def test_pontryagin_inner(self, embed):
        x = torch.randn(2, 8, 4, 4)
        z = embed(x)
        ip = embed.pontryagin_inner(z, z)
        assert ip.shape == (2, 4, 4)
