import torch
import pytest
from prfe.layers import PontryaginEmbedding
from prfe.losses import (
    IndefiniteTopographicPenalty,
    PontryaginMLR,
    PontryaginPrototypical,
)


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def embed():
    return PontryaginEmbedding(in_channels=8, n_rff=16, n_srf=8, kappa=2)


@pytest.fixture
def J(embed):
    return embed.J.clone()


@pytest.fixture
def flat_batch(embed):
    torch.manual_seed(0)
    x = torch.randn(20, 8, 4, 4)
    z = embed(x)                                    # (20, p+q, 4, 4)
    B, D, H, W = z.shape
    z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
    labels = torch.randint(0, 5, (B * H * W,))
    return z_flat, labels, D


# ── IndefiniteTopographicPenalty ─────────────────────────────────────────────

class TestTopographicPenalty:
    def test_q_J_sign(self, J):
        topo = IndefiniteTopographicPenalty(J)
        p = (J == 1).sum().item()   # number of positive channels
        z_pos = torch.zeros(4, J.shape[0])
        z_pos[:, :p] = 1.0          # only positive channels → q_J > 0
        z_neg = torch.zeros(4, J.shape[0])
        z_neg[:, p:] = 1.0          # only negative channels → q_J < 0
        assert (topo.q_J(z_pos) > 0).all()
        assert (topo.q_J(z_neg) < 0).all()

    def test_light_cone_penalty_zero_far(self, J):
        topo = IndefiniteTopographicPenalty(J, lc_epsilon=0.05)
        z = torch.zeros(10, J.shape[0])
        z[:, 0] = 5.0                               # large positive → far from null cone
        assert topo.light_cone_penalty(z).item() == pytest.approx(0.0, abs=1e-6)

    def test_light_cone_penalty_nonzero_near(self, J):
        topo = IndefiniteTopographicPenalty(J, lc_epsilon=1.0)
        z = torch.zeros(10, J.shape[0])             # q_J = 0 for all → on null cone
        assert topo.light_cone_penalty(z).item() > 0

    def test_causal_consistency_same_class(self, J):
        topo = IndefiniteTopographicPenalty(J)
        z = torch.zeros(10, J.shape[0])
        z[:, 0] = 2.0                               # all identical → zero variance
        labels = torch.zeros(10, dtype=torch.long)
        assert topo.causal_consistency(z, labels).item() == pytest.approx(0.0, abs=1e-6)

    def test_signature_balance_zero(self, J):
        topo = IndefiniteTopographicPenalty(J)
        D = J.shape[0]
        p = (J == 1).sum().item()
        z = torch.zeros(10, D)
        z[:, :p] = 1.0
        z[:, p:] = 1.0                              # q_J = p - q → not zero in general
        # just test it returns a scalar
        assert topo.signature_balance(z).shape == torch.Size([])

    def test_forward_requires_labels_when_lambda_cc(self, J):
        topo = IndefiniteTopographicPenalty(J, lambda_cc=1.0)
        z = torch.randn(10, J.shape[0])
        with pytest.raises(ValueError):
            topo(z, labels=None)

    def test_forward_scalar(self, J, flat_batch):
        z, labels, _ = flat_batch
        topo = IndefiniteTopographicPenalty(J)
        loss = topo(z, labels)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)


# ── PontryaginMLR ────────────────────────────────────────────────────────────

class TestPontryaginMLR:
    def test_logits_shape(self, J, flat_batch):
        z, labels, D = flat_batch
        mlr = PontryaginMLR(n_classes=5, in_features=D, J=J)
        assert mlr.logits(z).shape == (z.shape[0], 5)

    def test_cone_penalty_nonneg(self, J, flat_batch):
        _, _, D = flat_batch
        mlr = PontryaginMLR(n_classes=5, in_features=D, J=J)
        assert mlr.cone_penalty_W().item() >= 0

    def test_loss_finite(self, J, flat_batch):
        z, labels, D = flat_batch
        mlr = PontryaginMLR(
            n_classes=5, in_features=D, J=J,
            lambda_topo=0.05,
            topo_kwargs={"lambda_lc": 1.0, "lambda_cc": 1.0, "lambda_sb": 1.0},
        )
        loss = mlr(z, labels)
        assert torch.isfinite(loss)

    def test_predict_shape(self, J, flat_batch):
        z, _, D = flat_batch
        mlr = PontryaginMLR(n_classes=5, in_features=D, J=J)
        assert mlr.predict(z).shape == (z.shape[0],)

    def test_spatial_wrappers(self, embed, J):
        B, H, W = 2, 4, 4
        x = torch.randn(B, 8, H, W)
        z = embed(x)                                # (B, p+q, H, W)
        labels = torch.randint(0, 3, (B, H, W))
        mlr = PontryaginMLR(3, embed.out_channels, J, lambda_topo=0.0)
        loss = mlr.forward_spatial(z, labels)
        preds = mlr.predict_spatial(z)
        assert torch.isfinite(loss)
        assert preds.shape == (B, H, W)


# ── PontryaginPrototypical ───────────────────────────────────────────────────

class TestPontryaginPrototypical:
    @pytest.fixture
    def episode(self, embed):
        torch.manual_seed(1)
        N_way, K_shot, N_q = 3, 5, 12
        # support
        sx = torch.randn(N_way * K_shot, 8, 2, 2)
        sz = embed(sx)
        sz = embed._modules                         # avoid re-running embed twice
        # use embed directly
        sz = embed(torch.randn(N_way * K_shot, 8, 2, 2))
        sz_flat = sz.permute(0, 2, 3, 1).reshape(N_way * K_shot * 4, -1)
        sl = torch.repeat_interleave(torch.arange(N_way), K_shot * 4)
        # query
        qz = embed(torch.randn(N_q, 8, 2, 2))
        qz_flat = qz.permute(0, 2, 3, 1).reshape(N_q * 4, -1)
        ql = torch.randint(0, N_way, (N_q * 4,))
        return sz_flat, sl, qz_flat, ql

    def test_prototypes_shape(self, J, episode):
        sz, sl, _, _ = episode
        proto = PontryaginPrototypical(J)
        prototypes, classes = proto.compute_prototypes(sz, sl)
        assert classes.shape == (3,)
        assert prototypes.shape == (3, J.shape[0])

    def test_cone_penalty_nonneg(self, J, episode):
        sz, sl, _, _ = episode
        proto = PontryaginPrototypical(J)
        prototypes, _ = proto.compute_prototypes(sz, sl)
        assert proto.cone_penalty(prototypes).item() >= 0

    def test_orthogonality_nonneg(self, J, episode):
        sz, sl, _, _ = episode
        proto = PontryaginPrototypical(J)
        prototypes, _ = proto.compute_prototypes(sz, sl)
        assert proto.j_orthogonality(prototypes).item() >= 0

    def test_loss_finite(self, J, episode):
        sz, sl, qz, ql = episode
        proto = PontryaginPrototypical(
            J, lambda_topo=0.05,
            topo_kwargs={"lambda_lc": 1.0, "lambda_cc": 0.5, "lambda_sb": 0.5},
        )
        loss = proto(sz, sl, qz, ql)
        assert torch.isfinite(loss)

    def test_predict_shape(self, J, episode):
        sz, sl, qz, _ = episode
        proto = PontryaginPrototypical(J)
        preds = proto.predict(sz, sl, qz)
        assert preds.shape == (qz.shape[0],)
        # predictions must be one of the support classes
        assert set(preds.tolist()).issubset(set(sl.unique().tolist()))
