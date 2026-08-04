"""Likelihoods: moments, the Fisher, and its factorisation.

Two families of check matter most. The *identities* -- that $\\bm\\Lambda$ really
is the Hessian of the NLL and $\\bm\\nu$ really is its negative gradient -- pin the
implementation to the definitions rather than to a formula someone typed. And the
*rank* checks pin the softmax singularity, which is the property the whole choice
of update form rests on.
"""

from __future__ import annotations

import math

import pytest
import torch

from dekf_bench.likelihoods.base import Likelihood, LikelihoodError, reduce
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.likelihoods.gaussian import Gaussian

Q = 6
BATCH = 4
DTYPE = torch.float64


def logits(batch: int = BATCH, q: int = Q, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, q, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)


def targets(batch: int = BATCH, q: int = Q) -> torch.Tensor:
    return torch.arange(batch, dtype=torch.int64) % q


@pytest.fixture
def categorical() -> Categorical:
    return Categorical(output_dim=Q)


@pytest.fixture
def gaussian() -> Gaussian:
    return Gaussian(output_dim=3, variance=0.25)


# =========================================================================== #
# 1. both satisfy the interface
# =========================================================================== #


def test_both_satisfy_the_protocol(categorical: Categorical, gaussian: Gaussian) -> None:
    """An interface with one implementation is shaped around it whether or not
    anyone intended that."""
    assert isinstance(categorical, Likelihood)
    assert isinstance(gaussian, Likelihood)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_reduce_handles_every_mode(reduction: str) -> None:
    values = torch.tensor([1.0, 2.0, 3.0])
    result = reduce(values, reduction)  # type: ignore[arg-type]
    assert result.shape == ((3,) if reduction == "none" else ())


def test_unknown_reduction_is_rejected() -> None:
    with pytest.raises(LikelihoodError, match="unknown reduction"):
        reduce(torch.zeros(3), "average")  # type: ignore[arg-type]


# =========================================================================== #
# 2. softmax moments
# =========================================================================== #


def test_mu_is_a_probability_vector(categorical: Categorical) -> None:
    pi = categorical.mu(logits())
    assert torch.allclose(pi.sum(-1), torch.ones(BATCH, dtype=DTYPE))
    assert bool((pi > 0).all())


def test_mu_is_shift_invariant(categorical: Categorical) -> None:
    """softmax(h + c1) = softmax(h). This invariance is *why* the Fisher is
    singular, so it is worth pinning directly."""
    h = logits()
    shifted = h + 3.7
    assert torch.allclose(categorical.mu(h), categorical.mu(shifted), atol=1e-14)


def test_predictions_are_the_argmax(categorical: Categorical) -> None:
    h = logits()
    assert torch.equal(categorical.predictions(h), h.argmax(dim=-1))


# =========================================================================== #
# 3. the Fisher, and its singularity
# =========================================================================== #


def test_fisher_matches_the_closed_form(categorical: Categorical) -> None:
    h = logits()
    pi = categorical.mu(h)
    fisher = categorical.fisher(h)
    for b in range(BATCH):
        for i in range(Q):
            assert fisher[b, i, i] == pytest.approx(float(pi[b, i] * (1 - pi[b, i])))
            for j in range(Q):
                if i != j:
                    assert fisher[b, i, j] == pytest.approx(float(-pi[b, i] * pi[b, j]))


def test_fisher_is_symmetric(categorical: Categorical) -> None:
    fisher = categorical.fisher(logits())
    assert torch.allclose(fisher, fisher.transpose(-1, -2), atol=1e-15)


def test_fisher_is_positive_semidefinite(categorical: Categorical) -> None:
    eigenvalues = torch.linalg.eigvalsh(categorical.fisher(logits()))
    assert float(eigenvalues.min()) > -1e-14


def test_fisher_annihilates_the_ones_vector(categorical: Categorical) -> None:
    """Lambda 1 = 0: softmax cannot see a shift common to every logit, and the
    Fisher says so rather than pretending otherwise."""
    fisher = categorical.fisher(logits())
    ones = torch.ones(Q, dtype=DTYPE)
    assert float((fisher @ ones).abs().max()) < 1e-14


def test_fisher_rank_is_exactly_q_minus_one(categorical: Categorical) -> None:
    ranks = torch.linalg.matrix_rank(categorical.fisher(logits()))
    assert torch.all(ranks == Q - 1)
    assert categorical.fisher_rank == Q - 1


def test_the_factorisation_is_exact(categorical: Categorical) -> None:
    """Lambda = G G^T to machine precision, with no eigendecomposition: G is
    diag(sqrt(pi))(I - sqrt(pi) sqrt(pi)^T), and I - ss^T is a projector because
    ||s||^2 = sum(pi) = 1."""
    h = logits()
    factor = categorical.fisher_factor(h)
    assert torch.allclose(categorical.fisher(h), factor @ factor.transpose(-1, -2), atol=1e-14)


def test_the_factor_has_the_same_rank_as_the_fisher(categorical: Categorical) -> None:
    ranks = torch.linalg.matrix_rank(categorical.fisher_factor(logits()))
    assert torch.all(ranks == Q - 1)


def test_the_factor_also_annihilates_the_ones_vector(categorical: Categorical) -> None:
    factor = categorical.fisher_factor(logits())
    ones = torch.ones(Q, dtype=DTYPE)
    assert float((factor.transpose(-1, -2) @ ones).abs().max()) < 1e-14


def test_the_factorisation_survives_a_saturated_prediction() -> None:
    """Where a Cholesky would fail: one probability at ~1, the rest at ~0."""
    likelihood = Categorical(output_dim=Q)
    h = torch.zeros(1, Q, dtype=DTYPE)
    h[0, 2] = 60.0
    factor = likelihood.fisher_factor(h)
    assert torch.allclose(likelihood.fisher(h), factor @ factor.transpose(-1, -2), atol=1e-14)
    assert bool(torch.isfinite(factor).all())


# =========================================================================== #
# 4. the identities that pin Lambda and nu to their definitions
# =========================================================================== #


def test_the_innovation_is_the_negative_gradient_of_the_nll(
    categorical: Categorical,
) -> None:
    """dNLL/dh = pi - y = -nu. This is the link to gradient training: the filter's
    innovation term H^T nu *is* the negative loss gradient, preconditioned."""
    h = logits().clone().requires_grad_(True)
    y = targets()
    categorical.nll(h, y, reduction="sum").backward()
    assert h.grad is not None
    assert torch.allclose(h.grad, -categorical.innovation(h.detach(), y), atol=1e-12)


def test_the_fisher_is_the_hessian_of_the_nll(categorical: Categorical) -> None:
    """For a canonical link the Fisher and the Hessian coincide. Checking against
    autograd pins Lambda to its definition rather than to a typed formula."""
    y = targets(1)
    h = logits(1)

    def loss(z: torch.Tensor) -> torch.Tensor:
        return categorical.nll(z.unsqueeze(0), y, reduction="sum")

    hessian = torch.autograd.functional.hessian(loss, h.squeeze(0))
    assert torch.allclose(hessian, categorical.fisher(h)[0], atol=1e-10)


def test_the_innovation_lies_in_the_identifiable_subspace(
    categorical: Categorical,
) -> None:
    """1^T nu = 1 - 1 = 0, the same subspace Lambda spans. Data never arrives
    along the direction the model cannot identify."""
    nu = categorical.innovation(logits(), targets())
    assert float(nu.sum(-1).abs().max()) < 1e-14


def test_a_perfect_prediction_carries_no_information(categorical: Categorical) -> None:
    """If nu = 0 the sample was completely predicted and nothing is learned."""
    h = torch.full((1, Q), -60.0, dtype=DTYPE)
    h[0, 3] = 60.0
    nu = categorical.innovation(h, torch.tensor([3]))
    assert float(nu.abs().max()) < 1e-12


# =========================================================================== #
# 5. the loss
# =========================================================================== #


def test_nll_matches_cross_entropy(categorical: Categorical) -> None:
    import torch.nn.functional as TF

    h, y = logits(), targets()
    assert torch.allclose(
        categorical.nll(h, y, reduction="mean"), TF.cross_entropy(h, y, reduction="mean")
    )


def test_reductions_are_consistent(categorical: Categorical) -> None:
    h, y = logits(), targets()
    per_sample = categorical.nll(h, y, reduction="none")
    assert per_sample.shape == (BATCH,)
    assert categorical.nll(h, y, reduction="sum") == pytest.approx(float(per_sample.sum()))
    assert categorical.nll(h, y, reduction="mean") == pytest.approx(float(per_sample.mean()))


def test_reduction_has_no_default(categorical: Categorical) -> None:
    """Mean-versus-sum is one of the four preconditions of the X0 identity. A
    loss that quietly reduces one way while the pooled learner reduces the other
    produces a residual that reads as a numerical problem rather than a bug."""
    import inspect

    signature = inspect.signature(categorical.nll)
    assert signature.parameters["reduction"].default is inspect.Parameter.empty


def test_nll_of_a_perfect_prediction_is_near_zero(categorical: Categorical) -> None:
    h = torch.full((1, Q), -60.0, dtype=DTYPE)
    h[0, 1] = 60.0
    assert float(categorical.nll(h, torch.tensor([1]), reduction="mean")) < 1e-12


def test_nll_of_a_uniform_prediction_is_log_q(categorical: Categorical) -> None:
    h = torch.zeros(1, Q, dtype=DTYPE)
    value = float(categorical.nll(h, torch.tensor([0]), reduction="mean"))
    assert value == pytest.approx(math.log(Q))


# =========================================================================== #
# 6. validation
# =========================================================================== #


def test_wrong_logit_width_is_rejected(categorical: Categorical) -> None:
    with pytest.raises(LikelihoodError, match=f"expected {Q} logits"):
        categorical.mu(torch.zeros(2, Q + 1, dtype=DTYPE))


def test_float_targets_are_rejected(categorical: Categorical) -> None:
    """One-hot or float targets mean a caller has built something the interface
    builds internally."""
    with pytest.raises(LikelihoodError, match="must be int64 class indices"):
        categorical.innovation(logits(), torch.zeros(BATCH, dtype=torch.float32))


def test_out_of_range_targets_are_rejected(categorical: Categorical) -> None:
    with pytest.raises(LikelihoodError, match=r"must lie in \[0, 5\]"):
        categorical.innovation(logits(), torch.full((BATCH,), 99))


def test_mismatched_target_shape_is_rejected(categorical: Categorical) -> None:
    with pytest.raises(LikelihoodError, match="do not match logits"):
        categorical.innovation(logits(), targets(BATCH + 1))


def test_too_few_classes_is_rejected() -> None:
    with pytest.raises(LikelihoodError, match="output_dim must be >= 2"):
        Categorical(output_dim=1)


def test_likelihood_is_frozen(categorical: Categorical) -> None:
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(categorical, "output_dim", 3)  # noqa: B010


# =========================================================================== #
# 7. the Gaussian case
# =========================================================================== #


def test_gaussian_mean_is_the_identity_link(gaussian: Gaussian) -> None:
    h = logits(BATCH, 3)
    assert torch.equal(gaussian.mu(h), h)


def test_gaussian_fisher_is_the_inverse_noise(gaussian: Gaussian) -> None:
    h = logits(BATCH, 3)
    product = gaussian.fisher(h) @ gaussian.noise_covariance(h)
    assert torch.allclose(product, torch.eye(3, dtype=DTYPE).expand_as(product), atol=1e-14)


def test_gaussian_fisher_is_full_rank(gaussian: Gaussian) -> None:
    """The reason this likelihood exists: it is the only one here that can
    exercise the filter's gain form, which needs an invertible Lambda."""
    ranks = torch.linalg.matrix_rank(gaussian.fisher(logits(BATCH, 3)))
    assert torch.all(ranks == 3)
    assert gaussian.fisher_rank == 3


def test_gaussian_factorisation_is_exact(gaussian: Gaussian) -> None:
    h = logits(BATCH, 3)
    factor = gaussian.fisher_factor(h)
    assert torch.allclose(gaussian.fisher(h), factor @ factor.transpose(-1, -2), atol=1e-14)


def test_gaussian_innovation_is_the_residual(gaussian: Gaussian) -> None:
    h, y = logits(BATCH, 3), logits(BATCH, 3, seed=1)
    assert torch.allclose(gaussian.innovation(h, y), y - h)


def test_gaussian_nll_is_the_negative_gradient_relation(gaussian: Gaussian) -> None:
    h = logits(BATCH, 3).clone().requires_grad_(True)
    y = logits(BATCH, 3, seed=1)
    gaussian.nll(h, y, reduction="sum").backward()
    assert h.grad is not None
    expected = -gaussian.innovation(h.detach(), y) / gaussian.variance
    assert torch.allclose(h.grad, expected, atol=1e-12)


def test_gaussian_nll_keeps_the_normaliser(gaussian: Gaussian) -> None:
    """An NLL missing the constant is not comparable across noise levels, and
    calibration reporting compares exactly that."""
    h = torch.zeros(1, 3, dtype=DTYPE)
    value = float(gaussian.nll(h, h, reduction="mean"))
    assert value == pytest.approx(0.5 * 3 * math.log(2 * math.pi * gaussian.variance))


def test_gaussian_rejects_a_non_positive_variance() -> None:
    with pytest.raises(LikelihoodError, match="variance must be > 0"):
        Gaussian(output_dim=2, variance=0.0)


def test_gaussian_rejects_mismatched_targets(gaussian: Gaussian) -> None:
    with pytest.raises(LikelihoodError, match="must match logits"):
        gaussian.innovation(logits(BATCH, 3), logits(BATCH, 2))


# =========================================================================== #
# 8. what the filter will actually compute
# =========================================================================== #


def test_the_information_increment_is_low_rank(categorical: Categorical) -> None:
    """Delta Omega = B B^T with B = H^T G in R^{p x q}. Rank q-1 per sample is
    what lets Woodbury replace a p x p inverse with a small one."""
    from dekf_bench.models.mlp import MLP

    model = MLP(input_size=6, hidden=(5,), output_dim=Q, dtype=DTYPE)
    params = model.init_params(torch.Generator().manual_seed(0))
    x = torch.rand(1, 1, 6, 6, generator=torch.Generator().manual_seed(1), dtype=DTYPE)

    jac = model.jacobian(params, x)[0]  # (q, p)
    factor = categorical.fisher_factor(model.forward(params, x))[0]  # (q, q)

    increment = jac.T @ categorical.fisher(model.forward(params, x))[0] @ jac
    b = jac.T @ factor
    assert torch.allclose(increment, b @ b.T, atol=1e-12)
    assert int(torch.linalg.matrix_rank(increment)) == Q - 1


def test_the_innovation_term_is_the_loss_gradient(categorical: Categorical) -> None:
    """H^T nu is the negative gradient of the network log-loss. This is the sense
    in which the EKF update is a preconditioned gradient step (research note
    Rem. 3), and it is the bridge to the SGD baselines."""
    from dekf_bench.models.mlp import MLP

    model = MLP(input_size=6, hidden=(5,), output_dim=Q, dtype=DTYPE)
    params = model.init_params(torch.Generator().manual_seed(0))
    x = torch.rand(1, 1, 6, 6, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    y = torch.tensor([2])

    nu = categorical.innovation(model.forward(params, x), y)
    innovation_term = model.vjp(params, x, nu)

    grad_params = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    categorical.nll(model.forward(grad_params, x), y, reduction="sum").backward()
    loss_gradient = model.flatten({k: v.grad for k, v in grad_params.items()})  # type: ignore[misc]

    assert torch.allclose(innovation_term, -loss_gradient, atol=1e-12)
