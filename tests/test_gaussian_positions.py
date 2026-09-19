"""Per-position R for the series task: the diagonal Gaussian path.

The isotropic path is untouched and covered by `test_likelihoods.py`. These check
that the per-position diagonal agrees with it when the diagonal is constant, and is
internally consistent when it is not -- the score the filter uses must be the
derivative of the NLL the reports quote (D60).
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.likelihoods.base import LikelihoodError
from dekf_bench.likelihoods.gaussian import Gaussian

DTYPE = torch.float64
Q = 5


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(3, Q, dtype=DTYPE, generator=generator)
    targets = torch.randn(3, Q, dtype=DTYPE, generator=generator)
    return logits, targets


def test_a_constant_diagonal_is_the_isotropic_likelihood(batch) -> None:
    logits, targets = batch
    iso = Gaussian(output_dim=Q, variance=0.3)
    diag = Gaussian(output_dim=Q, variance=1.0, variances=(0.3,) * Q)
    assert torch.allclose(iso.fisher(logits), diag.fisher(logits))
    assert torch.allclose(iso.fisher_factor(logits), diag.fisher_factor(logits))
    assert torch.allclose(iso.noise_covariance(logits), diag.noise_covariance(logits))
    assert torch.allclose(iso.score(logits, targets), diag.score(logits, targets))
    assert torch.allclose(
        iso.nll(logits, targets, reduction="sum"), diag.nll(logits, targets, reduction="sum")
    )


def test_the_score_is_the_derivative_of_the_nll(batch) -> None:
    logits, targets = batch
    likelihood = Gaussian(output_dim=Q, variances=(0.1, 0.2, 0.5, 1.0, 2.0))
    h = logits.clone().requires_grad_(True)
    likelihood.nll(h, targets, reduction="sum").backward()
    assert torch.allclose(-h.grad, likelihood.score(logits, targets), atol=1e-12)


def test_the_factor_squares_to_the_fisher(batch) -> None:
    logits, _ = batch
    likelihood = Gaussian(output_dim=Q, variances=(0.1, 0.2, 0.5, 1.0, 2.0))
    factor = likelihood.fisher_factor(logits)
    assert torch.allclose(factor @ factor.transpose(-1, -2), likelihood.fisher(logits))
    assert torch.allclose(
        likelihood.fisher(logits)[0] @ likelihood.noise_covariance(logits)[0],
        torch.eye(Q, dtype=DTYPE),
    )


def test_the_nll_keeps_the_normaliser(batch) -> None:
    """Comparable across noise levels: the log-determinant term is per position."""
    logits, _ = batch
    likelihood = Gaussian(output_dim=Q, variances=(0.1, 0.2, 0.5, 1.0, 2.0))
    at_mean = likelihood.nll(logits, logits, reduction="none")
    expected = 0.5 * torch.log(2 * torch.pi * torch.tensor([0.1, 0.2, 0.5, 1.0, 2.0])).sum()
    assert torch.allclose(at_mean, expected.to(DTYPE).expand(3))


@pytest.mark.parametrize(
    ("variances", "match"), [((0.1, 0.2), "entries"), ((0.1, 0.2, -1.0, 1.0, 1.0), "> 0")]
)
def test_a_malformed_diagonal_is_refused(variances, match) -> None:
    with pytest.raises(LikelihoodError, match=match):
        Gaussian(output_dim=Q, variances=variances)


def test_the_registry_passes_the_diagonal_through() -> None:
    from dekf_bench.likelihoods.registry import build_likelihood
    from dekf_bench.utils.config import load_config

    variances = [0.02 + 0.001 * i for i in range(31)]
    config = load_config(
        "x1_stationary",
        overrides={
            "run": {"name": "probe", "horizon": 10, "seeds": [0]},
            "env": {"dataset": "mackey_glass"},
            "model": {"name": "causal_transformer", "likelihood": "gaussian", "output_dim": 31,
                      "observation_variances": variances},
        },
    )
    likelihood = build_likelihood(config)
    assert likelihood.variances == pytest.approx(tuple(variances))
