"""Turning a Gaussian belief over logits into a distribution over classes.

The load-bearing test is :func:`test_both_reduce_to_the_plugin_at_zero_spread`:
an uncertainty correction that does not vanish when there is no uncertainty is
wrong, and that limit is checkable exactly.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.metrics.classification import MetricError
from dekf_bench.metrics.predictive import (
    logit_variance,
    plugin_probabilities,
    probit_probabilities,
    sampled_probabilities,
)

BATCH, Q, DTYPE = 6, 4, torch.float64


def logits(seed: int = 0) -> torch.Tensor:
    return torch.randn(BATCH, Q, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)


def covariance(scale: float = 1.0, seed: int = 1) -> torch.Tensor:
    root = torch.randn(BATCH, Q, Q, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)
    return scale * (root @ root.transpose(-1, -2) + Q * torch.eye(Q, dtype=DTYPE))


# -- the limit that pins both approximations -------------------------------- #


def test_both_reduce_to_the_plugin_at_zero_spread() -> None:
    """With no uncertainty there is nothing to correct for.

    Exact for both, and the sampling path is where that is not free: a zero
    covariance is positive *semi*-definite, so a Cholesky-plus-jitter
    implementation would perturb the logits by the square root of its own jitter
    and miss the limit by ~1e-6. The eigendecomposition gets it exactly.
    """
    h = logits()
    plugin = plugin_probabilities(h)
    zero_diag = torch.zeros(BATCH, Q, dtype=DTYPE)
    zero_full = torch.zeros(BATCH, Q, Q, dtype=DTYPE)

    assert torch.allclose(probit_probabilities(h, zero_diag), plugin, atol=1e-12)
    sampled = sampled_probabilities(h, zero_full, samples=4, generator=torch.Generator())
    assert torch.allclose(sampled, plugin, atol=1e-12)


def test_a_singular_belief_is_handled_rather_than_rescued() -> None:
    """A rank-deficient $\\bm\\Sigma$ is legitimate, not an error to jitter away.

    $\\bm H\\bm P\\bm H^{\\mathsf T}$ is positive semi-definite; nothing
    guarantees $\\bm H$'s rows are independent. Sampling must respect a direction
    of exactly zero variance rather than inventing spread there.
    """
    h = logits()
    direction = torch.zeros(BATCH, Q, Q, dtype=DTYPE)
    direction[:, 0, 0] = 4.0  # variance in one logit only
    sampled = sampled_probabilities(
        h, direction, samples=2000, generator=torch.Generator().manual_seed(0)
    )
    assert torch.all(sampled >= 0)
    assert torch.allclose(sampled.sum(dim=-1), torch.ones(BATCH, dtype=DTYPE))


# -- both are distributions ------------------------------------------------- #


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_outputs_are_probability_vectors(scale: float) -> None:
    h, sigma = logits(), covariance(scale)
    for probabilities in (
        probit_probabilities(h, logit_variance(sigma)),
        sampled_probabilities(h, sigma, samples=64, generator=torch.Generator().manual_seed(0)),
    ):
        assert probabilities.shape == (BATCH, Q)
        assert torch.all(probabilities >= 0)
        assert torch.allclose(probabilities.sum(dim=-1), torch.ones(BATCH, dtype=DTYPE))


# -- uncertainty softens, and only softens ---------------------------------- #


def test_uncertainty_moves_mass_toward_uniform() -> None:
    """More spread means a less confident prediction, monotonically.

    Fixed by construction for the probit form -- it divides every logit by a
    factor that grows with the variance -- so this is a test rather than a
    measurement, and it is what makes 'the filter is better calibrated' a
    non-trivial claim rather than an arithmetic consequence.
    """
    h = logits()
    previous = float(plugin_probabilities(h).max(dim=-1).values.mean())
    for scale in (0.1, 1.0, 10.0, 100.0):
        variance = scale * torch.ones(BATCH, Q, dtype=DTYPE)
        confidence = float(probit_probabilities(h, variance).max(dim=-1).values.mean())
        assert confidence < previous, f"confidence did not fall at scale {scale}"
        previous = confidence
    assert previous == pytest.approx(1.0 / Q, abs=0.05), "huge spread should approach uniform"


def test_the_argmax_survives_isotropic_uncertainty() -> None:
    """Equal variance on every logit rescales them all, so the ranking holds.

    Which is the honest limit of what the probit form can do: it cannot change
    the decision unless the uncertainty is anisotropic, so any accuracy
    difference it produces comes from *unequal* per-class spread.
    """
    h = logits()
    variance = 3.0 * torch.ones(BATCH, Q, dtype=DTYPE)
    assert torch.equal(
        probit_probabilities(h, variance).argmax(dim=-1), plugin_probabilities(h).argmax(dim=-1)
    )


# -- the two approximations against each other ------------------------------ #


def test_the_probit_error_grows_with_the_spread_and_is_bounded() -> None:
    r"""How wrong the closed form is, measured against sampling (design note D63).

    The gap is **approximation error, not sampling noise** -- at $S=200{,}000$
    two seeds differ by $\le0.002$ while the distance to the probit form runs
    from 0.002 at $\sigma^2\approx0.05$ to 0.042 at $\sigma^2\approx16$. So the
    bound below is a measured property of the approximation, and the test says
    what it is rather than merely that the two are "close".

    Bounded above because both forms tend to uniform as the spread grows, so the
    error saturates instead of diverging. That is what makes the probit form
    usable as the headline metric with sampling as the check.
    """
    h = logits()
    for scale, tolerance in ((0.25, 0.015), (1.0, 0.030), (16.0, 0.050)):
        variance = scale * (
            torch.rand(BATCH, Q, generator=torch.Generator().manual_seed(5), dtype=DTYPE) + 0.5
        )
        sampled = sampled_probabilities(
            h, torch.diag_embed(variance), samples=50_000,
            generator=torch.Generator().manual_seed(7),
        )
        gap = float((sampled - probit_probabilities(h, variance)).abs().max())
        assert gap < tolerance, f"probit error {gap:.4f} exceeds {tolerance} at scale {scale}"


def test_sampling_is_reproducible_from_its_generator() -> None:
    h, sigma = logits(), covariance()
    first = sampled_probabilities(h, sigma, samples=32, generator=torch.Generator().manual_seed(2))
    second = sampled_probabilities(h, sigma, samples=32, generator=torch.Generator().manual_seed(2))
    assert torch.equal(first, second)


def test_sampling_sees_correlations_the_probit_form_cannot() -> None:
    """The distinction that justifies paying for Monte Carlo (design note D63).

    Two beliefs with identical marginals and different correlation structure are
    the same input to the probit form and different inputs to sampling.
    """
    h = logits()
    variance = torch.ones(BATCH, Q, dtype=DTYPE)
    independent = torch.diag_embed(variance)
    correlated = independent.clone()
    correlated[:, 0, 1] = correlated[:, 1, 0] = 0.9

    generator = torch.Generator().manual_seed(11)
    apart = sampled_probabilities(h, independent, samples=40_000, generator=generator)
    generator = torch.Generator().manual_seed(11)
    together = sampled_probabilities(h, correlated, samples=40_000, generator=generator)

    assert torch.allclose(
        logit_variance(independent), logit_variance(correlated)
    ), "the two beliefs must have identical marginals for this to mean anything"
    assert float((apart - together).abs().max()) > 1e-3


# -- malformed input is refused --------------------------------------------- #


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(MetricError, match="to match logits"):
        probit_probabilities(logits(), torch.zeros(BATCH, Q + 1, dtype=DTYPE))
    with pytest.raises(MetricError, match="to match logits"):
        sampled_probabilities(logits(), torch.zeros(BATCH, Q, dtype=DTYPE))


def test_negative_variance_is_refused() -> None:
    with pytest.raises(MetricError, match="non-negative"):
        probit_probabilities(logits(), -torch.ones(BATCH, Q, dtype=DTYPE))


def test_zero_samples_is_refused() -> None:
    with pytest.raises(MetricError, match="samples must be >= 1"):
        sampled_probabilities(logits(), covariance(), samples=0)
