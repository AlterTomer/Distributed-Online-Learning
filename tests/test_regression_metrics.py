"""Regression scores: exact on constructed cases, nominal on calibrated draws."""

from __future__ import annotations

import math

import pytest
import torch

from dekf_bench.metrics import regression as R

DTYPE = torch.float64


def test_rmse_on_a_known_case() -> None:
    predictions = torch.zeros(2, 3, dtype=DTYPE)
    targets = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=DTYPE)
    assert R.mse(predictions, targets) == pytest.approx(5.0)
    assert R.rmse(predictions, targets) == pytest.approx(math.sqrt(5.0))
    assert R.rmse_from(predictions, targets, 2) == pytest.approx(math.sqrt(5.0))


def test_full_context_rmse_ignores_the_early_positions() -> None:
    targets = torch.zeros(1, 31, dtype=DTYPE)
    predictions = torch.zeros(1, 31, dtype=DTYPE)
    predictions[0, :16] = 10.0  # short-context positions badly wrong
    assert R.rmse_from(predictions, targets, R.FULL_CONTEXT_FROM) == 0.0


@pytest.fixture(scope="module")
def calibrated():
    generator = torch.Generator().manual_seed(0)
    variance = torch.rand(20_000, 31, dtype=DTYPE, generator=generator) + 0.1
    residuals = torch.randn(20_000, 31, dtype=DTYPE, generator=generator) * variance.sqrt()
    return residuals, variance


def test_calibrated_draws_give_nominal_coverage(calibrated) -> None:
    residuals, variance = calibrated
    for level in (50, 90, 95):
        assert R.coverage(residuals, variance, level) == pytest.approx(level / 100, abs=0.005)


def test_calibrated_draws_give_unit_variance_ratio(calibrated) -> None:
    assert R.variance_ratio(*calibrated) == pytest.approx(1.0, abs=0.01)


def test_calibrated_draws_give_a_flat_pit(calibrated) -> None:
    histogram = R.pit_histogram(*calibrated)
    assert len(histogram) == R.PIT_BINS
    assert sum(histogram) == pytest.approx(1.0)
    assert max(abs(f - 0.1) for f in histogram) < 0.005


def test_over_confidence_raises_the_ratio_and_hollows_the_pit(calibrated) -> None:
    """A variance four times too small: ratio ~4, and mass piles into the end bins."""
    residuals, variance = calibrated
    assert R.variance_ratio(residuals, variance / 4) == pytest.approx(4.0, rel=0.02)
    histogram = R.pit_histogram(residuals, variance / 4)
    assert histogram[0] > 0.2 and histogram[-1] > 0.2


def test_the_nll_matches_the_formula() -> None:
    residuals = torch.tensor([[0.0, 1.0]], dtype=DTYPE)
    variance = torch.tensor([[1.0, 4.0]], dtype=DTYPE)
    expected = (0.5 * math.log(2 * math.pi) + 0.5 * math.log(8 * math.pi) + 0.125) / 2
    assert R.gaussian_nll(residuals, variance) == pytest.approx(expected)


def test_predictive_scores_name_every_metric(calibrated) -> None:
    scores = R.predictive_scores(*calibrated)
    assert {"predictive_nll", "variance_ratio", "coverage_50", "coverage_90", "coverage_95"} <= set(
        scores
    )
    assert [f"pit_{k}" for k in range(10)] == [k for k in scores if k.startswith("pit_")]


@pytest.mark.parametrize(
    ("variance", "match"),
    [(torch.ones(2, 2), "must match"), (torch.zeros(2, 3), "> 0")],
)
def test_bad_variances_are_refused(variance, match) -> None:
    with pytest.raises(R.RegressionMetricError, match=match):
        R.variance_ratio(torch.zeros(2, 3), variance)
