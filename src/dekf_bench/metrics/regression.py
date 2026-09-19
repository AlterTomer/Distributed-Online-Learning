r"""One-step regression scores: accuracy, and whether the stated uncertainty is right.

`docs/mackey_glass_plan.md`, decisions 15 and 18. Accuracy is RMSE in standardised
units, over every position of a block -- what the filter trains on -- and over the
positions with a whole delay of context. Calibration applies only to learners that
hold a covariance: their predictive variance is $\operatorname{diag}(\bm H\bm P\bm
H^{\mathsf T})+\operatorname{diag}(\bm R)$, a Gaussian per value, and it is scored
against the realised residual four ways:

* **predictive NLL** per value, keeping the normaliser so noise levels compare;
* **coverage** of the central 50/90/95% intervals -- nominal when calibrated;
* the **variance ratio** $\mathbb E[r^2/s^2]$ -- 1 when calibrated, above 1 when
  the filter is over-confident, which is the failure correlated innovations
  would produce (Mackey--Glass plan, risk 2);
* the **PIT histogram**, $\Phi(r/s)$ in ten bins -- uniform when calibrated, and
  the one of the four that shows *how* a miscalibration is shaped.
"""

from __future__ import annotations

import math

import torch

#: Two-sided standard-normal quantiles of the reported central intervals.
COVERAGE_LEVELS: dict[int, float] = {
    50: 0.6744897501960817,
    90: 1.6448536269514722,
    95: 1.959963984540054,
}
PIT_BINS = 10
#: Position 16 (0-based) is the first to predict from tau = 17 samples of context.
FULL_CONTEXT_FROM = 16


class RegressionMetricError(ValueError):
    """Raised for mismatched shapes or a non-positive variance."""


def _check(residuals: torch.Tensor, variance: torch.Tensor | None = None) -> None:
    if residuals.numel() == 0:
        raise RegressionMetricError("nothing to score")
    if variance is not None:
        if variance.shape != residuals.shape:
            raise RegressionMetricError(
                f"variance {tuple(variance.shape)} must match residuals {tuple(residuals.shape)}"
            )
        if bool((variance <= 0).any()):
            raise RegressionMetricError("predictive variance must be > 0 everywhere")


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    residuals = targets - predictions
    _check(residuals)
    return float((residuals**2).mean())


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return math.sqrt(mse(predictions, targets))


def rmse_from(predictions: torch.Tensor, targets: torch.Tensor, start: int) -> float:
    """RMSE over positions ``start`` onward, the last axis being position."""
    return rmse(predictions[..., start:], targets[..., start:])


def gaussian_nll(residuals: torch.Tensor, variance: torch.Tensor) -> float:
    """Mean per-value NLL of ``residuals`` under zero-mean Gaussians of ``variance``."""
    _check(residuals, variance)
    return float((0.5 * torch.log(2.0 * math.pi * variance) + 0.5 * residuals**2 / variance).mean())


def coverage(residuals: torch.Tensor, variance: torch.Tensor, level: int) -> float:
    """Fraction of values inside the central ``level``% interval."""
    _check(residuals, variance)
    if level not in COVERAGE_LEVELS:
        raise RegressionMetricError(f"level must be one of {sorted(COVERAGE_LEVELS)}")
    return float((residuals.abs() <= COVERAGE_LEVELS[level] * variance.sqrt()).double().mean())


def variance_ratio(residuals: torch.Tensor, variance: torch.Tensor) -> float:
    """$\\mathbb E[r^2/s^2]$: 1 when calibrated, above 1 when over-confident."""
    _check(residuals, variance)
    return float((residuals**2 / variance).mean())


def pit_histogram(
    residuals: torch.Tensor, variance: torch.Tensor, bins: int = PIT_BINS
) -> list[float]:
    """Fractions of $\\Phi(r/s)$ in ``bins`` equal bins: uniform when calibrated."""
    _check(residuals, variance)
    pit = 0.5 * (1.0 + torch.erf(residuals / variance.sqrt() / math.sqrt(2.0)))
    index = torch.clamp((pit * bins).long(), max=bins - 1)
    counts = torch.bincount(index.reshape(-1), minlength=bins).double()
    return (counts / counts.sum()).tolist()


def predictive_scores(residuals: torch.Tensor, variance: torch.Tensor) -> dict[str, float]:
    """Every calibration quantity, keyed by the metric name the schema records."""
    scores = {
        "predictive_nll": gaussian_nll(residuals, variance),
        "variance_ratio": variance_ratio(residuals, variance),
        **{f"coverage_{level}": coverage(residuals, variance, level) for level in COVERAGE_LEVELS},
    }
    for index, fraction in enumerate(pit_histogram(residuals, variance)):
        scores[f"pit_{index}"] = fraction
    return scores
