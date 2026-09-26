r"""Scoring the belief itself, and the covariance scale $\kappa^\star$ (P5.11, P5.14).

Every calibration number logged before this module was **plug-in**: softmax of the
belief's *mean*, which any point estimator has (D80's warning). The filter's
distinctive claim is that its covariance $\bm P$ knows what the mean does not, and
that claim is scored here, through the predictive distribution

$$p(y\mid\bm x)=\int\operatorname{softmax}(\bm h)\,
  \mathcal N\bigl(\bm h;\,\bm h(\bm m),\,\kappa\,\bm H\bm P\bm H^{\mathsf T}\bigr)\,d\bm h .$$

At $\kappa=1$ this is the belief as the filter reports it. At $\kappa=0$ it is the
plug-in prediction. **$\kappa^\star$ is the scale that minimises held-out NLL**:
the multiplier the covariance would need for its predictions to be as good as they
can be. So it reads directly as a verdict on $\bm P$:

* $\kappa^\star\approx1$ -- the reported covariance is the right size;
* $\kappa^\star<1$ -- $\bm P$ is too large: the belief is conservative;
* $\kappa^\star>1$ -- $\bm P$ is too small: the belief is over-confident.

$\kappa^\star=0$ is a legitimate answer, not a failure: it says *any* spread makes
the predictions worse, which is what an already under-confident mean produces
(D80). It sits on the grid's lower edge and is flagged as such.

**The scan uses the probit approximation**, which needs only the diagonal of
$\bm H\bm P\bm H^{\mathsf T}$ and makes a 160-point scan free. The Monte Carlo
predictive over the full covariance is scored at $\kappa=1$ beside it, so the
distortion of the cheap form is measured rather than inherited (D63).

For a Gaussian likelihood (Mackey--Glass) the same scan runs on the exact
predictive variance $\kappa\,\bm H\bm P\bm H^{\mathsf T}+\bm R$, with no
approximation at all; see :func:`gaussian_kappa_scan`.

**The tempered $\kappa^\star$.** A covariance can only soften the prediction, so
when the plug-in mean is itself *under*-confident -- every filter here is, by
0.016 to 0.040 (D80, and the X20 and N>10 cells) -- no positive scale helps and
$\kappa^\star$ sits at 0 whatever $\bm P$ is. That says the mean is miscalibrated,
not how large $\bm P$ is. So a second reading is taken: first the temperature
$\tau^\star$ that best calibrates the plug-in mean, $\operatorname{softmax}(\bm h/\tau)$,
then the $\kappa$ scan on the tempered predictive, logits $\bm h/\tau^\star$ with
variance $\kappa\,\bm\sigma^2/\tau^{\star2}$. With the mean's global
miscalibration removed, this asks whether the *per-input* spread of $\bm P$ is the
right size. It is not the pure scale of $\bm P$ -- $\tau^\star$ absorbs part of
the uncertainty -- which is why the raw $\kappa^\star$ stays the confirmatory
measure and this one is exploratory (decided 2026-09-27, D120).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from dekf_bench.metrics import calibration
from dekf_bench.metrics.classification import MetricError
from dekf_bench.metrics.predictive import (
    logit_variance,
    plugin_probabilities,
    probit_probabilities,
    sampled_probabilities,
)

#: Zero (the plug-in prediction) and 161 points at 0.05-decade spacing over
#: [1e-4, 1e4]. Wide on purpose: D80 found the diffusion filter's belief several
#: times too wide, and a grid that stopped at 1e-1 would report its edge as the
#: answer.
KAPPA_GRID: tuple[float, ...] = (0.0, *(10.0 ** (k / 20.0) for k in range(-80, 81)))

#: Temperatures for the tempered reading: 0.025-decade steps over [0.25, 4], so a
#: mean may be sharpened or softened by up to 4x. Holds 1 (no tempering).
TEMPERATURE_GRID: tuple[float, ...] = tuple(10.0 ** (k / 40.0) for k in range(-24, 25))

#: Probabilities are clamped here before the log, so one confidently wrong sample
#: cannot make the NLL infinite and swamp the scan.
_FLOOR = 1e-12


@dataclass(frozen=True)
class KappaScan:
    """Where the NLL is lowest along the covariance scale, and what it costs there."""

    kappa_star: float
    nll_at_star: float
    nll_at_one: float
    nll_at_zero: float
    #: True when the minimum sits on the grid's first or last point, so the true
    #: optimum may lie beyond it. Zero is a real value, but still an edge.
    at_edge: bool


def kappa_scan(nll_at: Callable[[float], float],
               grid: tuple[float, ...] = KAPPA_GRID) -> KappaScan:
    """Minimise ``nll_at(kappa)`` over ``grid``, which must hold 0 and 1."""
    if 0.0 not in grid or 1.0 not in grid:
        raise MetricError("the kappa grid must contain 0 (plug-in) and 1 (the belief)")
    values = [nll_at(kappa) for kappa in grid]
    if not all(math.isfinite(value) for value in values):
        raise MetricError("the NLL is not finite somewhere on the kappa grid")
    best = min(range(len(grid)), key=values.__getitem__)
    return KappaScan(
        kappa_star=grid[best],
        nll_at_star=values[best],
        nll_at_one=values[grid.index(1.0)],
        nll_at_zero=values[grid.index(0.0)],
        at_edge=best in (0, len(grid) - 1),
    )


def categorical_nll(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean negative log probability of the true class."""
    picked = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(-picked.clamp_min(_FLOOR).log().mean())


def classification_scores(
    logits: torch.Tensor,
    covariance: torch.Tensor,
    targets: torch.Tensor,
    mc_samples: int = 256,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    r"""Plug-in, probit and Monte Carlo scores on one batch, and the $\kappa$ scan.

    Args:
        logits: $\bm h(\bm m)$, shape ``(n, q)``.
        covariance: $\bm H\bm P\bm H^{\mathsf T}$, shape ``(n, q, q)``.
        targets: class indices, shape ``(n,)``.
    """
    variance = logit_variance(covariance)
    plugin = plugin_probabilities(logits)
    probit = probit_probabilities(logits, variance)
    sampled = sampled_probabilities(logits, covariance, mc_samples, generator)

    def scored(probabilities: torch.Tensor) -> tuple[float, calibration.Reliability]:
        return (categorical_nll(probabilities, targets),
                calibration.reliability(probabilities, targets))

    plugin_nll, plugin_curve = scored(plugin)
    probit_nll, probit_curve = scored(probit)
    sampled_nll, sampled_curve = scored(sampled)
    scan = kappa_scan(lambda kappa: categorical_nll(
        probit_probabilities(logits, kappa * variance), targets))
    temperature = temperature_star(logits, targets)
    tempered = kappa_scan(lambda kappa: categorical_nll(
        probit_probabilities(logits / temperature, kappa * variance / temperature**2), targets))
    return {
        "plugin_nll": plugin_nll,
        "plugin_ece": plugin_curve.ece,
        "plugin_overconfidence": plugin_curve.overconfidence,
        "belief_nll": probit_nll,
        "belief_brier": calibration.brier(probit, targets),
        "belief_ece": probit_curve.ece,
        "belief_overconfidence": probit_curve.overconfidence,
        "belief_mean_confidence": float(probit.max(dim=-1).values.mean()),
        "belief_nll_mc": sampled_nll,
        "belief_ece_mc": sampled_curve.ece,
        "kappa_star": scan.kappa_star,
        "nll_at_kappa_star": scan.nll_at_star,
        "kappa_star_at_edge": float(scan.at_edge),
        "temperature_star": temperature,
        "tempered_kappa_star": tempered.kappa_star,
        "tempered_kappa_star_at_edge": float(tempered.at_edge),
        "nll_at_tempered_kappa_star": tempered.nll_at_star,
        "logit_variance_mean": float(variance.mean()),
    }


def temperature_star(logits: torch.Tensor, targets: torch.Tensor,
                     grid: tuple[float, ...] = TEMPERATURE_GRID) -> float:
    r"""The $\tau$ minimising the NLL of $\operatorname{softmax}(\bm h/\tau)$: above 1
    softens an over-confident mean, below 1 sharpens an under-confident one."""
    return min(grid, key=lambda tau: categorical_nll(plugin_probabilities(logits / tau),
                                                     targets))


def gaussian_nll(residuals: torch.Tensor, variance: torch.Tensor) -> float:
    """Mean Gaussian negative log-likelihood of residuals under ``variance``."""
    return float(0.5 * (torch.log(2.0 * math.pi * variance) + residuals**2 / variance).mean())


def gaussian_kappa_scan(
    residuals: torch.Tensor, model_variance: torch.Tensor, noise_variance: torch.Tensor
) -> KappaScan:
    r"""The same scan on the exact Gaussian predictive $\kappa\,\bm H\bm P\bm H^{\mathsf T}+\bm R$.

    ``noise_variance`` must be strictly positive: at $\kappa=0$ it is the whole
    predictive variance, and a zero there would make the plug-in NLL infinite.
    """
    if bool((noise_variance <= 0).any()):
        raise MetricError("the observation variance R must be strictly positive")
    if bool((model_variance < 0).any()):
        raise MetricError("the model's predictive variance must be non-negative")
    return kappa_scan(lambda kappa: gaussian_nll(
        residuals, kappa * model_variance + noise_variance))
