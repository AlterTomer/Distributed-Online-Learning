r"""From a Gaussian belief over logits to a distribution over classes.

The filter's posterior is Gaussian in **parameter** space, and pushing it through
the network's linearisation gives a Gaussian in **logit** space,
$\bm h\sim\mathcal N(\bm h(\bm m),\,\bm H\bm P\bm H^{\mathsf T})$. Neither is a
distribution over classes. Getting one means integrating

$$\bm\pi = \int \operatorname{softmax}(\bm h)\,
  \mathcal N(\bm h;\bm h(\bm m),\bm\Sigma)\,\mathrm d\bm h,$$

which has no closed form. Two approximations are implemented, and **both are
reported**, because the gap between them is the only direct evidence of how much
the cheap one distorts (design note D63).

:func:`probit_probabilities` -- the MacKay/probit approximation, closed form and
free, using only $\operatorname{diag}\bm\Sigma$.

:func:`sampled_probabilities` -- Monte Carlo over the full $\bm\Sigma$, which
keeps the off-diagonal correlations the probit form discards, at the cost of $S$
softmaxes and a sampling seed.

**The plug-in baseline is $\operatorname{softmax}(\bm h(\bm m))$**, which is what
every SGD learner reports and what both of these reduce to as
$\bm\Sigma\to\bm0$. That limit is worth testing: an uncertainty correction that
does not vanish when there is no uncertainty is wrong.

Both approximations only ever **soften** the prediction -- they move mass toward
uniform, never away from it -- so neither can fix an over-confident *mean*. The
research note is explicit that the delta-method covariance is systematically
over-confident to begin with, so a calibration improvement here is a real
finding and its absence is not a bug.
"""

from __future__ import annotations

import math

import torch

from dekf_bench.metrics.classification import MetricError

#: The probit approximation's constant. $\kappa(\sigma^2)=(1+\pi\sigma^2/8)^{-1/2}$
#: comes from matching the logistic function to a probit and integrating exactly;
#: the multiclass form applies it per logit.
_PROBIT_SCALE = math.pi / 8.0



def _check(logits: torch.Tensor, spread: torch.Tensor, square: bool) -> None:
    if logits.ndim != 2:
        raise MetricError(f"expected (n, q) logits, got shape {tuple(logits.shape)}")
    batch, outputs = logits.shape
    expected = (batch, outputs, outputs) if square else (batch, outputs)
    if tuple(spread.shape) != expected:
        raise MetricError(
            f"expected {expected} to match logits {tuple(logits.shape)}, "
            f"got {tuple(spread.shape)}"
        )
    if bool((spread.diagonal(dim1=-2, dim2=-1) if square else spread).lt(0).any()):
        raise MetricError("logit variances must be non-negative")


def logit_variance(covariance: torch.Tensor) -> torch.Tensor:
    """The per-logit marginal variances, shape ``(n, q)``."""
    return covariance.diagonal(dim1=-2, dim2=-1)


def probit_probabilities(logits: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    r"""$\operatorname{softmax}\bigl(\bm h/\sqrt{1+\tfrac{\pi}{8}\bm\sigma^2}\bigr)$.

    Each logit is shrunk toward zero in proportion to its own uncertainty, which
    moves the softmax toward uniform. Exact for the binary case by construction,
    and the standard multiclass extension otherwise.

    **Only the diagonal is used**, so two beliefs with the same marginals and
    different correlations are indistinguishable here. That is the approximation
    :func:`sampled_probabilities` exists to measure rather than to inherit.

    Args:
        logits: $\bm h(\bm m)$, shape ``(n, q)``.
        variance: $\operatorname{diag}\bm H\bm P\bm H^{\mathsf T}$, shape ``(n, q)``.
    """
    _check(logits, variance, square=False)
    return torch.softmax(logits / torch.sqrt(1.0 + _PROBIT_SCALE * variance), dim=-1)


def sampled_probabilities(
    logits: torch.Tensor,
    covariance: torch.Tensor,
    samples: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""$\frac1S\sum_s\operatorname{softmax}(\bm h_s)$, $\bm h_s\sim\mathcal N(\bm h,\bm\Sigma)$.

    The full covariance enters through its Cholesky factor, so correlated logits
    stay correlated across a draw -- the thing the probit form cannot represent.

    The estimator is unbiased with standard error $O(S^{-1/2})$, which at the
    default $S=256$ is about 3% of a probability. That is comparable to the
    effect being measured, so this is a **check on the probit approximation
    rather than a headline metric**, and the generator is threaded explicitly so
    the check is reproducible (design note D63).

    Args:
        logits: $\bm h(\bm m)$, shape ``(n, q)``.
        covariance: $\bm H\bm P\bm H^{\mathsf T}$, shape ``(n, q, q)``.
        samples: $S$.
        generator: required for a reproducible run; ``None`` draws from the
            global RNG and makes the metric depend on whatever drew before it.
    """
    _check(logits, covariance, square=True)
    if samples < 1:
        raise MetricError(f"samples must be >= 1, got {samples}")

    # A symmetric eigendecomposition rather than a Cholesky, in one path with no
    # fallback. Cholesky needs strict positive definiteness, and $\bm H\bm P\bm
    # H^{\mathsf T}$ is only guaranteed positive *semi*-definite -- $\bm H$'s
    # rows need not be independent. Jittering the diagonal to rescue it would
    # inject spread that is not in the belief, and at an exactly-zero covariance
    # that shows up as a prediction that is not quite the plug-in one. This is
    # exact for singular and zero inputs alike, and q = 10 makes it free.
    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    factor = vectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(-2)

    batch, outputs = logits.shape
    noise = torch.randn(
        samples, batch, outputs, dtype=logits.dtype, device=logits.device, generator=generator
    )
    # (n, q, q) @ (S, n, q, 1) -> logits perturbed by L z, correlations intact.
    perturbed = logits + torch.einsum("nqr,snr->snq", factor, noise)
    return torch.softmax(perturbed, dim=-1).mean(dim=0)


def plugin_probabilities(logits: torch.Tensor) -> torch.Tensor:
    r"""$\operatorname{softmax}(\bm h(\bm m))$ -- the belief's mean, ignoring its spread.

    What every SGD learner reports, and the $\bm\Sigma\to\bm0$ limit of both
    approximations above. Present so the comparison is against a named baseline
    rather than an implicit one.
    """
    if logits.ndim != 2:
        raise MetricError(f"expected (n, q) logits, got shape {tuple(logits.shape)}")
    return torch.softmax(logits, dim=-1)
