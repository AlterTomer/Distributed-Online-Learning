"""How far apart the agents are, and how far they are from centralized.

$$E_{\\mathrm{agree}}(t) = \\frac1N\\sum_v \\lVert\\bm\\theta_{v,t}-\\bar{\\bm\\theta}_t\\rVert^2,
\\qquad
E_{\\mathrm{cent}}(t) = \\frac1N\\sum_v \\lVert\\bm\\theta_{v,t}-\\bm\\theta^{\\mathrm C}_t\\rVert^2$$

$E_{\\mathrm{agree}}$ is the direct check on whether the combine step is doing its
job. $E_{\\mathrm{cent}}$ is the phase-1 analogue of the quantity the Diff-EKF will
ultimately be judged on.

**The centralized reference shares $\\bm\\theta_0$ and runs its own trajectory.**
The research note asks for an "independently initialised" centralized run; that
is read here as *running its own trajectory from $t=0$* rather than *drawing a
different $\\bm\\theta_0$* (design note D30). With a different initialization,
$E_{\\mathrm{cent}}$ would carry an irreducible floor that never vanishes even for
a perfect method, and there would be no value of the metric meaning "these
coincide". Sharing $\\bm\\theta_0$ makes $E_{\\mathrm{cent}}(0) = 0$ exactly, so
the metric measures algorithmic divergence and nothing else. It is also what
WORKPLAN §4.5 mandates for every learner in a run.

**Both are absolute, not normalised.** They are squared distances in parameter
space and therefore scale with $p$ and with the weight magnitudes. The norm
$\\lVert\\bar{\\bm\\theta}_t\\rVert^2$ is logged alongside so any normalisation --
per-parameter, or relative to the mean's own size -- can be derived at plot time
without a re-run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class DisagreementError(ValueError):
    """Raised when parameter vectors cannot be compared."""


def _stack(parameters: dict[int, torch.Tensor]) -> torch.Tensor:
    if not parameters:
        raise DisagreementError("need at least one agent")
    vectors = [parameters[node] for node in sorted(parameters)]
    shapes = {tuple(vector.shape) for vector in vectors}
    if len(shapes) != 1:
        raise DisagreementError(f"agents hold different parameter shapes: {shapes}")
    if vectors[0].ndim != 1:
        raise DisagreementError(
            f"expected flat parameter vectors, got shape {tuple(vectors[0].shape)}"
        )
    return torch.stack(vectors).to(torch.float64)


def network_mean(parameters: dict[int, torch.Tensor]) -> torch.Tensor:
    """$\\bar{\\bm\\theta}_t$, the average estimate across agents."""
    return _stack(parameters).mean(dim=0)


def e_agree(parameters: dict[int, torch.Tensor]) -> float:
    """Mean squared distance from the network mean.

    Zero exactly when every agent holds the same vector, which is what a
    complete graph produces after one combine step -- and is the reason X0 holds
    inductively.
    """
    stacked = _stack(parameters)
    return float(((stacked - stacked.mean(dim=0)) ** 2).sum(dim=1).mean())


def e_cent(parameters: dict[int, torch.Tensor], centralized: torch.Tensor) -> float:
    """Mean squared distance from the centralized trajectory."""
    stacked = _stack(parameters)
    reference = centralized.to(torch.float64)
    if reference.shape != stacked.shape[1:]:
        raise DisagreementError(
            f"centralized vector {tuple(reference.shape)} does not match the agents' "
            f"{tuple(stacked.shape[1:])}"
        )
    return float(((stacked - reference) ** 2).sum(dim=1).mean())


def max_pairwise_distance(parameters: dict[int, torch.Tensor]) -> float:
    """The widest gap between any two agents.

    $E_{\\mathrm{agree}}$ is a mean and can stay small while one agent drifts far
    off; this catches that.
    """
    stacked = _stack(parameters)
    return float(torch.cdist(stacked, stacked).max())


@dataclass(frozen=True)
class Disagreement:
    """Everything worth logging about how far apart the agents are."""

    e_agree: float
    e_cent: float | None
    mean_norm_squared: float
    max_pairwise: float

    def as_rows(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = [
            {"metric": "e_agree", "value": self.e_agree},
            {"metric": "theta_mean_norm_sq", "value": self.mean_norm_squared},
            {"metric": "max_pairwise_distance", "value": self.max_pairwise},
        ]
        if self.e_cent is not None:
            rows.append({"metric": "e_cent", "value": self.e_cent})
        return rows


def measure(
    parameters: dict[int, torch.Tensor], centralized: torch.Tensor | None = None
) -> Disagreement:
    """All the disagreement quantities for one step.

    ``centralized`` is optional because ``local_only`` and the centralized
    learner itself have nothing to compare against -- rather than logging a zero
    or a NaN, the field is simply absent.
    """
    mean = network_mean(parameters)
    return Disagreement(
        e_agree=e_agree(parameters),
        e_cent=None if centralized is None else e_cent(parameters, centralized),
        mean_norm_squared=float((mean**2).sum()),
        max_pairwise=max_pairwise_distance(parameters),
    )
