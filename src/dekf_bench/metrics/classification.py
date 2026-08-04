"""Accuracy, error rate, and the gap that is the headline number.

Per-agent quantities are what get **stored**; means and spreads are derived at
plot time. Storing only per-agent rows means a new aggregate never requires a
re-run (IMPLEMENTATION.md §8.1), and it keeps the failure the mean hides
visible: a good mean with a terrible spread is not a working method.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class MetricError(ValueError):
    """Raised for mismatched predictions and targets."""


def _check(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    if predictions.shape != targets.shape:
        raise MetricError(
            f"predictions {tuple(predictions.shape)} do not match targets "
            f"{tuple(targets.shape)}"
        )
    if predictions.numel() == 0:
        raise MetricError("cannot score an empty batch")


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Correct-classification rate in $[0,1]$."""
    _check(predictions, targets)
    return float((predictions == targets).to(torch.float64).mean())


def error_rate(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """$1 - \\text{accuracy}$."""
    return 1.0 - accuracy(predictions, targets)


def correct_count(predictions: torch.Tensor, targets: torch.Tensor) -> int:
    """Raw count, for pooling scores across batches without weighting error."""
    _check(predictions, targets)
    return int((predictions == targets).sum())


@dataclass(frozen=True)
class AgentScores:
    """One score per agent, and the aggregates derived from them.

    The aggregates are properties rather than stored fields, so there is exactly
    one definition of "the mean error rate" and it cannot drift from the
    per-agent numbers it summarises.
    """

    error_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.error_rates:
            raise MetricError("need at least one agent")
        if any(not 0.0 <= rate <= 1.0 for rate in self.error_rates):
            raise MetricError(f"error rates must lie in [0, 1], got {self.error_rates}")

    @property
    def n_nodes(self) -> int:
        return len(self.error_rates)

    @property
    def mean(self) -> float:
        """$\\bar e_t = \\frac1N\\sum_v (1-\\mathrm{acc}_{v,t})$ -- the headline curve."""
        return sum(self.error_rates) / self.n_nodes

    @property
    def worst(self) -> float:
        return max(self.error_rates)

    @property
    def best(self) -> float:
        return min(self.error_rates)

    @property
    def spread(self) -> float:
        """$\\max_v e_v - \\min_v e_v$. A good mean with a terrible spread is not
        a working method, and the mean alone cannot show that."""
        return self.worst - self.best

    @property
    def std(self) -> float:
        if self.n_nodes == 1:
            return 0.0
        mean = self.mean
        variance = sum((rate - mean) ** 2 for rate in self.error_rates) / (self.n_nodes - 1)
        return variance**0.5

    def gap(self, reference_error: float) -> float:
        """$\\bar e_t - e^\\star$ against the offline reference.

        **The headline number.** Q1 asks what decentralization costs, and that is
        this difference rather than the raw error rate -- which also moves with
        the drift state, the architecture and the horizon.
        """
        return self.mean - reference_error

    def as_rows(self) -> list[dict[str, float | int | str]]:
        """Long-format rows, one per agent. Aggregates are *not* included:
        they are derived at plot time so a new one never requires a re-run."""
        return [
            {"node_id": node, "metric": "error_rate", "value": rate}
            for node, rate in enumerate(self.error_rates)
        ]


def score_agents(
    predictions: dict[int, torch.Tensor], targets: dict[int, torch.Tensor]
) -> AgentScores:
    """Error rate per agent, in node order."""
    if set(predictions) != set(targets):
        raise MetricError(
            f"predictions cover agents {sorted(predictions)} but targets cover "
            f"{sorted(targets)}"
        )
    return AgentScores(
        error_rates=tuple(
            error_rate(predictions[node], targets[node]) for node in sorted(predictions)
        )
    )
