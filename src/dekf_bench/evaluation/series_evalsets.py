r"""Held-out evaluation sets for the series task: the current law, and the canonical one.

`docs/mackey_glass_plan.md`, decision 16. On MNIST a held-out set is the test
split rotated to the current angle. Here the distribution is a *law*, so the
held-out set at step $t$ is **fresh trajectories integrated at the fixed law the
agents face at $t$** -- its stationary distribution, which is what "current"
means -- observed through the same sensor. They share nothing with the training
stream: separate histories and noise, from their own seed sub-stream.

* ``current``: at the channel's value at step $t$ -- the network's, or one agent's
  under ``per_node`` drift.
* ``current_mean``: at the mean of the agents' values, the protocol's companion to
  per-agent ``current``.
* ``canonical``: at zero displacement, the undrifted law.

With heterogeneous agents (decision 22) every set uses the **central** law -- no
offset, the configured $\tau$ and $\sigma$ -- for the reason MNIST's skew runs are
scored on the global test distribution: the shared predictor is measured against
the network's law, not against each agent's shard of it.

Built lazily and cached by channel value, so a schedule that revisits a value (a
recurring jump, or the stationary twin) integrates it once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dekf_bench.env.series import SeriesError, channel_value, law_blocks


@dataclass(frozen=True)
class SeriesEvalSet:
    """Blocks to be scored, tagged with the channel value they were built at."""

    name: str
    inputs: torch.Tensor
    targets: torch.Tensor
    rotation_degrees: float
    step: int

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def batches(self, batch_size: int):
        if batch_size < 1:
            raise SeriesError(f"batch_size must be >= 1, got {batch_size}")
        for start in range(0, len(self), batch_size):
            yield self.inputs[start : start + batch_size], self.targets[start : start + batch_size]


class SeriesEvalSets:
    """Builds and caches the held-out sets for one series run."""

    def __init__(self, config: Any, environment: Any):
        self._config = config
        self._environment = environment
        self._series = config.env.series
        self._cache: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}

    def at(self, name: str, step: int, node: int | None = None) -> SeriesEvalSet | None:
        drift = self._environment.drift
        if name == "current":
            displacement = drift.rotation_at(step, 0 if node is None else node)
        elif name == "current_mean":
            displacement = float(
                np.mean([drift.rotation_at(step, v) for v in range(self._environment.n_nodes)])
            )
        elif name == "canonical":
            displacement = 0.0
        elif name == "backward":
            raise SeriesError(
                "the series task has no backward set: it matters only under sinusoidal "
                "drift, which is deferred (docs/mackey_glass_plan.md, decision 16)"
            )
        else:
            raise SeriesError(f"unknown evaluation set {name!r}")
        value = float(channel_value(self._series, displacement))
        inputs, targets = self._build(value)
        return SeriesEvalSet(
            name=name, inputs=inputs, targets=targets, rotation_degrees=value, step=step
        )

    def _build(self, value: float) -> tuple[torch.Tensor, torch.Tensor]:
        key = round(value, 12)
        if key in self._cache:
            return self._cache[key]
        series = self._series
        seeds = self._environment.seeds
        per_trajectory = math.ceil(series.eval_blocks / series.eval_trajectories)
        rng = seeds.numpy_rng("stream", "eval", f"{key:.12f}")
        inputs, targets = law_blocks(series, value, series.eval_trajectories, per_trajectory, rng)
        width = series.length - 1
        env_inputs = self._environment.inputs
        built = tuple(
            torch.tensor(
                array.reshape(-1, width)[: series.eval_blocks],
                dtype=env_inputs.dtype,
                device=env_inputs.device,
            )
            for array in (inputs, targets)
        )
        self._cache[key] = built  # type: ignore[assignment]
        return built  # type: ignore[return-value]

    def summary(self) -> dict[str, Any]:
        return {"cached_values": sorted(self._cache), "eval_blocks": self._series.eval_blocks}


def build_series_evalsets(config: Any, environment: Any) -> SeriesEvalSets:
    return SeriesEvalSets(config, environment)
