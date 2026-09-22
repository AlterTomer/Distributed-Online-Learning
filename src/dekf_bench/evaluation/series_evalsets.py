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

from dekf_bench.env.series import SeriesError, channel_value, law_blocks, secondary_value


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
        # The secondary comes from the environment's resolved path, never re-derived
        # here: under coupling='independent' it follows its own schedule, which this
        # builder cannot reconstruct from `drift`. Re-deriving it would build the
        # held-out set at the wrong secondary value and mis-score the whole cell.
        second = self._secondary_at(name, step, node)
        inputs, targets = self._build(value, second)
        return SeriesEvalSet(
            name=name, inputs=inputs, targets=targets, rotation_degrees=value, step=step
        )

    def _secondary_at(self, name: str, step: int, node: int | None) -> float | None:
        """The secondary channel's value for this set, or None when one channel drifts."""
        if not self._series.secondary_channel:
            return None
        if name == "canonical":
            # Zero displacement on both channels: the undrifted law and rest sensor.
            return float(secondary_value(self._series, 0.0))
        resolved = getattr(self._environment, "second_values", None)
        if resolved is None:
            raise SeriesError(
                "a secondary channel is configured but the environment carries no "
                "second_values; the environment and the evaluation sets disagree"
            )
        if name == "current_mean":
            return float(np.mean(resolved[:, step]))
        return float(resolved[0 if node is None else node, step])

    def _build(self, value: float, second: float | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keyed on the PAIR: with two channels drifting, one scalar no longer
        # identifies a held-out set, and a single key would serve the first
        # secondary value seen for every later one.
        key = (round(value, 12), None if second is None else round(second, 12))
        if key in self._cache:
            return self._cache[key]
        series = self._series
        seeds = self._environment.seeds
        per_trajectory = math.ceil(series.eval_blocks / series.eval_trajectories)
        stamp = f"{key[0]:.12f}" if second is None else f"{key[0]:.12f}|{key[1]:.12f}"
        rng = seeds.numpy_rng("stream", "eval", stamp)
        inputs, targets = law_blocks(series, value, series.eval_trajectories, per_trajectory,
                                     rng, second=second)
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
