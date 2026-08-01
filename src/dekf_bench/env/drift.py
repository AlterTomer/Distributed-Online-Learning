"""Drift schedules: the map from a time step to the transform parameters.

This module decides *how much* the distribution has moved by step $t$. It never
applies the transform -- that is ``data/transforms.py``, and keeping the two
apart is what lets the evaluation sets be built at an arbitrary drift state
without duplicating the rotation code.

Four schedules, one interface:

``stationary``
    Nothing moves. The baseline for X1.
``linear``
    Rotation accumulates at a constant rate to ``total_degrees`` over the
    horizon. The rate is **derived**, never configured: $\\alpha =
    \\text{total\\_degrees}/T$, so changing the horizon cannot silently change
    how far the distribution travels.
``piecewise``
    Abrupt jumps at known steps. An abrupt change is what makes the adaptation
    transient measurable, which is the cleanest test of tracking.
``sinusoidal``
    The distribution returns to states it has already visited, which is what
    exposes forgetting.

**Global versus per-node.** By default every agent sees the same rotation, so a
single shared parameter vector remains the correct object to estimate. Under
``drift_scope: per_node`` agents drift at different rates, which is the regime
where a shared model starts to be the wrong assumption -- and therefore the
regime that motivates the hierarchical shared/local extension. It is available
but not the default, because whether it is the interesting story is still open.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

#: Beyond roughly this much rotation, MNIST labels stop being well defined --
#: a 6 becomes a 9. Duplicated from the config schema, where it is enforced.
MAX_WELL_POSED_DEGREES = 45.0


class DriftError(ValueError):
    """Raised for an unbuildable or inconsistent drift schedule."""


@dataclass(frozen=True)
class DriftState:
    """What the environment must know to build the data at one step.

    A dataclass rather than a bare float, because the transform will grow --
    translation, scale, per-node offsets -- and every consumer already unpacks a
    named field rather than positional numbers.
    """

    step: int
    rotation_degrees: float
    node: int | None = None

    @property
    def is_canonical(self) -> bool:
        """Whether this state is the unrotated one."""
        return self.rotation_degrees == 0.0


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #


class DriftSchedule(ABC):
    """Maps a step to a rotation in degrees."""

    @abstractmethod
    def rotation_at(self, step: int) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def total_travel(self, horizon: int) -> float:
        """The largest rotation reached over the run, in absolute value.

        Used to check the well-posedness cap against what the schedule actually
        does, rather than against the parameter that was configured.
        """
        return max(abs(self.rotation_at(step)) for step in range(horizon + 1))


@dataclass(frozen=True)
class Stationary(DriftSchedule):
    def rotation_at(self, step: int) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return "stationary"


@dataclass(frozen=True)
class Linear(DriftSchedule):
    """Constant rate, reaching ``total_degrees`` at ``horizon``."""

    total_degrees: float
    horizon: int

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise DriftError(f"linear drift needs horizon >= 1, got {self.horizon}")

    @property
    def alpha(self) -> float:
        """Degrees per step. Derived, never configured."""
        return self.total_degrees / self.horizon

    def rotation_at(self, step: int) -> float:
        return self.alpha * step

    @property
    def name(self) -> str:
        return "linear"


@dataclass(frozen=True)
class Piecewise(DriftSchedule):
    """A step function: ``jump_degrees`` added at each change point."""

    change_points: tuple[int, ...]
    jump_degrees: float

    def __post_init__(self) -> None:
        if any(point < 0 for point in self.change_points):
            raise DriftError(f"change points must be non-negative: {self.change_points}")
        if list(self.change_points) != sorted(self.change_points):
            raise DriftError(f"change points must be sorted: {self.change_points}")

    def rotation_at(self, step: int) -> float:
        return self.jump_degrees * sum(1 for point in self.change_points if step >= point)

    @property
    def name(self) -> str:
        return "piecewise"


@dataclass(frozen=True)
class Sinusoidal(DriftSchedule):
    """Oscillation of amplitude ``amplitude_degrees`` and the given period."""

    amplitude_degrees: float
    period: int

    def __post_init__(self) -> None:
        if self.period < 1:
            raise DriftError(f"sinusoidal drift needs period >= 1, got {self.period}")

    def rotation_at(self, step: int) -> float:
        return self.amplitude_degrees * math.sin(2.0 * math.pi * step / self.period)

    @property
    def name(self) -> str:
        return "sinusoidal"


# --------------------------------------------------------------------------- #
# scope
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Drift:
    """A schedule plus the scope it applies over.

    Under ``global`` scope every agent sees the same rotation. Under
    ``per_node`` each agent's rotation is scaled by a fixed multiplier, evenly
    spaced over ``[1 - spread, 1]``.

    The multipliers top out at **1, not 1 + spread**: scaling any agent above
    the configured rate would carry that agent past the well-posedness cap the
    schedule was validated against, so the spread slows the laggards rather than
    accelerating the leaders. The fastest agent therefore travels exactly
    ``total_degrees`` and the slowest ``(1 - spread) * total_degrees``.
    """

    schedule: DriftSchedule
    scope: str = "global"
    n_nodes: int = 1
    spread: float = 0.5

    def __post_init__(self) -> None:
        if self.scope not in ("global", "per_node"):
            raise DriftError(f"unknown drift scope {self.scope!r}")
        if self.n_nodes < 1:
            raise DriftError(f"n_nodes must be >= 1, got {self.n_nodes}")
        if not 0.0 <= self.spread < 1.0:
            raise DriftError(f"per-node spread must lie in [0, 1), got {self.spread}")

    def multiplier(self, node: int) -> float:
        """The rate scaling for one agent."""
        if self.scope == "global" or self.n_nodes == 1:
            return 1.0
        if not 0 <= node < self.n_nodes:
            raise DriftError(f"node {node} outside 0..{self.n_nodes - 1}")
        lowest = 1.0 - self.spread
        return lowest + (1.0 - lowest) * node / (self.n_nodes - 1)

    def rotation_at(self, step: int, node: int = 0) -> float:
        return self.multiplier(node) * self.schedule.rotation_at(step)

    def state_at(self, step: int, node: int | None = None) -> DriftState:
        """The drift state at ``step``, for one agent or for the network."""
        if node is None:
            if self.scope == "per_node" and self.n_nodes > 1:
                raise DriftError(
                    "per_node drift has no single network-wide state; ask for a node, "
                    "or use states_at() to get all of them"
                )
            return DriftState(step=step, rotation_degrees=self.schedule.rotation_at(step))
        return DriftState(step=step, rotation_degrees=self.rotation_at(step, node), node=node)

    def states_at(self, step: int) -> tuple[DriftState, ...]:
        """One state per agent."""
        return tuple(self.state_at(step, node) for node in range(self.n_nodes))

    @property
    def is_stationary(self) -> bool:
        return isinstance(self.schedule, Stationary)

    def distinct_rotations(self, horizon: int, every: int = 1) -> tuple[float, ...]:
        """Every rotation the run visits, sorted and deduplicated.

        The evaluation sets and the offline reference classifier are cached per
        rotation level, so this is what decides how many of each are needed.
        """
        values = {
            round(self.rotation_at(step, node), 9)
            for step in range(0, horizon + 1, every)
            for node in (range(self.n_nodes) if self.scope == "per_node" else (0,))
        }
        return tuple(sorted(values))

    def summary(self, horizon: int) -> dict[str, Any]:
        return {
            "schedule": self.schedule.name,
            "scope": self.scope,
            "alpha_per_step": (self.schedule.alpha if isinstance(self.schedule, Linear) else 0.0),
            "total_travel": self.schedule.total_travel(horizon),
            "rotation_at_start": self.schedule.rotation_at(0),
            "rotation_at_end": self.schedule.rotation_at(horizon),
        }


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def build_schedule(drift_config: Any, horizon: int) -> DriftSchedule:
    """The schedule a config's ``env.drift`` block asks for."""
    kind = drift_config.schedule
    if kind == "stationary":
        return Stationary()
    if kind == "linear":
        return Linear(total_degrees=drift_config.total_degrees, horizon=horizon)
    if kind == "piecewise":
        return Piecewise(
            change_points=tuple(drift_config.change_points),
            jump_degrees=drift_config.jump_degrees,
        )
    if kind == "sinusoidal":
        return Sinusoidal(
            amplitude_degrees=drift_config.amplitude_degrees, period=drift_config.period
        )
    raise DriftError(f"unknown drift schedule {kind!r}")


def build_drift(config: Any) -> Drift:
    """The drift a run's config asks for, with the well-posedness cap checked.

    The cap is verified against what the schedule *does* over the horizon, not
    against the parameter that was configured -- a piecewise schedule with
    several change points can travel past the cap while every individual field
    looks reasonable.
    """
    schedule = build_schedule(config.env.drift, config.run.horizon)
    drift = Drift(
        schedule=schedule,
        scope=config.env.drift_scope,
        n_nodes=config.graph.n_nodes,
        spread=config.env.drift.per_node_spread,
    )

    travel = schedule.total_travel(config.run.horizon)
    if travel > MAX_WELL_POSED_DEGREES + 1e-9:
        raise DriftError(
            f"the {schedule.name} schedule reaches {travel:.1f} degrees over "
            f"T={config.run.horizon}, above the {MAX_WELL_POSED_DEGREES} degree cap. "
            "Past that, rotated MNIST labels stop being well defined and the gap to the "
            "reference measures label ambiguity rather than decentralization cost. "
            "Individual fields can each look reasonable while the schedule as a whole "
            "travels too far -- several change points, for instance."
        )
    return drift
