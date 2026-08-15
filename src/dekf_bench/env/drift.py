"""Drift schedules: the map from a time step to the transform parameters.

This module decides *how much* the distribution has moved by step $t$. It never
applies the transform -- that is ``data/transforms.py``, and keeping the two
apart is what lets the evaluation sets be built at an arbitrary drift state
without duplicating the rotation code.

Five schedules, one interface:

``stationary``
    Nothing moves. The baseline for X1.
``linear``
    Rotation accumulates at a constant rate to ``total_degrees`` over the
    horizon. The rate is **derived**, never configured: $\\alpha =
    \\text{total\\_degrees}/T$, so changing the horizon cannot silently change
    how far the distribution travels.
``ramp``
    Accelerating: the rate starts at zero and grows, so a single run sweeps a
    range of rates and the step at which tracking gives way locates the
    critical one. Under a constant rate a tracker settles to a steady-state lag
    and stops degrading, so "after how many steps does it break" is only a
    question when the rate is changing.
``piecewise``
    Abrupt jumps at known steps. An abrupt change is what makes the adaptation
    transient measurable, which is the cleanest test of tracking.
``sinusoidal``
    The distribution returns to states it has already visited, which is what
    exposes forgetting -- and therefore what to avoid when the question is
    tracking, since a learner can score well by remembering rather than by
    keeping up.

**Rate is the axis, not displacement.** Rotation is capped at
``MAX_WELL_POSED_DEGREES`` because past it a 6 is a 9 and a rising error would
measure label ambiguity rather than tracking failure. That cap bounds *how far*
the distribution can go, so "how much change breaks it" has to be asked as "how
fast" -- ``rate_at`` and ``peak_rate`` are the quantities the break threshold is
stated in. The cap also means rate and duration trade off against each other
($\\alpha T \\le$ cap), which is why the class-prior channel exists: label shift
has no such ceiling.

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

import numpy as np

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
    """Maps a step to a normalised progress, and thence to each drift channel.

    **Progress is the primitive, not degrees.** A schedule says *how far along*
    the run is; a channel says what that means. Rotation multiplies progress by
    a scale in degrees; the class-prior channel interpolates between two
    distributions by the same number. One schedule therefore drives every
    channel coherently, and adding a channel does not touch this class.

    Progress is normalised so that 1.0 is "fully travelled". It is not confined
    to $[0, 1]$: ``sinusoidal`` runs over $[-1, 1]$, because it goes backwards.
    """

    @abstractmethod
    def progress_at(self, step: int) -> float: ...

    @property
    @abstractmethod
    def degrees_scale(self) -> float:
        """Degrees at progress 1.0."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    def rotation_at(self, step: int) -> float:
        return self.degrees_scale * self.progress_at(step)

    def rate_at(self, step: int) -> float:
        """Degrees moved between ``step - 1`` and ``step``.

        A difference rather than a derivative, deliberately: this is what a
        learner actually experiences per update, it needs no special case for
        ``piecewise`` (where the derivative is a delta function), and it is the
        quantity the break threshold is a threshold *on*.
        """
        if step <= 0:
            return 0.0
        return self.rotation_at(step) - self.rotation_at(step - 1)

    def peak_rate(self, horizon: int) -> float:
        """The fastest the distribution ever moves over the run."""
        return max((abs(self.rate_at(step)) for step in range(1, horizon + 1)), default=0.0)

    def mean_rate(self, horizon: int) -> float:
        """Path length per step -- distance travelled, not displacement.

        For ``linear`` this is exactly ``alpha``. For ``sinusoidal`` it stays
        positive across the turning point, which is the honest reading: the
        learner has to chase the distribution back down again.
        """
        if horizon < 1:
            return 0.0
        return sum(abs(self.rate_at(step)) for step in range(1, horizon + 1)) / horizon

    def total_travel(self, horizon: int) -> float:
        """The largest rotation reached over the run, in absolute value.

        Used to check the well-posedness cap against what the schedule actually
        does, rather than against the parameter that was configured.
        """
        return max(abs(self.rotation_at(step)) for step in range(horizon + 1))


@dataclass(frozen=True)
class Stationary(DriftSchedule):
    def progress_at(self, step: int) -> float:
        return 0.0

    @property
    def degrees_scale(self) -> float:
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

    def progress_at(self, step: int) -> float:
        return step / self.horizon

    @property
    def degrees_scale(self) -> float:
        return self.total_degrees

    @property
    def name(self) -> str:
        return "linear"


@dataclass(frozen=True)
class Ramp(DriftSchedule):
    r"""Accelerating drift: progress is $(t/T)^{p}$, so the rate grows with $t$.

    **Why this schedule exists.** Under a *constant* rate a tracking learner
    settles to a steady-state lag and then stops degrading, so "after how many
    steps does it break" has no non-trivial answer -- run it longer at the same
    speed and nothing new happens. A ramp sweeps the rate within a single run:
    it starts at zero and ends at $p$ times the constant rate that would cover
    the same ground, so the step at which tracking gives way *locates* the
    critical rate instead of merely confirming one.

    The exponent is the width of that sweep. At $p = 2$ the run ends at twice
    the equivalent linear rate; at $p = 4$, four times, at the cost of spending
    most of the run barely moving. $p = 1$ is exactly ``linear`` and is
    rejected, because a schedule that silently aliases another is a trap.

    **Read the located rate as an upper bound.** The learner lags, so it crosses
    the threshold slightly after the rate that would break it in steady state.
    Confirm with constant-rate ``linear`` runs bracketing the located value --
    the ramp is the cheap instrument that says where to look.
    """

    total_degrees: float
    horizon: int
    exponent: float = 2.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise DriftError(f"ramp drift needs horizon >= 1, got {self.horizon}")
        if self.exponent <= 1.0:
            raise DriftError(
                f"ramp exponent must be > 1, got {self.exponent}. At 1.0 the ramp is "
                "exactly the linear schedule, and below it the drift decelerates -- "
                "which measures nothing this schedule exists to measure."
            )

    def progress_at(self, step: int) -> float:
        return float((step / self.horizon) ** self.exponent)

    @property
    def degrees_scale(self) -> float:
        return self.total_degrees

    @property
    def final_rate(self) -> float:
        """Degrees per step at the end of the run -- the top of the swept range."""
        return self.rate_at(self.horizon)

    @property
    def name(self) -> str:
        return "ramp"


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

    def progress_at(self, step: int) -> float:
        if not self.change_points:
            return 0.0
        passed = sum(1 for point in self.change_points if step >= point)
        return passed / len(self.change_points)

    @property
    def degrees_scale(self) -> float:
        return self.jump_degrees * len(self.change_points)

    @property
    def name(self) -> str:
        return "piecewise"


@dataclass(frozen=True)
class Recurring(DriftSchedule):
    r"""Abrupt jumps of fixed magnitude at a fixed interval, direction unpredictable.

    X5 measures the transient after *one* jump. This repeats it, so the learner
    never gets to settle and the run measures recovery over and over rather than
    once. That is the regime a filter should own: a gradient method recovers
    only as fast as a step size tuned for the stationary regime allows, while a
    filter's covariance says how much to trust the new evidence.

    **The jumps cannot march in one direction.** Rotation is capped at
    ``MAX_WELL_POSED_DEGREES``, so a repeated shift must stay inside a band and
    therefore revisits states. What is controlled instead is that every jump has
    *exactly* the same magnitude -- so every transient is comparable -- and that
    the direction is unpredictable, so a learner cannot pre-position for the
    next one. At the band edge the direction is **reflected rather than
    clipped**: clipping would shorten that jump and quietly make the transients
    incomparable, which is the property the schedule exists to provide.

    ``jump_degrees / jump_every`` is the average speed, so this schedule and
    ``linear`` can be compared at matched speed -- which separates "the
    distribution moved" from "it moved *abruptly*".
    """

    jump_degrees: float
    jump_every: int
    horizon: int
    seed: int = 0
    cap: float = MAX_WELL_POSED_DEGREES

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise DriftError(f"recurring drift needs horizon >= 1, got {self.horizon}")
        if self.jump_every < 1:
            raise DriftError(f"jump_every must be >= 1, got {self.jump_every}")
        if self.jump_degrees <= 0:
            raise DriftError(f"jump_degrees must be > 0, got {self.jump_degrees}")
        if self.jump_degrees > self.cap:
            raise DriftError(
                f"jump_degrees {self.jump_degrees} exceeds the {self.cap} degree cap, so "
                "from a rotation of 0 neither direction lands inside the well-posed band "
                "and no jump of that size is possible."
            )
        object.__setattr__(self, "_rotations", self._plan())

    def _plan(self) -> tuple[float, ...]:
        """Every rotation the run visits, one per jump, decided up front.

        Precomputed rather than drawn per step so ``rotation_at`` stays a pure
        function of the step -- the same property that lets the evaluation sets
        ask about step 900 without walking there.
        """
        rng = np.random.default_rng(self.seed)
        n_jumps = self.horizon // self.jump_every
        current = 0.0
        rotations = [current]
        for _ in range(n_jumps):
            direction = 1.0 if rng.random() < 0.5 else -1.0
            proposed = current + direction * self.jump_degrees
            if abs(proposed) > self.cap + 1e-9:
                proposed = current - direction * self.jump_degrees
            current = round(proposed, 9)
            rotations.append(current)
        return tuple(rotations)

    @property
    def rotations(self) -> tuple[float, ...]:
        """The planned rotation after each jump, starting from the initial one."""
        return self._rotations  # type: ignore[attr-defined]

    def progress_at(self, step: int) -> float:
        index = min(max(step, 0) // self.jump_every, len(self.rotations) - 1)
        return self.rotations[index] / self.cap

    @property
    def degrees_scale(self) -> float:
        return self.cap

    @property
    def n_jumps(self) -> int:
        return len(self.rotations) - 1

    @property
    def name(self) -> str:
        return "recurring"


@dataclass(frozen=True)
class Sinusoidal(DriftSchedule):
    """Oscillation of amplitude ``amplitude_degrees`` and the given period."""

    amplitude_degrees: float
    period: int

    def __post_init__(self) -> None:
        if self.period < 1:
            raise DriftError(f"sinusoidal drift needs period >= 1, got {self.period}")

    def progress_at(self, step: int) -> float:
        return math.sin(2.0 * math.pi * step / self.period)

    @property
    def degrees_scale(self) -> float:
        return self.amplitude_degrees

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

    def progress_at(self, step: int, node: int = 0) -> float:
        """Normalised progress for one agent.

        Scaled by the same multiplier as the rotation, so every drift channel
        moves together for a given agent: under ``per_node`` a laggard is
        equally behind in rotation and in class prior, rather than behind in
        one and current in the other.
        """
        return self.multiplier(node) * self.schedule.progress_at(step)

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
            # Mean and peak rather than a single alpha: they coincide only for
            # `linear`, and their ratio is what says whether a run sweeps a
            # range of rates or sits at one.
            "mean_rate_per_step": self.schedule.mean_rate(horizon),
            "peak_rate_per_step": self.schedule.peak_rate(horizon),
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
    if kind == "ramp":
        return Ramp(
            total_degrees=drift_config.total_degrees,
            horizon=horizon,
            exponent=drift_config.ramp_exponent,
        )
    if kind == "recurring":
        return Recurring(
            jump_degrees=drift_config.jump_degrees,
            jump_every=drift_config.jump_every,
            horizon=horizon,
            seed=drift_config.jump_seed,
        )
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
