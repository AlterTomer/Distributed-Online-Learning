r"""Class-prior drift: label shift, the drift channel with no ceiling.

Rotation is covariate shift and it is **capped**. Past
``MAX_WELL_POSED_DEGREES`` a 6 is a 9, so a rising error would measure label
ambiguity rather than tracking failure. That cap bounds total travel, hence the
mean rate, so rotation trades rate against duration ($\alpha T \le$ cap) and
cannot be pushed arbitrarily hard.

Label shift has no such ceiling. Each agent's class distribution moves from one
Dirichlet draw to another along the schedule's progress:

.. math::
    \bm q_v(t) = (1 - \lambda_t)\,\bm q_v^{\text{start}} + \lambda_t\,\bm q_v^{\text{end}},
    \qquad \lambda_t = \texttt{total\_shift} \cdot \texttt{progress}(t)

The Bayes-optimal classifier moves the whole way, but no label ever becomes
ambiguous, so the distribution can travel as far as total variation allows and
the gap to the reference stays interpretable at every point.

**The plan is built up front, and it is the ground truth.** Rather than
sampling a class at each step -- which would make the stream stateful and the
run unreproducible from a step offset -- the whole $(N, T, n)$ table of classes
is drawn once. Everything downstream is then derived from it rather than
agreeing with it by convention:

* the *partition* holds exactly the per-class counts the plan will ask for,
  so a shard cannot run dry part-way through a run;
* the *stream* serves a shard in the plan's order, which makes prior drift a
  permutation of an existing shard rather than a new sampling mechanism.

That second point is why ``stream.py`` needs no new machinery: offsets stay a
prefix sum, and the exactly-once guarantee is untouched because each index is
still popped from its class queue exactly once.

**Feasibility is a property of the plan, checked before a run starts.** MNIST
holds about 6000 of each digit. If the agents' summed demand for a class
exceeds the pool, no partition exists, and the error says so rather than
letting a shard quietly come up short at step 1400.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dekf_bench.env.drift import DriftSchedule


class PriorDriftError(ValueError):
    """Raised when a class-prior schedule is malformed or infeasible."""


def _dirichlet_rows(
    concentration: float, n_classes: int, rows: int, generator: torch.Generator | None
) -> torch.Tensor:
    """``rows`` draws from ``Dir(concentration * 1_K)``, reproducibly.

    Mirrors ``partition._dirichlet``: ``torch.distributions.Dirichlet`` takes no
    generator and would draw from the global RNG, which is exactly the coupling
    the separable seed streams exist to prevent.
    """
    seed = 0 if generator is None else int(torch.randint(0, 2**63 - 1, (1,), generator=generator))
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.full(n_classes, float(concentration)), size=rows)
    return torch.from_numpy(draws).to(torch.float64)


@dataclass(frozen=True)
class ClassPriors:
    """Where each agent's class distribution starts and where it ends.

    Attributes:
        start: ``(N, K)`` -- the composition at progress 0.
        end: ``(N, K)`` -- the composition the run travels toward.
        total_shift: how far along that path the run actually goes. 1.0 arrives
            exactly at ``end``; smaller values stop short, which is the
            magnitude knob for "how much does the distribution have to change".
    """

    start: torch.Tensor
    end: torch.Tensor
    total_shift: float = 1.0

    def __post_init__(self) -> None:
        if self.start.shape != self.end.shape:
            raise PriorDriftError(
                f"start {tuple(self.start.shape)} and end {tuple(self.end.shape)} must agree"
            )
        if self.start.ndim != 2:
            raise PriorDriftError("class priors must be (n_nodes, n_classes)")
        if not 0.0 <= self.total_shift <= 1.0:
            raise PriorDriftError(
                f"total_shift must lie in [0, 1], got {self.total_shift}. Past 1 the "
                "interpolation leaves the simplex and would ask for negative class mass."
            )
        for name, table in (("start", self.start), ("end", self.end)):
            if not torch.allclose(table.sum(dim=1), torch.ones(len(table), dtype=table.dtype)):
                raise PriorDriftError(f"{name} rows must each sum to 1")

    @property
    def n_nodes(self) -> int:
        return int(self.start.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.start.shape[1])

    def lam(self, progress: float) -> float:
        """The interpolation weight at a schedule progress.

        Clamped to $[0, 1]$: ``sinusoidal`` progress goes negative, and there is
        no meaningful distribution "before the start" to extrapolate toward.
        """
        return float(min(1.0, max(0.0, self.total_shift * progress)))

    def at(self, progress: float) -> torch.Tensor:
        """``(N, K)`` -- every agent's class distribution at this progress."""
        weight = self.lam(progress)
        return (1.0 - weight) * self.start + weight * self.end

    def travel(self) -> torch.Tensor:
        """Per-agent total-variation distance between the endpoints actually visited.

        The magnitude axis. Unlike degrees of rotation this has a hard maximum
        of 1 that is a property of probability rather than of MNIST's labels,
        and nothing goes ill-posed on the way there.
        """
        moved = self.at(1.0) - self.at(0.0)
        return 0.5 * moved.abs().sum(dim=1)

    def summary(self) -> dict[str, Any]:
        travel = self.travel()
        return {
            "n_nodes": self.n_nodes,
            "n_classes": self.n_classes,
            "total_shift": self.total_shift,
            "mean_tv_travel": float(travel.mean()),
            "max_tv_travel": float(travel.max()),
        }


@dataclass(frozen=True)
class ClassPlan:
    """Which class each agent receives at each step, decided before the run.

    Attributes:
        classes: ``(N, T, n)`` int64 -- the class of every sample the run will
            serve. The experiment's ground truth: the partition is sized from
            it and the stream is ordered by it.
    """

    classes: torch.Tensor
    n_classes: int

    def __post_init__(self) -> None:
        if self.classes.ndim != 3:
            raise PriorDriftError("a class plan must be (n_nodes, horizon, samples_per_step)")
        if self.classes.dtype != torch.int64:
            raise PriorDriftError(f"class plan must be int64, got {self.classes.dtype}")

    @property
    def n_nodes(self) -> int:
        return int(self.classes.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.classes.shape[1])

    @property
    def samples_per_step(self) -> int:
        return int(self.classes.shape[2])

    def demand(self) -> torch.Tensor:
        """``(N, K)`` -- how many of each class each agent's shard must hold.

        Counted over the whole horizon regardless of label availability. With
        $\\pi_{\\text{lab}} < 1$ an agent idles and consumes less, so this is an
        upper bound and the shard is over-provisioned rather than short.
        """
        return torch.stack(
            [
                torch.bincount(self.classes[node].reshape(-1), minlength=self.n_classes)
                for node in range(self.n_nodes)
            ]
        )

    def classes_for(self, node: int) -> torch.Tensor:
        """The flat class sequence agent ``node`` will be served, in order."""
        return self.classes[node].reshape(-1)


def build_class_priors(
    n_nodes: int,
    n_classes: int,
    beta: float,
    total_shift: float = 1.0,
    generator: torch.Generator | None = None,
    uniform_start: bool = True,
) -> ClassPriors:
    """Draw the endpoints of the prior path.

    Args:
        beta: Dirichlet concentration for the endpoints. Small values give
            agents that concentrate on a few classes, so the path is long;
            large values approach uniform and the run barely moves.
        total_shift: how far along the path to travel. The magnitude knob.
        uniform_start: begin at $1/K$ rather than at a Dirichlet draw. Default
            true so that at progress 0 a prior-drift run is the *same*
            experiment as an ordinary one, and any difference measured is the
            drift rather than the starting point.
    """
    if beta <= 0:
        raise PriorDriftError(f"prior drift beta must be > 0, got {beta}")

    end = _dirichlet_rows(beta, n_classes, n_nodes, generator)
    if uniform_start:
        start = torch.full((n_nodes, n_classes), 1.0 / n_classes, dtype=torch.float64)
    else:
        start = _dirichlet_rows(beta, n_classes, n_nodes, generator)
    return ClassPriors(start=start, end=end, total_shift=total_shift)


def build_class_plan(
    priors: ClassPriors,
    schedule: DriftSchedule,
    horizon: int,
    samples_per_step: int,
    generator: torch.Generator | None = None,
    node_progress: dict[int, list[float]] | None = None,
) -> ClassPlan:
    """Draw the whole run's classes at once.

    Multinomial rather than a deterministic quota: the prior *is* a sampling
    distribution, and rounding $n = 2$ samples to a 10-class quota every step
    would serve a suspiciously tidy stream that no real deployment produces.
    The draws are made up front so the plan stays a pure function of
    ``(agent, step)`` however it is reached.

    Args:
        node_progress: per-agent progress sequences, for ``per_node`` drift
            where agents move at different speeds. Defaults to the schedule's
            own progress for every agent.
    """
    if horizon < 1:
        raise PriorDriftError(f"horizon must be >= 1, got {horizon}")
    if samples_per_step < 1:
        raise PriorDriftError(f"samples_per_step must be >= 1, got {samples_per_step}")

    rows = []
    for node in range(priors.n_nodes):
        progress = (
            [schedule.progress_at(step) for step in range(horizon)]
            if node_progress is None
            else node_progress[node]
        )
        # (T, K): this agent's class distribution at every step.
        rows.append(torch.stack([priors.at(value)[node] for value in progress]))
    probabilities = torch.cat(rows, dim=0).clamp(min=0.0)

    draws = torch.multinomial(
        probabilities, samples_per_step, replacement=True, generator=generator
    )
    classes = draws.reshape(priors.n_nodes, horizon, samples_per_step).to(torch.int64)
    return ClassPlan(classes=classes, n_classes=priors.n_classes)


def build_class_plan_from_config(
    config: Any, drift: Any, generator: torch.Generator | None = None
) -> ClassPlan | None:
    """The class plan a run's config asks for, or ``None`` when the channel is off.

    Returning ``None`` rather than a uniform plan is deliberate: a run without
    prior drift must be *byte-identical* to one from before this channel
    existed, and an all-classes-equal plan would still reorder every shard.
    """
    settings = config.env.prior_drift
    if not settings.enabled:
        return None

    n_nodes = config.graph.n_nodes
    priors = build_class_priors(
        n_nodes=n_nodes,
        n_classes=config.model.output_dim,
        beta=settings.beta,
        total_shift=settings.total_shift,
        generator=generator,
        uniform_start=settings.uniform_start,
    )
    horizon = config.run.horizon
    per_node = config.env.drift_scope == "per_node"
    node_progress = {
        node: [drift.progress_at(step, node if per_node else 0) for step in range(horizon)]
        for node in range(n_nodes)
    }
    return build_class_plan(
        priors=priors,
        schedule=drift.schedule,
        horizon=horizon,
        samples_per_step=config.env.samples_per_node_per_step,
        generator=generator,
        node_progress=node_progress,
    )


def check_plan_is_feasible(plan: ClassPlan, labels: torch.Tensor) -> None:
    """Reject a plan the training set cannot supply, before anything is built.

    The shards are disjoint, so the agents' demands compete for one finite pool
    per class. A plan that oversubscribes a class has no valid partition at all
    -- there is nothing to degrade gracefully into, and discovering it at step
    1400 would waste the run.
    """
    demand = plan.demand().sum(dim=0)
    available = torch.bincount(labels, minlength=plan.n_classes)
    over = [
        (int(c), int(demand[c]), int(available[c]))
        for c in range(plan.n_classes)
        if demand[c] > available[c]
    ]
    if over:
        detail = "; ".join(f"class {c}: needs {want}, pool holds {have}" for c, want, have in over)
        raise PriorDriftError(
            f"the class-prior plan oversubscribes the training set ({detail}). The agents' "
            "demands compete for one disjoint pool per class, so no partition satisfies this. "
            "Reduce env.prior_drift.total_shift, raise env.prior_drift.beta so the endpoints "
            "are less concentrated, shorten the horizon, or lower "
            "env.samples_per_node_per_step."
        )
