"""The held-out evaluation sets, and the drift state each carries.

**This module is the single place that guarantees train and eval see the same
rotation.** It does not compute a rotation of its own: it asks the environment's
``Drift`` for the state at step $t$ and applies the environment's own
``ImageTransform``. The two paths cannot disagree because they are the same two
objects (`environment.md` G4).

Three sets:

``current``
    The test split at the rotation the training data carries at step $t$. The
    headline metric. Evaluate on anything else while training on rotated data
    and every method appears to fail, for a reason that has nothing to do with
    any of them.
``backward``
    The test split at a rotation the model saw *earlier*. Measures forgetting.
``canonical``
    Unrotated. Comparable across every run and against published MNIST numbers.

**The backward set is anchored by rotation, not by a step count.** A fixed
offset $t' = t - \\Delta$ degenerates badly. At $\\Delta$ equal to the sinusoidal
period the separation is *identically zero* -- the schedule chosen to expose
forgetting could not measure it. Under piecewise it collapses to zero once $t$
passes the last change point plus the offset. Instead:

$$t' = \\max\\{\\,s < t \\;:\\; |\\varphi(s) - \\varphi(t)| \\ge \\Delta\\varphi\\,\\}$$

the most recent earlier step far enough away in *rotation*. This guarantees two
things a step offset does not: the state is separated from the current one, and
it is one the model actually visited -- forgetting of a distribution never seen
is not forgetting. Where no such step exists (early in a run, or a stationary
one) the probe is **undefined and reported as such**, never as "no forgetting".

**Sets are cached by rotation, not by step.** Two steps at the same rotation
share one tensor, so a stationary run builds exactly one set rather than 1500,
and a piecewise run builds one per regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.data.transforms import ImageTransform
from dekf_bench.env.drift import Drift, DriftState
from dekf_bench.env.partition import largest_remainder

#: Rotations closer than this are treated as the same state for caching, and as
#: indistinguishable when checking that the backward probe is separated.
ROTATION_TOLERANCE = 1e-6


class EvalSetError(ValueError):
    """Raised when an evaluation set cannot be built or is degenerate."""


@dataclass(frozen=True)
class EvalSet:
    """A scored-against tensor pair, tagged with the state it was built at.

    ``rotation_degrees`` is carried so a score can be *checked* against the data
    it is being compared with rather than trusted.
    """

    name: str
    images: torch.Tensor
    labels: torch.Tensor
    rotation_degrees: float
    step: int
    source_step: int | None = None
    #: The class composition this set was built to, when prior drift is on.
    #: Carried for the same reason as `rotation_degrees`: so a score can be
    #: checked against the distribution it claims to be measured on.
    class_prior: tuple[float, ...] | None = None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def batches(self, batch_size: int):
        """Iterate in chunks, so a 10 000-image set does not go through the model
        in one allocation."""
        if batch_size < 1:
            raise EvalSetError(f"batch_size must be >= 1, got {batch_size}")
        for start in range(0, len(self), batch_size):
            stop = start + batch_size
            yield self.images[start:stop], self.labels[start:stop]


class EvalSetBuilder:
    """Builds and caches the evaluation sets for one run.

    Holds the environment's *own* transform and drift objects, which is what
    makes "train and eval see the same rotation" structural rather than
    conventional.
    """

    def __init__(
        self,
        test: MnistSplit,
        transform: ImageTransform,
        drift: Drift,
        horizon: int,
        backward_separation_degrees: float = 15.0,
        priors: Any = None,
        n_classes: int = 10,
    ) -> None:
        if backward_separation_degrees <= 0:
            raise EvalSetError(
                f"backward_separation_degrees must be > 0, got {backward_separation_degrees}"
            )
        self.test = test
        self.transform = transform
        self.drift = drift
        self.horizon = horizon
        self.separation = backward_separation_degrees
        self.priors = priors
        self.n_classes = n_classes
        self._cache: dict[int, torch.Tensor] = {}
        self._by_class = [
            torch.nonzero(test.labels == c, as_tuple=True)[0] for c in range(n_classes)
        ]
        self._matched_size = self._plan_matched_size()

    # -- composition matching ----------------------------------------------- #

    def _plan_matched_size(self) -> int | None:
        """One evaluation-set size for the whole run, or ``None`` without priors.

        **Fixed for the run on purpose.** A composition-matched set can only be
        as large as its scarcest class allows, and that limit moves as the prior
        drifts. Letting the size follow it would make the sampling noise floor
        drift too -- and a break threshold read off a curve whose noise floor is
        changing is measuring the evaluation, not the learner.

        The maximum of a linear function over a segment is attained at an
        endpoint, so scanning the two endpoints bounds the whole path exactly;
        no step-by-step search is needed.
        """
        if self.priors is None:
            return None
        available = torch.tensor([len(pool) for pool in self._by_class], dtype=torch.float64)
        feasible = []
        for progress in (0.0, 1.0):
            table = self.priors.at(progress)
            for node in range(table.shape[0]):
                share = table[node]
                positive = share > 0
                feasible.append(float((available[positive] / share[positive]).min()))
        return max(1, int(min(feasible)))

    def _matched_indices(self, share: torch.Tensor) -> torch.Tensor:
        """Test indices whose class composition realises ``share``.

        Deterministic -- a fixed prefix of each class pool -- so two runs at the
        same drift state score against the *same* images and a difference
        between them cannot be sampling noise.
        """
        size = self._matched_size or len(self.test)
        counts = largest_remainder(share, size)
        parts = [self._by_class[c][: int(counts[c])] for c in range(self.n_classes)]
        return torch.cat([part for part in parts if len(part)])

    def _prior_at(self, step: int, node: int | None) -> torch.Tensor | None:
        if self.priors is None:
            return None
        table = self.priors.at(self.drift.progress_at(step, node or 0))
        return table[node or 0]

    # -- the sets ----------------------------------------------------------- #

    def current(self, step: int, node: int | None = None) -> EvalSet:
        """The test split at the drift state the data carries at ``step``.

        Under class-prior drift this means the *composition* too, not only the
        rotation: an agent trained on a drifting prior and scored on a uniform
        split would show a gap that is the mismatch rather than its tracking.
        """
        rotation = self._rotation(step, node)
        return self._compose("current", step, rotation, self._prior_at(step, node))

    def _compose(
        self,
        name: str,
        step: int,
        rotation: float,
        share: torch.Tensor | None,
        source_step: int | None = None,
    ) -> EvalSet:
        images = self._images_at(rotation)
        if share is None:
            return EvalSet(
                name=name,
                images=images,
                labels=self.test.labels,
                rotation_degrees=rotation,
                step=step,
                source_step=source_step,
            )
        indices = self._matched_indices(share)
        return EvalSet(
            name=name,
            images=images[indices],
            labels=self.test.labels[indices],
            rotation_degrees=rotation,
            step=step,
            source_step=source_step,
            class_prior=tuple(float(value) for value in share),
        )

    def canonical(self, step: int) -> EvalSet:
        """Unrotated, whatever the drift state. Comparable across every run."""
        return EvalSet(
            name="canonical",
            images=self._images_at(0.0),
            labels=self.test.labels,
            rotation_degrees=0.0,
            step=step,
        )

    def backward(self, step: int, node: int | None = None) -> EvalSet | None:
        """A rotation the model saw earlier, or ``None`` when there is none.

        ``None`` rather than a zero-separation set: a probe that cannot
        distinguish the past from the present has not measured "no forgetting",
        it has measured nothing, and the two must not log the same value.
        """
        source = self.backward_step(step, node)
        if source is None:
            return None
        # Composed at the *source* state, prior included: the probe asks how the
        # model does on a distribution it has already seen, and that means the
        # whole distribution, not its rotation with a present-day composition.
        return self._compose(
            "backward",
            step,
            self._rotation(source, node),
            self._prior_at(source, node),
            source_step=source,
        )

    def backward_step(self, step: int, node: int | None = None) -> int | None:
        """The most recent earlier step at least ``separation`` degrees away.

        Searched backwards from ``step`` so the answer is the *most recent*
        qualifying state -- the one most plausibly still "remembered" -- rather
        than the earliest.
        """
        current = self._rotation(step, node)
        for source in range(step - 1, -1, -1):
            if abs(self._rotation(source, node) - current) >= self.separation:
                return source
        return None

    def mean_rotation(self, step: int) -> float:
        """The network-average rotation at ``step``.

        Equals every agent's rotation under global drift. Under ``per_node`` it
        is the state *no agent actually faces*, which is why scoring against it
        is logged as a second view rather than as the metric.
        """
        states = self.drift.states_at(step)
        return sum(state.rotation_degrees for state in states) / len(states)

    def current_at_mean(self, step: int) -> EvalSet:
        """The test split at the network-mean rotation.

        Under ``per_node`` drift the agents sit at different rotations, so
        ``current`` is per-agent and the per-agent spread mixes "this agent
        learned worse" with "this agent is further from the mean rotation".
        Scoring everyone at the mean too separates the two, at the cost of a
        second evaluation pass in that regime only.
        """
        share = None
        if self.priors is not None:
            # The network-mean composition, matching the network-mean rotation:
            # a single reference state no agent occupies, which is the point.
            table = torch.stack(
                [
                    self.priors.at(self.drift.progress_at(step, node))[node]
                    for node in range(self.priors.n_nodes)
                ]
            )
            share = table.mean(dim=0)
        return self._compose("current_mean", step, self.mean_rotation(step), share)

    def at(self, name: str, step: int, node: int | None = None) -> EvalSet | None:
        if name == "current":
            return self.current(step, node)
        if name == "current_mean":
            return self.current_at_mean(step)
        if name == "canonical":
            return self.canonical(step)
        if name == "backward":
            return self.backward(step, node)
        raise EvalSetError(
            f"unknown evaluation set {name!r}; expected 'current', 'current_mean', "
            "'backward' or 'canonical'. 'prequential' is not built here -- it scores "
            "the incoming training batch."
        )

    # -- diagnostics -------------------------------------------------------- #

    def backward_is_available(self, step: int, node: int | None = None) -> bool:
        return self.backward_step(step, node) is not None

    def first_backward_step(self, node: int | None = None) -> int | None:
        """The earliest step at which the forgetting probe becomes defined.

        Worth reporting before a run: under a stationary schedule it is never,
        and a config asking for the backward set there is asking for nothing.
        """
        for step in range(self.horizon):
            if self.backward_is_available(step, node):
                return step
        return None

    def distinct_rotations(self, step_every: int = 1) -> int:
        """How many sets the run will actually build. One for a stationary run;
        one per regime for piecewise; one per step for linear."""
        return len(self.drift.distinct_rotations(self.horizon, step_every))

    def summary(self) -> dict[str, Any]:
        # Under per-node drift there is no network-wide state, and `state_at`
        # rightly refuses to invent one. A *diagnostic* may still name a
        # representative agent, so long as it says which -- that licence does
        # not extend to the metric path, where agent 0's rotation standing in
        # for everyone's is precisely the silent lie the error prevents.
        representative = 0 if self.drift.scope == "per_node" else None
        first = self.first_backward_step(representative)
        return {
            "n_test": len(self.test),
            "input_size": self.transform.size,
            "separation_degrees": self.separation,
            "backward_first_available_at": first,
            "backward_first_available_for_node": representative,
            "backward_ever_available": first is not None,
            "distinct_rotations": self.distinct_rotations(),
            "cached_sets": len(self._cache),
            "prior_matched": self.priors is not None,
            # The evaluation sample size, which sets the floor on how finely a
            # break threshold can be located. Worth reporting before a run.
            "evalset_size": self._matched_size or len(self.test),
        }

    # -- internals ---------------------------------------------------------- #

    def _rotation(self, step: int, node: int | None) -> float:
        if not 0 <= step < self.horizon:
            raise EvalSetError(f"step {step} outside 0..{self.horizon - 1}")
        state: DriftState = self.drift.state_at(step, node)
        return state.rotation_degrees

    def _images_at(self, rotation: float) -> torch.Tensor:
        """Cached by rotation, so two steps at the same state share one tensor."""
        key = round(rotation / ROTATION_TOLERANCE)
        if key not in self._cache:
            self._cache[key] = self.transform.apply(self.test.images, rotation)
        return self._cache[key]


def build_evalsets(config: Any, environment: Any, test: MnistSplit) -> EvalSetBuilder:
    """The builder a run's config asks for, wired to the environment's own
    transform and drift so the two paths cannot diverge.

    The test split is converted to the environment's dtype. The environment
    already converts *train* -- X0 runs everything in float64 -- and a test split
    left at float32 makes the first forward pass raise on a dtype mismatch. That
    is the loud version of the failure; the quiet one would be a silent
    promotion changing the numbers the exactness check is asserted against.
    """
    dtype = environment.train.images.dtype
    device = str(environment.train.images.device)
    if test.images.dtype != dtype or str(test.images.device) != device:
        # Device for the same reason as dtype, and with the same failure modes:
        # the loud one is a device mismatch on the first eval, the quiet one is
        # a per-eval copy that makes CUDA look slower than it is.
        test = test.to(dtype=dtype, device=device)

    plan = getattr(environment, "class_plan", None)
    return EvalSetBuilder(
        test=test,
        transform=environment.transform,
        drift=environment.drift,
        horizon=config.run.horizon,
        backward_separation_degrees=config.eval.backward_separation_degrees,
        priors=None if plan is None else plan.priors,
        n_classes=config.model.output_dim,
    )
