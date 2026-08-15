"""Per-agent sample streams: who receives which indices at which step.

Each agent draws from its own shard, ``n`` samples per step, in a fixed shuffled
order. This module decides *which* indices; it never touches an image and never
applies drift.

**The stream is a pure function of (agent, step), not a cursor.** The obvious
implementation keeps a per-agent pointer and advances it each step. That works
until something needs to look at step 900 without having walked steps 0..899 --
an evaluation set, a test, a runner that restarts -- and then the answer depends
on how you got there. Here the label-availability draws are made once up front,
so the offset into a shard at step $t$ is a prefix sum and any $(v, t)$ can be
answered directly and identically however it is reached.

**Exactly once.** Without ``allow_epochs`` no index is served twice over a run.
That is not tidiness: the research note's Assumption 5 requires every labelled
observation to enter the network likelihood exactly once, and a sample counted
twice makes the filter's covariance shrink faster than the evidence justifies.
The constructor checks the whole run up front rather than discovering the
shortfall at step 1400.

**An unlabelled step consumes nothing.** With $\\pi_{\\text{lab}} < 1$ an agent
idles: it receives no samples, contributes no measurement, and its shard is
untouched. The alternative -- serving the samples but withholding the labels --
destroys them, because shards are disjoint and finite and no learner here can
use an unlabelled sample. It would also make the maximum horizon independent of
$\\pi_{\\text{lab}}$ (the shard drains at $n$ per step either way), which
forecloses the experiment of running a sparse-label regime longer to see whether
it eventually catches up. See ``docs/design_notes.md`` D24.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dekf_bench.env.partition import Partition


class StreamError(ValueError):
    """Raised when a stream cannot supply the run it was asked for."""


@dataclass(frozen=True)
class Stream:
    """The full schedule of which indices each agent receives at each step.

    Attributes:
        order: per-agent shard indices in the order they will be consumed.
        has_label: ``(N, T)`` bool -- whether agent $v$ has labels at step $t$.
        offsets: ``(N, T)`` int64 -- where in ``order[v]`` step $t$ starts.
            A prefix sum over ``has_label``, which is what makes the stream
            positional rather than stateful.
        samples_per_step: $n$.
        allow_epochs: whether a shard may wrap around and repeat.
    """

    order: tuple[torch.Tensor, ...]
    has_label: torch.Tensor
    offsets: torch.Tensor
    samples_per_step: int
    allow_epochs: bool = False

    def __post_init__(self) -> None:
        if self.samples_per_step < 1:
            raise StreamError(f"samples_per_step must be >= 1, got {self.samples_per_step}")
        if self.has_label.dtype != torch.bool:
            raise StreamError(f"has_label must be bool, got {self.has_label.dtype}")
        if self.has_label.shape != self.offsets.shape:
            raise StreamError(
                f"has_label {tuple(self.has_label.shape)} and offsets "
                f"{tuple(self.offsets.shape)} must agree"
            )
        if len(self.order) != self.has_label.shape[0]:
            raise StreamError(
                f"{len(self.order)} shards but has_label covers "
                f"{self.has_label.shape[0]} agents"
            )

    @property
    def n_nodes(self) -> int:
        return len(self.order)

    @property
    def horizon(self) -> int:
        return int(self.has_label.shape[1])

    def is_labelled(self, node: int, step: int) -> bool:
        self._check(node, step)
        return bool(self.has_label[node, step])

    def indices_at(self, node: int, step: int) -> torch.Tensor:
        """The training-set indices agent ``node`` receives at ``step``.

        Empty when the agent idles. Never partially filled: an agent either gets
        its full ``n`` samples or none, so batch sizes stay equal across the
        agents that are active -- which is what the X0 identity requires.
        """
        self._check(node, step)
        if not self.has_label[node, step]:
            return torch.empty(0, dtype=torch.int64)

        shard = self.order[node]
        start = int(self.offsets[node, step])
        stop = start + self.samples_per_step
        if stop <= len(shard):
            return shard[start:stop]
        if not self.allow_epochs:  # pragma: no cover - the constructor rejects this
            raise StreamError(
                f"agent {node} exhausted its shard at step {step}; "
                "the constructor should have caught this"
            )
        positions = torch.arange(start, stop) % len(shard)
        return shard[positions]

    def batch_at(self, step: int) -> dict[int, torch.Tensor]:
        """Every agent's indices at ``step``. Idle agents map to an empty tensor."""
        return {node: self.indices_at(node, step) for node in range(self.n_nodes)}

    def active_at(self, step: int) -> tuple[int, ...]:
        """Which agents have labels at ``step``."""
        return tuple(torch.nonzero(self.has_label[:, step], as_tuple=True)[0].tolist())

    def consumed_by(self, node: int, through: int | None = None) -> torch.Tensor:
        """Every index agent ``node`` has been served up to and including ``through``."""
        through = self.horizon - 1 if through is None else through
        parts = [self.indices_at(node, step) for step in range(through + 1)]
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)

    def labelled_steps(self, node: int) -> int:
        return int(self.has_label[node].sum())

    def required_samples(self, node: int) -> int:
        """How many shard entries the whole run will consume for this agent."""
        return self.labelled_steps(node) * self.samples_per_step

    @property
    def empirical_label_rate(self) -> float:
        """The realised fraction of labelled agent-steps."""
        return float(self.has_label.to(torch.float64).mean())

    def summary(self) -> dict[str, Any]:
        required = [self.required_samples(node) for node in range(self.n_nodes)]
        return {
            "n_nodes": self.n_nodes,
            "horizon": self.horizon,
            "samples_per_step": self.samples_per_step,
            "allow_epochs": self.allow_epochs,
            "empirical_label_rate": self.empirical_label_rate,
            "min_required": min(required),
            "max_required": max(required),
            "total_consumed": sum(required),
        }

    def _check(self, node: int, step: int) -> None:
        if not 0 <= node < self.n_nodes:
            raise StreamError(f"node {node} outside 0..{self.n_nodes - 1}")
        if not 0 <= step < self.horizon:
            raise StreamError(f"step {step} outside 0..{self.horizon - 1}")


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def _label_mask(
    n_nodes: int, horizon: int, availability: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Bernoulli draws for every (agent, step), made once up front.

    Drawn as a block rather than per step, so the schedule does not depend on
    the order steps are visited in -- and so the realised rate can be checked
    before a run starts rather than inferred from it afterwards.
    """
    if not 0.0 <= availability <= 1.0:
        raise StreamError(f"label_availability must lie in [0, 1], got {availability}")
    if availability == 1.0:
        # Exact, not almost-surely: the X0 identity needs equal batch sizes
        # across agents at every step, and "1.0 minus a rounding accident" is
        # not a guarantee.
        return torch.ones(n_nodes, horizon, dtype=torch.bool)
    if availability == 0.0:
        return torch.zeros(n_nodes, horizon, dtype=torch.bool)
    draws = torch.rand(n_nodes, horizon, generator=generator)
    return draws < availability


def _class_ordered(
    shard: torch.Tensor,
    labels: torch.Tensor,
    wanted: torch.Tensor,
    n_classes: int,
    generator: torch.Generator | None,
    node: int,
) -> torch.Tensor:
    """Order a shard so that consuming it front-to-back realises ``wanted``.

    This is the whole of class-prior drift as far as the stream is concerned: a
    permutation, computed up front. Each index is still popped from its class
    queue exactly once, so disjointness and exactly-once are untouched, and the
    prefix-sum offsets keep working because nothing about *how many* samples a
    step consumes has changed -- only which.

    Whatever the plan does not name is shuffled onto the end. It is never
    served; it exists so the shard remains a partition of the training set.
    """
    shard_labels = labels[shard]
    queues: dict[int, list[int]] = {}
    for c in range(n_classes):
        held = shard[shard_labels == c]
        shuffled = held[torch.randperm(len(held), generator=generator)]
        queues[c] = shuffled.tolist()

    served: list[int] = []
    for target in wanted.tolist():
        queue = queues[int(target)]
        if not queue:
            raise StreamError(
                f"agent {node} ran out of class {int(target)} while ordering its shard. "
                "The partition must be built from the same plan the stream serves -- pass "
                "the plan's demand() to build_partition."
            )
        served.append(queue.pop())
    leftover = [index for queue in queues.values() for index in queue]
    rest = torch.tensor(leftover, dtype=torch.int64)
    if len(rest):
        rest = rest[torch.randperm(len(rest), generator=generator)]
    return torch.cat([torch.tensor(served, dtype=torch.int64), rest])


def build_stream(
    partition: Partition,
    horizon: int,
    samples_per_step: int = 2,
    label_availability: float = 1.0,
    generator: torch.Generator | None = None,
    allow_epochs: bool = False,
    class_plan: Any = None,
    labels: torch.Tensor | None = None,
) -> Stream:
    """Plan the whole run's sampling.

    Args:
        partition: the disjoint shards. Only indices are read; no images.
        horizon: $T$.
        samples_per_step: $n$.
        label_availability: $\\pi_{\\text{lab}}$, the per-(agent, step)
            probability of having labels.
        generator: from the ``stream`` seed stream, so sample order and label
            draws can vary while the partition is held fixed.
        allow_epochs: permit a shard to wrap and repeat. Forfeits exactly-once.
        class_plan: a ``priors.ClassPlan``. When given, each shard is ordered so
            that the classes served follow the plan -- which is how class-prior
            drift reaches the learner.
        labels: the training labels, needed only to sort a shard by class.
    """
    if horizon < 1:
        raise StreamError(f"horizon must be >= 1, got {horizon}")

    n_nodes = partition.n_nodes
    has_label = _label_mask(n_nodes, horizon, label_availability, generator)

    # Offsets are a prefix sum of the labelled steps *before* t, so step t
    # starts where step t-1 left off and an idle step advances nothing.
    labelled = has_label.to(torch.int64)
    before = torch.cumsum(labelled, dim=1) - labelled
    offsets = before * samples_per_step

    if class_plan is None:
        order = tuple(
            shard[torch.randperm(len(shard), generator=generator)] for shard in partition.shards
        )
    else:
        if labels is None:
            raise StreamError("a class plan needs labels to sort each shard by class")
        # Only the labelled steps are drawn from. An idle agent consumes
        # nothing, so taking its plan entry anyway would slide every later class
        # one step out of step with the drift schedule that chose it.
        order = tuple(
            _class_ordered(
                shard=shard,
                labels=labels,
                wanted=class_plan.classes[node][has_label[node]].reshape(-1),
                n_classes=class_plan.n_classes,
                generator=generator,
                node=node,
            )
            for node, shard in enumerate(partition.shards)
        )
    stream = Stream(
        order=order,
        has_label=has_label,
        offsets=offsets,
        samples_per_step=samples_per_step,
        allow_epochs=allow_epochs,
    )
    _check_shards_last(stream, partition, allow_epochs)
    return stream


def _check_shards_last(stream: Stream, partition: Partition, allow_epochs: bool) -> None:
    """Reject a run that would exhaust a shard, before it starts."""
    if allow_epochs:
        return
    for node in range(stream.n_nodes):
        required = stream.required_samples(node)
        available = int(partition.sizes[node])
        if required > available:
            raise StreamError(
                f"agent {node} would consume {required} samples over the run but its shard "
                f"holds {available}. At n={stream.samples_per_step} and "
                f"{stream.labelled_steps(node)} labelled steps the stream outruns the shard. "
                "Shorten the horizon, reduce n, reduce N, or set env.allow_epochs=true "
                "(which lets samples repeat and forfeits the exactly-once guarantee that "
                "the filter's covariance accounting depends on)."
            )


def build_stream_from_config(
    config: Any,
    partition: Partition,
    generator: torch.Generator | None = None,
    class_plan: Any = None,
    labels: torch.Tensor | None = None,
) -> Stream:
    """The stream a run's config asks for."""
    return build_stream(
        partition=partition,
        horizon=config.run.horizon,
        samples_per_step=config.env.samples_per_node_per_step,
        label_availability=config.env.label_availability,
        generator=generator,
        allow_epochs=config.env.allow_epochs,
        class_plan=class_plan,
        labels=labels,
    )
