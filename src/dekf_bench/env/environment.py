"""The environment: what each agent observes at each step.

Composes the graph, the partition, the stream, the drift schedule and the image
transform into one object, and answers a single question: *what does agent $v$
see at step $t$?* It contains no learner logic -- it does not know what a
gradient is, and the same observations are handed unchanged to every learner in
the run.

**Observations are shared and read-only.** ``simulate.py`` calls ``step(t)``
*once* and gives the result to centralized SGD, diffusion SGD and local-only
alike (design note D4). If any of them normalised, augmented or otherwise
mutated a tensor in place, the learners that ran after it in the same iteration
would silently train on different data -- and the exactness check, which
compares two of them, would fail for a reason having nothing to do with the
algebra it is testing. :class:`Observation` is frozen, and
:meth:`Environment.assert_unmodified` gives the runner a positive check.

**The environment is frozen and positional too.** ``step(t)`` is a pure function
of $t$: the same call returns the same tensors whenever it is made, in whatever
order. This follows the stream (D23) for the same reason -- an evaluation set,
a test, or a restarted run must be able to ask about an arbitrary step and get
the answer the run would have produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.data.transforms import ImageTransform, build_transform_from_config
from dekf_bench.env.drift import Drift, DriftState, build_drift
from dekf_bench.env.graph import Graph, Graphs, build_graphs
from dekf_bench.env.partition import Partition, build_partition_from_config
from dekf_bench.env.priors import build_class_plan_from_config, check_plan_is_feasible
from dekf_bench.env.stream import Stream, build_stream_from_config
from dekf_bench.runner.seeding import Seeds
from dekf_bench.utils.determinism import resolve_device


class EnvironmentError(RuntimeError):
    """Raised when the environment is misconfigured or an observation is corrupted."""


@dataclass(frozen=True)
class Observation:
    """What one agent receives at one step.

    Frozen, and the tensors must be treated as read-only: one instance is shared
    by every learner in the run.

    Attributes:
        x: ``(n, 1, size, size)`` transformed images. Empty when the agent idles.
        y: ``(n,)`` int64 labels, or ``None`` when the agent idles.
        has_label: whether this agent has a measurement this step.
        n_samples: number of samples, 0 when idle.
        node: which agent.
        step: which step.
        rotation_degrees: the drift state these images were built at. Recorded so
            an evaluation set can be checked against it rather than trusted.
    """

    x: torch.Tensor
    y: torch.Tensor | None
    has_label: bool
    n_samples: int
    node: int
    step: int
    rotation_degrees: float

    def __post_init__(self) -> None:
        if self.has_label != (self.n_samples > 0):
            raise EnvironmentError(
                f"agent {self.node} at step {self.step}: has_label={self.has_label} "
                f"disagrees with n_samples={self.n_samples}"
            )
        if self.x.shape[0] != self.n_samples:
            raise EnvironmentError(
                f"agent {self.node} at step {self.step}: {self.x.shape[0]} images "
                f"but n_samples={self.n_samples}"
            )
        if self.has_label:
            if self.y is None:
                raise EnvironmentError(
                    f"agent {self.node} at step {self.step}: labelled but y is None"
                )
            if self.y.shape[0] != self.n_samples:
                raise EnvironmentError(
                    f"agent {self.node} at step {self.step}: {self.y.shape[0]} labels "
                    f"but {self.n_samples} samples"
                )
        elif self.y is not None:
            raise EnvironmentError(
                f"agent {self.node} at step {self.step}: unlabelled but y is not None"
            )

    def __len__(self) -> int:
        return self.n_samples

    def checksum(self) -> tuple[float, float]:
        """A cheap fingerprint, for detecting in-place mutation by a learner."""
        x_sum = float(self.x.to(torch.float64).sum()) if self.n_samples else 0.0
        y_sum = float(self.y.sum()) if self.y is not None else 0.0
        return x_sum, y_sum


def pool(observations: dict[int, Observation]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack every labelled agent's samples into one batch.

    What the centralized learner trains on: $\\mathcal D_t = \\bigcup_v
    \\mathcal D_t^v$. Lives here rather than in the learner because it is data
    handling, and because the exactness identity depends on the pooled batch
    being exactly the union of the per-agent ones -- worth having in one place
    that both the learner and its test can call.
    """
    labelled = [obs for obs in observations.values() if obs.has_label]
    if not labelled:
        example = next(iter(observations.values()))
        # The device is carried from the example rather than defaulted: an empty
        # batch that lands on the CPU while the belief is on CUDA fails only on
        # the steps where nothing is labelled, which is the hardest kind of bug
        # to reproduce.
        return (
            torch.empty((0, *example.x.shape[1:]), dtype=example.x.dtype, device=example.x.device),
            torch.empty(0, dtype=torch.int64, device=example.x.device),
        )
    xs = torch.cat([obs.x for obs in labelled])
    ys = torch.cat([obs.y for obs in labelled])  # type: ignore[misc]
    return xs, ys


@dataclass(frozen=True)
class Environment:
    """Graph, shards, stream, drift and transform, composed."""

    config: Any
    seeds: Seeds
    train: MnistSplit
    graphs: Graphs
    partition: Partition
    stream: Stream
    drift: Drift
    transform: ImageTransform
    #: The class-prior plan, when that channel is on. Retained rather than
    #: discarded after the shards are built, because the *evaluation* sets need
    #: the same priors: scoring a prior-drifted learner on a uniform split would
    #: report the mismatch as though it were tracking error.
    class_plan: Any = None

    # -- what the runner needs -------------------------------------------- #

    @property
    def graph(self) -> Graph:
        """The communication graph. The data graph is ``graphs.data``."""
        return self.graphs.comm

    @property
    def horizon(self) -> int:
        return int(self.config.run.horizon)

    @property
    def n_nodes(self) -> int:
        return self.graphs.n_nodes

    def drift_state(self, step: int, node: int | None = None) -> DriftState:
        """The drift the data carries at ``step``.

        The evaluation sets read this rather than recomputing a rotation, so
        train and eval cannot disagree about where the distribution is.
        """
        self._check_step(step)
        return self.drift.state_at(step, node)

    def step(self, step: int) -> dict[int, Observation]:
        """Every agent's observation at ``step``.

        Called once per step by the runner; the result is shared by all
        learners, so nothing here may be mutated downstream.
        """
        self._check_step(step)
        return {node: self.observe(node, step) for node in range(self.n_nodes)}

    def observe(self, node: int, step: int) -> Observation:
        """One agent's observation. Empty when the agent idles."""
        self._check_step(step)
        indices = self.stream.indices_at(node, step)
        rotation = self._rotation_for(step, node)

        if indices.numel() == 0:
            empty = torch.empty(
                (0, 1, self.transform.size, self.transform.size),
                dtype=self.train.images.dtype,
                device=self.train.images.device,
            )
            return Observation(
                x=empty,
                y=None,
                has_label=False,
                n_samples=0,
                node=node,
                step=step,
                rotation_degrees=rotation,
            )

        images = self.train.images[indices]
        return Observation(
            x=self.transform.apply(images, rotation),
            y=self.train.labels[indices].clone(),
            has_label=True,
            n_samples=int(indices.numel()),
            node=node,
            step=step,
            rotation_degrees=rotation,
        )

    def _rotation_for(self, step: int, node: int) -> float:
        scope = self.config.env.drift_scope
        return self.drift.rotation_at(step, node if scope == "per_node" else 0)

    # -- integrity --------------------------------------------------------- #

    def assert_unmodified(self, observations: dict[int, Observation], step: int) -> None:
        """Check that no learner mutated a shared observation in place.

        A frozen dataclass stops a field being rebound but cannot stop
        ``obs.x.add_(1)``. This recomputes the observations and compares
        fingerprints, so the runner can verify cheaply on the evaluation steps
        rather than trusting every learner to behave.
        """
        for node, observed in observations.items():
            expected = self.observe(node, step).checksum()
            if observed.checksum() != expected:
                raise EnvironmentError(
                    f"agent {node}'s observation at step {step} was modified in place. "
                    "Observations are shared by every learner in the run, so a learner "
                    "that normalises or augments its input in place corrupts the data "
                    "the learners after it will see."
                )

    def summary(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "horizon": self.horizon,
            **{f"graph_{k}": v for k, v in self.graph.summary().items()},
            **{f"stream_{k}": v for k, v in self.stream.summary().items()},
            **{f"drift_{k}": v for k, v in self.drift.summary(self.horizon).items()},
            "partition_kind": self.partition.kind,
            "partition_skew": self.partition.skew(self.train.labels),
            "input_dim": self.transform.input_dim,
        }

    def reset(self, master_seed: int) -> Environment:
        """A fresh environment at another master seed.

        Returns a new instance rather than mutating this one. The spec sketches
        ``reset(seed) -> None``, but every other object here is frozen and
        positional; a mutable reset would be the only place in the environment
        where a stale reference could hand back data from a previous seed.
        """
        return build_environment(self.config, master_seed, self.train)

    def _check_step(self, step: int) -> None:
        if not 0 <= step < self.horizon:
            raise EnvironmentError(f"step {step} outside 0..{self.horizon - 1}")


def build_environment(config: Any, master_seed: int, train: MnistSplit) -> Environment:
    """Assemble an environment for one seed.

    Each component draws from its own seed stream, so the graph realization, the
    shard assignment and the sample order can each be held fixed while the
    others vary.
    """
    seeds = Seeds.from_master(master_seed)

    graphs = build_graphs(config, seeds.torch_generator("graph"))
    drift = build_drift(config)

    # The class-prior plan comes first when it is on: it decides the per-class
    # counts each shard must hold, so the partition is sized from it rather
    # than drawn independently and hoped to fit.
    plan = build_class_plan_from_config(config, drift, seeds.torch_generator("priors"))
    if plan is not None:
        check_plan_is_feasible(plan, train.labels)

    partition = build_partition_from_config(
        config,
        train.labels,
        seeds.torch_generator("partition"),
        demand=None if plan is None else plan.demand(),
    )
    stream = build_stream_from_config(
        config,
        partition,
        seeds.torch_generator("stream"),
        class_plan=plan,
        labels=train.labels,
    )
    transform = build_transform_from_config(config, train.images)

    dtype = torch.float64 if config.run.dtype == "float64" else torch.float32
    device = resolve_device(config.run.device)
    if train.images.dtype != dtype or str(train.images.device) != device:
        # Moved once here rather than per batch. The images are the only large
        # tensor in a run, and copying a shard to the device 1500 times would
        # cost more than the filter step it feeds (design note D58).
        train = train.to(dtype=dtype, device=device)

    return Environment(
        config=config,
        seeds=seeds,
        train=train,
        graphs=graphs,
        partition=partition,
        stream=stream,
        drift=drift,
        transform=transform,
        class_plan=plan,
    )
