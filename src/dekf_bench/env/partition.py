"""Assigning disjoint shards of the training set to agents.

Each agent owns a shard nobody else touches. This module decides *which* indices
go where; it never sees an image, only labels and indices.

Two regimes:

**IID** -- a uniform random split. Every agent sees roughly the same class
distribution, so drift is the only source of non-stationarity and "what does
decentralization cost" is not confounded with "what does heterogeneity cost".

**Dirichlet label skew** -- each agent draws a class preference
$\\bm q_v \\sim \\mathrm{Dir}(\\beta \\bm 1_K)$. Small $\\beta$ gives agents that
see two or three digits; $\\beta \\to \\infty$ approaches IID. An agent that never
sees a 7 can only learn 7s through the combine step, which is the sharpest
statement of "does communication help".

**Shard sizes stay equal even under skew** (``balance_sizes``, default true).
The textbook construction -- draw a distribution over *agents* per class -- makes
shard sizes vary wildly, and at $\\beta = 0.1$ one agent can end up with a
handful of samples. Two things then go wrong. The experiment confounds label
skew with data volume, so X6 no longer isolates the effect it is named after.
And the config-time shard budget check assumes equal shards, so a small agent
silently exhausts its data part-way through a run. Balancing sizes and skewing
only the *composition* keeps X6 measuring one thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _dirichlet(
    concentration: float, size: int, draws: int, generator: torch.Generator | None
) -> torch.Tensor:
    """``draws`` samples from ``Dir(concentration * 1_size)``, reproducibly.

    ``torch.distributions.Dirichlet.sample()`` takes **no** generator argument
    and draws from the global RNG. Using it would make the partition depend on
    whatever else happened to consume random numbers first, so a run could not
    be reproduced from ``seed_partition`` -- exactly the failure the separable
    seed streams exist to prevent. numpy's ``Generator`` does accept explicit
    state, so the torch generator is consumed once to seed it.
    """
    seed = 0 if generator is None else int(torch.randint(0, 2**63 - 1, (1,), generator=generator))
    rng = np.random.default_rng(seed)
    sample = rng.dirichlet(np.full(size, float(concentration)), size=draws)
    return torch.from_numpy(sample).to(torch.float64)


class PartitionError(ValueError):
    """Raised when a partition is impossible or comes out malformed."""


@dataclass(frozen=True)
class Partition:
    """Disjoint index shards, one per agent.

    Attributes:
        shards: one int64 index tensor per agent, indexing into the training
            split. Disjoint, and together covering every index.
        kind: ``"iid"`` or ``"dirichlet"``.
        beta: the Dirichlet concentration, or ``None`` for IID.
        n_classes: number of label values, for the class-count helpers.
    """

    shards: tuple[torch.Tensor, ...]
    kind: str
    beta: float | None
    n_classes: int

    def __post_init__(self) -> None:
        if not self.shards:
            raise PartitionError("a partition needs at least one shard")
        for node, shard in enumerate(self.shards):
            if shard.dtype != torch.int64:
                raise PartitionError(f"shard {node}: indices must be int64, got {shard.dtype}")
            if shard.ndim != 1:
                raise PartitionError(f"shard {node}: expected a flat index tensor")

        stacked = torch.cat(self.shards)
        if stacked.numel() != len(torch.unique(stacked)):
            raise PartitionError(
                "shards overlap: at least one sample was assigned to two agents, which "
                "breaks the disjointness the exactly-once guarantee rests on"
            )

    @property
    def n_nodes(self) -> int:
        return len(self.shards)

    @property
    def sizes(self) -> torch.Tensor:
        """Samples per agent."""
        return torch.tensor([len(shard) for shard in self.shards], dtype=torch.int64)

    @property
    def total(self) -> int:
        return int(self.sizes.sum())

    def __getitem__(self, node: int) -> torch.Tensor:
        return self.shards[node]

    def class_counts(self, labels: torch.Tensor) -> torch.Tensor:
        """``(N, n_classes)`` counts: how many of each class each agent holds."""
        return torch.stack(
            [torch.bincount(labels[shard], minlength=self.n_classes) for shard in self.shards]
        )

    def class_distribution(self, labels: torch.Tensor) -> torch.Tensor:
        """``(N, n_classes)`` row-normalised class frequencies per agent."""
        counts = self.class_counts(labels).to(torch.float64)
        return counts / counts.sum(dim=1, keepdim=True).clamp(min=1)

    def skew(self, labels: torch.Tensor) -> float:
        """Mean total-variation distance from the global class distribution.

        0 means every agent mirrors the pooled data; it approaches
        $1 - 1/K$ when each agent holds a single class. A single number for the
        x-axis of X6, and a check that ``beta`` did what was asked.
        """
        per_agent = self.class_distribution(labels)
        overall = torch.bincount(labels, minlength=self.n_classes).to(torch.float64)
        overall = overall / overall.sum()
        return float(0.5 * (per_agent - overall).abs().sum(dim=1).mean())

    def classes_present(self, labels: torch.Tensor) -> torch.Tensor:
        """How many distinct classes each agent actually holds."""
        return (self.class_counts(labels) > 0).sum(dim=1)

    def summary(self, labels: torch.Tensor) -> dict[str, Any]:
        sizes = self.sizes
        present = self.classes_present(labels)
        return {
            "kind": self.kind,
            "beta": self.beta,
            "n_nodes": self.n_nodes,
            "total": self.total,
            "min_size": int(sizes.min()),
            "max_size": int(sizes.max()),
            "skew": self.skew(labels),
            "min_classes_present": int(present.min()),
            "max_classes_present": int(present.max()),
        }


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def _target_sizes(n_samples: int, n_nodes: int) -> list[int]:
    """Shard sizes differing by at most one, together covering everything."""
    base, remainder = divmod(n_samples, n_nodes)
    return [base + (1 if node < remainder else 0) for node in range(n_nodes)]


def _largest_remainder(weights: torch.Tensor, total: int) -> torch.Tensor:
    """Apportion ``total`` items across ``weights``, summing exactly to ``total``.

    Plain rounding does not sum to ``total``; the largest-remainder method
    assigns the floor to everyone and hands the leftovers to whoever was cut
    hardest. Without it a shard ends up a few samples short of its target and
    the "sizes are equal" guarantee quietly fails.
    """
    exact = weights * total
    counts = exact.floor().to(torch.int64)
    shortfall = total - int(counts.sum())
    if shortfall > 0:
        order = torch.argsort(exact - counts, descending=True)
        counts[order[:shortfall]] += 1
    return counts


def iid_partition(
    n_samples: int, n_nodes: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, ...]:
    """A uniform random split into ``n_nodes`` disjoint shards."""
    permutation = torch.randperm(n_samples, generator=generator)
    shards, start = [], 0
    for size in _target_sizes(n_samples, n_nodes):
        shards.append(permutation[start : start + size].sort().values)
        start += size
    return tuple(shards)


def dirichlet_partition(
    labels: torch.Tensor,
    n_nodes: int,
    beta: float,
    n_classes: int,
    generator: torch.Generator | None = None,
    balance_sizes: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Label-skewed shards, with sizes held equal by default.

    Each agent draws $\\bm q_v \\sim \\mathrm{Dir}(\\beta\\bm 1_K)$ and is then
    filled to its target size, taking as much of each class as its preference
    asks for and as the remaining pool can supply. Agents are served in a random
    order so that no agent systematically gets first choice.

    With ``balance_sizes=False`` the classical construction is used instead --
    a Dirichlet draw over *agents* per class -- and shard sizes vary.
    """
    if beta <= 0:
        raise PartitionError(f"dirichlet beta must be > 0, got {beta}")

    pools = [
        _shuffled(torch.nonzero(labels == c, as_tuple=True)[0], generator) for c in range(n_classes)
    ]

    if not balance_sizes:
        return _classical_dirichlet(pools, n_nodes, beta, generator)

    preferences = _dirichlet(beta, n_classes, n_nodes, generator)

    targets = _target_sizes(int(labels.numel()), n_nodes)
    # Explicit ints: Tensor.tolist() is typed as returning nested lists, so the
    # index type would otherwise widen to `int | list[int]`.
    order = [int(node) for node in torch.randperm(n_nodes, generator=generator)]
    shards: list[list[torch.Tensor]] = [[] for _ in range(n_nodes)]

    for node in order:
        wanted = _largest_remainder(preferences[node], targets[node])
        taken = 0
        for c in range(n_classes):
            take = min(int(wanted[c]), len(pools[c]))
            if take:
                shards[node].append(pools[c][:take])
                pools[c] = pools[c][take:]
                taken += take
        # The pool may not have been able to satisfy the draw; make up the
        # shortfall from whichever classes still have the most left, so the
        # shard still reaches its target size.
        while taken < targets[node]:
            fullest = max(range(n_classes), key=lambda c: len(pools[c]))
            if not len(pools[fullest]):
                raise PartitionError(  # pragma: no cover - totals make this unreachable
                    "ran out of samples while balancing shard sizes"
                )
            take = min(targets[node] - taken, len(pools[fullest]))
            shards[node].append(pools[fullest][:take])
            pools[fullest] = pools[fullest][take:]
            taken += take

    return tuple(torch.cat(parts).sort().values for parts in shards)


def demand_partition(
    labels: torch.Tensor,
    demand: torch.Tensor,
    n_classes: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Shards holding exactly the per-class counts a class-prior plan will ask for.

    Under a drifting class prior the composition an agent needs is decided by
    the plan, not by a one-off Dirichlet draw, so the shard is *sized from the
    plan*. Each agent is first given precisely ``demand[v, k]`` samples of
    class ``k``; the shards are then topped up from whatever is left so that
    sizes stay equal and the partition still covers the training set.

    The filler is never served -- ``build_stream`` pops only the classes the
    plan names -- but it has to exist, because "shards partition the training
    set" is what disjointness and the exactly-once accounting rest on.
    """
    if demand.shape[1] != n_classes:
        raise PartitionError(f"demand has {demand.shape[1]} classes, expected {n_classes}")
    if int(demand.min()) < 0:
        raise PartitionError("demand counts must be non-negative")

    n_nodes = int(demand.shape[0])
    pools = [
        _shuffled(torch.nonzero(labels == c, as_tuple=True)[0], generator) for c in range(n_classes)
    ]
    shards: list[list[torch.Tensor]] = [[] for _ in range(n_nodes)]

    # Demand first, for every agent, before anyone is topped up: the filler is
    # interchangeable but the demanded samples are not, and serving one agent
    # completely before the next would let filler consume a class a later agent
    # still needs.
    for node in range(n_nodes):
        for c in range(n_classes):
            want = int(demand[node, c])
            if want > len(pools[c]):
                raise PartitionError(
                    f"agent {node} needs {want} samples of class {c} but only "
                    f"{len(pools[c])} remain. The plan should have been checked with "
                    "priors.check_plan_is_feasible before the partition was built."
                )
            if want:
                shards[node].append(pools[c][:want])
                pools[c] = pools[c][want:]

    targets = _target_sizes(int(labels.numel()), n_nodes)
    shortfall = []
    for node in range(n_nodes):
        taken = int(sum(len(part) for part in shards[node]))
        if taken > targets[node]:
            raise PartitionError(
                f"agent {node} is demanded {taken} samples but its equal-size share is "
                f"{targets[node]}. Shorten the horizon or lower samples_per_node_per_step."
            )
        shortfall.append(targets[node] - taken)

    # Spread the filler across classes rather than handing each agent its whole
    # shortfall from whichever pool is currently largest. The filler is never
    # served, but it lands in `partition.skew()`, and a lumpy remainder would
    # report a skew the experiment does not have.
    for c in range(n_classes):
        remaining = shortfall.copy()
        if not len(pools[c]) or not sum(remaining):
            continue
        share = _largest_remainder(
            torch.tensor(remaining, dtype=torch.float64) / sum(remaining),
            min(len(pools[c]), sum(remaining)),
        )
        for node in range(n_nodes):
            take = min(int(share[node]), shortfall[node], len(pools[c]))
            if take:
                shards[node].append(pools[c][:take])
                pools[c] = pools[c][take:]
                shortfall[node] -= take

    # Anything still short -- rounding, or a class pool that emptied -- is made
    # up from whatever is left, which is now a small remainder rather than the
    # bulk of the shard.
    for node in range(n_nodes):
        while shortfall[node] > 0:
            fullest = max(range(n_classes), key=lambda c: len(pools[c]))
            if not len(pools[fullest]):  # pragma: no cover - totals make this unreachable
                raise PartitionError("ran out of samples while balancing shard sizes")
            take = min(shortfall[node], len(pools[fullest]))
            shards[node].append(pools[fullest][:take])
            pools[fullest] = pools[fullest][take:]
            shortfall[node] -= take

    return tuple(torch.cat(parts).sort().values for parts in shards)


def _classical_dirichlet(
    pools: list[torch.Tensor],
    n_nodes: int,
    beta: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, ...]:
    """Per-class Dirichlet over agents. Sizes vary, sometimes drastically."""
    proportions_per_class = _dirichlet(beta, n_nodes, len(pools), generator)
    shards: list[list[torch.Tensor]] = [[] for _ in range(n_nodes)]
    for pool, proportions in zip(pools, proportions_per_class, strict=True):
        counts = _largest_remainder(proportions, int(pool.numel()))
        start = 0
        for node in range(n_nodes):
            take = int(counts[node])
            shards[node].append(pool[start : start + take])
            start += take
    return tuple(
        (
            torch.cat(parts).sort().values if parts else torch.empty(0, dtype=torch.int64)
        )  # pragma: no cover
        for parts in shards
    )


def _shuffled(indices: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return indices[torch.randperm(int(indices.numel()), generator=generator)]


def build_partition(
    labels: torch.Tensor,
    n_nodes: int,
    kind: str = "iid",
    beta: float = 1.0,
    n_classes: int = 10,
    generator: torch.Generator | None = None,
    balance_sizes: bool = True,
    min_shard_size: int | None = None,
    demand: torch.Tensor | None = None,
) -> Partition:
    """Assign disjoint shards.

    Args:
        labels: the training labels. Only used to see class membership; images
            are never touched here.
        n_nodes: $N$.
        kind: ``"iid"`` or ``"dirichlet"``.
        beta: Dirichlet concentration. Ignored for IID.
        n_classes: label cardinality $K$.
        generator: seeded from the ``partition`` stream, so the split can be
            held fixed while initialization or stream order varies.
        balance_sizes: keep shard sizes equal under skew. See the module
            docstring for why this is the default.
        min_shard_size: reject a partition that leaves any agent with fewer than
            this many samples. Pass ``n * T`` to guarantee the run cannot
            exhaust a shard.
        demand: ``(N, K)`` per-class counts from a class-prior plan. When given
            it overrides ``kind``: the composition is dictated by the plan, so
            a Dirichlet draw would only fight it.
    """
    n_samples = int(labels.numel())
    if n_nodes < 1:
        raise PartitionError(f"n_nodes must be >= 1, got {n_nodes}")
    if n_nodes > n_samples:
        raise PartitionError(f"cannot split {n_samples} samples across {n_nodes} agents")

    if demand is not None:
        shards = demand_partition(labels, demand, n_classes, generator)
        return _finish(
            Partition(shards=shards, kind="prior_drift", beta=None, n_classes=n_classes),
            n_samples,
            min_shard_size,
        )

    if kind == "iid":
        shards = iid_partition(n_samples, n_nodes, generator)
        beta_recorded = None
    elif kind == "dirichlet":
        shards = dirichlet_partition(labels, n_nodes, beta, n_classes, generator, balance_sizes)
        beta_recorded = beta
    else:
        raise PartitionError(f"unknown partition kind {kind!r}; expected 'iid' or 'dirichlet'")

    partition = Partition(shards=shards, kind=kind, beta=beta_recorded, n_classes=n_classes)
    return _finish(partition, n_samples, min_shard_size)


def _finish(partition: Partition, n_samples: int, min_shard_size: int | None) -> Partition:
    """The checks every construction shares: coverage, then the shard budget."""
    _check_covers(partition, n_samples)
    if min_shard_size is not None:
        smallest = int(partition.sizes.min())
        if smallest < min_shard_size:
            raise PartitionError(
                f"smallest shard holds {smallest} samples but the run needs "
                f"{min_shard_size} per agent. With balance_sizes=False the Dirichlet draw "
                "leaves some agents far smaller than 60000/N, so the config-time shard "
                "budget check -- which assumes equal shards -- does not protect the run. "
                "Set balance_sizes=true, raise beta, or shorten the horizon."
            )
    return partition


def _check_covers(partition: Partition, n_samples: int) -> None:
    """Every index assigned exactly once. Disjointness is checked in the dataclass."""
    if partition.total != n_samples:
        raise PartitionError(
            f"shards hold {partition.total} of {n_samples} samples; a partition must "
            "cover the training set"
        )
    stacked = torch.cat(partition.shards)
    if int(stacked.min()) < 0 or int(stacked.max()) >= n_samples:
        raise PartitionError("shard indices fall outside the training set")


def build_partition_from_config(
    config: Any,
    labels: torch.Tensor,
    generator: torch.Generator | None = None,
    demand: torch.Tensor | None = None,
) -> Partition:
    """The partition a run's config asks for, with the shard budget enforced."""
    needed = config.env.samples_per_node_per_step * config.run.horizon
    return build_partition(
        labels=labels,
        n_nodes=config.graph.n_nodes,
        kind=config.env.partition.kind,
        beta=config.env.partition.beta,
        n_classes=config.model.output_dim,
        generator=generator,
        min_shard_size=None if config.env.allow_epochs else needed,
        demand=demand,
    )
