"""Shard assignment: disjointness, coverage, and the shape of the skew.

Most tests run on synthetic labels, which is fast and lets the class balance be
controlled exactly. The few that need real MNIST are marked ``needs_data``.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.env.partition import (
    Partition,
    PartitionError,
    build_partition,
    build_partition_from_config,
)
from dekf_bench.utils.config import load_config

N_CLASSES = 10
BETAS = [0.1, 0.5, 1.0, 10.0, 100.0]


def labels_of(n_samples: int = 2000, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Exactly balanced synthetic labels, so any skew comes from the partition."""
    return torch.arange(n_samples, dtype=torch.int64) % n_classes


def make(
    n_nodes: int = 10,
    kind: str = "iid",
    beta: float = 1.0,
    seed: int = 0,
    labels: torch.Tensor | None = None,
    **kwargs,
) -> Partition:
    labels = labels_of() if labels is None else labels
    return build_partition(
        labels, n_nodes, kind, beta, N_CLASSES, torch.Generator().manual_seed(seed), **kwargs
    )


ALL_KINDS = [("iid", 1.0), *[("dirichlet", beta) for beta in BETAS]]


# =========================================================================== #
# 1. the guarantees every partition must give
# =========================================================================== #


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_shards_are_disjoint(kind: str, beta: float) -> None:
    """No sample belongs to two agents. Everything downstream -- exactly-once
    consumption, the no-leakage claim -- rests on this."""
    partition = make(kind=kind, beta=beta)
    stacked = torch.cat(partition.shards)
    assert stacked.numel() == len(torch.unique(stacked))


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_shards_cover_the_training_set(kind: str, beta: float) -> None:
    labels = labels_of()
    partition = make(kind=kind, beta=beta, labels=labels)
    stacked = torch.cat(partition.shards).sort().values
    assert torch.equal(stacked, torch.arange(labels.numel(), dtype=torch.int64))


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_indices_are_int64_and_in_range(kind: str, beta: float) -> None:
    labels = labels_of()
    partition = make(kind=kind, beta=beta, labels=labels)
    for shard in partition.shards:
        assert shard.dtype == torch.int64
        assert int(shard.min()) >= 0
        assert int(shard.max()) < labels.numel()


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_every_agent_gets_a_shard(kind: str, beta: float) -> None:
    partition = make(kind=kind, beta=beta)
    assert partition.n_nodes == 10
    assert int(partition.sizes.min()) > 0


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_class_counts_agree_with_shard_sizes(kind: str, beta: float) -> None:
    labels = labels_of()
    partition = make(kind=kind, beta=beta, labels=labels)
    assert torch.equal(partition.class_counts(labels).sum(dim=1), partition.sizes)


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_class_distributions_are_normalised(kind: str, beta: float) -> None:
    labels = labels_of()
    distribution = make(kind=kind, beta=beta, labels=labels).class_distribution(labels)
    row_sums = distribution.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-12)


# =========================================================================== #
# 2. sizes
# =========================================================================== #


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_sizes_are_equal_by_default(kind: str, beta: float) -> None:
    """Balanced sizes are what keep X6 measuring label skew alone."""
    partition = make(kind=kind, beta=beta)
    assert int(partition.sizes.min()) == int(partition.sizes.max()) == 200


def test_sizes_differ_by_at_most_one_when_n_does_not_divide() -> None:
    partition = make(n_nodes=7, labels=labels_of(1000))
    assert int(partition.sizes.max()) - int(partition.sizes.min()) <= 1
    assert partition.total == 1000


def test_unbalanced_dirichlet_produces_wildly_uneven_shards() -> None:
    """The classical construction, and the reason it is not the default."""
    partition = make(kind="dirichlet", beta=0.1, balance_sizes=False)
    sizes = partition.sizes
    assert int(sizes.max()) > 3 * int(sizes.min())


def test_balancing_removes_the_size_variation_but_keeps_the_skew() -> None:
    labels = labels_of()
    balanced = make(kind="dirichlet", beta=0.1, labels=labels, balance_sizes=True)
    unbalanced = make(kind="dirichlet", beta=0.1, labels=labels, balance_sizes=False)

    assert int(balanced.sizes.min()) == int(balanced.sizes.max())
    assert int(unbalanced.sizes.min()) != int(unbalanced.sizes.max())
    assert balanced.skew(labels) > 0.4, "balancing must not wash the skew out"


# =========================================================================== #
# 3. the skew itself
# =========================================================================== #


#: Skew is measured against the *global* distribution, so an IID split still
#: shows a little of it from finite-sample noise -- about 0.06 at 200 samples
#: per agent, 0.017 at MNIST's 6000. These tests use a larger synthetic set so
#: the noise floor does not swamp the effect being measured.
SKEW_SAMPLES = 20_000


def mean_skew(kind: str, beta: float, seeds: int = 5, **kwargs) -> float:
    labels = labels_of(SKEW_SAMPLES)
    values = [
        make(kind=kind, beta=beta, seed=seed, labels=labels, **kwargs).skew(labels)
        for seed in range(seeds)
    ]
    return sum(values) / len(values)


def test_iid_shards_mirror_the_global_class_distribution() -> None:
    """Not exactly zero: an IID draw has sampling noise, and `skew` measures it
    honestly rather than pretending a finite sample is uniform."""
    assert mean_skew("iid", 1.0) < 0.05


def test_skew_increases_as_beta_falls() -> None:
    """The defining property of the axis X6 sweeps."""
    skews = [mean_skew("dirichlet", beta) for beta in BETAS]
    assert skews == sorted(skews, reverse=True), dict(zip(BETAS, skews, strict=True))


def test_large_beta_approaches_the_iid_partition() -> None:
    assert mean_skew("dirichlet", 1000.0) == pytest.approx(mean_skew("iid", 1.0), abs=0.05)


def test_small_beta_leaves_agents_missing_classes_entirely() -> None:
    """The sharpest form of "does communication help": an agent that never sees
    a 7 can only learn 7s from its neighbours."""
    labels = labels_of()
    partition = make(kind="dirichlet", beta=0.1, labels=labels)
    assert int(partition.classes_present(labels).min()) < N_CLASSES


def test_classes_present_falls_as_beta_falls() -> None:
    labels = labels_of()
    wide = make(kind="dirichlet", beta=100.0, labels=labels).classes_present(labels)
    narrow = make(kind="dirichlet", beta=0.1, labels=labels).classes_present(labels)
    assert float(narrow.to(torch.float64).mean()) < float(wide.to(torch.float64).mean())


def test_skew_is_zero_for_a_perfectly_uniform_partition() -> None:
    """Sanity on the metric itself, on a partition built by hand."""
    labels = labels_of(20, n_classes=2)
    shards = (torch.arange(0, 10, dtype=torch.int64), torch.arange(10, 20, dtype=torch.int64))
    partition = Partition(shards=shards, kind="hand", beta=None, n_classes=2)
    assert partition.skew(labels) == pytest.approx(0.0, abs=1e-12)


def test_skew_is_maximal_when_each_agent_holds_one_class() -> None:
    """Two agents, two classes, perfectly split: TV distance is 1 - 1/K = 0.5."""
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    shards = (torch.tensor([0, 1]), torch.tensor([2, 3]))
    partition = Partition(shards=shards, kind="hand", beta=None, n_classes=2)
    assert partition.skew(labels) == pytest.approx(0.5)


@pytest.mark.parametrize("beta", BETAS)
def test_skew_stays_within_its_theoretical_bounds(beta: float) -> None:
    labels = labels_of()
    skew = make(kind="dirichlet", beta=beta, labels=labels).skew(labels)
    assert 0.0 <= skew <= 1.0 - 1.0 / N_CLASSES + 1e-9


# =========================================================================== #
# 4. determinism and seed separability
# =========================================================================== #


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_the_same_seed_gives_the_same_shards(kind: str, beta: float) -> None:
    first = make(kind=kind, beta=beta, seed=3)
    second = make(kind=kind, beta=beta, seed=3)
    assert all(torch.equal(a, b) for a, b in zip(first.shards, second.shards, strict=True))


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_different_seeds_give_different_shards(kind: str, beta: float) -> None:
    first = make(kind=kind, beta=beta, seed=1)
    second = make(kind=kind, beta=beta, seed=2)
    assert not all(torch.equal(a, b) for a, b in zip(first.shards, second.shards, strict=True))


def test_partition_depends_only_on_the_partition_stream() -> None:
    """Holding the partition fixed while initialization varies is the ablation
    the four separable seed streams exist to enable."""
    from dekf_bench.runner.seeding import Seeds, derive_seed

    labels = labels_of()
    base = Seeds.from_master(0)
    other_init = Seeds(
        master=0,
        init=derive_seed(999, "init"),
        partition=base.partition,
        stream=base.stream,
        graph=base.graph,
    )

    first = build_partition(labels, 10, generator=base.torch_generator("partition"))
    second = build_partition(labels, 10, generator=other_init.torch_generator("partition"))
    assert all(torch.equal(a, b) for a, b in zip(first.shards, second.shards, strict=True))


# =========================================================================== #
# 5. degenerate sizes
# =========================================================================== #


def test_single_agent_takes_everything() -> None:
    labels = labels_of(100)
    partition = make(n_nodes=1, labels=labels)
    assert partition.n_nodes == 1
    assert int(partition.sizes[0]) == 100


def test_one_sample_per_agent_is_allowed() -> None:
    labels = labels_of(10)
    partition = make(n_nodes=10, labels=labels)
    assert torch.equal(partition.sizes, torch.ones(10, dtype=torch.int64))


@pytest.mark.parametrize(("kind", "beta"), ALL_KINDS)
def test_two_agents_still_split_cleanly(kind: str, beta: float) -> None:
    labels = labels_of(100)
    partition = make(n_nodes=2, kind=kind, beta=beta, labels=labels)
    assert partition.total == 100
    assert int(partition.sizes.min()) == 50


# =========================================================================== #
# 6. rejected inputs
# =========================================================================== #


def test_more_agents_than_samples_is_rejected() -> None:
    with pytest.raises(PartitionError, match="cannot split"):
        make(n_nodes=50, labels=labels_of(10))


def test_zero_agents_is_rejected() -> None:
    with pytest.raises(PartitionError, match="n_nodes must be >= 1"):
        make(n_nodes=0)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(PartitionError, match="unknown partition kind"):
        make(kind="power_law")


@pytest.mark.parametrize("beta", [0.0, -1.0])
def test_non_positive_beta_is_rejected(beta: float) -> None:
    with pytest.raises(PartitionError, match="beta must be > 0"):
        make(kind="dirichlet", beta=beta)


def test_overlapping_shards_are_rejected() -> None:
    with pytest.raises(PartitionError, match="shards overlap"):
        Partition(
            shards=(torch.tensor([0, 1, 2]), torch.tensor([2, 3])),
            kind="hand",
            beta=None,
            n_classes=N_CLASSES,
        )


def test_non_integer_indices_are_rejected() -> None:
    with pytest.raises(PartitionError, match="must be int64"):
        Partition(shards=(torch.tensor([0.0, 1.0]),), kind="hand", beta=None, n_classes=N_CLASSES)


def test_empty_partition_is_rejected() -> None:
    with pytest.raises(PartitionError, match="at least one shard"):
        Partition(shards=(), kind="hand", beta=None, n_classes=N_CLASSES)


def test_partition_is_frozen() -> None:
    import dataclasses

    partition = make()
    # setattr rather than a direct assignment: see the note in test_graph.py.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(partition, "kind", "dirichlet")  # noqa: B010


# =========================================================================== #
# 7. the shard budget, which is where balancing earns its keep
# =========================================================================== #


def test_min_shard_size_is_enforced() -> None:
    with pytest.raises(PartitionError, match="smallest shard holds"):
        make(labels=labels_of(100), min_shard_size=50)


def test_min_shard_size_message_explains_the_balanced_size_interaction() -> None:
    with pytest.raises(PartitionError, match="does not protect the run"):
        make(labels=labels_of(100), min_shard_size=50)


def test_balanced_shards_satisfy_a_budget_that_unbalanced_ones_do_not() -> None:
    """The concrete hazard: the config-time check N*n*T <= 60000 assumes equal
    shards. Unbalanced Dirichlet violates that assumption."""
    labels = labels_of(2000)
    per_agent = 2000 // 10

    balanced = make(kind="dirichlet", beta=0.1, labels=labels, balance_sizes=True)
    assert int(balanced.sizes.min()) == per_agent

    unbalanced = make(kind="dirichlet", beta=0.1, labels=labels, balance_sizes=False)
    assert int(unbalanced.sizes.min()) < per_agent


# =========================================================================== #
# 8. integration with the configs
# =========================================================================== #


def test_config_partition_enforces_the_run_budget() -> None:
    """n * T samples per agent, or the stream exhausts a shard mid-run."""
    config = load_config("x1_stationary")
    labels = labels_of(60_000)
    partition = build_partition_from_config(config, labels, torch.Generator().manual_seed(0))
    needed = config.env.samples_per_node_per_step * config.run.horizon
    assert int(partition.sizes.min()) >= needed


def test_config_partition_rejects_a_horizon_the_shards_cannot_feed() -> None:
    """T=1500 at n=2 needs 3000 per agent; 2000 labels over 10 agents gives 200."""
    config = load_config("x1_stationary")
    assert not config.env.allow_epochs
    with pytest.raises(PartitionError, match="smallest shard holds 200 samples"):
        build_partition_from_config(config, labels_of(2000), torch.Generator().manual_seed(0))


def test_allow_epochs_waives_the_budget_check() -> None:
    config = load_config("x1_stationary", overrides={"env": {"allow_epochs": True}})
    labels = labels_of(2000)
    partition = build_partition_from_config(config, labels, torch.Generator().manual_seed(0))
    assert partition.total == 2000


def test_config_selects_the_dirichlet_kind_and_beta() -> None:
    config = load_config(
        "x1_stationary",
        overrides={"env": {"partition": {"kind": "dirichlet", "beta": 0.1}}},
    )
    labels = labels_of(60_000)
    partition = build_partition_from_config(config, labels, torch.Generator().manual_seed(0))
    assert partition.kind == "dirichlet"
    assert partition.beta == 0.1
    assert partition.skew(labels) > 0.4


# =========================================================================== #
# 9. against the real dataset
# =========================================================================== #


@pytest.fixture(scope="session")
def mnist_labels() -> torch.Tensor:
    from dekf_bench.data.mnist import is_cached, load_split

    if not is_cached():
        pytest.skip("MNIST not cached; run scripts/check_data.py once")
    return load_split("train", download=False).labels


@pytest.mark.needs_data
def test_mnist_iid_shards_are_exactly_six_thousand(mnist_labels: torch.Tensor) -> None:
    """N=10 over 60000: the default, and the number the shard budget assumes."""
    partition = build_partition(mnist_labels, 10, generator=torch.Generator().manual_seed(0))
    assert torch.equal(partition.sizes, torch.full((10,), 6000, dtype=torch.int64))


@pytest.mark.needs_data
def test_mnist_dirichlet_keeps_sizes_equal_despite_unequal_class_counts(
    mnist_labels: torch.Tensor,
) -> None:
    """MNIST classes are not balanced -- 5421 fives against 6742 ones -- so
    equal shard sizes are not automatic."""
    partition = build_partition(
        mnist_labels, 10, "dirichlet", 0.1, 10, torch.Generator().manual_seed(0)
    )
    assert torch.equal(partition.sizes, torch.full((10,), 6000, dtype=torch.int64))
    assert partition.skew(mnist_labels) > 0.5


@pytest.mark.needs_data
@pytest.mark.parametrize(("beta", "min_starving_seeds"), [(0.1, 10), (0.5, 6)])
def test_mnist_unbalanced_dirichlet_would_starve_a_default_run(
    mnist_labels: torch.Tensor, beta: float, min_starving_seeds: int
) -> None:
    """The hazard, quantified. A default run consumes n*T = 3000 samples per
    agent and N*n*T = 30000 passes the config-time check -- but that check
    assumes equal shards. Under the classical construction the smallest shard
    falls below 3000 for 8 of 10 seeds at beta=0.5 and for every seed at
    beta=0.1, so the run would exhaust an agent mid-flight.

    Asserted over seeds rather than on one draw: the shortfall is a property of
    the construction, not of a particular realization.
    """
    starving = 0
    for seed in range(10):
        partition = build_partition(
            mnist_labels,
            10,
            "dirichlet",
            beta,
            10,
            torch.Generator().manual_seed(seed),
            balance_sizes=False,
        )
        starving += int(partition.sizes.min()) < 3000
    assert starving >= min_starving_seeds


@pytest.mark.needs_data
def test_mnist_balanced_dirichlet_never_starves(mnist_labels: torch.Tensor) -> None:
    """The same betas, with balancing on: always exactly 6000, always safe."""
    for beta in (0.1, 0.5, 1.0):
        for seed in range(5):
            partition = build_partition(
                mnist_labels, 10, "dirichlet", beta, 10, torch.Generator().manual_seed(seed)
            )
            assert int(partition.sizes.min()) == 6000


@pytest.mark.needs_data
def test_mnist_shards_are_disjoint_and_complete(mnist_labels: torch.Tensor) -> None:
    partition = build_partition(
        mnist_labels, 10, "dirichlet", 0.5, 10, torch.Generator().manual_seed(0)
    )
    stacked = torch.cat(partition.shards).sort().values
    assert torch.equal(stacked, torch.arange(60_000, dtype=torch.int64))


@pytest.mark.needs_data
def test_summary_reports_every_field(mnist_labels: torch.Tensor) -> None:
    summary = build_partition(mnist_labels, 10).summary(mnist_labels)
    assert set(summary) == {
        "kind",
        "beta",
        "n_nodes",
        "total",
        "min_size",
        "max_size",
        "skew",
        "min_classes_present",
        "max_classes_present",
    }
