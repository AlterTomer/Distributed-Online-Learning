"""Per-agent sample streams.

The guarantees under test are the ones everything downstream leans on: no sample
is served twice, no agent sees another's shard, and the answer for $(v, t)$ does
not depend on how the run reached step $t$.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.env.partition import build_partition
from dekf_bench.env.stream import Stream, StreamError, build_stream, build_stream_from_config
from dekf_bench.utils.config import load_config

N_NODES = 10
HORIZON = 100
SAMPLES = 2
AVAILABILITIES = [1.0, 0.75, 0.5, 0.25]


def labels_of(n_samples: int = 2000, n_classes: int = 10) -> torch.Tensor:
    return torch.arange(n_samples, dtype=torch.int64) % n_classes


def make(
    availability: float = 1.0,
    horizon: int = HORIZON,
    samples: int = SAMPLES,
    n_nodes: int = N_NODES,
    seed: int = 0,
    n_samples: int = 2000,
    **kwargs,
) -> Stream:
    partition = build_partition(
        labels_of(n_samples), n_nodes, generator=torch.Generator().manual_seed(0)
    )
    return build_stream(
        partition,
        horizon=horizon,
        samples_per_step=samples,
        label_availability=availability,
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    )


def partition_of(n_nodes: int = N_NODES, n_samples: int = 2000):
    return build_partition(
        labels_of(n_samples), n_nodes, generator=torch.Generator().manual_seed(0)
    )


# =========================================================================== #
# 1. exactly once
# =========================================================================== #


@pytest.mark.parametrize("availability", AVAILABILITIES)
def test_no_sample_is_served_twice_to_an_agent(availability: float) -> None:
    """Assumption 5 of the research note: every labelled observation enters the
    likelihood exactly once. A repeat makes the filter's covariance shrink
    faster than the evidence justifies."""
    stream = make(availability)
    for node in range(stream.n_nodes):
        seen = stream.consumed_by(node)
        assert seen.numel() == len(torch.unique(seen))


@pytest.mark.parametrize("availability", AVAILABILITIES)
def test_no_sample_is_served_to_two_agents(availability: float) -> None:
    stream = make(availability)
    everything = torch.cat([stream.consumed_by(node) for node in range(stream.n_nodes)])
    assert everything.numel() == len(torch.unique(everything))


@pytest.mark.parametrize("availability", AVAILABILITIES)
def test_agents_only_ever_see_their_own_shard(availability: float) -> None:
    partition = partition_of()
    stream = build_stream(
        partition,
        horizon=HORIZON,
        samples_per_step=SAMPLES,
        label_availability=availability,
        generator=torch.Generator().manual_seed(0),
    )
    for node in range(stream.n_nodes):
        own = set(partition[node].tolist())
        assert set(stream.consumed_by(node).tolist()) <= own


def test_the_stream_is_a_permutation_of_the_shard_prefix() -> None:
    """Exactly-once plus own-shard-only means the consumed set is a prefix of a
    shuffle of the shard, not an arbitrary subset."""
    partition = partition_of()
    stream = build_stream(
        partition,
        horizon=HORIZON,
        samples_per_step=SAMPLES,
        generator=torch.Generator().manual_seed(0),
    )
    for node in range(stream.n_nodes):
        consumed = stream.consumed_by(node)
        assert torch.equal(consumed, stream.order[node][: consumed.numel()])


def test_consumed_grows_monotonically() -> None:
    stream = make()
    previous = 0
    for step in range(0, HORIZON, 7):
        current = stream.consumed_by(0, through=step).numel()
        assert current >= previous
        previous = current


def test_consumed_through_is_a_prefix_of_the_whole_run() -> None:
    stream = make()
    partial = stream.consumed_by(0, through=HORIZON // 2)
    whole = stream.consumed_by(0)
    assert torch.equal(whole[: partial.numel()], partial)


# =========================================================================== #
# 2. batches
# =========================================================================== #


@pytest.mark.parametrize("availability", AVAILABILITIES)
def test_a_batch_is_either_full_or_empty(availability: float) -> None:
    """Never partially filled: the X0 identity needs equal batch sizes across
    every agent that is active at a step."""
    stream = make(availability)
    for step in range(HORIZON):
        for node in range(stream.n_nodes):
            size = stream.indices_at(node, step).numel()
            assert size in (0, SAMPLES)


def test_active_agents_all_receive_the_same_count() -> None:
    stream = make(0.5)
    for step in range(HORIZON):
        sizes = {stream.indices_at(node, step).numel() for node in stream.active_at(step)}
        assert len(sizes) <= 1


def test_idle_agents_receive_nothing() -> None:
    stream = make(0.5)
    for step in range(HORIZON):
        for node in range(stream.n_nodes):
            if not stream.is_labelled(node, step):
                assert stream.indices_at(node, step).numel() == 0


def test_batch_at_covers_every_agent() -> None:
    stream = make(0.5)
    batch = stream.batch_at(3)
    assert set(batch) == set(range(N_NODES))


def test_active_at_agrees_with_is_labelled() -> None:
    stream = make(0.5)
    for step in (0, 17, 99):
        expected = tuple(n for n in range(N_NODES) if stream.is_labelled(n, step))
        assert stream.active_at(step) == expected


def test_batch_size_scales_with_n() -> None:
    for samples in (1, 2, 4):
        stream = make(samples=samples, n_samples=10_000)
        assert stream.indices_at(0, 0).numel() == samples


# =========================================================================== #
# 3. label availability
# =========================================================================== #


def test_full_availability_labels_every_agent_at_every_step() -> None:
    """Exact, not almost-surely -- X0 depends on it."""
    stream = make(1.0)
    assert bool(stream.has_label.all())
    assert stream.empirical_label_rate == 1.0


def test_zero_availability_labels_nobody() -> None:
    stream = make(0.0)
    assert not bool(stream.has_label.any())
    assert stream.consumed_by(0).numel() == 0


@pytest.mark.parametrize("availability", [0.25, 0.5, 0.75])
def test_empirical_rate_matches_the_requested_one(availability: float) -> None:
    stream = make(availability, horizon=1000, n_samples=30_000)
    assert stream.empirical_label_rate == pytest.approx(availability, abs=0.02)


def test_availability_varies_across_agents_and_steps() -> None:
    """Per-(agent, step), not a global on/off switch: at some steps a few agents
    are labelled and others idle, which is the regime where the combine step
    carries an agent that learned nothing this round."""
    stream = make(0.5, horizon=200, n_samples=10_000)

    per_agent = stream.has_label.to(torch.float64).mean(dim=1)
    assert float(per_agent.std()) > 0.0, "agents must not share one schedule"

    active_counts = stream.has_label.sum(dim=0)
    assert int(active_counts.min()) < N_NODES, "some step must leave an agent idle"
    assert int(active_counts.max()) > 0, "some step must have an active agent"
    assert len(set(active_counts.tolist())) > 1, "the active count must vary by step"


def test_an_idle_step_consumes_nothing_from_the_shard() -> None:
    """The offset must not advance, or a sparser label stream would silently
    also be a shorter run -- confounding the two axes X4 sweeps separately."""
    stream = make(0.5)
    for node in range(stream.n_nodes):
        for step in range(1, HORIZON):
            if not stream.is_labelled(node, step - 1):
                assert int(stream.offsets[node, step]) == int(stream.offsets[node, step - 1])


def test_sparser_labels_consume_less_of_the_shard() -> None:
    dense = make(1.0).summary()["total_consumed"]
    sparse = make(0.25).summary()["total_consumed"]
    assert sparse < dense / 2


def test_offsets_are_the_prefix_sum_of_labelled_steps() -> None:
    stream = make(0.5)
    for node in range(stream.n_nodes):
        for step in range(HORIZON):
            expected = int(stream.has_label[node, :step].sum()) * SAMPLES
            assert int(stream.offsets[node, step]) == expected


def test_availability_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(StreamError, match=r"must lie in \[0, 1\]"):
        make(1.5)


# =========================================================================== #
# 4. positional, not stateful
# =========================================================================== #


def test_a_step_gives_the_same_answer_however_it_is_reached() -> None:
    """The reason the stream is a prefix sum rather than a cursor: an evaluation
    set or a restarted runner must be able to ask about step 90 directly."""
    stream = make(0.5)
    direct = stream.indices_at(4, 90)
    for step in range(91):
        walked = stream.indices_at(4, step)
    assert torch.equal(direct, walked)
    assert torch.equal(stream.indices_at(4, 90), direct)


def test_querying_out_of_order_is_stable() -> None:
    stream = make(0.5)
    forward = [stream.indices_at(2, step) for step in range(20)]
    backward = [stream.indices_at(2, step) for step in reversed(range(20))][::-1]
    assert all(torch.equal(a, b) for a, b in zip(forward, backward, strict=True))


def test_repeated_queries_do_not_advance_anything() -> None:
    stream = make()
    first = stream.indices_at(0, 5)
    for _ in range(10):
        assert torch.equal(stream.indices_at(0, 5), first)


# =========================================================================== #
# 5. determinism
# =========================================================================== #


@pytest.mark.parametrize("availability", AVAILABILITIES)
def test_the_same_seed_gives_the_same_stream(availability: float) -> None:
    a, b = make(availability, seed=5), make(availability, seed=5)
    assert torch.equal(a.has_label, b.has_label)
    assert all(torch.equal(x, y) for x, y in zip(a.order, b.order, strict=True))


def test_different_seeds_give_different_streams() -> None:
    a, b = make(0.5, seed=1), make(0.5, seed=2)
    assert not torch.equal(a.has_label, b.has_label)


def test_stream_seed_does_not_disturb_the_partition() -> None:
    """The ablation the separable seed streams exist for: vary the sample order
    while the shard assignment is held fixed."""
    partition = partition_of()
    first = build_stream(partition, horizon=HORIZON, generator=torch.Generator().manual_seed(1))
    second = build_stream(partition, horizon=HORIZON, generator=torch.Generator().manual_seed(2))
    for node in range(N_NODES):
        assert set(first.order[node].tolist()) == set(partition[node].tolist())
        assert set(second.order[node].tolist()) == set(partition[node].tolist())
    assert not all(torch.equal(a, b) for a, b in zip(first.order, second.order, strict=True))


def test_the_order_is_a_shuffle_not_the_shard_order() -> None:
    partition = partition_of()
    stream = build_stream(partition, horizon=HORIZON, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(stream.order[0], partition[0])
    assert torch.equal(stream.order[0].sort().values, partition[0].sort().values)


# =========================================================================== #
# 6. the shard budget
# =========================================================================== #


def test_a_run_that_would_exhaust_a_shard_is_rejected_up_front() -> None:
    """At step 0, not at step 1400 of a five-seed sweep."""
    with pytest.raises(StreamError, match="would consume"):
        make(horizon=1000, samples=4, n_samples=2000)


def test_the_message_names_the_way_out() -> None:
    with pytest.raises(StreamError, match="allow_epochs=true"):
        make(horizon=1000, samples=4, n_samples=2000)


def test_sparse_labels_let_a_longer_horizon_fit() -> None:
    """Half the labels, so roughly twice the horizon fits in the same shard."""
    with pytest.raises(StreamError):
        make(availability=1.0, horizon=200, samples=2, n_samples=2000)
    stream = make(availability=0.25, horizon=200, samples=2, n_samples=2000)
    assert stream.horizon == 200


def test_allow_epochs_permits_wrapping() -> None:
    stream = make(horizon=1000, samples=4, n_samples=2000, allow_epochs=True)
    seen = stream.consumed_by(0)
    assert seen.numel() > len(torch.unique(seen)), "wrapping means repeats"


def test_wrapping_stays_inside_the_shard() -> None:
    partition = partition_of()
    stream = build_stream(
        partition,
        horizon=1000,
        samples_per_step=4,
        generator=torch.Generator().manual_seed(0),
        allow_epochs=True,
    )
    own = set(partition[0].tolist())
    assert set(stream.consumed_by(0).tolist()) <= own


def test_exactly_once_holds_right_up_to_the_shard_boundary() -> None:
    """200 samples per agent, n=2, T=100 -- exactly full, nothing left over."""
    stream = make(horizon=100, samples=2, n_samples=2000)
    seen = stream.consumed_by(0)
    assert seen.numel() == 200 == len(torch.unique(seen))
    assert seen.numel() == int(partition_of().sizes[0])


# =========================================================================== #
# 7. rejected inputs
# =========================================================================== #


def test_zero_horizon_is_rejected() -> None:
    with pytest.raises(StreamError, match="horizon must be >= 1"):
        make(horizon=0)


def test_zero_samples_per_step_is_rejected() -> None:
    with pytest.raises(StreamError, match="samples_per_step must be >= 1"):
        make(samples=0)


def test_out_of_range_node_is_rejected() -> None:
    stream = make()
    with pytest.raises(StreamError, match="node 99 outside"):
        stream.indices_at(99, 0)


def test_out_of_range_step_is_rejected() -> None:
    stream = make()
    with pytest.raises(StreamError, match="step 999 outside"):
        stream.indices_at(0, 999)


def test_stream_is_frozen() -> None:
    import dataclasses

    stream = make()
    # setattr rather than a direct assignment: see the note in test_graph.py.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(stream, "samples_per_step", 8)  # noqa: B010


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(StreamError, match="must agree"):
        Stream(
            order=(torch.arange(10),),
            has_label=torch.ones(1, 5, dtype=torch.bool),
            offsets=torch.zeros(1, 4, dtype=torch.int64),
            samples_per_step=1,
        )


def test_non_boolean_label_mask_is_rejected() -> None:
    with pytest.raises(StreamError, match="has_label must be bool"):
        Stream(
            order=(torch.arange(10),),
            has_label=torch.ones(1, 5, dtype=torch.int64),
            offsets=torch.zeros(1, 5, dtype=torch.int64),
            samples_per_step=1,
        )


# =========================================================================== #
# 8. summary and config integration
# =========================================================================== #


def test_summary_reports_every_field() -> None:
    assert set(make().summary()) == {
        "n_nodes",
        "horizon",
        "samples_per_step",
        "allow_epochs",
        "empirical_label_rate",
        "min_required",
        "max_required",
        "total_consumed",
    }


def test_config_stream_matches_the_run_parameters() -> None:
    config = load_config("x1_stationary")
    partition = build_partition(
        labels_of(60_000), config.graph.n_nodes, generator=torch.Generator().manual_seed(0)
    )
    stream = build_stream_from_config(config, partition, torch.Generator().manual_seed(0))
    assert stream.horizon == config.run.horizon
    assert stream.samples_per_step == config.env.samples_per_node_per_step
    assert stream.empirical_label_rate == 1.0


def test_x0_config_gives_every_agent_labels_at_every_step() -> None:
    """A precondition of the exactness identity: equal batch sizes."""
    config = load_config("x0_exactness")
    partition = build_partition(
        labels_of(60_000), config.graph.n_nodes, generator=torch.Generator().manual_seed(0)
    )
    stream = build_stream_from_config(config, partition, torch.Generator().manual_seed(0))
    for step in range(config.run.horizon):
        assert len(stream.active_at(step)) == config.graph.n_nodes


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
def test_a_default_run_consumes_the_whole_training_set(mnist_labels) -> None:
    """At n=4 the default run sits exactly on the shard budget: 10*4*1500 =
    60000, the entire MNIST training split with nothing spare. That is
    deliberate (WORKPLAN 4.2) and this test is what would catch a change that
    silently pushed it over into reuse."""
    config = load_config("x1_stationary")
    partition = build_partition(
        mnist_labels, config.graph.n_nodes, generator=torch.Generator().manual_seed(0)
    )
    stream = build_stream_from_config(config, partition, torch.Generator().manual_seed(0))
    assert stream.summary()["total_consumed"] == 60_000
    assert stream.summary()["min_required"] == 6_000


@pytest.mark.needs_data
def test_exactly_once_holds_over_a_full_length_run(mnist_labels) -> None:
    config = load_config("x1_stationary")
    partition = build_partition(
        mnist_labels, config.graph.n_nodes, generator=torch.Generator().manual_seed(0)
    )
    stream = build_stream_from_config(config, partition, torch.Generator().manual_seed(0))
    everything = torch.cat([stream.consumed_by(node) for node in range(stream.n_nodes)])
    assert everything.numel() == 60_000 == len(torch.unique(everything))


@pytest.mark.needs_data
def test_no_training_index_falls_outside_the_split(mnist_labels) -> None:
    config = load_config("x1_stationary")
    partition = build_partition(
        mnist_labels, config.graph.n_nodes, generator=torch.Generator().manual_seed(0)
    )
    stream = build_stream_from_config(config, partition, torch.Generator().manual_seed(0))
    everything = torch.cat([stream.consumed_by(node) for node in range(stream.n_nodes)])
    assert int(everything.min()) >= 0
    assert int(everything.max()) < 60_000
