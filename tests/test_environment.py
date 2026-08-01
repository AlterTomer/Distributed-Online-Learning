"""The environment: what each agent observes at each step.

The properties under test are the ones the runner and the exactness check lean
on: observations are shared and unmutated, ``step(t)`` is positional, the pooled
batch is exactly the union of the per-agent ones, and images stay paired with
their labels through the transform.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.env.environment import (
    EnvironmentError,
    Observation,
    build_environment,
    pool,
)
from dekf_bench.utils.config import load_config

N_NODES = 10
HORIZON = 40


def synthetic_split(n: int = 4000) -> MnistSplit:
    """Distinct images, so an image can be traced back to its index."""
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(n, 1, 28, 28, generator=generator)
    labels = torch.arange(n, dtype=torch.int64) % 10
    return MnistSplit(images=images, labels=labels, split="synthetic")


@pytest.fixture(scope="module")
def train() -> MnistSplit:
    return synthetic_split()


def make(train: MnistSplit, experiment: str = "x1_stationary", seed: int = 0, **overrides):
    """An environment on the synthetic split, with a short horizon.

    A short horizon costs nothing in coverage: alpha is derived as
    total_degrees / T, so a 40-step run traverses the same 0-45 degrees a
    1500-step one does. It only changes how finely.
    """
    from dekf_bench.utils.config import deep_merge

    config = load_config(experiment, overrides=deep_merge({"run": {"horizon": HORIZON}}, overrides))
    return build_environment(config, seed, train)


# =========================================================================== #
# 1. shape of an observation
# =========================================================================== #


def test_step_covers_every_agent(train: MnistSplit) -> None:
    env = make(train)
    assert set(env.step(0)) == set(range(N_NODES))


def test_observation_shapes_match_the_transform(train: MnistSplit) -> None:
    env = make(train)
    obs = env.step(0)[0]
    assert obs.x.shape == (2, 1, 14, 14)
    assert obs.y is not None and obs.y.shape == (2,)
    assert obs.n_samples == 2
    assert obs.has_label


def test_labels_are_int64_and_in_range(train: MnistSplit) -> None:
    env = make(train)
    for obs in env.step(0).values():
        assert obs.y is not None
        assert obs.y.dtype == torch.int64
        assert int(obs.y.min()) >= 0 and int(obs.y.max()) < 10


def test_idle_agents_yield_an_empty_observation(train: MnistSplit) -> None:
    env = make(train, env={"label_availability": 0.5})
    idle = [obs for step in range(HORIZON) for obs in env.step(step).values() if not obs.has_label]
    assert idle, "the fixture must actually produce idle agents"
    for obs in idle:
        assert obs.n_samples == 0
        assert obs.x.shape == (0, 1, 14, 14)
        assert obs.y is None
        assert len(obs) == 0


def test_observation_records_where_it_came_from(train: MnistSplit) -> None:
    env = make(train)
    obs = env.step(7)[3]
    assert obs.node == 3
    assert obs.step == 7


def test_every_active_agent_receives_the_same_count(train: MnistSplit) -> None:
    """A precondition of the exactness identity."""
    env = make(train)
    for step in range(HORIZON):
        counts = {obs.n_samples for obs in env.step(step).values()}
        assert counts == {2}


# =========================================================================== #
# 2. images stay paired with their labels
# =========================================================================== #


def test_labels_match_the_images_they_came_with(train: MnistSplit) -> None:
    """The transform must not permute the batch. If it did, every method would
    train on mismatched pairs and simply fail to learn -- with no error."""
    env = make(train)
    for step in (0, 5, 20):
        for node in range(N_NODES):
            indices = env.stream.indices_at(node, step)
            obs = env.observe(node, step)
            assert obs.y is not None
            assert torch.equal(obs.y, train.labels[indices])
            expected = env.transform.apply(train.images[indices], obs.rotation_degrees)
            assert torch.equal(obs.x, expected)


def test_agents_never_see_another_agents_shard(train: MnistSplit) -> None:
    env = make(train)
    for node in range(N_NODES):
        own = set(env.partition[node].tolist())
        for step in range(HORIZON):
            assert set(env.stream.indices_at(node, step).tolist()) <= own


def test_no_sample_reaches_two_agents_over_a_run(train: MnistSplit) -> None:
    env = make(train)
    served = torch.cat([env.stream.consumed_by(node, HORIZON - 1) for node in range(N_NODES)])
    assert served.numel() == len(torch.unique(served))


# =========================================================================== #
# 3. positional, and shared
# =========================================================================== #


def test_the_same_step_returns_the_same_data(train: MnistSplit) -> None:
    """step(t) is called once and shared by every learner, so calling it again
    must not advance anything."""
    env = make(train)
    first, second = env.step(3), env.step(3)
    for node in range(N_NODES):
        assert torch.equal(first[node].x, second[node].x)
        assert torch.equal(first[node].y, second[node].y)  # type: ignore[arg-type]


def test_steps_can_be_visited_out_of_order(train: MnistSplit) -> None:
    env = make(train)
    direct = env.step(30)[0].x.clone()
    for step in range(31):
        env.step(step)
    assert torch.equal(env.step(30)[0].x, direct)


def test_observation_is_frozen(train: MnistSplit) -> None:
    import dataclasses

    obs = make(train).step(0)[0]
    # setattr rather than a direct assignment: see the note in test_graph.py.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obs, "has_label", False)  # noqa: B010


def test_environment_is_frozen(train: MnistSplit) -> None:
    import dataclasses

    env = make(train)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(env, "config", None)  # noqa: B010


def test_in_place_mutation_of_a_shared_observation_is_caught(train: MnistSplit) -> None:
    """A frozen dataclass cannot stop `obs.x.add_(1)`; this is the positive
    check the runner makes so a misbehaving learner cannot corrupt the ones
    that run after it."""
    env = make(train)
    observations = env.step(2)
    env.assert_unmodified(observations, 2)

    observations[4].x.add_(1.0)
    with pytest.raises(EnvironmentError, match="modified in place"):
        env.assert_unmodified(observations, 2)


def test_label_mutation_is_caught_too(train: MnistSplit) -> None:
    env = make(train)
    observations = env.step(2)
    observations[1].y.add_(1)  # type: ignore[union-attr]
    with pytest.raises(EnvironmentError, match="modified in place"):
        env.assert_unmodified(observations, 2)


def test_labels_are_copied_not_aliased(train: MnistSplit) -> None:
    """Mutating an observation must not corrupt the underlying dataset."""
    env = make(train)
    before = train.labels.clone()
    env.step(0)[0].y.add_(5)  # type: ignore[union-attr]
    assert torch.equal(train.labels, before)


# =========================================================================== #
# 4. pooling
# =========================================================================== #


def test_pooled_batch_is_exactly_the_union(train: MnistSplit) -> None:
    """The exactness identity depends on the centralized learner's batch being
    the union of the per-agent ones, with nothing added or dropped."""
    env = make(train)
    observations = env.step(0)
    xs, ys = pool(observations)

    assert xs.shape[0] == sum(obs.n_samples for obs in observations.values()) == 20
    assert torch.equal(xs, torch.cat([observations[v].x for v in range(N_NODES)]))
    assert torch.equal(ys, torch.cat([observations[v].y for v in range(N_NODES)]))  # type: ignore[misc]


def test_pooling_skips_idle_agents(train: MnistSplit) -> None:
    env = make(train, env={"label_availability": 0.5})
    for step in range(HORIZON):
        observations = env.step(step)
        xs, _ = pool(observations)
        assert xs.shape[0] == sum(obs.n_samples for obs in observations.values())


def test_pooling_an_entirely_idle_step_gives_an_empty_batch(train: MnistSplit) -> None:
    env = make(train, env={"label_availability": 0.0})
    xs, ys = pool(env.step(0))
    assert xs.shape == (0, 1, 14, 14)
    assert ys.shape == (0,)


def test_pooled_dtype_follows_the_run(train: MnistSplit) -> None:
    env = make(train, "x0_exactness")
    xs, _ = pool(env.step(0))
    assert xs.dtype == torch.float64


# =========================================================================== #
# 5. drift reaches the pixels
# =========================================================================== #


def test_stationary_runs_show_no_rotation(train: MnistSplit) -> None:
    env = make(train)
    for step in (0, 10, 39):
        assert env.drift_state(step).rotation_degrees == 0.0
        assert env.step(step)[0].rotation_degrees == 0.0


def test_rotation_recorded_on_the_observation_matches_the_schedule(
    train: MnistSplit,
) -> None:
    env = make(train, "x2_rotating")
    for step in (0, 20, 39):
        expected = env.drift_state(step).rotation_degrees
        assert env.step(step)[0].rotation_degrees == pytest.approx(expected)


def test_drift_actually_changes_the_pixels(train: MnistSplit) -> None:
    """Otherwise the rotation would be recorded but never applied -- the exact
    failure mode 'the same transform for train and eval' is meant to prevent.

    The same images at the run's first and last drift states, so the only
    difference is the rotation.
    """
    env = make(train, "x2_rotating")
    indices = env.stream.indices_at(0, 0)
    last = env.horizon - 1
    assert env.drift_state(last).rotation_degrees > 40.0, "the run must actually drift"

    early = env.transform.apply(train.images[indices], env.drift_state(0).rotation_degrees)
    late = env.transform.apply(train.images[indices], env.drift_state(last).rotation_degrees)
    assert not torch.allclose(early, late)


def test_per_node_drift_gives_agents_different_rotations(train: MnistSplit) -> None:
    env = make(train, "x2_rotating", env={"drift_scope": "per_node"})
    rotations = {obs.rotation_degrees for obs in env.step(HORIZON - 1).values()}
    assert len(rotations) == N_NODES


def test_global_drift_gives_every_agent_the_same_rotation(train: MnistSplit) -> None:
    env = make(train, "x2_rotating")
    rotations = {obs.rotation_degrees for obs in env.step(HORIZON - 1).values()}
    assert len(rotations) == 1


# =========================================================================== #
# 6. transform reaches the pixels
# =========================================================================== #


def test_images_are_downsampled_and_normalized(train: MnistSplit) -> None:
    env = make(train)
    obs = env.step(0)[0]
    assert obs.x.shape[-1] == env.transform.size == 14
    assert float(obs.x.min()) < 0.0, "normalized data straddles zero"


def test_full_resolution_model_gets_full_resolution_input(train: MnistSplit) -> None:
    env = make(train, include={"model": "mlp"})
    assert env.transform.size == 28
    assert env.step(0)[0].x.shape == (2, 1, 28, 28)
    assert env.transform.input_dim == 784


def test_exactness_run_is_float64_end_to_end(train: MnistSplit) -> None:
    env = make(train, "x0_exactness")
    assert env.step(0)[0].x.dtype == torch.float64


# =========================================================================== #
# 7. determinism and seeds
# =========================================================================== #


def test_the_same_seed_gives_the_same_environment(train: MnistSplit) -> None:
    a, b = make(train, seed=3), make(train, seed=3)
    assert torch.equal(a.step(5)[2].x, b.step(5)[2].x)


def test_different_seeds_give_different_environments(train: MnistSplit) -> None:
    a, b = make(train, seed=1), make(train, seed=2)
    assert not torch.equal(a.step(5)[2].x, b.step(5)[2].x)


def test_reset_returns_a_new_environment_at_another_seed(train: MnistSplit) -> None:
    env = make(train, seed=0)
    other = env.reset(1)
    assert other is not env
    assert other.seeds.master == 1
    assert not torch.equal(env.step(0)[0].x, other.step(0)[0].x)


def test_the_graphs_stay_separable(train: MnistSplit) -> None:
    env = make(train)
    assert env.graph is env.graphs.comm
    assert env.graphs.data.n_edges == 0
    assert env.graphs.data.weights is None


# =========================================================================== #
# 8. bounds, summary
# =========================================================================== #


def test_step_outside_the_horizon_is_rejected(train: MnistSplit) -> None:
    env = make(train)
    with pytest.raises(EnvironmentError, match="outside 0..39"):
        env.step(HORIZON)
    with pytest.raises(EnvironmentError, match="outside"):
        env.step(-1)


def test_horizon_and_node_count_agree_with_the_config(train: MnistSplit) -> None:
    env = make(train)
    assert env.horizon == HORIZON
    assert env.n_nodes == N_NODES == env.graph.n_nodes


def test_summary_covers_every_component(train: MnistSplit) -> None:
    summary = make(train).summary()
    assert summary["n_nodes"] == N_NODES
    for prefix in ("graph_", "stream_", "drift_"):
        assert any(key.startswith(prefix) for key in summary)
    assert summary["input_dim"] == 196


# =========================================================================== #
# 9. observation validation
# =========================================================================== #


def blank(**overrides) -> dict:
    base = dict(
        x=torch.zeros(2, 1, 14, 14),
        y=torch.zeros(2, dtype=torch.int64),
        has_label=True,
        n_samples=2,
        node=0,
        step=0,
        rotation_degrees=0.0,
    )
    base.update(overrides)
    return base


def test_has_label_must_agree_with_the_sample_count() -> None:
    with pytest.raises(EnvironmentError, match="disagrees with n_samples"):
        Observation(**blank(has_label=False))


def test_image_count_must_match_n_samples() -> None:
    with pytest.raises(EnvironmentError, match="but n_samples"):
        Observation(**blank(x=torch.zeros(3, 1, 14, 14)))


def test_label_count_must_match_sample_count() -> None:
    with pytest.raises(EnvironmentError, match="labels but"):
        Observation(**blank(y=torch.zeros(3, dtype=torch.int64)))


def test_a_labelled_observation_needs_labels() -> None:
    with pytest.raises(EnvironmentError, match="labelled but y is None"):
        Observation(**blank(y=None))


def test_an_idle_observation_must_not_carry_labels() -> None:
    with pytest.raises(EnvironmentError, match="unlabelled but y is not None"):
        Observation(**blank(x=torch.zeros(0, 1, 14, 14), has_label=False, n_samples=0))


# =========================================================================== #
# 10. against real MNIST
# =========================================================================== #


@pytest.fixture(scope="session")
def mnist_train() -> MnistSplit:
    from dekf_bench.data.mnist import is_cached, load_split

    if not is_cached():
        pytest.skip("MNIST not cached; run scripts/check_data.py once")
    return load_split("train", download=False)


@pytest.mark.needs_data
def test_a_default_run_builds_and_steps(mnist_train: MnistSplit) -> None:
    env = build_environment(load_config("x1_stationary"), 0, mnist_train)
    assert env.horizon == 1500
    xs, ys = pool(env.step(0))
    assert xs.shape == (20, 1, 14, 14)
    assert ys.shape == (20,)


@pytest.mark.needs_data
def test_the_whole_rotating_run_stays_inside_the_cap(mnist_train: MnistSplit) -> None:
    env = build_environment(load_config("x2_rotating"), 0, mnist_train)
    rotations = [env.drift_state(step).rotation_degrees for step in range(0, 1500, 100)]
    assert max(rotations) <= 45.0
    assert rotations == sorted(rotations)


@pytest.mark.needs_data
@pytest.mark.slow
def test_exactly_once_holds_across_a_full_run(mnist_train: MnistSplit) -> None:
    env = build_environment(load_config("x1_stationary"), 0, mnist_train)
    served = torch.cat([env.stream.consumed_by(node) for node in range(env.n_nodes)])
    assert served.numel() == 30_000 == len(torch.unique(served))
