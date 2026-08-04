"""Evaluation sets and the scoring protocol.

The two properties that carry weight: the evaluation set must carry the *same*
rotation as the training data at that step, and the backward probe must be
separated from the current one on every schedule -- which a fixed step offset
was not.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.data.transforms import build_transform
from dekf_bench.env.drift import build_drift
from dekf_bench.env.environment import build_environment
from dekf_bench.evaluation import protocol
from dekf_bench.evaluation.evalsets import EvalSetBuilder, EvalSetError, build_evalsets
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.utils.config import load_config

HORIZON = 120
SEPARATION = 15.0


def split(n: int = 300, seed: int = 0) -> MnistSplit:
    generator = torch.Generator().manual_seed(seed)
    return MnistSplit(
        images=torch.rand(n, 1, 28, 28, generator=generator),
        labels=torch.arange(n, dtype=torch.int64) % 10,
        split="synthetic",
    )


def builder_for(experiment: str = "x2_rotating", **overrides) -> EvalSetBuilder:
    from dekf_bench.utils.config import deep_merge

    config = load_config(experiment, overrides=deep_merge({"run": {"horizon": HORIZON}}, overrides))
    train, test = split(600), split(300, seed=1)
    return EvalSetBuilder(
        test=test,
        transform=build_transform(train.images, 14),
        drift=build_drift(config),
        horizon=config.run.horizon,
        backward_separation_degrees=SEPARATION,
    )


# =========================================================================== #
# 1. the sets carry the right rotation
# =========================================================================== #


def test_current_carries_the_training_rotation() -> None:
    """G4: evaluate on anything else while training on rotated data and every
    method appears to fail, for a reason unrelated to any of them."""
    builder = builder_for()
    drift = builder.drift
    for step in (0, 40, HORIZON - 1):
        assert builder.current(step).rotation_degrees == pytest.approx(drift.rotation_at(step))


def test_canonical_is_always_unrotated() -> None:
    builder = builder_for()
    for step in (0, 60, HORIZON - 1):
        assert builder.canonical(step).rotation_degrees == 0.0


def test_current_and_canonical_differ_once_the_run_has_drifted() -> None:
    builder = builder_for()
    late = HORIZON - 1
    assert not torch.allclose(builder.current(late).images, builder.canonical(late).images)


def test_labels_are_untouched_by_rotation() -> None:
    builder = builder_for()
    assert torch.equal(builder.current(50).labels, builder.canonical(50).labels)


def test_sets_are_cached_by_rotation_not_by_step() -> None:
    """A stationary run builds one set, not one per step."""
    builder = builder_for("x1_stationary")
    for step in range(0, HORIZON, 7):
        builder.current(step)
    assert len(builder._cache) == 1


def test_a_piecewise_run_builds_one_set_per_regime() -> None:
    # The shipped change point is at t=500; this fixture runs 120 steps, so the
    # jump has to be moved inside the horizon or there is only one regime.
    builder = builder_for("x5_abrupt_shift", env={"drift": {"change_points": [60]}})
    for step in range(HORIZON):
        builder.current(step)
    assert len(builder._cache) == 2


# =========================================================================== #
# 2. the backward probe, on every schedule
# =========================================================================== #


def test_the_backward_probe_is_separated_under_linear_drift() -> None:
    builder = builder_for("x2_rotating")
    for step in (80, 100, HORIZON - 1):
        current, backward = builder.current(step), builder.backward(step)
        assert backward is not None
        assert abs(current.rotation_degrees - backward.rotation_degrees) >= SEPARATION


def test_the_backward_probe_survives_a_piecewise_schedule() -> None:
    """A fixed step offset collapsed to zero separation once t passed the last
    change point plus the offset; anchoring by rotation does not."""
    builder = builder_for("x5_abrupt_shift", env={"drift": {"change_points": [40]}})
    for step in (60, 100, HORIZON - 1):
        current, backward = builder.current(step), builder.backward(step)
        assert backward is not None
        assert abs(current.rotation_degrees - backward.rotation_degrees) >= SEPARATION
        assert backward.source_step is not None and backward.source_step < 40


def test_the_backward_probe_survives_a_sinusoidal_schedule() -> None:
    """The case that was identically degenerate: with offset == period the old
    anchor gave phi(t - offset) == phi(t) at *every* step, so the schedule chosen
    to expose forgetting could not measure it."""
    builder = builder_for(
        "x2_rotating",
        include={"env": "mnist_rotating_sinusoidal"},
        env={"drift": {"period": 60}},
    )
    separations = []
    for step in range(40, HORIZON):
        backward = builder.backward(step)
        if backward is not None:
            separations.append(
                abs(builder.current(step).rotation_degrees - backward.rotation_degrees)
            )
    assert separations, "the probe must be defined somewhere"
    assert min(separations) >= SEPARATION


def test_the_backward_state_is_one_the_model_actually_visited() -> None:
    """Forgetting of a distribution never seen is not forgetting."""
    builder = builder_for()
    step = HORIZON - 1
    backward = builder.backward(step)
    assert backward is not None
    assert backward.source_step is not None
    assert builder.drift.rotation_at(backward.source_step) == pytest.approx(
        backward.rotation_degrees
    )


def test_the_probe_picks_the_most_recent_qualifying_state() -> None:
    """The most recently visited one is the most plausibly still remembered."""
    builder = builder_for()
    step = HORIZON - 1
    source = builder.backward_step(step)
    assert source is not None
    current = builder.drift.rotation_at(step)
    for later in range(source + 1, step):
        assert abs(builder.drift.rotation_at(later) - current) < SEPARATION


def test_a_stationary_run_has_no_backward_probe() -> None:
    """Reported as undefined, never as 'no forgetting' -- the two are different
    findings and must not log the same value."""
    builder = builder_for("x1_stationary")
    assert builder.backward(HORIZON - 1) is None
    assert builder.first_backward_step() is None
    assert not builder.backward_is_available(50)


def test_the_probe_is_undefined_early_in_a_run() -> None:
    """Before the run has drifted far enough, there is no qualifying state."""
    builder = builder_for()
    assert builder.backward(0) is None
    first = builder.first_backward_step()
    assert first is not None and first > 0
    assert builder.backward(first) is not None


def test_the_summary_reports_whether_the_probe_ever_works() -> None:
    assert builder_for("x1_stationary").summary()["backward_ever_available"] is False
    assert builder_for("x2_rotating").summary()["backward_ever_available"] is True


# =========================================================================== #
# 3. per-node drift: both views
# =========================================================================== #


def test_per_node_drift_gives_each_agent_its_own_current_set() -> None:
    builder = builder_for("x2_rotating", env={"drift_scope": "per_node"})
    rotations = {builder.current(HORIZON - 1, node).rotation_degrees for node in range(10)}
    assert len(rotations) == 10


def test_the_mean_rotation_is_a_state_no_agent_faces() -> None:
    """Which is why scoring against it is a second view rather than the metric."""
    builder = builder_for("x2_rotating", env={"drift_scope": "per_node"})
    step = HORIZON - 1
    mean = builder.mean_rotation(step)
    own = [builder.current(step, node).rotation_degrees for node in range(10)]
    assert mean == pytest.approx(sum(own) / len(own))
    assert all(abs(rotation - mean) > 1e-9 for rotation in own)


def test_global_drift_makes_both_views_identical() -> None:
    builder = builder_for("x2_rotating")
    step = HORIZON - 1
    assert builder.current(step).rotation_degrees == pytest.approx(
        builder.current_at_mean(step).rotation_degrees
    )


def test_an_unknown_evalset_name_is_rejected() -> None:
    with pytest.raises(EvalSetError, match="unknown evaluation set"):
        builder_for().at("future", 10)


def test_prequential_is_not_built_here() -> None:
    """It scores the incoming training batch, which no held-out set can supply."""
    with pytest.raises(EvalSetError, match="scores the incoming training batch"):
        builder_for().at("prequential", 10)


# =========================================================================== #
# 4. the protocol
# =========================================================================== #


@pytest.fixture(scope="module")
def scored():
    """A small run wired end to end, with an untrained model."""
    from dekf_bench.models.registry import build_model_from_config

    config = load_config("x2_rotating", overrides={"run": {"horizon": HORIZON}})
    train, test = split(4000), split(300, seed=1)
    environment = build_environment(config, 0, train)
    model = build_model_from_config(config)
    params = model.init_params(environment.seeds.torch_generator("init"))
    return {
        "config": config,
        "env": environment,
        "builder": build_evalsets(config, environment, test),
        "predict": lambda node, x: model.forward(params, x),
        "likelihood": Categorical(10),
    }


def test_prequential_scores_every_labelled_agent(scored) -> None:
    result = protocol.prequential(
        scored["env"].step(5), scored["predict"], scored["likelihood"], step=5
    )
    assert len(result.scores) == 10
    assert result.skipped == ()


def test_prequential_skips_idle_agents_rather_than_scoring_them(scored) -> None:
    """A 0% or 100% error rate for an agent with no samples would bias the mean
    by an amount growing as labels get sparser -- along the axis X4 sweeps."""
    config = load_config(
        "x2_rotating",
        overrides={"run": {"horizon": HORIZON}, "env": {"label_availability": 0.5}},
    )
    environment = build_environment(config, 0, split(4000))
    result = protocol.prequential(
        environment.step(5), scored["predict"], scored["likelihood"], step=5
    )
    assert result.skipped
    assert len(result.scores) + len(result.skipped) == 10
    assert all(score.n_samples > 0 for score in result.scores)


def test_prequential_records_the_drift_state_it_scored_at(scored) -> None:
    result = protocol.prequential(
        scored["env"].step(60), scored["predict"], scored["likelihood"], step=60
    )
    expected = scored["env"].drift_state(60).rotation_degrees
    assert all(score.rotation_degrees == pytest.approx(expected) for score in result.scores)


def test_an_untrained_model_scores_near_chance(scored) -> None:
    result = protocol.prequential(
        scored["env"].step(5), scored["predict"], scored["likelihood"], step=5
    )
    mean_error = sum(score.error_rate for score in result.scores) / len(result.scores)
    assert 0.80 < mean_error <= 1.0


def test_full_evaluation_scores_each_agent_on_each_set(scored) -> None:
    result = protocol.full_evaluate(
        scored["builder"],
        scored["predict"],
        scored["likelihood"],
        step=HORIZON - 1,
        nodes=[0, 1, 2],
        evalsets=["current", "backward", "canonical"],
        batch_size=128,
    )
    assert len(result.for_evalset("current")) == 3
    assert len(result.for_evalset("canonical")) == 3
    assert len(result.for_evalset("backward")) == 3


def test_full_evaluation_omits_an_undefined_backward_probe(scored) -> None:
    """Rather than logging a zero: 'cannot measure' and 'measured no forgetting'
    must not produce the same row."""
    result = protocol.full_evaluate(
        scored["builder"],
        scored["predict"],
        scored["likelihood"],
        step=0,
        nodes=[0],
        evalsets=["current", "backward"],
        batch_size=128,
    )
    assert result.for_evalset("current")
    assert not result.for_evalset("backward")


def test_batching_does_not_change_the_score(scored) -> None:
    scores = [
        protocol.full_evaluate(
            scored["builder"],
            scored["predict"],
            scored["likelihood"],
            step=60,
            nodes=[0],
            evalsets=["canonical"],
            batch_size=size,
        ).for_evalset("canonical")[0]
        for size in (16, 128, 10_000)
    ]
    assert len({score.n_correct for score in scores}) == 1


def test_calibration_travels_with_the_score(scored) -> None:
    result = protocol.full_evaluate(
        scored["builder"],
        scored["predict"],
        scored["likelihood"],
        step=60,
        nodes=[0],
        evalsets=["canonical"],
        batch_size=256,
    )
    score = result.for_evalset("canonical")[0]
    assert score.calibration is not None
    assert 0.0 <= score.calibration.ece <= 1.0
    assert score.nll is not None and score.nll > 0


def test_rows_are_per_agent_and_carry_the_drift_state(scored) -> None:
    result = protocol.full_evaluate(
        scored["builder"],
        scored["predict"],
        scored["likelihood"],
        step=60,
        nodes=[0, 1],
        evalsets=["current"],
        batch_size=256,
    )
    rows = result.as_rows()
    assert {row["node_id"] for row in rows} == {0, 1}
    assert all("drift_state" in row for row in rows)
    assert "error_rate" in {row["metric"] for row in rows}


def test_per_node_drift_adds_the_mean_view(scored) -> None:
    config = load_config(
        "x2_rotating",
        overrides={"run": {"horizon": HORIZON}, "env": {"drift_scope": "per_node"}},
    )
    environment = build_environment(config, 0, split(4000))
    builder = build_evalsets(config, environment, split(300, seed=1))

    result = protocol.full_evaluate(
        builder,
        scored["predict"],
        scored["likelihood"],
        step=HORIZON - 1,
        nodes=[0, 1, 2],
        evalsets=["current"],
        batch_size=256,
        per_node_drift=True,
    )
    own = {score.rotation_degrees for score in result.for_evalset("current")}
    mean = {score.rotation_degrees for score in result.for_evalset("current_mean")}
    assert len(own) == 3, "each agent scored at its own rotation"
    assert len(mean) == 1, "and all at the shared mean rotation"


# =========================================================================== #
# 5. cadence
# =========================================================================== #


def test_evaluation_happens_on_the_cadence() -> None:
    assert protocol.should_evaluate(0, 25, 100)
    assert protocol.should_evaluate(50, 25, 100)
    assert not protocol.should_evaluate(51, 25, 100)


def test_the_final_step_is_always_evaluated() -> None:
    """Otherwise the last point of every curve would move with T mod K."""
    assert protocol.should_evaluate(99, 25, 100)
    assert protocol.should_evaluate(1499, 25, 1500)


def test_a_non_positive_cadence_is_rejected() -> None:
    with pytest.raises(protocol.ProtocolError, match="eval_every must be >= 1"):
        protocol.should_evaluate(0, 0, 100)
