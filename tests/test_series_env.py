"""The series environment and its held-out sets: the contract the runner relies on.

What the image environment guarantees, re-established for a generated task:
shapes, determinism, twins that differ in the drift alone, block availability,
a pooled batch, mutation detection, and held-out sets that share nothing with the
training stream.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dekf_bench.env.series import (
    SeriesError,
    agent_laws,
    build_series_environment,
    channel_value,
    pool,
    secondary_value,
)
from dekf_bench.evaluation.series_evalsets import build_series_evalsets
from dekf_bench.utils.config import load_config

HORIZON = 12


#: The series task needs a sequence model; the config refuses anything else.
SEQUENCE_MODEL = {"name": "causal_transformer", "likelihood": "gaussian", "output_dim": 31}


def make_config(series: dict | None = None, **env):
    overrides = {
        "run": {"name": "probe", "horizon": HORIZON, "seeds": [0], "dtype": "float64",
                "device": "cpu"},
        "env": {"dataset": "mackey_glass", "series": {"burn_in": 50.0, **(series or {})}, **env},
        "model": SEQUENCE_MODEL,
    }
    return load_config("x1_stationary", overrides=overrides)


LINEAR = {"schedule": "linear", "total_degrees": 45.0}


@pytest.fixture(scope="module")
def environment():
    return build_series_environment(make_config(), 0)


def test_observations_are_blocks_with_one_step_targets(environment) -> None:
    observation = environment.step(0)[3]
    assert observation.x.shape == observation.y.shape == (1, 31)
    assert torch.equal(observation.y[:, :-1], observation.x[:, 1:])
    assert observation.has_label and observation.n_samples == 1


def test_the_data_are_deterministic_per_seed() -> None:
    first = build_series_environment(make_config(), 4)
    second = build_series_environment(make_config(), 4)
    other = build_series_environment(make_config(), 5)
    assert torch.equal(first.inputs, second.inputs)
    assert not torch.equal(first.inputs, other.inputs)


def test_agents_hold_different_trajectories(environment) -> None:
    assert not torch.allclose(environment.inputs[0], environment.inputs[1])


def test_a_twin_differs_in_the_drift_alone() -> None:
    """Same seed: identical until the law moves, different after."""
    still = build_series_environment(make_config(), 0)
    drifting = build_series_environment(make_config(drift=LINEAR), 0)
    assert torch.equal(still.inputs[:, 0], drifting.inputs[:, 0])
    assert not torch.allclose(still.inputs[:, -1], drifting.inputs[:, -1])


def test_a_sensor_channel_leaves_the_undrifted_data_alone() -> None:
    """At zero displacement every channel is the identity sensor on the same law."""
    beta = build_series_environment(make_config(), 0)
    gain = build_series_environment(make_config(series={"channel": "gain", "span": 0.5}), 0)
    assert torch.equal(beta.inputs, gain.inputs)


def test_gain_drift_changes_the_measurement_not_the_dynamics() -> None:
    still = build_series_environment(make_config(series={"channel": "gain", "span": 0.5}), 0)
    drifting = build_series_environment(
        make_config(series={"channel": "gain", "span": 0.5}, drift=LINEAR), 0
    )
    assert torch.equal(still.inputs[:, 0], drifting.inputs[:, 0])
    assert not torch.allclose(still.inputs[:, -1], drifting.inputs[:, -1])


def test_the_channel_moves_by_span_at_the_cap() -> None:
    series = make_config(series={"span": 0.04, "beta": 0.2}).env.series
    assert channel_value(series, 45.0) == pytest.approx(0.24)
    assert channel_value(series, 0.0) == pytest.approx(0.2)
    assert channel_value(series, -22.5) == pytest.approx(0.18)


# --------------------------------------------------------------- combined drift


def combined(coupling: str, secondary: str = "gain", **series):
    """A two-channel series config: beta primary, a sensor channel secondary."""
    return {"channel": "beta", "span": 0.02, "secondary_channel": secondary,
            "secondary_span": 0.1, "coupling": coupling, **series}


def test_one_channel_runs_carry_no_secondary() -> None:
    """The M12 fields are inert unless a config sets them -- every run before it."""
    series = make_config(series={"channel": "gain", "span": 0.1}).env.series
    assert series.secondary_channel == ""
    assert secondary_value(series, 45.0) is None


def test_the_secondary_takes_its_own_span_and_rest_position() -> None:
    series = make_config(series=combined("correlated")).env.series
    # beta rests at its base; a gain rests at 1, and moves by its OWN span, not beta's.
    assert channel_value(series, 0.0) == pytest.approx(0.22)
    assert secondary_value(series, 0.0) == pytest.approx(1.0)
    assert secondary_value(series, 45.0) == pytest.approx(1.1)
    bias = make_config(series=combined("correlated", secondary="bias")).env.series
    assert secondary_value(bias, 0.0) == pytest.approx(0.0)
    assert secondary_value(bias, 45.0) == pytest.approx(0.1)


def test_anti_coupling_mirrors_the_secondary() -> None:
    """The amplitude-cancelling arm: beta up while the gain goes down (D108)."""
    together = build_series_environment(
        make_config(series=combined("correlated"), drift=LINEAR), 0)
    opposed = build_series_environment(
        make_config(series=combined("anti"), drift=LINEAR), 0)
    # Same law path, mirrored sensor path.
    assert np.allclose(together.values, opposed.values)
    assert together.second_values.max() > 1.0 and opposed.second_values.min() < 1.0
    assert together.second_values[0, -1] - 1.0 == pytest.approx(
        1.0 - opposed.second_values[0, -1])


def test_independent_coupling_is_refused_on_a_deterministic_schedule() -> None:
    """A linear ramp has one path however it is seeded, so this would duplicate
    `correlated` in silence rather than test anything."""
    with pytest.raises(SeriesError, match="same displacement path"):
        build_series_environment(
            make_config(series=combined("independent"), drift=LINEAR), 0)


def test_independent_coupling_draws_its_own_jumps() -> None:
    # drift=ABRUPT is not optional: without it `jumping_config` falls back to a
    # stationary schedule, both displacement paths are all-zero, and the guard in
    # build_series_environment refuses the run -- failing for the wrong reason.
    config = jumping_config(series=combined("independent"), drift=ABRUPT)
    other = jumping_config(series=combined("correlated"), drift=ABRUPT)
    independent = build_series_environment(config, 0)
    correlated = build_series_environment(other, 0)
    # The primary is untouched by the coupling; only the secondary's path changes.
    assert np.allclose(independent.values, correlated.values)
    # Compare where jumps actually land: the rounds before the first are constant
    # by construction, so a prefix slice would pass vacuously.
    jump_every = config.env.drift.jump_every
    jumped = slice(jump_every, None)
    assert not np.allclose(
        independent.second_values[:, jumped], correlated.second_values[:, jumped]
    )


ABRUPT = {"schedule": "recurring", "jump_degrees": 15.0, "jump_every": 25, "jump_seed": 0}


def jumping_config(**env):
    """A horizon that actually contains jumps.

    At the module's 12-round horizon, `horizon // jump_every` is 0, so `recurring`
    plans no jumps and sits at displacement 0 -- which made a first version of the
    two tests below pass vacuously in one direction and fail in the other.
    """
    config = make_config(**env)
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": "probe", "horizon": 100, "seeds": [0], "dtype": "float64",
                    "device": "cpu"},
            "env": {"dataset": "mackey_glass", "series": {"burn_in": 50.0}, **env},
            "model": SEQUENCE_MODEL,
        },
    ) if env else config


def test_the_jump_draw_varies_with_the_run_seed() -> None:
    """D98: one fixed draw makes a run's mean law a fixed offset from its twin's.

    Across draws the realised mean displacement has a 12-degree spread, so with a
    single jump_seed the damage metric measures the draw as much as the drift.
    """
    first = build_series_environment(jumping_config(drift=ABRUPT), 0)
    second = build_series_environment(jumping_config(drift=ABRUPT), 1)
    assert not np.allclose(first.values, second.values)


def test_a_twin_still_shares_its_drifting_run_s_data() -> None:
    """Deriving the jump seed must not disturb the histories or the noise: those
    come from streams that do not depend on the drift, which is what makes a twin."""
    still = build_series_environment(jumping_config(drift={"schedule": "stationary",
                                                          "total_degrees": 0.0}), 3)
    drifting = build_series_environment(jumping_config(drift=ABRUPT), 3)
    assert torch.equal(still.inputs[:, 0], drifting.inputs[:, 0])
    assert not torch.allclose(still.inputs[:, -1], drifting.inputs[:, -1])


def test_the_image_path_keeps_its_configured_jump_seed() -> None:
    """MNIST pins jump_seed so a shift pattern is held while the data vary (X11,
    X18, X25). The override is opt-in, so build_drift's default must not move."""
    from dekf_bench.env.drift import build_drift

    config = load_config(
        "x1_stationary",
        overrides={"run": {"name": "probe", "horizon": 100, "seeds": [0]},
                   "env": {"drift": ABRUPT}},
    )
    assert build_drift(config).schedule.rotations == build_drift(config, jump_seed=None).schedule.rotations
    assert build_drift(config, jump_seed=7).schedule.rotations != build_drift(config).schedule.rotations


def test_per_node_drift_moves_agents_at_different_rates() -> None:
    env = build_series_environment(make_config(drift=LINEAR, drift_scope="per_node"), 0)
    last = env.values[:, -1]
    assert last[0] < last[-1]
    assert np.all(np.diff(last) > 0)


def test_block_availability_is_a_sensor_dropout() -> None:
    env = build_series_environment(make_config(label_availability=0.5), 0)
    idle = [obs for t in range(HORIZON) for obs in env.step(t).values() if not obs.has_label]
    assert idle, "a 0.5 availability never dropped a block"
    assert all(obs.y is None and obs.x.shape == (0, 31) for obs in idle)
    assert 0.25 < float(env.available.float().mean()) < 0.75


def test_heterogeneity_is_evenly_spread() -> None:
    series = make_config(
        series={"beta": 0.2, "sigma": 0.05, "beta_spread": 0.02, "sigma_spread": 0.5,
                "tau_values": [17.0, 20.0]}
    ).env.series
    laws = agent_laws(series, 10)
    assert laws.beta[0] == pytest.approx(0.18) and laws.beta[-1] == pytest.approx(0.22)
    assert laws.sigma[0] == pytest.approx(0.025) and laws.sigma[-1] == pytest.approx(0.075)
    assert list(laws.tau[:4]) == [17.0, 20.0, 17.0, 20.0]


def test_pool_stacks_available_blocks(environment) -> None:
    x, y = pool(environment.step(2))
    assert x.shape == y.shape == (10, 31)


def test_pool_of_nothing_is_empty_with_the_block_width() -> None:
    env = build_series_environment(make_config(label_availability=0.0), 0)
    x, y = env.pool(env.step(0))
    assert x.shape == y.shape == (0, 31)


def test_an_in_place_mutation_is_caught(environment) -> None:
    observations = environment.step(1)
    environment.assert_unmodified(observations, 1)
    observations[0].x.add_(1.0)
    with pytest.raises(SeriesError, match="modified in place"):
        environment.assert_unmodified(observations, 1)
    # ...and the environment's own copy is untouched, because observations are clones.
    environment.assert_unmodified(environment.step(1), 1)


def test_current_equals_canonical_without_drift(environment) -> None:
    evalsets = build_series_evalsets(environment.config, environment)
    current = evalsets.at("current", 5)
    canonical = evalsets.at("canonical", 5)
    assert current.inputs.shape == (32, 31)
    assert torch.equal(current.inputs, canonical.inputs)


def test_current_follows_the_drift() -> None:
    config = make_config(drift=LINEAR)
    env = build_series_environment(config, 0)
    evalsets = build_series_evalsets(config, env)
    late = evalsets.at("current", HORIZON - 1)
    assert late.rotation_degrees == pytest.approx(float(env.values[0, HORIZON - 1]))
    assert not torch.allclose(late.inputs, evalsets.at("canonical", HORIZON - 1).inputs)


def test_held_out_blocks_never_appear_in_training(environment) -> None:
    evalsets = build_series_evalsets(environment.config, environment)
    held = evalsets.at("current", 0).inputs
    train = environment.inputs.reshape(-1, 31)
    matches = (held[:, None, :] == train[None, :, :]).all(dim=-1)
    assert not bool(matches.any())


def test_the_backward_set_is_refused(environment) -> None:
    with pytest.raises(SeriesError, match="backward"):
        build_series_evalsets(environment.config, environment).at("backward", 3)


def test_the_shard_budget_does_not_apply_to_a_generated_series() -> None:
    """N x n x T far beyond MNIST's 60 000 is fine: every round is integrated fresh."""
    config = load_config(
        "x1_stationary",
        overrides={"run": {"name": "probe", "horizon": 100_000, "seeds": [0]},
                   "env": {"dataset": "mackey_glass"}, "model": SEQUENCE_MODEL},
    )
    assert config.env.is_series


def test_the_series_task_refuses_an_image_model() -> None:
    from dekf_bench.utils.config import ConfigError

    with pytest.raises(ConfigError, match="sequence model"):
        load_config(
            "x1_stationary",
            overrides={"run": {"name": "probe", "horizon": 10, "seeds": [0]},
                       "env": {"dataset": "mackey_glass"}},
        )


def test_the_model_context_must_match_the_block() -> None:
    from dekf_bench.utils.config import ConfigError

    with pytest.raises(ConfigError, match="context"):
        make_config(series={"length": 64})
