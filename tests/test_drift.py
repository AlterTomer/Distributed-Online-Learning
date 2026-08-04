"""Drift schedules: the map from a step to a rotation.

The schedules are pure functions of the step, so these tests are exact rather
than statistical. The properties that matter are that a zero-drift run is
*identical* to a stationary one, that jumps land where specified, and that no
schedule can travel past the well-posedness cap.
"""

from __future__ import annotations

import math

import pytest

from dekf_bench.env.drift import (
    MAX_WELL_POSED_DEGREES,
    Drift,
    DriftError,
    DriftState,
    Linear,
    Piecewise,
    Sinusoidal,
    Stationary,
    build_drift,
    build_schedule,
)
from dekf_bench.utils.config import load_config

HORIZON = 1500


# =========================================================================== #
# 1. the schedules
# =========================================================================== #


def test_stationary_never_moves() -> None:
    schedule = Stationary()
    assert all(schedule.rotation_at(t) == 0.0 for t in (0, 1, 500, HORIZON))
    assert schedule.total_travel(HORIZON) == 0.0


def test_linear_reaches_exactly_the_total_at_the_horizon() -> None:
    schedule = Linear(total_degrees=45.0, horizon=HORIZON)
    assert schedule.rotation_at(0) == 0.0
    assert schedule.rotation_at(HORIZON) == pytest.approx(45.0)
    assert schedule.alpha == pytest.approx(45.0 / HORIZON)


def test_linear_accumulates_at_a_constant_rate() -> None:
    schedule = Linear(total_degrees=45.0, horizon=HORIZON)
    steps = schedule.rotation_at(101) - schedule.rotation_at(100)
    assert steps == pytest.approx(schedule.alpha)
    assert schedule.rotation_at(HORIZON // 2) == pytest.approx(22.5)


def test_shortening_the_horizon_does_not_change_the_total_travel() -> None:
    """The whole point of deriving alpha: total travel is horizon-invariant."""
    long_run = Linear(total_degrees=45.0, horizon=1500)
    short_run = Linear(total_degrees=45.0, horizon=500)
    assert long_run.rotation_at(1500) == pytest.approx(short_run.rotation_at(500))
    assert short_run.alpha > long_run.alpha


def test_zero_total_degrees_is_bit_identical_to_stationary() -> None:
    """`alpha = 0` must not merely be close to stationary -- the two runs have
    to be the same run, or the drift ablation has a confound in it."""
    linear = Linear(total_degrees=0.0, horizon=HORIZON)
    stationary = Stationary()
    assert all(linear.rotation_at(t) == stationary.rotation_at(t) for t in range(0, HORIZON, 37))


def test_piecewise_jumps_exactly_at_the_change_points() -> None:
    schedule = Piecewise(change_points=(500,), jump_degrees=15.0)
    assert schedule.rotation_at(499) == 0.0
    assert schedule.rotation_at(500) == pytest.approx(15.0)
    assert schedule.rotation_at(1499) == pytest.approx(15.0)


def test_piecewise_accumulates_across_several_change_points() -> None:
    schedule = Piecewise(change_points=(200, 600, 1000), jump_degrees=10.0)
    assert schedule.rotation_at(0) == 0.0
    assert schedule.rotation_at(200) == pytest.approx(10.0)
    assert schedule.rotation_at(600) == pytest.approx(20.0)
    assert schedule.rotation_at(1000) == pytest.approx(30.0)


def test_piecewise_with_no_change_points_is_stationary() -> None:
    schedule = Piecewise(change_points=(), jump_degrees=15.0)
    assert schedule.total_travel(HORIZON) == 0.0


def test_sinusoidal_returns_to_previously_seen_states() -> None:
    """The property that makes it a forgetting probe."""
    schedule = Sinusoidal(amplitude_degrees=30.0, period=500)
    assert schedule.rotation_at(0) == pytest.approx(0.0, abs=1e-9)
    assert schedule.rotation_at(125) == pytest.approx(30.0)
    assert schedule.rotation_at(250) == pytest.approx(0.0, abs=1e-9)
    assert schedule.rotation_at(375) == pytest.approx(-30.0)
    assert schedule.rotation_at(500) == pytest.approx(0.0, abs=1e-9)


def test_sinusoidal_is_periodic() -> None:
    schedule = Sinusoidal(amplitude_degrees=30.0, period=500)
    for step in (13, 77, 301):
        assert schedule.rotation_at(step) == pytest.approx(schedule.rotation_at(step + 500))


def test_sinusoidal_never_exceeds_its_amplitude() -> None:
    schedule = Sinusoidal(amplitude_degrees=30.0, period=500)
    assert schedule.total_travel(HORIZON) == pytest.approx(30.0, abs=1e-3)


def test_every_schedule_reports_its_name() -> None:
    names = {
        Stationary().name,
        Linear(45.0, HORIZON).name,
        Piecewise((500,), 15.0).name,
        Sinusoidal(30.0, 500).name,
    }
    assert names == {"stationary", "linear", "piecewise", "sinusoidal"}


# =========================================================================== #
# 2. rejected schedules
# =========================================================================== #


def test_linear_needs_a_positive_horizon() -> None:
    with pytest.raises(DriftError, match="horizon >= 1"):
        Linear(total_degrees=45.0, horizon=0)


def test_sinusoidal_needs_a_positive_period() -> None:
    with pytest.raises(DriftError, match="period >= 1"):
        Sinusoidal(amplitude_degrees=30.0, period=0)


def test_unsorted_change_points_are_rejected() -> None:
    with pytest.raises(DriftError, match="must be sorted"):
        Piecewise(change_points=(900, 300), jump_degrees=15.0)


def test_negative_change_points_are_rejected() -> None:
    with pytest.raises(DriftError, match="non-negative"):
        Piecewise(change_points=(-1,), jump_degrees=15.0)


def test_unknown_schedule_name_is_rejected() -> None:
    config = load_config("x1_stationary")
    object.__setattr__(config.env.drift, "schedule", "quadratic")
    with pytest.raises(DriftError, match="unknown drift schedule"):
        build_schedule(config.env.drift, HORIZON)


# =========================================================================== #
# 3. the well-posedness cap, checked against behaviour
# =========================================================================== #


def test_the_cap_is_checked_against_what_the_schedule_does() -> None:
    """Each field can look reasonable while the schedule as a whole travels too
    far. Four 15-degree jumps is 60 degrees, and no single field is out of range.
    """
    config = load_config(
        "x5_abrupt_shift",
        overrides={"env": {"drift": {"change_points": [200, 500, 800, 1100]}}},
    )
    with pytest.raises(DriftError, match="reaches 60.0 degrees"):
        build_drift(config)


def test_a_schedule_inside_the_cap_is_accepted() -> None:
    config = load_config(
        "x5_abrupt_shift",
        overrides={"env": {"drift": {"change_points": [400, 800, 1200]}}},
    )
    drift = build_drift(config)
    assert drift.schedule.total_travel(HORIZON) == pytest.approx(45.0)


def test_every_shipped_experiment_stays_inside_the_cap() -> None:
    for name in ("x1_stationary", "x2_rotating", "x5_abrupt_shift", "x0_exactness"):
        config = load_config(name)
        drift = build_drift(config)
        assert drift.schedule.total_travel(config.run.horizon) <= MAX_WELL_POSED_DEGREES + 1e-9


# =========================================================================== #
# 4. scope
# =========================================================================== #


def test_global_scope_gives_every_agent_the_same_rotation() -> None:
    drift = Drift(Linear(45.0, HORIZON), scope="global", n_nodes=10)
    rotations = {drift.rotation_at(750, node) for node in range(10)}
    assert len(rotations) == 1


def test_per_node_scope_spreads_the_rates() -> None:
    drift = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=10, spread=0.5)
    rotations = [drift.rotation_at(HORIZON, node) for node in range(10)]
    assert len(set(rotations)) == 10
    assert rotations == sorted(rotations)


def test_per_node_multipliers_top_out_at_one_not_above_it() -> None:
    """Scaling any agent *above* the configured rate would carry it past the cap
    the schedule was validated against, so the spread slows the laggards."""
    drift = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=10, spread=0.5)
    assert drift.multiplier(9) == pytest.approx(1.0)
    assert drift.multiplier(0) == pytest.approx(0.5)
    assert max(drift.rotation_at(HORIZON, node) for node in range(10)) == pytest.approx(45.0)


def test_per_node_drift_never_exceeds_the_cap() -> None:
    drift = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=10, spread=0.9)
    worst = max(abs(drift.rotation_at(HORIZON, node)) for node in range(10))
    assert worst <= MAX_WELL_POSED_DEGREES + 1e-9


def test_zero_spread_makes_per_node_identical_to_global() -> None:
    per_node = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=10, spread=0.0)
    glob = Drift(Linear(45.0, HORIZON), scope="global", n_nodes=10)
    assert all(per_node.rotation_at(700, node) == glob.rotation_at(700, node) for node in range(10))


def test_single_agent_ignores_the_scope() -> None:
    drift = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=1, spread=0.5)
    assert drift.multiplier(0) == 1.0


def test_states_at_covers_every_agent() -> None:
    drift = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=4)
    states = drift.states_at(HORIZON)
    assert len(states) == 4
    assert [state.node for state in states] == [0, 1, 2, 3]
    assert all(state.step == HORIZON for state in states)


def test_asking_for_one_network_state_under_per_node_drift_is_an_error() -> None:
    """There is no single answer, and returning agent 0's would be a silent lie."""
    drift = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=10)
    with pytest.raises(DriftError, match="no single network-wide state"):
        drift.state_at(500)


def test_global_scope_has_a_single_network_state() -> None:
    drift = Drift(Linear(45.0, HORIZON), scope="global", n_nodes=10)
    state = drift.state_at(HORIZON)
    assert state.rotation_degrees == pytest.approx(45.0)
    assert state.node is None


def test_unknown_scope_is_rejected() -> None:
    with pytest.raises(DriftError, match="unknown drift scope"):
        Drift(Stationary(), scope="per_edge")


def test_spread_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(DriftError, match=r"spread must lie in \[0, 1\)"):
        Drift(Stationary(), scope="per_node", n_nodes=4, spread=1.0)


def test_node_outside_the_network_is_rejected() -> None:
    drift = Drift(Linear(45.0, HORIZON), scope="per_node", n_nodes=4)
    with pytest.raises(DriftError, match="outside 0..3"):
        drift.rotation_at(0, 9)


# =========================================================================== #
# 5. states and helpers
# =========================================================================== #


def test_drift_state_knows_when_it_is_canonical() -> None:
    assert DriftState(step=0, rotation_degrees=0.0).is_canonical
    assert not DriftState(step=10, rotation_degrees=0.1).is_canonical


def test_stationary_runs_visit_exactly_one_rotation() -> None:
    """Which is why a stationary run needs one evaluation set, not 1500."""
    drift = build_drift(load_config("x1_stationary"))
    assert drift.distinct_rotations(HORIZON) == (0.0,)
    assert drift.is_stationary


def test_piecewise_runs_visit_one_rotation_per_regime() -> None:
    drift = build_drift(load_config("x5_abrupt_shift"))
    assert drift.distinct_rotations(HORIZON) == (0.0, 15.0)


def test_linear_runs_visit_a_rotation_per_step() -> None:
    drift = build_drift(load_config("x2_rotating"))
    assert len(drift.distinct_rotations(HORIZON)) == HORIZON + 1


def test_summary_reports_the_derived_rate() -> None:
    drift = build_drift(load_config("x2_rotating"))
    summary = drift.summary(HORIZON)
    assert summary["schedule"] == "linear"
    assert summary["alpha_per_step"] == pytest.approx(0.03)
    assert summary["rotation_at_end"] == pytest.approx(45.0)


# =========================================================================== #
# 6. built from the shipped configs
# =========================================================================== #


def test_x1_is_stationary() -> None:
    drift = build_drift(load_config("x1_stationary"))
    assert drift.is_stationary
    assert drift.rotation_at(HORIZON) == 0.0


def test_x2_derives_the_documented_rate() -> None:
    drift = build_drift(load_config("x2_rotating"))
    assert isinstance(drift.schedule, Linear)
    assert drift.schedule.alpha == pytest.approx(0.03)
    assert drift.rotation_at(HORIZON) == pytest.approx(45.0)


def test_x5_jumps_fifteen_degrees_at_five_hundred() -> None:
    drift = build_drift(load_config("x5_abrupt_shift"))
    assert drift.rotation_at(499) == 0.0
    assert drift.rotation_at(500) == pytest.approx(15.0)


def test_the_backward_probe_sees_a_distinguishable_rotation() -> None:
    """The separation is a property of the schedule, so it lives here rather
    than in the config tests. Anchored by rotation now, not by a step count:
    a fixed offset gave identically zero separation under the sinusoidal
    schedule, whose whole purpose is to expose forgetting (design note D32).
    """
    config = load_config("x2_rotating")
    drift = build_drift(config)
    separation = config.eval.backward_separation_degrees
    current = drift.rotation_at(config.run.horizon - 1)

    source = next(
        step
        for step in range(config.run.horizon - 2, -1, -1)
        if abs(drift.rotation_at(step) - current) >= separation
    )
    assert abs(current - drift.rotation_at(source)) >= separation


def test_sinusoidal_config_builds() -> None:
    config = load_config("x2_rotating", overrides={"include": {"env": "mnist_rotating_sinusoidal"}})
    drift = build_drift(config)
    assert isinstance(drift.schedule, Sinusoidal)
    assert drift.rotation_at(125) == pytest.approx(30.0)


def test_a_full_sine_period_returns_to_the_start() -> None:
    config = load_config("x2_rotating", overrides={"include": {"env": "mnist_rotating_sinusoidal"}})
    drift = build_drift(config)
    assert drift.rotation_at(0) == pytest.approx(drift.rotation_at(500), abs=1e-9)
    assert math.isclose(drift.rotation_at(250), 0.0, abs_tol=1e-9)
