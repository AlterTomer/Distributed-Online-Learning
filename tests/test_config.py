"""Configuration loading, composition and validation.

The point of these tests is that a *wrong* config fails at load with a message
naming the problem, rather than producing a plausible curve nobody questions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dekf_bench.utils.config import (
    MNIST_TRAIN_SIZE,
    Config,
    ConfigError,
    DriftConfig,
    LearnerConfig,
    ModelConfig,
    deep_merge,
    default_configs_dir,
    dump_config,
    load_config,
)

EXPERIMENTS = ["x0_exactness", "x1_stationary", "x1b_atc_vs_cta", "x2_rotating", "x5_abrupt_shift"]


# --------------------------------------------------------------------------- #
# shipped configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_shipped_experiments_load(name: str) -> None:
    config = load_config(name)
    assert config.run.name == name
    assert config.learners


def test_defaults_match_the_agreed_phase_1_values() -> None:
    config = load_config("x1_stationary")
    assert config.graph.n_nodes == 10
    assert config.env.samples_per_node_per_step == 2
    assert config.run.horizon == 1500
    assert config.model.input_size == 14
    assert config.eval.backward_offset == 500


def test_backward_probe_sees_a_distinguishable_rotation() -> None:
    """The forgetting probe must differ enough from `current` to mean anything.

    At the capped drift rate the separation is backward_offset * alpha. A
    200-step offset gives 6 degrees, which is not a distribution shift a
    classifier will visibly forget; 500 gives 15.
    """
    config = load_config("x2_rotating")
    t = config.run.horizon - 1
    separation = config.rotation_at(t) - config.rotation_at(t - config.eval.backward_offset)
    assert separation >= 10.0, f"backward probe only {separation:.1f} degrees behind current"


def test_small_mlp_hits_the_phase_5_parameter_budget() -> None:
    """196-14-10 must land under 3e3, or a dense covariance is not affordable."""
    config = load_config("x1_stationary")
    assert config.model.num_params == 2908
    assert config.model.num_params <= 3_000


def test_full_size_mlp_is_two_orders_larger() -> None:
    model = ModelConfig(name="mlp", input_size=28, hidden=[128], output_dim=10)
    assert model.num_params == 101_770


def test_exactness_config_pins_every_precondition() -> None:
    """Each of these is required by the X0 identity (WORKPLAN.md section 7.1)."""
    config = load_config("x0_exactness")
    assert config.run.dtype == "float64"
    assert config.graph.topology == "complete"
    assert config.graph.weights == "uniform"
    assert config.env.label_availability == 1.0
    for learner in config.learners:
        assert learner.optimizer == "sgd", "momentum breaks the algebraic identity"
        assert learner.momentum == 0.0
    assert {learner.name for learner in config.learners} == {
        "centralized_sgd",
        "diffusion_sgd_atc",
    }


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #


def test_deep_merge_recurses_into_mappings() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_merge(base, {"a": {"y": 20, "z": 30}})
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}


def test_deep_merge_replaces_lists_wholesale() -> None:
    """Element-wise merging would make it impossible to shorten a list."""
    merged = deep_merge({"seeds": [1, 2, 3]}, {"seeds": [9]})
    assert merged["seeds"] == [9]


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_experiment_keys_win_over_included_files(tmp_path: Path) -> None:
    write_experiment(
        tmp_path,
        {
            "include": {"env": "mnist_rotating_linear", "graph": "ring", "model": "mlp_small"},
            "env": {"drift": {"total_degrees": 20.0}},
            "run": {"name": "override", "horizon": 1000},
        },
    )
    config = load_config(tmp_path / "experiment" / "e.yaml", configs_dir=tmp_path)
    assert config.env.drift.schedule == "linear"  # from the include
    assert config.env.drift.total_degrees == 20.0  # from the experiment


def test_learner_entry_may_be_a_name_or_a_mapping_with_overrides() -> None:
    config = load_config("x1_stationary", overrides={"run": {"horizon": 100}})
    plain = config.learner("local_only")
    assert plain.lr == 0.05

    tuned = load_config(
        "x1_stationary",
        overrides={"learners": [{"name": "local_only", "lr": 0.5, "optimizer": "sgd"}]},
    )
    assert tuned.learners[0].lr == 0.5


def test_overrides_apply_above_the_files() -> None:
    config = load_config("x1_stationary", overrides={"graph": {"n_nodes": 4}})
    assert config.graph.n_nodes == 4


def test_bare_experiment_name_resolves_inside_configs() -> None:
    by_name = load_config("x1_stationary")
    by_path = load_config(default_configs_dir() / "experiment" / "x1_stationary.yaml")
    assert by_name.to_dict() == by_path.to_dict()


# --------------------------------------------------------------------------- #
# strictness
# --------------------------------------------------------------------------- #


def test_unknown_key_is_an_error_not_a_silent_default() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_config("x1_stationary", overrides={"env": {"label_availabilty": 0.5}})


def test_unknown_key_suggests_the_intended_one() -> None:
    with pytest.raises(ConfigError, match="did you mean 'label_availability'"):
        load_config("x1_stationary", overrides={"env": {"label_availabilty": 0.5}})


def test_unknown_learner_lists_the_available_ones() -> None:
    with pytest.raises(ConfigError, match="unknown learner 'diffusion_sgd_atk'"):
        load_config("x1_stationary", overrides={"learners": ["diffusion_sgd_atk"]})


def test_unknown_include_section_points_at_the_learners_list() -> None:
    with pytest.raises(ConfigError, match="not an includable section"):
        load_config("x1_stationary", overrides={"include": {"learner": "local_only"}})


def test_string_where_a_number_belongs_is_rejected() -> None:
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config("x1_stationary", overrides={"run": {"horizon": "1500"}})


def test_bool_is_not_accepted_as_an_integer() -> None:
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config("x1_stationary", overrides={"graph": {"n_nodes": True}})


def test_learners_must_be_a_list() -> None:
    with pytest.raises(ConfigError, match="expected a list"):
        load_config("x1_stationary", overrides={"learners": "local_only"})


def test_duplicate_learner_names_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unique names"):
        load_config("x1_stationary", overrides={"learners": ["local_only", "local_only"]})


# --------------------------------------------------------------------------- #
# the two rules added in the 2026-07-30 spec revision
# --------------------------------------------------------------------------- #


def test_configuring_alpha_directly_is_rejected() -> None:
    with pytest.raises(ConfigError, match="alpha is derived"):
        load_config("x2_rotating", overrides={"env": {"drift": {"alpha": 0.2}}})


def test_alpha_is_derived_from_total_degrees_and_horizon() -> None:
    config = load_config("x2_rotating")
    assert config.alpha_per_step == pytest.approx(45.0 / 1500)
    assert config.rotation_at(config.run.horizon) == pytest.approx(45.0)


def test_shortening_the_horizon_does_not_change_the_total_rotation() -> None:
    """The whole point of deriving alpha: total travel is horizon-invariant."""
    long_run = load_config("x2_rotating", overrides={"run": {"horizon": 1500}})
    short_run = load_config("x2_rotating", overrides={"run": {"horizon": 500}})
    assert long_run.rotation_at(1500) == pytest.approx(short_run.rotation_at(500))
    assert short_run.alpha_per_step > long_run.alpha_per_step


def test_rotation_beyond_the_well_posed_cap_is_rejected() -> None:
    with pytest.raises(ConfigError, match="above the 45.0 degree cap"):
        load_config("x2_rotating", overrides={"env": {"drift": {"total_degrees": 300.0}}})


def test_shard_budget_rejects_the_combination_from_the_early_drafts() -> None:
    """N=10, n=4, T=2000 needs 80000 samples and MNIST has 60000."""
    with pytest.raises(ConfigError, match="shard budget exceeded"):
        load_config(
            "x1_stationary",
            overrides={
                "graph": {"n_nodes": 10},
                "env": {"samples_per_node_per_step": 4},
                "run": {"horizon": 2000},
            },
        )


def test_shard_budget_message_states_the_largest_workable_horizon() -> None:
    with pytest.raises(ConfigError, match="horizon can be at most 1500"):
        load_config(
            "x1_stationary",
            overrides={"env": {"samples_per_node_per_step": 4}, "run": {"horizon": 2000}},
        )


def test_shard_budget_is_waived_when_epochs_are_allowed() -> None:
    config = load_config(
        "x1_stationary",
        overrides={
            "env": {"samples_per_node_per_step": 4, "allow_epochs": True},
            "run": {"horizon": 2000},
        },
    )
    assert config.env.allow_epochs


def test_default_horizon_sits_exactly_inside_the_budget() -> None:
    config = load_config("x1_stationary")
    used = config.graph.n_nodes * config.env.samples_per_node_per_step * config.run.horizon
    assert used <= MNIST_TRAIN_SIZE


def test_scaling_n_agents_shortens_the_feasible_horizon() -> None:
    """Raising N does not buy a longer run; it buys a shorter one."""
    with pytest.raises(ConfigError, match="shard budget exceeded"):
        load_config("x1_stationary", overrides={"graph": {"n_nodes": 100}})


# --------------------------------------------------------------------------- #
# per-field validation
# --------------------------------------------------------------------------- #


def test_adaptive_optimizer_with_unmixed_state_is_rejected() -> None:
    """A known failure mode, not an open question (WORKPLAN.md section 3.4)."""
    with pytest.raises(ConfigError, match="known failure mode"):
        LearnerConfig(name="x", optimizer="adamw", mix_optimizer_state="none")


def test_plain_sgd_may_leave_state_unmixed() -> None:
    learner = LearnerConfig(name="x", optimizer="sgd", momentum=0.0, mix_optimizer_state="none")
    assert learner.mix_optimizer_state == "none"


def test_lambda_and_process_noise_cannot_both_be_active() -> None:
    with pytest.raises(ConfigError, match="unidentifiable"):
        LearnerConfig(name="ekf", lambda_forget=0.99, process_noise_q=1e-4)


def test_grid_dimensions_must_match_the_node_count() -> None:
    with pytest.raises(ConfigError, match="holds 6 nodes"):
        load_config(
            "x1_stationary",
            overrides={"graph": {"topology": "grid2d", "params": {"rows": 2, "cols": 3}}},
        )


def test_label_availability_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"must lie in \[0, 1\]"):
        load_config("x1_stationary", overrides={"env": {"label_availability": 1.5}})


def test_duplicate_seeds_are_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicates"):
        load_config("x1_stationary", overrides={"run": {"seeds": [0, 0, 1]}})


def test_backward_evalset_needs_an_offset_inside_the_horizon() -> None:
    with pytest.raises(ConfigError, match="backward evaluation set never exists"):
        load_config(
            "x1_stationary",
            overrides={
                "run": {"horizon": 100},
                "eval": {"evalsets": ["backward"], "backward_offset": 200},
            },
        )


# --------------------------------------------------------------------------- #
# drift schedules
# --------------------------------------------------------------------------- #


def test_stationary_drift_never_rotates() -> None:
    config = load_config("x1_stationary")
    assert all(config.rotation_at(t) == 0.0 for t in (0, 1, 500, 1500))


def test_zero_total_degrees_is_identical_to_stationary() -> None:
    stationary = load_config("x1_stationary")
    zero_drift = load_config("x2_rotating", overrides={"env": {"drift": {"total_degrees": 0.0}}})
    assert all(stationary.rotation_at(t) == zero_drift.rotation_at(t) for t in (0, 250, 1500))


def test_piecewise_drift_jumps_at_the_change_points() -> None:
    config = load_config("x5_abrupt_shift")
    assert config.rotation_at(499) == 0.0
    assert config.rotation_at(500) == pytest.approx(15.0)
    assert config.rotation_at(1499) == pytest.approx(15.0)


def test_sinusoidal_drift_returns_to_previously_seen_states() -> None:
    drift = DriftConfig(schedule="sinusoidal", amplitude_degrees=30.0, period=500)
    assert drift.rotation_at(0, 1500) == pytest.approx(0.0, abs=1e-9)
    assert drift.rotation_at(125, 1500) == pytest.approx(30.0)
    assert drift.rotation_at(500, 1500) == pytest.approx(0.0, abs=1e-9)


def test_change_points_must_be_sorted() -> None:
    with pytest.raises(ConfigError, match="must be sorted"):
        DriftConfig(schedule="piecewise", change_points=[900, 300])


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #


def test_resolved_config_round_trips_through_yaml(tmp_path: Path) -> None:
    """A result must be traceable to the exact settings that produced it."""
    config = load_config("x2_rotating")
    path = dump_config(config, tmp_path / "resolved.yaml")

    with path.open("r", encoding="utf-8") as handle:
        reloaded = yaml.safe_load(handle)

    from dekf_bench.utils.config import _build

    assert _build(Config, reloaded, "").to_dict() == config.to_dict()


def test_dumped_config_names_every_field() -> None:
    """Nothing may rely on a default that is not written down."""
    config = load_config("x1_stationary")
    dumped = config.to_dict()
    assert set(dumped) == {"run", "graph", "env", "model", "learners", "eval"}
    assert "total_degrees" in dumped["env"]["drift"]
    assert all("adapt_scope" in learner for learner in dumped["learners"])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def write_experiment(root: Path, body: dict) -> Path:
    """Build a minimal configs/ tree in a tmp dir, for composition tests."""
    source = default_configs_dir()
    for section in ("env", "graph", "model", "learner"):
        target = root / section
        target.mkdir(parents=True, exist_ok=True)
        for item in (source / section).glob("*.yaml"):
            (target / item.name).write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
    (root / "base.yaml").write_text(
        (source / "base.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    experiment_dir = root / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    path = experiment_dir / "e.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(body, handle)
    return path
