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
    assert config.eval.backward_separation_degrees == 15.0


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


def test_the_state_model_has_two_independent_axes() -> None:
    """F_t acts on the mean, the forgetting rule on the covariance. All four
    combinations are legal so the comparison can be measured (design note D26)."""
    for transition, gamma in (("identity", 1.0), ("scalar", 0.999)):
        for forgetting in ("lambda", "process_noise"):
            learner = LearnerConfig(
                name="ekf", transition=transition, gamma=gamma, forgetting=forgetting
            )
            assert learner.transition == transition
            assert learner.forgetting == forgetting


def test_gamma_under_identity_transition_is_rejected() -> None:
    """Rather than silently ignored: F_t = I means gamma does nothing, and a
    config that sets it is asking for behaviour it will not get."""
    with pytest.raises(ConfigError, match="has no effect under"):
        LearnerConfig(name="ekf", transition="identity", gamma=0.99)


def test_forgetting_default_is_multiplicative_inflation() -> None:
    """Exactly structure-preserving in the information domain, unlike +Q."""
    assert LearnerConfig(name="ekf").forgetting == "lambda"


def test_the_lambda_default_has_a_memory_shorter_than_the_horizon() -> None:
    """Effective memory is ~1/(1-lambda). A default whose memory exceeds the
    horizon means no forgetting at all, and the tracking claim could not be
    tested."""
    memory = 1.0 / (1.0 - LearnerConfig(name="ekf").lambda_forget)
    assert memory < load_config("x1_stationary").run.horizon


def test_out_of_range_gamma_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"gamma must lie in \(0, 1\]"):
        LearnerConfig(name="ekf", transition="scalar", gamma=1.5)


def test_non_positive_process_noise_is_rejected() -> None:
    with pytest.raises(ConfigError, match="process_noise_q must be > 0"):
        LearnerConfig(name="ekf", process_noise_q=0.0)


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


def test_a_backward_separation_the_schedule_cannot_reach_is_rejected() -> None:
    """Checked against what the schedule *does*: a separation larger than the
    run's total travel leaves the probe undefined at every step, and reporting
    that as "no forgetting" would be a silent lie."""
    with pytest.raises(ConfigError, match="only travels"):
        load_config(
            "x2_rotating",
            overrides={"eval": {"evalsets": ["backward"], "backward_separation_degrees": 90.0}},
        )


def test_a_reachable_backward_separation_is_accepted() -> None:
    config = load_config(
        "x2_rotating",
        overrides={"eval": {"evalsets": ["backward"], "backward_separation_degrees": 20.0}},
    )
    assert config.eval.backward_separation_degrees == 20.0


def test_a_stationary_run_cannot_ask_for_the_backward_probe() -> None:
    """It travels zero degrees, so the probe is undefined everywhere."""
    with pytest.raises(ConfigError, match="only travels 0.0 degrees"):
        load_config("x1_stationary", overrides={"eval": {"evalsets": ["backward"]}})


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
    assert set(dumped) == {"run", "graph", "env", "model", "reference", "learners", "eval"}
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
