r"""The offline reference classifier.

Most of these run on a tiny synthetic dataset with a two-epoch budget, so the
suite stays fast. What they check is the *machinery* -- that selection never
touches the test split, that interpolation behaves, that the three init
strategies really differ -- rather than the accuracy, which is the cached
asset's job.
"""

from __future__ import annotations

import json

import pytest
import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.data.transforms import build_transform
from dekf_bench.evaluation.reference import (
    EpochRecord,
    Reference,
    ReferenceError,
    ReferenceResult,
    cache_path,
    load,
    repeat_seeds,
    save,
    seed_spread,
    train_one,
    train_reference,
)
from dekf_bench.models.mlp import MLP
from dekf_bench.utils.config import ConfigError, load_config


def split(n: int = 400, seed: int = 0) -> MnistSplit:
    """Learnable-but-tiny: the label is a function of the image, so training
    actually reduces the error rather than fitting noise."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n, dtype=torch.int64) % 10
    images = torch.rand(n, 1, 28, 28, generator=generator) * 0.2
    for index, label in enumerate(labels):
        images[index, 0, int(label) * 2 : int(label) * 2 + 2, :] = 1.0
    return MnistSplit(images=images, labels=labels, split="synthetic")


@pytest.fixture(scope="module")
def pieces():
    train, test = split(400), split(200, seed=1)
    model = MLP(input_size=14, hidden=(8,), output_dim=10)
    return train, test, model, build_transform(train.images, 14)


def config_for(**reference):
    settings = {
        "epochs": 2,
        "validation_size": 50,
        "rotation_min_degrees": 0.0,
        "rotation_max_degrees": 10.0,
        "rotation_step_degrees": 5.0,
    }
    settings.update(reference)
    return load_config("x1_stationary", overrides={"reference": settings})


# =========================================================================== #
# 1. the rotation grid
# =========================================================================== #


def test_the_default_grid_covers_every_schedule_the_configs_use() -> None:
    """linear [0, 45], piecewise [0, 15], sinusoidal [-30, +30]. No run should
    have to extrapolate."""
    settings = load_config("x1_stationary").reference
    assert settings.rotations[0] == -30.0
    assert settings.rotations[-1] == 45.0
    assert len(settings.rotations) == 16


def test_grid_endpoints_are_inclusive() -> None:
    settings = config_for().reference
    assert settings.rotations == [0.0, 5.0, 10.0]


def test_an_empty_rotation_range_is_rejected() -> None:
    with pytest.raises(ConfigError, match="rotation range is empty"):
        config_for(rotation_min_degrees=10.0, rotation_max_degrees=0.0)


def test_a_non_positive_step_is_rejected() -> None:
    with pytest.raises(ConfigError, match="rotation_step_degrees must be > 0"):
        config_for(rotation_step_degrees=0.0)


# =========================================================================== #
# 2. training, and what selection is allowed to see
# =========================================================================== #


def test_training_reduces_the_error(pieces) -> None:
    train, test, model, transform = pieces
    result, _ = train_one(0.0, train, test, model, transform, config_for(epochs=8).reference)
    assert result.error_rate < 0.9
    assert result.epochs[-1].train_loss < result.epochs[0].train_loss


def test_the_validation_slice_comes_out_of_train_not_test(pieces) -> None:
    """The test split must stay untouched by anything that influences the model."""
    train, test, model, transform = pieces
    result, _ = train_one(0.0, train, test, model, transform, config_for().reference)
    assert result.n_validation == 50
    assert result.n_train == len(train) - 50


def test_selection_never_reads_the_test_error(pieces) -> None:
    """The epoch chosen must be the validation optimum, never the test one.

    Early-stopping on test would leak into the quantity every gap is measured
    against -- invisibly, and always in the same direction.
    """
    train, test, model, transform = pieces
    result, _ = train_one(0.0, train, test, model, transform, config_for(epochs=6).reference)

    validation = [record.validation_error for record in result.epochs]
    assert result.selected_epoch == validation.index(min(validation))


def test_the_test_curve_is_still_recorded_for_inspection(pieces) -> None:
    """Recorded, never selected on. Having it makes convergence auditable."""
    train, test, model, transform = pieces
    result, _ = train_one(0.0, train, test, model, transform, config_for().reference)
    assert all(record.test_error is not None for record in result.epochs)


def test_fixed_budget_trains_on_everything_and_takes_the_last_epoch(pieces) -> None:
    train, test, model, transform = pieces
    settings = config_for(selection="fixed_budget", validation_size=0, epochs=3).reference
    result, _ = train_one(0.0, train, test, model, transform, settings)

    assert result.n_validation == 0
    assert result.n_train == len(train)
    assert result.selected_epoch == 2
    assert all(record.validation_error is None for record in result.epochs)


def test_fixed_budget_with_a_validation_size_is_rejected() -> None:
    """Rather than silently ignored: a config setting it is asking for a holdout
    it will not get."""
    with pytest.raises(ConfigError, match="would be silently ignored"):
        config_for(selection="fixed_budget", validation_size=5000)


def test_convergence_is_reported_not_assumed(pieces) -> None:
    """Selecting the final epoch means the budget decided where training
    stopped, so 'trained to convergence' would be an assumption."""
    train, test, model, transform = pieces
    settings = config_for(selection="fixed_budget", validation_size=0, epochs=2).reference
    result, _ = train_one(0.0, train, test, model, transform, settings)
    assert not result.converged


def test_a_validation_slice_larger_than_the_split_is_rejected(pieces) -> None:
    train, test, model, transform = pieces
    settings = config_for().reference
    settings.validation_size = len(train) + 1
    with pytest.raises(ReferenceError, match="leaves no training data"):
        train_one(0.0, train, test, model, transform, settings)


def test_the_epoch_budget_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="epochs must be >= 1"):
        config_for(epochs=0)


# =========================================================================== #
# 3. the three init strategies
# =========================================================================== #


def test_shared_seed_trains_every_level_independently(pieces) -> None:
    train, test, model, transform = pieces
    reference = train_reference(
        config_for(init_strategy="shared_seed"), train, test, model, transform, progress=False
    )
    assert reference.init_strategy == "shared_seed"
    assert len(reference.results) == 3


def test_warm_start_differs_from_independent_training(pieces) -> None:
    """Otherwise the option would be a no-op. e*(10) under warm_start depends on
    having passed through 5 -- contaminated by exactly the history the reference
    exists to be free of."""
    train, test, model, transform = pieces
    cold = train_reference(
        config_for(init_strategy="shared_seed", epochs=3),
        train,
        test,
        model,
        transform,
        progress=False,
    )
    warm = train_reference(
        config_for(init_strategy="warm_start", epochs=3),
        train,
        test,
        model,
        transform,
        progress=False,
    )
    assert cold.results[-1].error_rate != warm.results[-1].error_rate


def test_independent_seeds_gives_each_level_its_own_initialisation(pieces) -> None:
    train, test, model, transform = pieces
    reference = train_reference(
        config_for(init_strategy="independent_seeds"),
        train,
        test,
        model,
        transform,
        progress=False,
    )
    assert reference.init_strategy == "independent_seeds"


def test_an_unknown_init_strategy_is_rejected() -> None:
    with pytest.raises(ConfigError, match="reference.init_strategy"):
        config_for(init_strategy="pretrained")


def test_the_reference_seed_is_separate_from_the_run_seed() -> None:
    """e* is a fixed asset: re-running an experiment must never silently
    retrain the thing it is measured against."""
    settings = load_config("x1_stationary", overrides={"run": {"seeds": [7]}}).reference
    assert settings.seed == 0


# =========================================================================== #
# 4. run-to-run noise
# =========================================================================== #


def test_repeats_vary_only_the_seed(pieces) -> None:
    """Same rotation, same everything else. If the results were identical the
    diagnostic would measure nothing."""
    train, test, model, transform = pieces
    results = repeat_seeds(
        0.0, train, test, model, transform, config_for(epochs=3).reference, 3, progress=False
    )
    assert len(results) == 3
    assert all(result.rotation_degrees == 0.0 for result in results)
    assert len({result.error_rate for result in results}) > 1


def test_seed_spread_reports_against_the_binomial_floor(pieces) -> None:
    """The floor is what scoring a finite test set imposes. A spread at that
    level means the measurement is at its resolution limit, and differences of
    that size across the rotation grid are not signal."""
    train, test, model, transform = pieces
    results = repeat_seeds(
        0.0, train, test, model, transform, config_for(epochs=2).reference, 3, progress=False
    )
    spread = seed_spread(results)
    assert set(spread) == {"mean", "std", "range", "binomial_standard_error", "n_repeats"}
    assert spread["n_repeats"] == 3
    assert spread["range"] >= 0.0
    assert spread["binomial_standard_error"] > 0.0


def test_a_single_repeat_has_no_spread() -> None:
    spread = seed_spread((synthetic_reference({0.0: 0.05}).results[0],))
    assert spread["std"] == 0.0
    assert spread["range"] == 0.0


# =========================================================================== #
# 5. lookup and interpolation
# =========================================================================== #


def synthetic_reference(points: dict[float, float]) -> Reference:
    return Reference(
        results=tuple(
            ReferenceResult(
                rotation_degrees=rotation,
                error_rate=error,
                selected_epoch=1,
                epochs=(EpochRecord(0, 1.0, 0.5, 0.5), EpochRecord(1, 0.5, 0.4, 0.4)),
                n_train=100,
                n_validation=10,
                seconds=0.1,
            )
            for rotation, error in points.items()
        ),
        init_strategy="shared_seed",
        selection="validation",
    )


def test_lookup_at_a_grid_point_is_exact() -> None:
    reference = synthetic_reference({0.0: 0.05, 5.0: 0.06})
    assert reference.at(0.0) == pytest.approx(0.05)
    assert reference.at(5.0) == pytest.approx(0.06)


def test_lookup_between_grid_points_interpolates() -> None:
    """Nearest-neighbour would put a sawtooth into the gap curve at the same
    scale as the effect being measured."""
    reference = synthetic_reference({0.0: 0.04, 10.0: 0.06})
    assert reference.at(2.5) == pytest.approx(0.045)
    assert reference.at(5.0) == pytest.approx(0.05)


def test_lookup_outside_the_grid_raises_rather_than_extrapolating() -> None:
    """A gap against an extrapolated reference is not a measurement."""
    reference = synthetic_reference({0.0: 0.05, 10.0: 0.06})
    with pytest.raises(ReferenceError, match="outside the reference grid"):
        reference.at(20.0)
    with pytest.raises(ReferenceError, match="outside the reference grid"):
        reference.at(-5.0)


def test_a_single_point_grid_still_answers_at_that_point() -> None:
    assert synthetic_reference({0.0: 0.05}).at(0.0) == pytest.approx(0.05)


def test_symmetry_is_measured_not_assumed() -> None:
    """The full grid is trained precisely so e*(-phi) == e*(+phi) need not be
    assumed. Reported as the worst mismatch across the grid."""
    reference = synthetic_reference({-10.0: 0.052, 0.0: 0.05, 10.0: 0.055})
    assert reference.symmetry_error() == pytest.approx(0.003)


def test_a_one_sided_grid_has_no_symmetry_to_report() -> None:
    assert synthetic_reference({0.0: 0.05, 10.0: 0.06}).symmetry_error() is None


def test_the_summary_flags_a_level_that_hit_its_budget() -> None:
    unconverged = Reference(
        results=(
            ReferenceResult(
                rotation_degrees=0.0,
                error_rate=0.05,
                selected_epoch=1,
                epochs=(EpochRecord(0, 1.0, 0.5, 0.5), EpochRecord(1, 0.5, 0.4, 0.4)),
                n_train=100,
                n_validation=10,
                seconds=0.1,
            ),
        ),
        init_strategy="shared_seed",
        selection="validation",
    )
    assert unconverged.summary()["all_converged"] is False


# =========================================================================== #
# 5. caching
# =========================================================================== #


def test_a_reference_round_trips_through_the_cache(tmp_path) -> None:
    original = synthetic_reference({0.0: 0.05, 5.0: 0.055})
    path = save(original, tmp_path / "e_star.json")
    reloaded = load(path)

    assert reloaded.rotations == original.rotations
    assert reloaded.at(2.5) == pytest.approx(original.at(2.5))
    assert reloaded.init_strategy == original.init_strategy
    assert reloaded.results[0].epochs == original.results[0].epochs


def test_each_variant_caches_to_its_own_file(tmp_path) -> None:
    """So building several and comparing them never overwrites anything."""
    names = {
        cache_path(tmp_path, strategy, selection).name
        for strategy in ("shared_seed", "independent_seeds", "warm_start")
        for selection in ("validation", "fixed_budget")
    }
    assert len(names) == 6


def test_a_missing_cache_points_at_the_script(tmp_path) -> None:
    with pytest.raises(ReferenceError, match="scripts/train_reference.py"):
        load(tmp_path / "absent.json")


def test_the_cache_records_how_it_was_built(tmp_path) -> None:
    """A reference that cannot say which strategy produced it is not traceable."""
    reference = synthetic_reference({0.0: 0.05})
    payload = json.loads(save(reference, tmp_path / "e.json").read_text(encoding="utf-8"))
    assert payload["init_strategy"] == "shared_seed"
    assert payload["selection"] == "validation"
    assert "results" in payload


# =========================================================================== #
# 6. the cached asset itself
# =========================================================================== #


@pytest.fixture(scope="session")
def cached_reference():
    from dekf_bench.utils.config import default_configs_dir

    path = cache_path(default_configs_dir().parent / "data", "shared_seed", "validation")
    if not path.is_file():
        pytest.skip("no cached reference; run scripts/train_reference.py once")
    return load(path)


@pytest.mark.needs_data
def test_the_cached_reference_covers_the_configured_grid(cached_reference) -> None:
    expected = load_config("x1_stationary").reference.rotations
    assert list(cached_reference.rotations) == expected


@pytest.mark.needs_data
def test_the_cached_error_is_plausible_for_this_architecture(cached_reference) -> None:
    """A 196-14-10 MLP on MNIST: a few percent. Well outside that in either
    direction means something is wrong -- a broken transform, or a leak."""
    for result in cached_reference.results:
        assert 0.01 < result.error_rate < 0.15


@pytest.mark.needs_data
def test_the_cached_reference_answers_every_rotation_a_schedule_visits(
    cached_reference,
) -> None:
    from dekf_bench.env.drift import build_drift

    for experiment, overrides in (
        ("x1_stationary", None),
        ("x2_rotating", None),
        ("x5_abrupt_shift", None),
        ("x2_rotating", {"include": {"env": "mnist_rotating_sinusoidal"}}),
    ):
        config = load_config(experiment, overrides=overrides)
        drift = build_drift(config)
        for step in range(0, config.run.horizon, 97):
            cached_reference.at(drift.rotation_at(step))  # must not raise
