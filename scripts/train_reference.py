r"""Train the offline reference classifiers and cache $e^\star$ per rotation.

Run it from the IDE with no arguments. This is a **one-time** asset: every
experiment reads the cache, and nothing rebuilds it automatically, so an
experiment can never silently retrain the thing it is measured against.

``INIT_STRATEGY`` and ``SELECTION`` pick which variant to build. Each
combination caches to its own file, so building several and comparing them costs
nothing but time.

Three modes:

- default: build the grid for ``INIT_STRATEGY`` / ``SELECTION``.
- ``COMPARE = True``: build all three init strategies and print them side by
  side. Roughly three times the runtime.
- ``REPEAT_SEEDS > 0``: retrain *one* rotation that many times, varying only the
  seed, to measure how much of the grid's variation is run-to-run noise. The
  grid has one draw per point and no stated uncertainty, so without this a
  difference between two levels cannot be told from noise by looking at it.
"""

from __future__ import annotations

import time

from dekf_bench.data.mnist import is_cached, load_mnist
from dekf_bench.data.transforms import build_transform
from dekf_bench.evaluation import reference as ref
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.utils.config import default_configs_dir, load_config

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
EXPERIMENT = "x1_stationary"
INIT_STRATEGY = "shared_seed"  # shared_seed | independent_seeds | warm_start
SELECTION = "validation"  # validation | fixed_budget
EPOCHS: int | None = None  # None keeps the config's value
COMPARE = False  # build all three init strategies and compare
REPEAT_SEEDS = 0  # >0: retrain one rotation this many times to measure seed noise
REPEAT_AT_DEGREES = 0.0  # which rotation to repeat
DATA_ROOT = default_configs_dir().parent / "data"


def overrides() -> dict:
    reference: dict = {"init_strategy": INIT_STRATEGY, "selection": SELECTION}
    if SELECTION == "fixed_budget":
        reference["validation_size"] = 0
    if EPOCHS is not None:
        reference["epochs"] = EPOCHS
    return {"reference": reference}


def build(strategy: str) -> ref.Reference:
    config = load_config(EXPERIMENT, overrides=overrides())
    config.reference.init_strategy = strategy

    train, test = load_mnist(DATA_ROOT, download=False)
    model = build_model_from_config(config)
    transform = build_transform(train.images, config.model.input_size)

    settings = config.reference
    print(
        f"\n{strategy} / {settings.selection}: {len(settings.rotations)} rotations, "
        f"{settings.epochs} epochs, {len(train) - settings.validation_size} train "
        f"/ {settings.validation_size} val / {len(test)} test\n"
    )
    started = time.perf_counter()
    reference = ref.train_reference(config, train, test, model, transform)
    path = ref.save(reference, ref.cache_path(DATA_ROOT, strategy, settings.selection))
    print(f"\n  cached to {path.name}   ({time.perf_counter() - started:.0f}s)")
    return reference


def report(reference: ref.Reference) -> None:
    summary = reference.summary()
    print("\n" + "=" * 62)
    print(f"e* summary: {summary['init_strategy']} / {summary['selection']}")
    print("=" * 62)
    print(f"  best  {summary['best_error']:.4f}   worst {summary['worst_error']:.4f}")
    print(f"  every level converged inside its budget: {summary['all_converged']}")
    if summary["symmetry_error"] is not None:
        print(
            f"  |e*(-phi) - e*(+phi)| at most {summary['symmetry_error']:.4f}  "
            "(measured, not assumed)"
        )
    print("\n  interpolation at rotations the schedules actually visit:")
    for rotation in (0.0, 11.25, 22.5, 33.75, 45.0):
        print(f"    {rotation:>6.2f} deg -> e* = {reference.at(rotation):.4f}")


def measure_seed_noise() -> int:
    """Retrain one rotation several times, varying only the seed.

    The grid has one draw per point, so a difference between two rotation levels
    cannot be distinguished from run-to-run noise by looking at the curve. This
    measures the noise directly, and compares it against both the grid's spread
    and the binomial floor a 10 000-image test set imposes.
    """
    config = load_config(EXPERIMENT, overrides=overrides())
    train, test = load_mnist(DATA_ROOT, download=False)
    model = build_model_from_config(config)
    transform = build_transform(train.images, config.model.input_size)

    print(
        f"\nseed noise at {REPEAT_AT_DEGREES} degrees: {REPEAT_SEEDS} runs, "
        "identical except for the seed\n"
    )
    results = ref.repeat_seeds(
        REPEAT_AT_DEGREES, train, test, model, transform, config.reference, REPEAT_SEEDS
    )
    spread = ref.seed_spread(results)

    print("\n" + "=" * 62)
    print(f"e* at {REPEAT_AT_DEGREES} degrees over {spread['n_repeats']} seeds")
    print("=" * 62)
    print(f"  mean            {spread['mean']:.4f}")
    print(f"  std             {spread['std']:.4f}")
    print(f"  range           {spread['range']:.4f}")
    print(f"  binomial floor  {spread['binomial_standard_error']:.4f}   sqrt(e(1-e)/n) on 10k")

    try:
        grid = ref.load(ref.cache_path(DATA_ROOT, INIT_STRATEGY, SELECTION))
    except ref.ReferenceError:
        return 0

    errors = [result.error_rate for result in grid.results]
    grid_range = max(errors) - min(errors)
    print(f"\n  spread across the whole rotation grid: {grid_range:.4f}")
    if grid_range <= 1.5 * spread["range"]:
        print("  within seed noise: rotation has no detectable effect on difficulty here")
    else:
        print("  larger than seed noise: rotation does affect difficulty")
    return 0


def main() -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1

    if REPEAT_SEEDS:
        return measure_seed_noise()

    strategies = ["shared_seed", "independent_seeds", "warm_start"] if COMPARE else [INIT_STRATEGY]
    built = {strategy: build(strategy) for strategy in strategies}

    for reference in built.values():
        report(reference)

    if len(built) > 1:
        print("\n" + "=" * 62)
        print("side by side")
        print("=" * 62)
        header = f"{'rotation':>10}" + "".join(f"{name:>20}" for name in built)
        print(header)
        print("-" * len(header))
        for index, rotation in enumerate(next(iter(built.values())).rotations):
            row = f"{rotation:>10.1f}"
            for reference in built.values():
                row += f"{reference.results[index].error_rate:>20.4f}"
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
