r"""X12 -- smooth drift at constant rate, to bracket X11's abrupt shifts.

Run this file directly.

    python scripts/run_linear_sweep.py
    python scripts/run_linear_sweep.py --fresh

X9's ramp cannot answer "does abrupt cost more than smooth at the same average
speed", because its damage at rate $\alpha$ is measured by a learner that
arrived there through every lower rate: at 0.15 deg/step the ramp is 1447 steps
in and sits 36 degrees from where it started, while an X11 cell at the same
average speed oscillates within the band and keeps returning. Same units,
different quantities.

These are the constant-rate runs D47 said would be needed to confirm a ramp,
and they double as the smooth-drift side of the X11 comparison.

## The horizon is forced, not chosen -- by two ceilings

Rotation is capped at 45 degrees, and a constant rate reaches it at
$T = 45/\alpha$. Separately the shard budget caps $T$ at 1500, since $N n T$
samples must fit in MNIST's 60000. Whichever binds first wins, and the total
travel is then derived so the realised rate is exactly the one asked for:

    alpha 0.025 -> T 1500, 37.5 deg   (shard budget binds)
    alpha 0.050 -> T  900, 45.0 deg   (well-posedness binds)
    alpha 0.100 -> T  450, 45.0 deg
    alpha 0.150 -> T  300, 45.0 deg

This is not a nuisance to work around, it is the asymmetry itself. Smooth
monotone drift **runs out of room**; repeated bounded shifts do not, which is
why an X11 cell can sustain 0.15 deg/step for 1500 steps and a linear run
cannot. Any comparison has to be made over a step window both regimes reach,
which is what `report_linear.py` does.

## Each speed gets its own control

The horizon differs per speed, and `assert_paired_runs` requires the control to
match on horizon -- rightly, since the subtraction is step by step. One control
at the longest horizon would *probably* be reusable, because a stationary run is
deterministic given its seed and the partition does not depend on $T$, but
"probably" is not a basis for a paired measurement.

**Ring, not the new Erdos-Renyi standard.** These exist to be compared against
X11, which is on ring; using the standard here would put topology into the very
difference being measured (design note D52).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from dekf_bench.env.environment import build_environment  # noqa: E402
from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: E402
from dekf_bench.learners.registry import build_learners  # noqa: E402
from dekf_bench.likelihoods.categorical import Categorical  # noqa: E402
from dekf_bench.models.registry import build_model_from_config  # noqa: E402
from dekf_bench.recording import recorder as rec  # noqa: E402
from dekf_bench.recording.schema import RunContext  # noqa: E402
from dekf_bench.runner import simulate  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402
from dekf_bench.utils.determinism import git_revision  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
#: Chosen to land on X11 cells: 0.05 is (100, 5), 0.10 is (50, 5), and 0.15 is
#: *two* cells -- (100, 15) and (200, 30) -- so the frequent-small and
#: rare-large deliveries can both be read against the same smooth run.
SPEEDS = [0.025, 0.05, 0.10, 0.15]

#: The well-posedness cap. A constant rate reaches it at T = 45 / alpha.
MAX_DEGREES = 45.0
#: The *other* ceiling, and the one that bites at low speeds: N*n*T samples must
#: fit in MNIST's 60000, so T <= 1500 at the default N=10, n=4. Slow drift is
#: therefore limited by data rather than by degrees, and asking for T=1800 at
#: alpha=0.025 is refused by the shard-budget check -- which is how this was
#: found, before the run rather than after.
MAX_HORIZON = 1500
EVAL_EVERY = 5
SEEDS = [0, 1, 2, 3, 4]
LEARNERS = [
    "centralized_sgd",
    "diffusion_sgd_atc",
    "diffusion_sgd_atc_plain",
    "local_only",
    "frozen_atc",
]
FRESH = False

DATA_ROOT = ROOT / "data"


def horizon_for(speed: float) -> int:
    """The longest run this speed allows, under whichever ceiling binds first."""
    return min(MAX_HORIZON, int(round(MAX_DEGREES / speed)))


def degrees_for(speed: float) -> float:
    """Total travel, derived so the realised rate is exactly ``speed``.

    Not fixed at 45: when the shard budget caps the horizon before the
    well-posedness cap does, travelling the full 45 degrees would give a faster
    rate than the one the cell is named after.
    """
    return speed * horizon_for(speed)


def names_for(speed: float) -> tuple[str, str]:
    tag = f"{speed:g}".replace(".", "p")
    return f"x12_linear_a{tag}", f"x12_control_a{tag}"


def config_for(speed: float, *, drifting: bool):
    drift_name, control_name = names_for(speed)
    horizon = horizon_for(speed)
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": drift_name if drifting else control_name,
                "horizon": horizon,
                "eval_every": EVAL_EVERY,
                "seeds": SEEDS,
            },
            "env": {
                "drift": {
                    "schedule": "linear" if drifting else "stationary",
                    "total_degrees": degrees_for(speed) if drifting else 0.0,
                }
            },
            "learners": [{"name": name} for name in LEARNERS],
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def run_one(config, train, test, fresh: bool) -> None:
    out_dir = ROOT / config.run.out_dir / config.run.name
    if fresh and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    rec.write_metadata(out_dir, config, {"experiment": config.run.name})

    for seed in config.run.seeds:
        environment = build_environment(config, seed, train)
        model = build_model_from_config(config)
        likelihood = Categorical(config.model.output_dim)
        learners = build_learners(config, model, likelihood)
        theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
        sha, _dirty = git_revision(ROOT)
        recorder = rec.Recorder(
            out_dir,
            RunContext.from_config(
                config, seed, environment.graph, run_id=rec.new_run_id(), git_sha=sha
            ),
            config,
        )
        simulate.run(
            config,
            environment,
            learners,
            build_evalsets(config, environment, test),
            likelihood,
            theta0,
            recorder=recorder,
            verify_observations=False,
            progress_every=0,
        )
        recorder.finalize()


def main(fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)

    print(f"X12: {len(SPEEDS)} speeds x 2 runs x {len(SEEDS)} seeds")
    print("horizon is forced by the 45-degree cap: T = 45 / alpha\n")

    started = time.time()
    for index, speed in enumerate(SPEEDS, start=1):
        horizon = horizon_for(speed)
        for drifting in (True, False):
            config = config_for(speed, drifting=drifting)
            kind = "linear" if drifting else "control"
            print(
                f"[{index}/{len(SPEEDS)}] {config.run.name}  {kind}, "
                f"alpha {speed:g} deg/step, T={horizon}",
                flush=True,
            )
            run_one(config, train, test, fresh)
            print(f"    done, {(time.time() - started) / 60:.1f} min elapsed", flush=True)

    print(f"\nX12 complete in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv))
