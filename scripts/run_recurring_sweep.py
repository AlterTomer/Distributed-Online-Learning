r"""X11 — recovery under *repeated* abrupt shifts.

Run this file directly.

    python scripts/run_recurring_sweep.py
    python scripts/run_recurring_sweep.py --fresh

X5 measures the transient after **one** 15-degree jump (F7). This repeats the
shift every ``jump_every`` steps, so the learner never gets to settle and the
run measures recovery over and over rather than once.

**Why this regime and not another.** A gradient method recovers from an abrupt
shift only as fast as its step size allows, and that step size was tuned for the
stationary regime -- so shifts arriving faster than the recovery time should
compound into a standing error. A filter can in principle respond faster,
because its covariance says how much to trust new evidence rather than applying
a fixed gain. That makes this the regime where Diff-EKF should show a real
advantage rather than a marginal one, and it is why the SGD numbers are worth
having *first*: they are the baseline the filter has to beat, and measuring them
after the filter exists would invite tuning one against the other.

## The grid, and why it is two-dimensional

$t' \in \{25, 50, 100, 200\}$ steps between shifts, crossed with $J \in \{5, 15,
30\}$ degrees per shift. Twelve cells, five seeds.

$J / t'$ is the *average* speed, so the two axes are not independent -- and that
is exactly the point. The same average speed can arrive as rare-large or
frequent-small shifts, and those are different problems: one asks whether the
learner can recover at all before the next shift, the other whether many small
perturbations accumulate. A one-dimensional sweep along $J/t'$ could not tell
them apart, and the diagonal of this grid is where that comparison lives.

Cells with matched $J/t'$: $(t'{=}50, J{=}15)$ and $(t'{=}100, J{=}30)$ both
average 0.30 deg/step; $(t'{=}25, J{=}5)$ and $(t'{=}100, J{=}30)$ bracket it.
Those pairs are the ones to read against each other.

## What the schedule does

Every $t'$ steps the rotation jumps by exactly $J$ degrees in an unpredictable
direction, reflected at the 45-degree well-posedness cap so that the magnitude
is *always* exactly $J$ -- clipping at the edge would shorten those jumps and
quietly make the transients incomparable. Between jumps nothing moves, so each
transient is attributable to one event rather than to a trend.

`frozen_atc` rides along so the comparative break is available here too: under
frequent shifts it is genuinely unclear whether continuing to adapt beats having
stopped, which is the question that definition exists to answer.
"""

from __future__ import annotations

import itertools
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
JUMP_EVERY = [25, 50, 100, 200]
JUMP_DEGREES = [5.0, 15.0, 30.0]
HORIZON = 1500

#: Held fixed across cells so the shift *pattern* is one less thing varying
#: between them. The data still varies with the run seed.
JUMP_SEED = 0

LEARNERS = [
    "centralized_sgd",
    "diffusion_sgd_atc",
    "diffusion_sgd_atc_plain",
    "local_only",
    "frozen_atc",
]
SEEDS = [0, 1, 2, 3, 4]
FRESH = False

#: Denser than the default 25 because a transient is only a few tens of steps
#: wide, and at t'=25 the default cadence would sample roughly once per shift --
#: which measures the standing error and misses the recovery entirely.
EVAL_EVERY = 5

DATA_ROOT = ROOT / "data"


def cell_name(jump_every: int, jump_degrees: float) -> str:
    return f"x11_every{jump_every}_jump{jump_degrees:g}"


def config_for(jump_every: int, jump_degrees: float):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": cell_name(jump_every, jump_degrees),
                "horizon": HORIZON,
                "eval_every": EVAL_EVERY,
                "seeds": SEEDS,
            },
            "env": {
                "drift": {
                    "schedule": "recurring",
                    "jump_every": jump_every,
                    "jump_degrees": jump_degrees,
                    "jump_seed": JUMP_SEED,
                }
            },
            "learners": [{"name": name} for name in LEARNERS],
            # Current-only: the question is recovery, and a canonical set would
            # score a rotation the run has long since left.
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

    cells = list(itertools.product(JUMP_EVERY, JUMP_DEGREES))
    print(f"X11: {len(cells)} cells x {len(SEEDS)} seeds, T={HORIZON}\n")

    started = time.time()
    for index, (jump_every, jump_degrees) in enumerate(cells, start=1):
        config = config_for(jump_every, jump_degrees)
        schedule_speed = jump_degrees / jump_every
        print(
            f"[{index}/{len(cells)}] {config.run.name}  "
            f"{jump_degrees:g} deg every {jump_every} steps "
            f"(avg {schedule_speed:.3f} deg/step)",
            flush=True,
        )
        run_one(config, train, test, fresh)
        print(f"    done, {(time.time() - started) / 60:.1f} min elapsed", flush=True)

    print(f"\nX11 complete in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv))
