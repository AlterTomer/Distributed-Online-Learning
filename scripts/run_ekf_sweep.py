r"""X13 -- tuning the centralised EKF, under drift and against a stationary twin.

Run this file directly.

    python scripts/run_ekf_sweep.py            # pilot: 2 seeds, drift only
    python scripts/run_ekf_sweep.py --full     # survivors at 5 seeds, + control
    python scripts/run_ekf_sweep.py --fresh    # discard and redo

## Why the drifting condition is the one that tunes

Tuning on stationary data would pick $\lambda\to1$ and $Q\to0$, because with
nothing moving there is nothing to forget and any inflation is pure variance.
That is the *wrong* answer for a tracking filter, and it would be arrived at
honestly. So the grid runs under drift, and the stationary twin runs only for
the settings that survive -- as a control on "does it also win when nothing
moves?", not as the thing being optimised.

**The rate is $\alpha=0.025$ deg/step, and it is the only one available at full
horizon.** X12 measured damage against a paired stationary control at four
speeds; a constant rate hits the 45-degree cap at $T=45/\alpha$, so only the
slowest speed reaches $T=1500$:

    alpha 0.025 -> T 1500   ATC damage 0.0126   frozen 0.2859
    alpha 0.050 -> T  900   ATC damage 0.0196   frozen 0.2922
    alpha 0.100 -> T  450   ATC damage 0.0257   frozen 0.0943
    alpha 0.150 -> T  300   ATC damage 0.0303   frozen 0.0303

0.0126 is about 2.5x the seed-noise floor, so the damage is real, and the frozen
baseline's 0.2859 confirms the displacement is large -- it is adaptation that
keeps ATC's number down, which is exactly the thing the filter is meant to do
better. The shorter horizons would also make forgetting hard to distinguish from
not having converged yet.

(At alpha 0.15 the frozen baseline's damage *equals* ATC's, because T=300 and
`freeze_after` is 300: it never froze. A row that agrees when it must is worth
more than one that merely looks plausible.)

## What a cell contains

One EKF learner and nothing else. The baselines run once, separately, in
`x13_baselines`. Given the same seed the environment and $\bm\theta_0$ do not
depend on the learners list, so every cell is paired with the baselines by
construction rather than by seed -- and the baselines are not recomputed eighty
times.

**Erdos-Renyi(0.3), not X12's ring.** The centralized filter pools every agent's
batch and transmits nothing, so its trajectory is *identical* on any topology;
the choice only reaches the baselines, and there the standing standard applies
(design note D52).

## Divergence is a result, not a crash

sigma_0^2 is a trust region and the grid deliberately brackets its edge, so some
corners will diverge (design note D61). A diverged cell is recorded as diverged
and the sweep continues; killing the run would lose the seventy cells that were
fine.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from dekf_bench.env.environment import build_environment  # noqa: E402
from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: E402
from dekf_bench.learners.base import LearnerError  # noqa: E402
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
#: The X12 speed that reaches the full horizon. See the module docstring.
ALPHA = 0.025
HORIZON = 1500

#: Two seeds to find the stable region, five to measure inside it. The pilot is
#: not a cheaper version of the answer -- it is a different question ("does this
#: cell run at all, and roughly where is the optimum?"), and two seeds answer it.
PILOT_SEEDS = [0, 1]
FULL_SEEDS = [0, 1, 2, 3, 4]

#: The pilot evaluates coarsely and the full pass finely, because they are asked
#: different questions. Measured on one cell at T=1500: eval_every=5 costs 1.57x
#: (425s per seed against 271s) and moves the settled error by 0.0001 -- 0.0840
#: against 0.0841. Tuning reads the settled error, so it gets the cheap cadence;
#: break and recovery analysis needs the time resolution and only the survivors
#: pay for it.
PILOT_EVAL_EVERY = 25
FULL_EVAL_EVERY = 5

#: sigma_0^2. Bounded above at 1e-1: past that the Gauss-Newton step overshoots
#: the linearisation on step one and the run diverges rather than converging
#: slowly (design note D61).
PRIOR_SCALES = [3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1]

#: gamma^1500 is what matters, not gamma: 0.9999 -> 0.86, 0.9995 -> 0.47,
#: 0.999 -> 0.22. Below that the mean is pulled to the origin faster than the
#: data can move it, so the usable range is narrow and close to 1.
GAMMAS = [1.0, 0.9999, 0.9995, 0.999]
PROCESS_NOISES = [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]

#: Effective memory is ~1/(1-lambda) steps. Measured: 0.99 (100 steps) inflates
#: without bound -- trace/p reaches 898 -- because forgetting outruns the
#: information arriving at n=4. 0.995 is the edge and is included to bracket it.
LAMBDAS = [0.9999, 0.999, 0.997, 0.995]

BASELINES = ["centralized_sgd", "diffusion_sgd_atc", "frozen_atc"]

#: How many of the best cells get a stationary twin. The control answers "does
#: this setting also win when nothing moves?", which is a question about the
#: setting being *chosen* -- running it for all fifty-odd survivors would cost
#: about as much as the drift pass itself to answer it for cells nobody will use.
CONTROLS_FOR_BEST = 8
DEVICE = "auto"
DTYPE = "float64"
FRESH = False

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x13_status.json"


# ---------------------------------------------------------------------------


def cells() -> list[dict]:
    r"""Every grid point, most-informative-first.

    Ordered so that a sweep stopped early has still answered something. The
    interior comes before the edges: a cell at $\sigma_0^2=10^{-1}$ with
    $\lambda=0.995$ is two extremes at once, and learning that it diverges is
    worth less than learning where the optimum sits.
    """
    grid = [
        {
            "name": "centralized_ekf_gamma",
            "transition": "scalar",
            "gamma": gamma,
            "process_noise_q": noise,
            "lambda_forget": 1.0,
            "prior_scale": prior,
        }
        for prior, gamma, noise in product(PRIOR_SCALES, GAMMAS, PROCESS_NOISES)
    ] + [
        {
            "name": "centralized_ekf_lambda",
            "transition": "identity",
            "gamma": 1.0,
            "process_noise_q": 0.0,
            "lambda_forget": lam,
            "prior_scale": prior,
        }
        for prior, lam in product(PRIOR_SCALES, LAMBDAS)
    ]

    def distance_from_centre(cell: dict) -> tuple[float, float]:
        """How extreme a cell is, on each axis it actually uses."""
        prior = PRIOR_SCALES.index(cell["prior_scale"])
        prior_extremity = abs(prior - (len(PRIOR_SCALES) - 1) / 2)
        if cell["name"].endswith("gamma"):
            other = GAMMAS.index(cell["gamma"]) + PROCESS_NOISES.index(cell["process_noise_q"])
            span = len(GAMMAS) + len(PROCESS_NOISES) - 2
        else:
            other = LAMBDAS.index(cell["lambda_forget"])
            span = len(LAMBDAS) - 1
        return prior_extremity, abs(other - span / 2)

    return sorted(grid, key=distance_from_centre)


def cell_name(cell: dict) -> str:
    """A directory name that can be read back into the settings it came from."""
    def tag(value: float) -> str:
        return f"{value:g}".replace(".", "p").replace("-", "m")

    if cell["name"].endswith("gamma"):
        return f"x13_g_s{tag(cell['prior_scale'])}_g{tag(cell['gamma'])}_q{tag(cell['process_noise_q'])}"
    return f"x13_l_s{tag(cell['prior_scale'])}_l{tag(cell['lambda_forget'])}"


def config_for(
    learners: list[dict],
    name: str,
    seeds: list[int],
    *,
    drifting: bool,
    eval_every: int = PILOT_EVAL_EVERY,
):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": name,
                "horizon": HORIZON,
                "eval_every": eval_every,
                "seeds": seeds,
                "device": DEVICE,
                "dtype": DTYPE,
            },
            "env": {
                "drift": {
                    "schedule": "linear" if drifting else "stationary",
                    "total_degrees": ALPHA * HORIZON if drifting else 0.0,
                }
            },
            "learners": learners,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def run_one(config, train, test, fresh: bool) -> str:
    """One config over its seeds. Returns 'ok', 'cached', or a divergence note.

    **A diverged cell is poisoned rather than left half-written.** The recorder
    checkpoints as it goes, so a run that dies at step 300 leaves rows on disk;
    a later pass would resume the *recorder* at step 301 while the filter starts
    again from $\\bm\\theta_0$, and report a clean 'ok' for a run whose first 300
    steps came from a different belief. Deleting the partial output and leaving
    a marker makes that impossible, and keeps the verdict stable across passes.
    """
    import shutil

    out_dir = ROOT / config.run.out_dir / config.run.name
    if fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    if (out_dir / "_complete").exists():
        return "cached"
    if (out_dir / "_diverged").exists():
        return (out_dir / "_diverged").read_text(encoding="utf-8").strip()
    rec.write_metadata(out_dir, config, {"experiment": config.run.name})

    for seed in config.run.seeds:
        environment = build_environment(config, seed, train)
        model = build_model_from_config(config)
        likelihood = Categorical(config.model.output_dim)
        learners = build_learners(config, model, likelihood)
        theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
        theta0 = theta0.to(environment.train.images.device)
        sha, _dirty = git_revision(ROOT)
        recorder = rec.Recorder(
            out_dir,
            RunContext.from_config(
                config, seed, environment.graph, run_id=rec.new_run_id(), git_sha=sha
            ),
            config,
        )
        try:
            simulate.run(
                config, environment, learners,
                build_evalsets(config, environment, test), likelihood, theta0,
                recorder=recorder, verify_observations=False, progress_every=0,
            )
        except LearnerError as failure:
            # A diverged cell is a measurement (design note D61). Recording it and
            # continuing keeps the other seventy-nine.
            note = f"diverged (seed {seed}): {failure}"
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "_diverged").write_text(note, encoding="utf-8")
            return note
        recorder.finalize()

    (out_dir / "_complete").write_text("", encoding="utf-8")
    return "ok"


def rank_cells(grid: list[dict], suffix: str) -> list[tuple[float, dict]]:
    """Cells by settled error, best first.

    Settled means the last fifth of the run, which is what tuning is about: a
    filter that converges fast and then tracks badly is not the one to carry into
    the diffusion version, and the full curve would let the early advantage hide
    the late failure. Cells with no readable output sort last rather than being
    dropped, so a ranking is always returned.
    """
    import pandas as pd

    scored = []
    for cell in grid:
        directory = ROOT / "results" / f"{cell_name(cell)}{suffix}"
        files = sorted(directory.glob("*.parquet"))
        if not files:
            scored.append((float("inf"), cell))
            continue
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        rows = frame[
            (frame["metric"] == "error_rate")
            & (frame["evalset"] == "current")
            & (frame["t"] >= int(0.8 * HORIZON))
        ]
        scored.append((float(rows["value"].mean()) if len(rows) else float("inf"), cell))
    return sorted(scored, key=lambda pair: pair[0])


def load_status() -> dict:
    if STATUS.exists():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(full: bool = False, fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)

    seeds = FULL_SEEDS if full else PILOT_SEEDS
    eval_every = FULL_EVAL_EVERY if full else PILOT_EVAL_EVERY
    status = load_status()
    grid = cells()
    if full:
        # Only what survived the pilot. A cell that diverged at two seeds will
        # not stop diverging at five.
        survivors = {name for name, note in status.items() if note in ("ok", "cached")}
        grid = [cell for cell in grid if cell_name(cell) in survivors]
        print(f"X13 full: {len(grid)} survivors of {len(cells())} at {len(seeds)} seeds")
    else:
        print(f"X13 pilot: {len(cells())} cells at {len(seeds)} seeds")
    print(f"alpha {ALPHA} deg/step, T={HORIZON}, ~{HORIZON * ALPHA:.1f} degrees total")
    print(f"device {DEVICE}, {DTYPE}, eval_every {eval_every}\n")

    started = time.time()

    suffix = "_full" if full else ""
    print("baselines (once; every cell is paired with these by construction)", flush=True)
    note = run_one(
        config_for(
            [{"name": n} for n in BASELINES], f"x13_baselines{suffix}", seeds,
            drifting=True, eval_every=eval_every,
        ),
        train, test, fresh,
    )
    print(f"    {note}, {(time.time() - started) / 60:.1f} min\n", flush=True)

    for index, cell in enumerate(grid, start=1):
        name = f"{cell_name(cell)}{suffix}"
        note = run_one(
            config_for([cell], name, seeds, drifting=True, eval_every=eval_every),
            train, test, fresh,
        )
        status[name] = note
        save_status(status)
        elapsed = (time.time() - started) / 60
        remaining = elapsed / index * (len(grid) - index)
        print(
            f"[{index}/{len(grid)}] {name:<34} {note:<28} "
            f"{elapsed:.0f} min, ~{remaining:.0f} left",
            flush=True,
        )

    if full:
        best = rank_cells(grid, suffix)[:CONTROLS_FOR_BEST]
        print(f"\nstationary controls for the best {len(best)} of {len(grid)}", flush=True)
        for index, (score, cell) in enumerate(best, start=1):
            print(f"    {cell_name(cell)} settled at {score:.4f}", flush=True)
            name = f"{cell_name(cell)}_control"
            note = run_one(
                config_for([cell], name, seeds, drifting=False, eval_every=eval_every),
                train, test, fresh,
            )
            status[name] = note
            save_status(status)
            print(f"[{index}/{len(best)}] {name:<42} {note}", flush=True)
        run_one(
            config_for(
                [{"name": n} for n in BASELINES], "x13_baselines_control", seeds,
                drifting=False, eval_every=eval_every,
            ),
            train, test, fresh,
        )

    diverged = sum(1 for note in status.values() if note.startswith("diverged"))
    print(f"\nX13 complete in {(time.time() - started) / 60:.1f} min")
    print(f"{len(status) - diverged} ran, {diverged} diverged")
    print(f"status: {STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(full="--full" in sys.argv, fresh="--fresh" in sys.argv))
