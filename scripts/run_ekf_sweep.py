r"""X13 -- tuning the centralised EKF, under drift and against a stationary twin.

Run this file directly.

    python scripts/run_ekf_sweep.py             # pilot: 80 cells, 2 seeds  (~12h)
    python scripts/run_ekf_sweep.py --baselines # re-tune SGD lr under drift (~2h)
    python scripts/run_ekf_sweep.py --full      # refined grid, 5 seeds     (~32h)
    python scripts/run_ekf_sweep.py --fresh     # discard and redo

`--full` requires `--baselines` to have run: it refuses to start otherwise,
rather than silently comparing a drift-tuned filter against a baseline that was
tuned on stationary data.

## What the pilot found, and how the refined grid answers it

80 cells at 2 seeds, none diverged, seed noise 0.0021. Best cell 0.0620 against
`centralized_sgd` 0.0948 and `diffusion_sgd_atc` 0.0970 — but with the baselines
still on their stationary-tuned `lr 0.01`, so that gap is provisional.

The axes turned out badly proportioned. Best-per-level, span in the last column:

    Q          1e-6 0.0825   1e-5 0.0679   1e-4 0.0620   1e-3 0.0852    0.0232
    lambda   0.9999 0.0911   .999 0.0815   .997 0.0694   .995 0.0698    0.0217
    gamma         1 0.0650  .9999 0.0644  .9995 0.0626   .999 0.0620    0.0030
    sigma_0^2 0.003 0.0626   0.01 0.0635   0.03 0.0629    0.1 0.0620    0.0016

64 of the 80 cells went to the gamma family, where gamma moves the result by
0.0030 and sigma_0^2 by less than the noise, while Q — worth 0.0232 — got four
points three decades apart. Every one of the seven cells statistically tied with
the best has Q = 1e-4.

So the refined grid puts five Q values inside one decade, gives the lambda family
its own downward-extended prior axis (its pilot optimum was a tie between the two
lowest values tested, with sigma_0^2 at its edge — it lost while less well
tuned), and cuts gamma to three points and sigma_0^2 to two.

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

# --- the pilot grid, as it ran -------------------------------------------- #
#
# Kept verbatim so the pilot's results stay readable. What it found, at a seed
# noise of 0.0021 (median |seed0 - seed1| over 80 cells):
#
#   Q          1e-6 0.0825  1e-5 0.0679  1e-4 0.0620  1e-3 0.0852   span 0.0232
#   lambda   0.9999 0.0911  .999 0.0815  .997 0.0694  .995 0.0698   span 0.0217
#   gamma         1 0.0650  .9999 0.0644 .9995 0.0626 .999 0.0620   span 0.0030
#   sigma_0^2 0.003 0.0626  0.01 0.0635  0.03 0.0629  0.1  0.0620   span 0.0016
#
# So Q and lambda carry the result, gamma is marginal, and sigma_0^2 does not
# move it at all within [0.003, 0.1]. All 80 cells ran; none diverged.
PRIOR_SCALES = [3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1]
GAMMAS = [1.0, 0.9999, 0.9995, 0.999]
PROCESS_NOISES = [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]
LAMBDAS = [0.9999, 0.999, 0.997, 0.995]

# --- the refined grid, proportioned by what the pilot measured ------------- #
#
# The pilot spent 64 of 80 cells on the gamma family, where gamma is worth 0.0030
# and sigma_0^2 is worth nothing, while Q -- worth 0.0232 -- got four points
# three decades apart. This grid spends the budget the other way round.
#
#: Five points inside one decade, bracketing the pilot's optimum at 1e-4 on both
#: sides. The pilot's V is steep to the right (1e-4 -> 1e-3 costs 0.023) and
#: shallow to the left, so the true optimum is somewhere in [5e-5, 3e-4].
FULL_PROCESS_NOISES = [3.0e-5, 6.0e-5, 1.0e-4, 2.0e-4, 3.0e-4]

#: Two values, not four. Flat across 16 pilot cells, and kept plural only so the
#: 5-seed pass can confirm the flatness rather than inherit it.
FULL_PRIOR_SCALES = [1.0e-2, 1.0e-1]

#: gamma = 1 is the random walk and stays in as the reference model even though
#: the pilot put it slightly behind. The optimum sits between 0.9995 and 0.999,
#: where the curve flattens, so three points bracket it. Marginal but probably
#: real: at Q=1e-4, gamma=0.999 beat gamma=1 in all four sigma_0^2 blocks, and
#: the margin grew with sigma_0^2 -- shrinking the mean offsets a looser prior.
FULL_GAMMAS = [1.0, 0.9995, 0.999]

#: The lambda family gets a fair bracket. Its pilot optimum was a tie between
#: the two LOWEST values tested (0.997 at 0.0694, 0.995 at 0.0698) with
#: sigma_0^2 pinned at its low edge -- so it lost to the gamma family while
#: less well tuned, which is not a comparison worth reporting. 0.99 is known to
#: inflate without bound, so 0.993 is the floor.
FULL_LAMBDAS = [0.998, 0.997, 0.996, 0.995, 0.993]

#: Extended DOWNWARD, and for this family only. sigma_0^2 is flat for the gamma
#: family but not here: at lambda=0.997 the error runs 0.0694 -> 0.0751 as
#: sigma_0^2 goes 0.003 -> 0.1, monotonically preferring small. That is
#: mechanistic -- P <- P/lambda preserves scale, so P_0 never washes out, while
#: additive Q erases it -- and it means the pilot's low edge was a real edge.
FULL_LAMBDA_PRIOR_SCALES = [1.0e-3, 3.0e-3, 1.0e-2]

BASELINES = ["centralized_sgd", "diffusion_sgd_atc", "frozen_atc"]

#: Learning rates for the baseline re-tune. `lr 0.01` was selected on 2026-08-05
#: against STATIONARY data (X1); under drift a larger step may track better, and
#: comparing a drift-tuned filter with a stationary-tuned baseline is exactly the
#: mistake design note D39 exists to prevent. X3 re-tunes per topology and X4 per
#: cell, so re-tuning per drift condition is the project's own convention.
BASELINE_LRS = [0.01, 0.02, 0.05, 0.1, 0.2]

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


def grid_levels(full: bool) -> tuple[list, list, list, list, list]:
    """The four axes plus the lambda family's own prior axis, per pass."""
    if full:
        return (
            FULL_PRIOR_SCALES, FULL_GAMMAS, FULL_PROCESS_NOISES,
            FULL_LAMBDAS, FULL_LAMBDA_PRIOR_SCALES,
        )
    return PRIOR_SCALES, GAMMAS, PROCESS_NOISES, LAMBDAS, PRIOR_SCALES


def cells(full: bool = False) -> list[dict]:
    r"""Every grid point, most-informative-first.

    Ordered so that a sweep stopped early has still answered something. The
    interior comes before the edges: a cell at $\sigma_0^2=10^{-1}$ with
    $\lambda=0.995$ is two extremes at once, and learning that it diverges is
    worth less than learning where the optimum sits.

    **The two families no longer share a prior axis under ``full``.** The pilot
    measured $\sigma_0^2$ as flat for the $\gamma$ family and monotone for the
    $\lambda$ family, so giving them one shared list would either waste cells on
    a flat axis or leave the $\lambda$ optimum at an edge again.
    """
    priors, gammas, noises, lambdas, lambda_priors = grid_levels(full)
    grid = [
        {
            "name": "centralized_ekf_gamma",
            "transition": "scalar",
            "gamma": gamma,
            "process_noise_q": noise,
            "lambda_forget": 1.0,
            "prior_scale": prior,
        }
        for prior, gamma, noise in product(priors, gammas, noises)
    ] + [
        {
            "name": "centralized_ekf_lambda",
            "transition": "identity",
            "gamma": 1.0,
            "process_noise_q": 0.0,
            "lambda_forget": lam,
            "prior_scale": prior,
        }
        for prior, lam in product(lambda_priors, lambdas)
    ]

    def distance_from_centre(cell: dict) -> tuple[float, float]:
        """How extreme a cell is, on each axis it actually uses."""
        if cell["name"].endswith("gamma"):
            prior_extremity = abs(priors.index(cell["prior_scale"]) - (len(priors) - 1) / 2)
            other = gammas.index(cell["gamma"]) + noises.index(cell["process_noise_q"])
            span = len(gammas) + len(noises) - 2
        else:
            prior_extremity = abs(
                lambda_priors.index(cell["prior_scale"]) - (len(lambda_priors) - 1) / 2
            )
            other = lambdas.index(cell["lambda_forget"])
            span = len(lambdas) - 1
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


def baseline_name(lr: float) -> str:
    return f"x13_lr{f'{lr:g}'.replace('.', 'p')}"


def settled_for(directory: str, learner: str) -> float:
    """One learner's settled error in one run directory, or +inf if absent."""
    import pandas as pd

    files = sorted((ROOT / "results" / directory).glob("*.parquet"))
    if not files:
        return float("inf")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    rows = frame[
        (frame["learner"] == learner)
        & (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
        & (frame["t"] >= int(0.8 * HORIZON))
    ]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def best_baseline_lr() -> float | None:
    r"""The learning rate the re-tune selected, by ATC's settled error.

    Chosen on **ATC** rather than on the centralized learner because ATC is the
    distributed baseline the filter's claim is actually against, and because a
    single lr has to serve both -- the two agreed on a cell when they were last
    tuned together (2026-08-05), so one number is the convention here.

    Returns ``None`` when the re-tune has not been run, which the caller reports
    rather than papering over with the shipped default.
    """
    scored = [
        (settled_for(baseline_name(lr), "diffusion_sgd_atc"), lr)
        for lr in BASELINE_LRS
        if (ROOT / "results" / baseline_name(lr)).exists()
    ]
    scored = [(value, lr) for value, lr in scored if value != float("inf")]
    return min(scored)[1] if scored else None


def tune_baselines(train, test, fresh: bool) -> int:
    r"""Give the SGD baselines their best shot at *this* drift condition.

    `lr 0.01` was selected on 2026-08-05 against stationary data. Reporting a
    filter that was tuned under drift against a baseline that was not is the
    mistake design note D39 exists to prevent, and the gap here is large enough
    (0.033) that it deserves a baseline which had every chance.

    Two seeds and the coarse cadence, because this asks the same question the
    pilot did -- where is the optimum, roughly -- and the winner is then re-run
    at five seeds alongside the filter.
    """
    status = load_status()
    print(f"X13 baseline re-tune: {len(BASELINE_LRS)} learning rates at "
          f"{len(PILOT_SEEDS)} seeds")
    print(f"alpha {ALPHA} deg/step, T={HORIZON}, eval_every {PILOT_EVAL_EVERY}\n")

    started = time.time()
    for index, lr in enumerate(BASELINE_LRS, start=1):
        # frozen_atc carries ATC's learning rate too. It froze at the same point
        # having taken the same steps, so a different lr would make it a
        # different algorithm rather than the same one stopped -- which is the
        # whole basis of the comparative break.
        learners = [
            {"name": "centralized_sgd", "optimizer": "sgd_momentum", "lr": lr, "momentum": 0.9},
            {"name": "diffusion_sgd_atc", "optimizer": "sgd_momentum", "lr": lr, "momentum": 0.9},
            {"name": "frozen_atc", "optimizer": "sgd_momentum", "lr": lr, "momentum": 0.9,
             "freeze_after": 300},
        ]
        name = baseline_name(lr)
        note = run_one(
            config_for(learners, name, PILOT_SEEDS, drifting=True, eval_every=PILOT_EVAL_EVERY),
            train, test, fresh,
        )
        status[name] = note
        save_status(status)
        elapsed = (time.time() - started) / 60
        print(f"[{index}/{len(BASELINE_LRS)}] lr {lr:<6g} {note:<12} "
              f"{elapsed:.0f} min, ~{elapsed / index * (len(BASELINE_LRS) - index):.0f} left",
              flush=True)

    print(f"\nBaseline re-tune complete in {(time.time() - started) / 60:.1f} min")
    print("Read it with: python scripts/report_ekf_sweep.py --baselines")
    return 0


def main(full: bool = False, fresh: bool = FRESH, baselines: bool = False) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)

    if baselines:
        return tune_baselines(train, test, fresh)

    seeds = FULL_SEEDS if full else PILOT_SEEDS
    eval_every = FULL_EVAL_EVERY if full else PILOT_EVAL_EVERY
    status = load_status()
    grid = cells(full=full)
    if full:
        # NOT the pilot's survivors. Nothing diverged, so survivor-filtering would
        # be a no-op; the refined grid is a different set of points, chosen from
        # what the pilot measured about which axes carry the result.
        print(f"X13 full: {len(grid)} refined cells at {len(seeds)} seeds")
    else:
        print(f"X13 pilot: {len(grid)} cells at {len(seeds)} seeds")
    print(f"alpha {ALPHA} deg/step, T={HORIZON}, ~{HORIZON * ALPHA:.1f} degrees total")
    print(f"device {DEVICE}, {DTYPE}, eval_every {eval_every}\n")

    started = time.time()

    suffix = "_full" if full else ""

    # The full pass runs its baselines at the re-tuned learning rate, so the
    # headline is tuned-against-tuned. The pilot's baselines used the shipped
    # 0.01, which is why its 0.033 gap is provisional.
    baseline_entries = [{"name": name} for name in BASELINES]
    if full:
        lr = best_baseline_lr()
        if lr is None:
            print(
                "The baseline re-tune has not been run, so the full pass would compare a\n"
                "drift-tuned filter against a stationary-tuned baseline (design note D39).\n"
                "  python scripts/run_ekf_sweep.py --baselines\n"
                "Re-run with --full afterwards, or delete this guard if that is intended."
            )
            return 1
        print(f"baselines re-tuned under this drift: lr {lr:g}")
        baseline_entries = [
            {"name": name, "optimizer": "sgd_momentum", "lr": lr, "momentum": 0.9}
            | ({"freeze_after": 300} if name == "frozen_atc" else {})
            for name in BASELINES
        ]

    print("baselines (once; every cell is paired with these by construction)", flush=True)
    note = run_one(
        config_for(
            baseline_entries, f"x13_baselines{suffix}", seeds,
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
                baseline_entries, "x13_baselines_control", seeds,
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
    raise SystemExit(
        main(
            full="--full" in sys.argv,
            fresh="--fresh" in sys.argv,
            baselines="--baselines" in sys.argv,
        )
    )
