r"""X14 -- does the tuned filter hold up where the learners actually break?

Run this file directly.

    python scripts/run_ekf_generalization.py --lr    # re-tune baselines (~0.5h)
    python scripts/run_ekf_generalization.py         # the conditions   (~21h)

Run both **after** `run_ekf_sweep.py --full`, and not alongside it. The `--lr`
pass has no dependency on X13 and would start happily, but it would contend for
the same GPU and slow both; the main pass refuses outright, since the two filter
settings are read from X13's grid rather than chosen here. **One tuned setting per family, tested
across regimes** -- tuning per condition would answer "can the filter be tuned to
win anywhere", which is a weaker and less useful claim than "one setting tracks
across regimes", and it is the latter the diffusion version needs.

## The conditions are chosen by measured damage, not by which sound hardest

X13 tuned at linear $\alpha=0.025$, which turns out to be among the **mildest**
conditions the project has. Damage (drifting minus its paired stationary twin,
over the settled window) across everything already run:

    abrupt every25 jump30   0.0677     linear a=0.15    0.0303  (T=299)
    abrupt every25 jump15   0.0518     linear a=0.10    0.0257  (T=449)
    abrupt every50 jump15   0.0498     linear a=0.05    0.0196  (T=899)
    abrupt every200 jump30  0.0398     linear a=0.025   0.0126  (T=1499)
    abrupt every100 jump30  0.0345
    abrupt every200 jump15  0.0300
    abrupt every100 jump15  0.0183
    abrupt every25 jump5    0.0112
    abrupt every200 jump5   0.0054

So the tuning condition sits 5x below the worst abrupt cell.

**The faster linear rates are not used, and the reason is not that they are
mild.** A constant rate hits the 45-degree cap at $T=45/\alpha$, so $\alpha=0.15$
runs only 299 steps, and its damage is then part drift and part "had less time to
converge" with no way to separate them afterwards. The giveaway is in the table
above: at $\alpha=0.15$ the *frozen* baseline's damage equals ATC's exactly
(0.0303), because `freeze_after` is 300 and the run ends at 299 -- it never
froze. Recurring shifts have neither problem: reflecting at the cap lets them
sustain any rate for the full 1500 steps (design note D51), so every cell here
runs the same horizon and the comparison is clean.

## Everything shares one run and one control

Each condition runs the two filter settings and the three baselines **together**,
so they see one environment and one $\bm\theta_0$ and the comparison is paired by
construction. Every drifting condition is $T=1500$ and differs from the others
only in the drift block, so **one** stationary control serves all of them -- the
same reasoning `x11_control` used for twelve cells.

## The baselines are re-tuned once, on the worst condition

D65 established re-tuning per condition rather than once per project. Doing that
for all seven here would cost more than the experiment. Instead the learning rate
is re-swept on `every25 jump30` -- the most damaging cell, and the one where a
larger step is most likely to pay -- and if the X13 choice survives there it is
kept throughout. A jump of 30 degrees every 25 steps is the strongest case for
fast adaptation the project has; a rate that does not want to move there will not
want to move at 0.0054 damage either.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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
from run_ekf_sweep import BASELINE_LRS, cell_name, cells, settled_for  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
HORIZON = 1500
SEEDS = [0, 1, 2, 3, 4]
LR_SEEDS = [0, 1]

#: Dense, because a transient after a jump is only tens of steps wide. At
#: t'=25 the default cadence of 25 would sample about once per shift and measure
#: the standing error while missing the recovery entirely -- which is the whole
#: quantity this experiment exists to compare.
EVAL_EVERY = 5

#: How rarely the recorder may write, in steps. **Not the eval cadence** -- every
#: eval point is still recorded, and the parquet is byte-identical either way.
#: This only decides how much work a crash costs.
#:
#: A flush rebuilds a frame from every accumulated row and rewrites the whole
#: parquet, so it is O(rows so far) and a run is O(T^2) in flushing. Measured at
#: the real row width: 39 ms at 2k rows against 1573 ms at 52k. At EVAL_EVERY=5
#: over T=3000 that is 600 rewrites of a frame growing past 100k rows -- roughly
#: 15 minutes a seed spent writing. Every 100 steps costs at most 100 steps of
#: lost work on a crash and cuts that by ~20x (design note D69).
MIN_FLUSH_STEPS = 100

#: Samples per agent per step in the default block. The shard budget is
#: $N n T \le 60000$, so this and the horizon trade directly against each other.
SAMPLES = 4

#: (label, drift override, n, horizon). Ordered mildest-first so a run stopped
#: early has still crossed the interesting boundary. Chosen to span the measured
#: damage range 0.0054 -> 0.0677 and both ends of the frequent-small /
#: rare-large axis, which is the contrast X11 was built around.
#:
#: **Two ceilings decide the horizon, and which one binds differs by schedule.**
#: Linear drift reaches the 45-degree well-posedness cap at $T=45/\alpha$; every
#: schedule also obeys the shard budget $Nn T \le 60000$. Recurring shifts
#: reflect at the cap (design note D51), so only the budget binds them.
#:
#:     alpha   45/alpha   T at n=4   binds
#:     0.025       1800       1500   shard budget
#:     0.030       1500       1500   both, exactly
#:     0.050        900        900   45-degree cap
#:     0.150        300        300   45-degree cap
#:
#: So **alpha=0.03 is the fastest linear rate that still runs a full 1500-step
#: horizon**, and above alpha=0.04 the cap binds at every n -- lowering n cannot
#: rescue a fast linear run, because monotone drift runs out of *room* rather
#: than out of data. The faster rates are therefore still excluded: their damage
#: would mix drift with "had less time to converge", exactly as X12 found.
#:
#: The low-n block buys horizon where the budget *is* the binding constraint.
#: At n=2 the budget allows T=3000, which doubles the shifts a recurring cell
#: delivers (120 at t'=25 instead of 60) and halves the data each step supplies.
#: Both matter here: scarce data is the regime where a second-order method should
#: have the most to offer, and it is an axis X4 already established as real.
CONDITIONS: list[tuple[str, dict, int, int]] = [
    ("linear_a0p025", {"schedule": "linear", "total_degrees": 37.5}, 4, 1500),
    ("every200_jump5", {"schedule": "recurring", "jump_every": 200, "jump_degrees": 5.0}, 4, 1500),
    ("every25_jump5", {"schedule": "recurring", "jump_every": 25, "jump_degrees": 5.0}, 4, 1500),
    ("linear_a0p03", {"schedule": "linear", "total_degrees": 45.0}, 4, 1500),
    ("every100_jump15", {"schedule": "recurring", "jump_every": 100, "jump_degrees": 15.0},
     4, 1500),
    ("every200_jump15", {"schedule": "recurring", "jump_every": 200, "jump_degrees": 15.0},
     4, 1500),
    ("every200_jump30", {"schedule": "recurring", "jump_every": 200, "jump_degrees": 30.0},
     4, 1500),
    ("every50_jump15", {"schedule": "recurring", "jump_every": 50, "jump_degrees": 15.0}, 4, 1500),
    ("every25_jump30", {"schedule": "recurring", "jump_every": 25, "jump_degrees": 30.0}, 4, 1500),
    # Half the data, twice the horizon. One smooth and one abrupt condition at
    # the same n and T, so the contrast between them is not confounded by either.
    # alpha=0.015 x 3000 = 45 degrees, the same total travel as linear_a0p03
    # delivered at half the rate -- which is what makes the pair readable.
    ("lowN_linear_a0p015", {"schedule": "linear", "total_degrees": 45.0}, 2, 3000),
    ("lowN_every25_jump30", {"schedule": "recurring", "jump_every": 25, "jump_degrees": 30.0},
     2, 3000),
]

#: The condition the learning rate is re-swept on: the most damaging one.
LR_CONDITION = "every25_jump30"

#: Held fixed across cells so the shift *pattern* is one less thing varying
#: between them, exactly as X11 did. The data still varies with the run seed.
JUMP_SEED = 0

BASELINES = ["centralized_sgd", "diffusion_sgd_atc", "frozen_atc"]
DEVICE = "auto"
DTYPE = "float64"
FRESH = False

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x14_status.json"
SETTLED_FROM = int(0.8 * HORIZON)


# ---------------------------------------------------------------------------


#: The three settings X14 carries, as (label, predicate on an X13 cell). Each is
#: filled with whichever cell minimises settled error under drift among the cells
#: matching it -- the criterion X1--X12 used for every baseline.
#:
#: **gamma=1 and gamma<1 are carried separately, on purpose.** gamma's whole span
#: on the refined grid is 0.0012 against a 0.0013 significance threshold, so the
#: data does not separate the driftless random walk from a shrinking one, and
#: picking the argmin would be choosing between cells the measurement cannot tell
#: apart. Reporting both leaves the call to be made on grounds the numbers do not
#: supply -- parsimony, or what the diffusion filter should have to carry
#: (design note D71).
#: (label, predicate, learner name to run it under). The third entry exists
#: because two entries in one run need two names -- the config refuses duplicates,
#: loudly, which is how this was caught rather than silently dropping a setting.
SETTING_GROUPS: list[tuple[str, Any, str]] = [
    ("gamma = 1 (random walk)",
     lambda c: c["name"].endswith("gamma") and c["gamma"] == 1.0,
     "centralized_ekf_walk"),
    ("gamma < 1 (shrinking)",
     lambda c: c["name"].endswith("gamma") and c["gamma"] < 1.0,
     "centralized_ekf_gamma"),
    ("lambda family",
     lambda c: c["name"].endswith("lambda"),
     "centralized_ekf_lambda"),
]


def tuned_settings() -> list[dict]:
    r"""The settings X13 selected, one per group in :data:`SETTING_GROUPS`.

    Read rather than hard-coded, so this experiment cannot drift out of step with
    the tuning that justifies it. The two families are different *models* --
    $\bm F=\gamma\bm I$ against $\bm F=\bm I$ -- and which generalises is part of
    the question; within the $\gamma$ family, $\gamma=1$ is the canonical random
    walk and is carried alongside the empirical argmin for the reason above.
    """
    chosen = []
    for label, matches, runs_as in SETTING_GROUPS:
        scored = []
        for cell in cells(full=True):
            if not matches(cell):
                continue
            directory = f"{cell_name(cell)}_full"
            if not (ROOT / "results" / directory).exists():
                continue
            value = settled_for(directory, cell["name"])
            if value != float("inf"):
                scored.append((value, cell))
        if scored:
            score, best = min(scored, key=lambda pair: pair[0])
            print(f"  {label:<26} {score:.4f}  {cell_name(best)}  -> {runs_as}")
            chosen.append({**best, "name": runs_as})
    return chosen


def control_name(samples: int, horizon: int) -> str:
    """The stationary twin shared by every condition at this ``(n, T)``.

    Keyed on both, because `assert_paired_runs` rightly refuses a pairing whose
    two runs differ in anything but the drift -- and a control at a different
    horizon or sample count is exactly that kind of mismatch, producing
    well-formed rows whose subtraction is meaningless.
    """
    return f"x14_control_n{samples}_T{horizon}"


def config_for(
    name: str,
    drift: dict | None,
    learners: list[dict],
    seeds: list[int],
    samples: int = SAMPLES,
    horizon: int = HORIZON,
):
    """One condition. ``drift=None`` is the stationary control."""
    block = {"schedule": "stationary", "total_degrees": 0.0}
    if drift is not None:
        block = dict(drift)
        if block["schedule"] == "recurring":
            block["jump_seed"] = JUMP_SEED
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": name,
                "horizon": horizon,
                "eval_every": EVAL_EVERY,
                "seeds": seeds,
                "device": DEVICE,
                "dtype": DTYPE,
            },
            "env": {"samples_per_node_per_step": samples, "drift": block},
            "learners": learners,
            # Current-only: the question is tracking and recovery, and a
            # canonical set would score a rotation the run has long since left.
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def run_one(config, train, test, fresh: bool) -> str:
    """One config over its seeds. Divergence is recorded, poisoned, and skipped."""
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
            min_flush_steps=MIN_FLUSH_STEPS,
        )
        try:
            simulate.run(
                config, environment, learners,
                build_evalsets(config, environment, test), likelihood, theta0,
                recorder=recorder, verify_observations=False, progress_every=0,
            )
        except LearnerError as failure:
            note = f"diverged (seed {seed}): {failure}"
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "_diverged").write_text(note, encoding="utf-8")
            return note
        recorder.finalize()

    (out_dir / "_complete").write_text("", encoding="utf-8")
    # Nothing to resume from once a run is complete, and the checkpoints are the
    # largest thing in the directory (design note D68).
    for stale in out_dir.glob("*.checkpoint.pt"):
        stale.unlink()
    return "ok"


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def lr_name(lr: float) -> str:
    return f"x14_lr{f'{lr:g}'.replace('.', 'p')}"


def baseline_entries(lr: float) -> list[dict]:
    """The three baselines at one learning rate.

    `frozen_atc` takes ATC's rate rather than its own best: it is ATC stopped at
    step 300, and tuning it separately would make it a different algorithm that
    happens to be good at being frozen (design note D65).
    """
    return [
        {"name": name, "optimizer": "sgd_momentum", "lr": lr, "momentum": 0.9}
        | ({"freeze_after": 300} if name == "frozen_atc" else {})
        for name in BASELINES
    ]


def best_lr() -> float | None:
    """The rate the X14 re-tune selected, by ATC under the worst condition."""
    scored = [
        (settled_for(lr_name(lr), "diffusion_sgd_atc"), lr)
        for lr in BASELINE_LRS
        if (ROOT / "results" / lr_name(lr)).exists()
    ]
    scored = [pair for pair in scored if pair[0] != float("inf")]
    return min(scored)[1] if scored else None


def tune_learning_rate(train, test, fresh: bool) -> int:
    drift, samples, horizon = next(
        (dict(block), n, t) for label, block, n, t in CONDITIONS if label == LR_CONDITION
    )
    status = load_status()
    print(f"X14 baseline re-tune on {LR_CONDITION} -- the most damaging condition")
    print(f"{len(BASELINE_LRS)} rates at {len(LR_SEEDS)} seeds, T={HORIZON}\n")

    started = time.time()
    for index, lr in enumerate(BASELINE_LRS, start=1):
        name = lr_name(lr)
        note = run_one(
            config_for(name, drift, baseline_entries(lr), LR_SEEDS, samples, horizon),
            train, test, fresh,
        )
        status[name] = note
        save_status(status)
        elapsed = (time.time() - started) / 60
        print(f"[{index}/{len(BASELINE_LRS)}] lr {lr:<6g} {note:<12} {elapsed:.0f} min",
              flush=True)

    print(f"\nRe-tune complete in {(time.time() - started) / 60:.1f} min")
    print("Read it with: python scripts/report_ekf_generalization.py --lr")
    return 0


def main(tune: bool = False, fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)

    if tune:
        return tune_learning_rate(train, test, fresh)

    print("filter settings, read from the X13 refined grid:")
    settings = tuned_settings()
    if len(settings) < len(SETTING_GROUPS):
        print(
            "\nThe X13 refined grid has not finished, so there is no tuned setting to\n"
            "generalise. Run it first:\n"
            "  python scripts/run_ekf_sweep.py --full"
        )
        return 1

    lr = best_lr()
    if lr is None:
        print(
            "\nThe baselines have not been re-tuned on the worst condition, so this would\n"
            "compare a tuned filter against a baseline tuned for a 5x milder regime.\n"
            "  python scripts/run_ekf_generalization.py --lr"
        )
        return 1
    print(f"baselines at lr {lr:g}, re-tuned on {LR_CONDITION}\n")

    learners = settings + baseline_entries(lr)
    status = load_status()
    started = time.time()

    # One control per distinct (n, T), shared by every condition at that setting.
    settings_used = sorted({(samples, horizon) for _, _, samples, horizon in CONDITIONS})
    print(f"X14: {len(CONDITIONS)} conditions + {len(settings_used)} controls, "
          f"{len(learners)} learners, {len(SEEDS)} seeds")
    for samples, horizon in settings_used:
        count = sum(1 for _, _, n, t in CONDITIONS if (n, t) == (samples, horizon))
        print(f"  n={samples} T={horizon}: {count} conditions, "
              f"budget {10 * samples * horizon}/60000")
    print()

    # Controls first: every drifting condition subtracts one, so a failure here
    # invalidates all of them and should surface in minutes rather than in hours.
    for samples, horizon in settings_used:
        name = control_name(samples, horizon)
        note = run_one(
            config_for(name, None, learners, SEEDS, samples, horizon), train, test, fresh
        )
        status[name] = note
        save_status(status)
        print(f"  {name:<26} {note}  ({(time.time() - started) / 60:.0f} min)", flush=True)
    print()

    ran = 0
    for index, (label, drift, samples, horizon) in enumerate(CONDITIONS, start=1):
        name = f"x14_{label}"
        note = run_one(
            config_for(name, drift, learners, SEEDS, samples, horizon), train, test, fresh
        )
        status[name] = note
        save_status(status)
        elapsed = (time.time() - started) / 60
        # Averaged over conditions that actually ran; a cached one costs nothing
        # and would otherwise drag the mean down and make the estimate climb.
        if note != "cached":
            ran += 1
        remaining = (elapsed / ran * (len(CONDITIONS) - index)) if ran else 0.0
        print(f"[{index}/{len(CONDITIONS)}] {name:<26} {note:<28} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    diverged = sum(1 for note in status.values() if note.startswith("diverged"))
    print(f"\nX14 complete in {(time.time() - started) / 60:.1f} min, {diverged} diverged")
    print("Read it with: python scripts/report_ekf_generalization.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tune="--lr" in sys.argv, fresh="--fresh" in sys.argv))
