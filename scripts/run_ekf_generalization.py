r"""X14 -- does the tuned filter hold up where the learners actually break?

Run this file directly.

    python scripts/run_ekf_generalization.py --lr    # re-tune baselines (~0.5h)
    python scripts/run_ekf_generalization.py         # the conditions   (~12h)

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

#: (label, drift override). Ordered mildest-first so a run stopped early has
#: still crossed the interesting boundary. Chosen to span the measured damage
#: range 0.0054 -> 0.0677 and both ends of the frequent-small / rare-large axis,
#: which is the contrast X11 was built around.
CONDITIONS: list[tuple[str, dict]] = [
    ("linear_a0p025", {"schedule": "linear", "total_degrees": 0.025 * HORIZON}),
    ("every200_jump5", {"schedule": "recurring", "jump_every": 200, "jump_degrees": 5.0}),
    ("every25_jump5", {"schedule": "recurring", "jump_every": 25, "jump_degrees": 5.0}),
    ("every100_jump15", {"schedule": "recurring", "jump_every": 100, "jump_degrees": 15.0}),
    ("every200_jump15", {"schedule": "recurring", "jump_every": 200, "jump_degrees": 15.0}),
    ("every200_jump30", {"schedule": "recurring", "jump_every": 200, "jump_degrees": 30.0}),
    ("every50_jump15", {"schedule": "recurring", "jump_every": 50, "jump_degrees": 15.0}),
    ("every25_jump30", {"schedule": "recurring", "jump_every": 25, "jump_degrees": 30.0}),
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


def tuned_settings() -> list[dict]:
    r"""The best cell of each family from the X13 refined grid.

    Read rather than hard-coded, so this experiment cannot drift out of step with
    the tuning that justifies it. One setting per family because the two are
    different *models* -- $\bm F=\gamma\bm I$ against $\bm F=\bm I$ -- and which
    of them generalises is part of the question.
    """
    chosen = []
    for family in ("centralized_ekf_gamma", "centralized_ekf_lambda"):
        scored = []
        for cell in cells(full=True):
            if cell["name"] != family:
                continue
            directory = f"{cell_name(cell)}_full"
            if not (ROOT / "results" / directory).exists():
                continue
            value = settled_for(directory, family)
            if value != float("inf"):
                scored.append((value, cell))
        if scored:
            best = min(scored, key=lambda pair: pair[0])
            print(f"  {family:<24} best {best[0]:.4f}  {cell_name(best[1])}")
            chosen.append(best[1])
    return chosen


def config_for(
    name: str,
    drift: dict | None,
    learners: list[dict],
    seeds: list[int],
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
                "horizon": HORIZON,
                "eval_every": EVAL_EVERY,
                "seeds": seeds,
                "device": DEVICE,
                "dtype": DTYPE,
            },
            "env": {"drift": block},
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
    drift = dict(next(block for label, block in CONDITIONS if label == LR_CONDITION))
    status = load_status()
    print(f"X14 baseline re-tune on {LR_CONDITION} -- the most damaging condition")
    print(f"{len(BASELINE_LRS)} rates at {len(LR_SEEDS)} seeds, T={HORIZON}\n")

    started = time.time()
    for index, lr in enumerate(BASELINE_LRS, start=1):
        name = lr_name(lr)
        note = run_one(
            config_for(name, drift, baseline_entries(lr), LR_SEEDS), train, test, fresh
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
    if len(settings) < 2:
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

    print(f"X14: {len(CONDITIONS)} conditions + 1 shared control, "
          f"{len(learners)} learners, {len(SEEDS)} seeds\n")

    # The control first: every drifting condition subtracts it, so a failure here
    # invalidates all of them and should surface in minutes rather than in hours.
    note = run_one(config_for("x14_control", None, learners, SEEDS), train, test, fresh)
    status["x14_control"] = note
    save_status(status)
    print(f"  x14_control  {note}  ({(time.time() - started) / 60:.0f} min)\n", flush=True)

    for index, (label, drift) in enumerate(CONDITIONS, start=1):
        name = f"x14_{label}"
        note = run_one(config_for(name, drift, learners, SEEDS), train, test, fresh)
        status[name] = note
        save_status(status)
        elapsed = (time.time() - started) / 60
        print(f"[{index}/{len(CONDITIONS)}] {name:<26} {note:<28} "
              f"{elapsed:.0f} min, ~{elapsed / index * (len(CONDITIONS) - index):.0f} left",
              flush=True)

    diverged = sum(1 for note in status.values() if note.startswith("diverged"))
    print(f"\nX14 complete in {(time.time() - started) / 60:.1f} min, {diverged} diverged")
    print("Read it with: python scripts/report_ekf_generalization.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tune="--lr" in sys.argv, fresh="--fresh" in sys.argv))
