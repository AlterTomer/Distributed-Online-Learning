r"""The abrupt condition re-run at a finer cadence, so a shift *transient* is visible.

    python scripts/run_m6_shift_cycles.py --device cuda
    python scripts/run_m6_shift_cycles.py --eval-every 1 --device cuda
    python scripts/run_m6_shift_cycles.py --report-only

## Why this run exists

M6 recorded the abrupt condition at ``eval_every = 25``. Its schedule jumps at
``jump_every = 25``. **Those coincide**, so every recorded step lands exactly on a
jump boundary and the within-cycle response -- the wound, and the recovery from it
-- is unobserved. M6 can say what the drift *costs* (MG11) and cannot say what a
single shift *looks like*, which is the question the image task's shift-cycle
figures answer.

The fix is cadence, not more seeds: record every 5th step (or every step) so each
25-step cycle carries 5 (or 25) phase points, then average over the ~60 cycles a
run contains.

## What it runs, and what it deliberately does not

Four arms, not fourteen. The transient is a property of the *method*, so the
figure needs the pooled reference, the two adapt scopes, and one gradient
baseline to show the contrast -- adding the full-sharing variants and the rest of
the gradient family would multiply the cost for curves that lie on top of the ones
already drawn.

Settings come from `m4_selection.json`, `m5_selection.json` and `m3_rates.json`,
exactly as M6 takes them, so these cells are the same learners M6 ran and their
curves are comparable to it. **It writes no selection and nothing M6 reads.**

⚠ Cost scales with the cadence, not the horizon: at ``--eval-every 5`` there are
300 evaluation points against M6's 60. The held-out sets are cached per law value
and `recurring` reflects at the cap, so only a handful of distinct laws are ever
built -- which is what keeps this affordable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, settled  # noqa: E402

from dekf_bench.utils.config import load_config  # noqa: E402

CONDITION = "m_abrupt"
CELL = "m6cyc_abrupt"

#: The pooled reference, both adapt scopes, and one gradient baseline for contrast.
#: The transient is a property of the method; twelve curves would overlap.
FILTERS = ["centralized_ekf_walk", "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
BASELINE = "diffusion_atc_adamw"

HORIZON, SEEDS = 1500, [0, 1, 2, 3, 4]
EVAL_EVERY = 5

M3_RATES = ROOT / "results" / "m3_rates.json"
M4_SELECTION = ROOT / "results" / "m4_selection.json"
M5_SELECTION = ROOT / "results" / "m5_selection.json"
STATUS = ROOT / "results" / "m6cyc_status.json"


def as_filter(selection: dict) -> dict:
    return {
        "transition": "scalar", "gamma": selection["gamma"],
        "forgetting": "process_noise", "process_noise_q": selection["process_noise_q"],
        "lambda_forget": 1.0, "prior_scale": selection["prior_scale"],
    }


def settings():
    """M6's own settings, or exit naming what is missing."""
    missing = [p.name for p in (M3_RATES, M4_SELECTION, M5_SELECTION) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing {missing}: this run carries M6's settings so its curves are\n"
            "comparable to M6's. Run the tuning chain first."
        )
    m4 = json.loads(M4_SELECTION.read_text(encoding="utf-8"))
    m5 = json.loads(M5_SELECTION.read_text(encoding="utf-8"))
    rates = json.loads(M3_RATES.read_text(encoding="utf-8"))
    return as_filter(m4), as_filter(m5), rates.get("abrupt", {})


def entries(centralised: dict, diffusion: dict, rates: dict) -> list[dict]:
    learners = [{"name": "centralized_ekf_walk", **centralised}]
    learners += [{"name": n, **diffusion} for n in FILTERS[1:]]
    if BASELINE in rates:
        learners.append({"name": BASELINE, "lr": rates[BASELINE]["lr"],
                         **ADAMW_FAMILY[BASELINE]})
    return learners


def config_for(args, learners: list[dict]):
    # R is a model-level setting M4 selected at x1, so nothing is overridden here:
    # these cells must match M6's model exactly or the curves are not comparable.
    return load_config(
        CONDITION,
        overrides={
            "run": {"name": CELL, "horizon": args.horizon, "seeds": args.seeds,
                    "eval_every": args.eval_every, "device": args.device,
                    "dtype": args.dtype},
            "learners": learners,
        },
    )


def sweep(args) -> int:
    centralised, diffusion, rates = settings()
    learners = entries(centralised, diffusion, rates)
    jump_every = load_config(CONDITION).env.drift.jump_every
    if args.eval_every >= jump_every:
        raise SystemExit(
            f"eval_every={args.eval_every} is not finer than jump_every={jump_every}.\n"
            "That is the exact collision this run exists to break: every recorded step\n"
            "would land on a jump boundary again. Use --eval-every 5 or 1."
        )

    print(f"shift cycles: 1 cell x {len(args.seeds)} seeds, {len(learners)} learners, "
          f"eval_every {args.eval_every} against jump_every {jump_every} "
          f"({jump_every // args.eval_every} phase points per cycle), "
          f"device {args.device}\n", flush=True)
    started = time.time()
    note = run_one(config_for(args, learners), None, None, args.fresh)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    status[CELL] = {"note": note, "eval_every": args.eval_every}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"  {CELL}: {note}, {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    directory = ROOT / "results" / CELL
    if not (directory / "_complete").exists():
        print(f"  {CELL} has not completed; run the sweep first")
        return
    print(f"  settled RMSE on {CONDITION} at the finer cadence\n")
    print(f"    {'learner':<44}{'this run':>11}")
    for name in [*FILTERS, BASELINE]:
        value = settled(CELL, name)
        print(f"    {name:<44}" + (f"{value:>11.4f}" if value != float("inf")
                                   else f"{'-':>11}"))
    print("\n  These should reproduce M6's abrupt column. If they do not, the cadence")
    print("  changed something it should not have, and the transient figure is unsafe")
    print("  to read. The phase-aligned figure is MG12, in make_mg_results_figures.py.")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                        help="record every Nth step; must be finer than jump_every")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.report_only:
        report()
        return 0
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
