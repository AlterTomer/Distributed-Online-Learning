r"""X24 -- was covariance sharing wasted because what it shipped was pessimistic?

    python scripts/run_diffusion_combine.py            # the alpha x beta grid
    python scripts/run_diffusion_combine.py --help     # every knob

## Why

X19 measured full covariance sharing against mean-only and found it buys +0.0002
to +0.0007 across six cells --- below the 0.0013 threshold --- for $2909\times$ the
bandwidth. The obvious reading is that the combine axis is empty (D79).

That reading assumes the covariance being shipped was worth having. It was not
necessarily: `eq:cov_combine` is $\sum_u a_{vu}\bm P^{\psi}_u$, which is
`lem:conservative` --- the bound that holds for *any* cross-correlation because it
assumes the neighbours' errors **coincide**. That is the worst case, and a filter
handed a worst-case covariance runs a smaller gain than its information deserves.

$\beta$ interpolates to the other end. $\sum_u a_{vu}^{\beta}\bm P^{\psi}_u$ with
$\beta=2$ is what *independent* errors give, and on uniform weights it divides the
combined covariance by about $\lvert\mathcal M_v\rvert$ --- the same factor D79
says the belief is inflated by. So the sharper question, and the one this sweep
asks: **was full sharing wasted, rather than useless?**

## Why $\alpha$ is in the grid rather than fixed at X22's answer

D82's lesson, and here there is a mechanism rather than merely an absence of one.
$\alpha$ inflates the information *entering* the adapt step; $\beta$ deflates the
covariance *leaving* the combine. Both shrink $\bm P$, so they are substitutes to
first order, and a $\beta$ line at X22's $\alpha$ would measure $\beta$ at a value
chosen while $\beta$ was pinned at 1. They differ in one way that keeps the grid
honest: $\alpha$ also scales the score and so moves the *estimate*, while $\beta$
touches only the covariance. If the grid were separable that difference is where
it would show.

## Why this is a separate script, and one learner per cell

$\beta$ is meaningless for X22's learners: `combine_exponent` is read only under
`covariance_sharing: full` (D85), and both deployable variants share means only.
So the full-sharing pair needs its own sweep regardless.

**One learner per cell, which is the part that is not obvious.** Cells already run
sequentially --- `run_one` builds the environment and learners fresh for each and
drops them after --- so the peak is set by a single cell, and splitting the grid
across invocations would not lower it. What sets the peak is how many filters live
in one cell at once. X22, carrying *two local-sharing* learners, was measured at
5.0 GiB of the 8.2 GiB card; full sharing duplicates the covariance set during the
mix, so the same pairing here would land near 6.3 GiB. Eighteen cells of one
learner cost exactly what nine cells of two would, and halve the high-water mark.
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

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: 1.0 is lem:conservative and what every run so far has used; 2.0 is the
#: independent-errors end. The midpoint is there because the truth is expected
#: between, the errors being correlated through shared history but not perfectly.
COMBINE_EXPONENTS = [1.0, 1.5, 2.0]

#: X22's axis, carried so the two can be read jointly rather than in turn.
EXPONENTS = [0.0, 0.5, 1.0]

#: The full-sharing variants -- the only ones beta reaches (D85).
LEARNERS = ["diffusion_ekf_full", "diffusion_ekf_onehop"]

GAMMA, PROCESS_NOISE, PRIOR_SCALE = 0.9995, 6.0e-4, 1.0e-3
CONDITION = {"schedule": "recurring", "jump_every": 25, "jump_degrees": 15.0, "jump_seed": 0}
TOPOLOGY = ("erdos_renyi", {"p": 0.3})

#: Defaults for the CLI (`--horizon`, `--seeds`); see scripts/_args.py.
HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2], 5

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x24_status.json"


def run_name(learner: str, alpha: float, beta: float) -> str:
    stem = learner.replace("diffusion_ekf_", "")
    return f"x24_{stem}_a{alpha:g}_b{beta:g}".replace(".", "p")


def settled(run: str, learner: str, metric: str = "error_rate") -> float:
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    files = sorted(directory.glob("seed_*.parquet"))
    if not (directory / "_complete").exists() or not files:
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"])
         for f in files], ignore_index=True)
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                 & (frame["t"] >= int(0.8 * frame["t"].max()))]
    if metric == "error_rate":
        rows = rows[rows["evalset"] == "current"]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def config_for(args, learner: str, alpha: float, beta: float):
    topology, params = TOPOLOGY
    entries = [
        {"name": learner, "transition": "scalar", "gamma": GAMMA, "lambda_forget": 1.0,
         "process_noise_q": PROCESS_NOISE, "prior_scale": PRIOR_SCALE,
         "information_exponent": alpha, "combine_exponent": beta}
    ]
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": run_name(learner, alpha, beta), "horizon": args.horizon,
                    "eval_every": EVAL_EVERY, "seeds": args.seeds,
                    "device": args.device, "dtype": args.dtype},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": args.dataset, "drift": dict(CONDITION)},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def report() -> None:
    r"""Error and ECE per learner. Divergence is the expected failure at beta=2:
    an over-confident EKF is the one that blows up (D61), and beta=2 is the
    over-confident end by construction."""
    for learner in LEARNERS:
        for metric, label in (("error_rate", "settled error"), ("ece", "ECE")):
            print(f"  {learner}, {label}")
            print(f"    {'alpha \\ beta':>13}"
                  + "".join(f"{b:>10g}" for b in COMBINE_EXPONENTS))
            for alpha in EXPONENTS:
                line = f"    {alpha:>13g}"
                for beta in COMBINE_EXPONENTS:
                    if (ROOT / "results" / run_name(learner, alpha, beta) / "_diverged").exists():
                        line += f"{'DIV':>10}"
                        continue
                    value = settled(run_name(learner, alpha, beta), learner, metric)
                    line += f"{value:>10.4f}" if value != float("inf") else f"{'-':>10}"
                print(line)
            print()

    # The comparison the sweep exists for: full sharing at its best beta against
    # the mean-only filter X20 reported.
    print("  full sharing against mean-only (X20's diffusion_ekf: 0.1189 at 5 seeds)")
    for learner in LEARNERS:
        scored = [(settled(run_name(learner, a, b), learner), a, b)
                  for a in EXPONENTS for b in COMBINE_EXPONENTS]
        scored = [(v, a, b) for v, a, b in scored if v != float("inf")]
        if not scored:
            continue
        value, alpha, beta = min(scored)
        at_one = settled(run_name(learner, alpha, 1.0), learner)
        gained = f"{at_one - value:+.4f} from beta" if at_one != float("inf") else "-"
        print(f"    {learner:<28} best {value:.4f} at alpha={alpha:g}, beta={beta:g}"
              f"   ({gained})")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--report-only", action="store_true",
                        help="print the surface from cells already on disk")
    args = parser.parse_args(argv)

    if args.report_only:
        report()
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    cells = [(lr, a, b) for lr in LEARNERS
             for a in EXPONENTS for b in COMBINE_EXPONENTS]
    print(f"X24: {len(EXPONENTS)} alpha x {len(COMBINE_EXPONENTS)} beta = {len(cells)} "
          f"cells at {len(args.seeds)} seeds, both full-sharing variants per cell")
    print("     memory: two full-sharing filters hold 2 x 10 x 64.5 MiB, and the mix "
          "needs a second set (~3.3 GiB peak)\n", flush=True)

    started = time.time()
    for index, (learner, alpha, beta) in enumerate(cells, start=1):
        name = run_name(learner, alpha, beta)
        note = run_one(config_for(args, learner, alpha, beta), train, test, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(cells)}] {name:<34} {note:<28} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nX24 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
