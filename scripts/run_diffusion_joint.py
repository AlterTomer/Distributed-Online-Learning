r"""X23 -- tune the filter's three axes jointly, having so far tuned them in turn.

    python scripts/run_diffusion_joint.py            # the 3-way grid
    python scripts/run_diffusion_joint.py --help     # every knob
    python scripts/run_diffusion_joint.py --fresh    # redo

## Why

X20 swept $(q,\sigma_0^2)$ with $\gamma$ pinned; X21 swept $(\gamma,q)$ with
$\sigma_0^2$ pinned. That is coordinate descent, and each pass fixed an axis at a
value chosen by a previous pass that had itself fixed something.

The axes are not separable, and X20's own data says so: the $\sigma_0^2$ optimum
**moves with $q$**, from $10^{-3}$ at $q=6\times10^{-4}$ to $0.1$ at
$6\times10^{-5}$ and $6\times10^{-6}$. "$\sigma_0^2$ is flat", which justified
pinning it in X21, holds only at the selected $q$ --- a lucky place to stand
rather than a property of the axis. The untested corner is $(\gamma,\sigma_0^2)$,
and after that it would be careless to assume it flat because it was at one
$\gamma$.

## Why the grid is cut where it is

$q\in\{6\times10^{-4},6\times10^{-5}\}$ only. X21 measured the whole
$q=6\times10^{-3}$ column as destroyed at *every* $\gamma$ --- error 0.43 to 0.89
against a chance level of 0.9, with $\lVert\bm\theta\rVert^2$ reaching
$1.3\times10^{7}$ --- and only one of those cells tripped the trust-region guard,
so "diverged" understates it. $6\times10^{-6}$ and below is uniformly poor. Two
hours over the live region beats seven over rubble.
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

#: The experiment itself, so not a flag.
GAMMAS = [1.0, 0.9999, 0.9995, 0.999]
PROCESS_NOISE = [6.0e-4, 6.0e-5]
PRIOR_SCALES = [0.1, 0.03, 0.01, 0.003, 0.001]

CONDITION = {"schedule": "recurring", "jump_every": 25, "jump_degrees": 15.0, "jump_seed": 0}
TOPOLOGY = ("erdos_renyi", {"p": 0.3})
LEARNER = "diffusion_ekf"

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2], 5
THRESHOLD = 0.0013
STATUS = ROOT / "results" / "x23_status.json"


def run_name(gamma: float, q: float, prior: float) -> str:
    stem = f"x23_g{gamma:g}_q{q:g}_s{prior:g}"
    return stem.replace(".", "p").replace("-", "m")


def settled(run: str, metric: str = "error_rate") -> float:
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    files = sorted(directory.glob("seed_*.parquet"))
    if not (directory / "_complete").exists() or not files:
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"])
         for f in files], ignore_index=True)
    # The last fifth of *this* run, so the window does not depend on whatever
    # horizon the caller happens to carry.
    rows = frame[(frame["learner"] == LEARNER) & (frame["metric"] == metric)
                 & (frame["t"] >= int(0.8 * frame["t"].max()))]
    if metric == "error_rate":
        rows = rows[rows["evalset"] == "current"]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def config_for(args, gamma: float, q: float, prior: float):
    topology, params = TOPOLOGY
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": run_name(gamma, q, prior), "horizon": args.horizon,
                    "eval_every": EVAL_EVERY, "seeds": args.seeds,
                    "device": args.device, "dtype": args.dtype},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": args.dataset, "drift": dict(CONDITION)},
            "learners": [{
                "name": LEARNER, "transition": "scalar", "gamma": gamma,
                "lambda_forget": 1.0, "process_noise_q": q, "prior_scale": prior,
            }],
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def report() -> None:
    cells = [(g, q, p) for g in GAMMAS for q in PROCESS_NOISE for p in PRIOR_SCALES]
    for q in PROCESS_NOISE:
        print(f"  settled error at q = {q:g}")
        print(f"    {'gamma \\ sigma':>14}" + "".join(f"{p:>10g}" for p in PRIOR_SCALES))
        for gamma in GAMMAS:
            line = f"    {gamma:>14g}"
            for prior in PRIOR_SCALES:
                if (ROOT / "results" / run_name(gamma, q, prior) / "_diverged").exists():
                    line += f"{'DIV':>10}"
                    continue
                value = settled(run_name(gamma, q, prior))
                line += f"{value:>10.4f}" if value != float("inf") else f"{'-':>10}"
            print(line)
        print()

    scored = [(settled(run_name(g, q, p)), (g, q, p)) for g, q, p in cells]
    scored = [(v, c) for v, c in scored if v != float("inf")]
    if not scored:
        return
    value, (gamma, q, prior) = min(scored)
    print(f"  joint best: gamma={gamma:g}, q={q:g}, sigma_0^2={prior:g} -> {value:.4f}")
    print("  coordinate-descent choice was gamma=0.9995, q=0.0006, sigma_0^2=0.001 -> 0.1173")
    coordinate = settled(run_name(0.9995, 6.0e-4, 1.0e-3))
    if coordinate != float("inf"):
        gap = coordinate - value
        verdict = ("below the 0.0013 threshold, so coordinate descent found the joint "
                   "optimum" if gap <= THRESHOLD else
                   "ABOVE threshold -- sweeping in turn cost us this much")
        print(f"  gap: {gap:+.4f}, {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--report-only", action="store_true",
                        help="print the surface from cells already on disk")
    args = parser.parse_args(argv)

    if args.report_only:
        report()
        return 0
    if not dataset_is_cached(args.dataset, ROOT / "data"):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, ROOT / "data", download=False)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    cells = [(g, q, p) for g in GAMMAS for q in PROCESS_NOISE for p in PRIOR_SCALES]
    print(f"X23: {len(GAMMAS)}x{len(PROCESS_NOISE)}x{len(PRIOR_SCALES)} = {len(cells)} cells "
          f"at {len(args.seeds)} seeds\n", flush=True)

    started = time.time()
    for index, (gamma, q, prior) in enumerate(cells, start=1):
        name = run_name(gamma, q, prior)
        note = run_one(config_for(args, gamma, q, prior), train, test, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(cells)}] {name:<34} {note:<28} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nX23 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
