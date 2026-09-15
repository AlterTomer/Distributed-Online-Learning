r"""X27 -- where should a one-hop agent linearise its neighbours' batches?

    python scripts/run_linearization_point.py                # the cells
    python scripts/run_linearization_point.py --report-only  # the table, from disk

## The question

D93 found that `diffusion_ekf_onehop_mean` rebuilds each neighbour's block at the
**sender's** predictive mean $\bm\theta_u^-$ and sums blocks from different points
into one update. That is not one \ac{ekf} update at a common point, and it costs a
$\bm\theta_u^-$ in the first message. The **receiver** alternative linearises every
batch at the receiving agent's own $\bm\theta_v^-$: one point, the textbook
diffusion \ac{ekf}, and 3 696 scalars per link per direction instead of 6 604 --
below momentum \ac{atc}'s 5 816 rather than above it (D94).

The two coincide whenever the agents' means agree, so the exactness gate passes
both and cannot choose. Only a sparse graph can, and only where the agents
disagree -- which is why the cells are X25's: skew is where they disagree most.

## The design

X25's five conditions on its \ac{er} $p=0.3$ graph and its filter settings, with
both points in **one run** per cell, so every comparison is paired by seed against
the same data stream. The sender arm doubles as a reproduction check: it must
match X25's `diffusion_ekf_onehop_mean` to the digit, because D94 changed the
ledger and not the filter.

Nothing is re-tuned. The receiver point might want different settings, but a
comparison at X25's selection isolates the linearisation point; tuning it is a
second question and only worth asking if this one comes back close.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _args import sweep_parser  # noqa: E402
from run_diffusion_skew import (  # noqa: E402
    CONDITIONS,
    DRIFTS,
    EVAL_EVERY,
    FILTER,
    HORIZON,
    SEEDS,
    SKEWS,
    TOPOLOGY,
)
from run_diffusion_skew import cell_name as x25_cell  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

SENDER, RECEIVER = "diffusion_ekf_onehop_mean", "diffusion_ekf_onehop_mean_receiver"
PAIR = [SENDER, RECEIVER]

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x27_status.json"


def cell_name(condition: str, twin: bool = False) -> str:
    return f"x27_control_{condition}" if twin else f"x27_{condition}"


def config_for(args, name: str, skew: float, drift: dict | None):
    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    topology, params = TOPOLOGY
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": args.seeds, "device": args.device, "dtype": args.dtype},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": args.dataset,
                    "partition": {"kind": "dirichlet", "beta": skew},
                    "drift": block},
            "learners": [{"name": n, **FILTER} for n in PAIR],
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def per_seed(run: str, learner: str, metric: str = "error_rate") -> dict[int, float]:
    """Settled value per seed: the mean over the last 20% of the horizon."""
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    if not (directory / "_complete").exists():
        return {}
    values: dict[int, float] = {}
    for path in sorted(directory.glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["learner", "metric", "evalset", "t", "value"])
        rows = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                     & (frame["t"] >= int(0.8 * frame["t"].max()))]
        if metric == "error_rate":
            rows = rows[rows["evalset"] == "current"]
        if len(rows):
            values[int(path.stem.split("_")[1])] = float(rows["value"].mean())
    return values


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, int]:
    """Mean of a - b over shared seeds, its t statistic, and the seed count."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    if not diffs:
        return math.nan, math.nan, 0
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return mean, math.nan, len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    return mean, (mean / (sd / math.sqrt(len(diffs))) if sd > 0 else math.inf), len(diffs)


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--report-only", action="store_true",
                        help="print the table from cells already on disk")
    args = parser.parse_args(argv)

    if args.report_only:
        report()
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)

    cells: list[tuple[str, float, dict | None]] = []
    for condition, skew, drift in CONDITIONS:
        cells.append((cell_name(condition), skew, drift))
        if drift is not None:
            cells.append((cell_name(condition, twin=True), skew, None))

    print(f"\nX27: {len(cells)} cells at {len(args.seeds)} seeds, T={args.horizon}, "
          f"learners {PAIR}")
    for name, skew, drift in cells:
        kind = "still" if drift is None else drift["schedule"]
        print(f"  {name:<30} skew {skew:<6g} {kind}")
    print(flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (name, skew, drift) in enumerate(cells, start=1):
        note = run_one(config_for(args, name, skew, drift), train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(cells) - index)) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<30} {note:<28} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nX27 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    """Receiver minus sender per cell, the reproduction check, and ATC for scale."""
    print("  settled error, receiver minus sender (negative: the receiver point wins)")
    print(f"    {'cell':>16}{'sender':>10}{'receiver':>10}{'diff':>10}{'t':>8}{'n':>4}"
          f"{'vs X25':>12}{'ATC (2psi)':>12}")
    rows = [(f"still_b{s:g}", False) for s in SKEWS]
    rows += [(c, False) for c in DRIFTS] + [(c, True) for c in DRIFTS]
    for condition, twin in rows:
        run = cell_name(condition, twin)
        sender, receiver = per_seed(run, SENDER), per_seed(run, RECEIVER)
        if not sender or not receiver:
            print(f"    {run.removeprefix('x27_'):>16}  -")
            continue
        diff, t, n = paired(receiver, sender)
        mean = lambda d: sum(d.values()) / len(d)  # noqa: E731
        # The reproduction check runs on the seeds both experiments share: X25 ran
        # three, so comparing means over different seed sets would test nothing.
        old = per_seed(x25_cell(condition, twin), SENDER)
        shared = sorted(set(old) & set(sender))
        drift = max((abs(sender[s] - old[s]) for s in shared), default=math.nan)
        atc = per_seed(x25_cell(condition, twin), "diffusion_sgd_atc")
        print(f"    {run.removeprefix('x27_'):>16}{mean(sender):>10.4f}{mean(receiver):>10.4f}"
              f"{diff:>+10.4f}{t:>8.2f}{n:>4}"
              f"{(f'{drift:.1e} ({len(shared)})' if shared else '-'):>12}"
              f"{(f'{mean(atc):.4f}' if atc else '-'):>12}")
    print("\n  'vs X25' is the largest per-seed |sender - X25 sender| over the seeds both")
    print("  ran, in brackets. It must be zero: D94 changed the ledger, not the filter.")
    print("  If it is not, stop -- something else moved. ATC is X25's, over its seeds.")


if __name__ == "__main__":
    raise SystemExit(main())
