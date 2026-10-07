r"""D3a -- the block-diagonal covariance against the full one, on MNIST (D139).

    python scripts/run_covariance_blocks.py               # the block cells (GPU)
    python scripts/run_covariance_blocks.py --report-only
    python scripts/run_covariance_blocks.py --smoke       # the path, on CPU

## The question

The decoupled EKF keeps the full filter's gain -- the innovation matrix couples every
block -- and its mean is exact for a block-diagonal prior; what it drops is the
cross-block terms of each posterior covariance (`ekf.BlockCovariance.woodbury`). That
is what lets a larger model carry a covariance at all (D3b's GNN), so its price must be
measured where the full covariance still fits: here, $p=2\,908$.

## Named before the run (confirmatory under D118)

1. **What does the neuron-block covariance cost?** Settled error, neuron minus full,
   per seed, for every filter (the centralised filter, local adapt, one-hop, and both
   full-sharing variants) in each cell; Holm across the ten. **No direction
   predicted**: the question is the size, read against the $\varepsilon=0.002$ the
   codec uses for "costs nothing".

The layer-block structure, the cost of each against the other, and the speed and
memory of each are exploratory.

## The design (decided 2026-10-07)

D1's cells at $K=1$: stationary IID and abrupt rotation -- the harder drift, where a
belief has to re-open and a structural approximation would show. The **full
covariance is reused**, not re-run: P5.4's stationary cell and D1's abrupt $K=1$ cell
are exactly these experiments. The filters **carry X20's selection** under every
structure (the X14 discipline); the gradient learners hold no covariance and stay out.
Groups as D1's: a (centralised, local adapt, one-hop) and b (full sharing).
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
from run_diffusion_skew import CENTRALIZED, FILTER  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_inner_steps import (  # noqa: E402
    CENTRAL,
    FULL_SHARING,
    HORIZON,
    MEAN_ONLY,
    SEEDS,
    config_for,
)
from run_inner_steps import cell_name as d1_cell  # noqa: E402
from run_linearization_point import per_seed  # noqa: E402
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402

CELLS = ["still", "abrupt"]
STRUCTURES = ["neuron", "layer"]
GROUPS = ["a", "b"]
FILTERS = [CENTRAL, *MEAN_ONLY, *FULL_SHARING]
DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "cvb_status.json"


def cell_name(cell: str, structure: str, group: str, suffix: str = "") -> str:
    """The run for (cell, structure, group): full is D1's K = 1 cell, itself reused."""
    if structure == "full":
        return d1_cell(f"{cell}_k1", group, suffix)
    return f"cvb_{cell}_{structure}_{group}{suffix}"


def entries(group: str, structure: str) -> list[dict]:
    if group == "b":
        learners = [{"name": n, **FILTER, "combine_exponent": 1.0} for n in FULL_SHARING]
    else:
        learners = [{"name": CENTRAL, **CENTRALIZED}] + [{"name": n, **FILTER} for n in MEAN_ONLY]
    return [{**entry, "covariance_blocks": structure} for entry in learners]


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="20 steps at one seed, into _smoke-suffixed runs")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.horizon, args.seeds = 20, [0]
    if args.report_only:
        report(suffix)
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)

    cells = [(c, s, g) for s in STRUCTURES for c in CELLS for g in GROUPS]
    print(f"D3a{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds, "
          f"T={args.horizon}\n", flush=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started = time.time()
    for index, (cell, structure, group) in enumerate(cells, start=1):
        name = cell_name(cell, structure, group, suffix)
        cell_started = time.time()
        note = run_one(config_for(args, name, cell, entries(group, structure)), train, test,
                       args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(cells)}] {name:<28} {note:<26} "
              f"{(time.time() - cell_started) / 60:.1f} min this cell, "
              f"{(time.time() - started) / 60:.0f} total", flush=True)
    print(f"\nD3a{' smoke' if suffix else ''} complete in {(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


def report(suffix: str = "") -> None:
    def seeds_of(learner: str, cell: str, structure: str) -> dict:
        group = "b" if learner in FULL_SHARING else "a"
        return per_seed(cell_name(cell, structure, group, suffix), learner)

    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 rounds at one seed, against the smoke's own K = 1 cells.")
        print("  These numbers mean nothing; the point is that every path ran.\n")
    for cell in CELLS:
        print(f"\n  settled error, {cell} (last 20%, current set), by structure\n")
        print(f"    {'filter':<36}" + "".join(f"{s:>9}" for s in ["full", *STRUCTURES]))
        for learner in FILTERS:
            print(f"    {learner:<36}" + "".join(
                f"{mean(seeds_of(learner, cell, s)):>9.4f}" for s in ["full", *STRUCTURES]))

    def table(title: str, lines: list[str], rows: list[tuple[str, object]]) -> None:
        print(f"\n  {title}")
        for line in lines:
            print(f"    {line}")
        width = max(len(r[0]) for r in rows) + 2
        print(f"\n    {'':<{width}}{STAT_HEADER}")
        live = [(lab, r) for lab, r in rows if r is not None]
        adjusted = dict(zip([lab for lab, _r in live], holm([r.p for _l, r in live]),
                            strict=True))
        for lab, result in rows:
            shown = stat_columns(result, adjusted[lab]) if result is not None else \
                f"{'-':>9}   (cells missing)"
            print(f"    {lab:<{width}}{shown}")

    def cost(learner: str, cell: str, structure: str, against: str = "full") -> object:
        a, b = seeds_of(learner, cell, structure), seeds_of(learner, cell, against)
        return compare(a, b) if a and b else None

    table("Q1 [confirmatory]: neuron blocks minus the full covariance",
          ["positive = the block approximation costs accuracy. No direction predicted;",
           "read the size against epsilon = 0.002."],
          [(f"{f} {c}", cost(f, c, "neuron")) for f in FILTERS for c in CELLS])
    table("exploratory: layer blocks minus the full covariance", [],
          [(f"{f} {c}", cost(f, c, "layer")) for f in FILTERS for c in CELLS])
    table("exploratory: neuron minus layer blocks", ["negative = finer blocks do better."],
          [(f"{f} {c}", cost(f, c, "neuron", "layer")) for f in FILTERS for c in CELLS])
    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
