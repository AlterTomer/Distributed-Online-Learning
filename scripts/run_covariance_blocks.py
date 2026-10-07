r"""D3a -- the block-diagonal covariance against the full one, on Mackey--Glass (D139).

    python scripts/run_covariance_blocks.py               # the block cells (GPU)
    python scripts/run_covariance_blocks.py --report-only [--smoke]
    python scripts/run_covariance_blocks.py --smoke       # the path, on CPU

The image task's runner states the question and the design (main's
`scripts/run_covariance_blocks.py`): neuron blocks against the full covariance for
every filter, Holm across the ten (five filters, two cells), no direction predicted,
the size read against $\varepsilon=0.002$; layer blocks exploratory. Here the cells
are D1's at $K=1$ -- stationary and **linear** $\beta$ drift (MG's harder drift,
D106) -- and the **full covariance is M6's** stationary and linear cells, reused
exactly. The filters carry M4's and M5's selections and M4's $\bm R$ scale under every
structure (decided 2026-10-07). The Transformer's neurons are its linear layers'
output units; its LayerNorm gains and biases are one block each.
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
from run_ekf_generalization import run_one  # noqa: E402
from run_inner_steps import (  # noqa: E402
    CENTRAL,
    HORIZON,
    MEAN_ONLY,
    SEEDS,
    SMOKE_HORIZON,
    config_for,
    per_seed,
)
from run_inner_steps import cell_name as d1_cell  # noqa: E402
from run_m6_comparison import GROUP_B_FILTERS, SMOKE_FILTER, load_settings, r_scale_override  # noqa: E402

from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402

CELLS = ["stationary", "linear"]
STRUCTURES = ["neuron", "layer"]
GROUPS = ["a", "b"]
FILTERS = [CENTRAL, *MEAN_ONLY, *GROUP_B_FILTERS]
STATUS = ROOT / "results" / "cvb_status.json"
ALPHA = 0.05
STAT_HEADER = f"{'diff':>9}  {'95% CI':^20}  {'t':>7}  {'p_holm':>7}"


def cell_name(cell: str, structure: str, group: str, suffix: str = "") -> str:
    """The run for (cell, structure, group): full is D1's K = 1 cell (M6's, reused)."""
    if structure == "full":
        return d1_cell(f"{cell}_k1", group, suffix)
    return f"cvb_{cell}_{structure}_{group}{suffix}"


def entries(group: str, structure: str, centralised: dict, diffusion: dict) -> list[dict]:
    if group == "b":
        learners = [{"name": n, **diffusion, "combine_exponent": 1.0} for n in GROUP_B_FILTERS]
    else:
        learners = [{"name": CENTRAL, **centralised}] + [{"name": n, **diffusion}
                                                         for n in MEAN_ONLY]
    return [{**entry, "covariance_blocks": structure} for entry in learners]


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="a short horizon at one seed, placeholder filter settings")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0
    if args.smoke:
        args.horizon, args.seeds = SMOKE_HORIZON, [0]
    model_override = r_scale_override(args.smoke)
    centralised, diffusion, _rates = load_settings(args.smoke)
    if args.smoke:
        centralised = diffusion = SMOKE_FILTER

    cells = [(c, s, g) for s in STRUCTURES for c in CELLS for g in GROUPS]
    print(f"D3a{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds, "
          f"T={args.horizon}, device {args.device}\n", flush=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started = time.time()
    for index, (cell, structure, group) in enumerate(cells, start=1):
        name = cell_name(cell, structure, group, suffix)
        cell_started = time.time()
        learners = entries(group, structure, centralised, diffusion)
        note = run_one(config_for(args, name, cell, learners, args.seeds, model_override),
                       None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(cells)}] {name:<30} {note:<14} "
              f"{(time.time() - cell_started) / 60:.1f} min this cell, "
              f"{(time.time() - started) / 60:.0f} total", flush=True)
    print(f"\nD3a{' smoke' if suffix else ''} complete in {(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


def report(suffix: str = "") -> None:
    def seeds_of(learner: str, cell: str, structure: str) -> dict:
        group = "b" if learner in GROUP_B_FILTERS else "a"
        values = per_seed(cell_name(cell, structure, group, suffix), learner)
        return {s: v for s, v in values.items() if math.isfinite(v)}

    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: a short horizon at one seed. These numbers mean nothing.\n")
    for cell in CELLS:
        print(f"\n  settled RMSE, {cell} (last 20%, current set), by structure\n")
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
        adjusted = holm([r.p if r is not None else float("nan") for _l, r in rows])
        for (lab, result), p_adjusted in zip(rows, adjusted, strict=True):
            if result is None:
                print(f"    {lab:<{width}}{'-':>9}   (cells missing or diverged)")
                continue
            lo, hi = result.ci(0.95)
            star = " *" if p_adjusted < ALPHA else ""
            print(f"    {lab:<{width}}{result.mean:>+9.4f}  {f'[{lo:+.4f}, {hi:+.4f}]':<20}"
                  f"  {result.t:>+7.2f}  {p_adjusted:>7.3f}{star}")

    def cost(learner: str, cell: str, structure: str, against: str = "full") -> object:
        a, b = seeds_of(learner, cell, structure), seeds_of(learner, cell, against)
        result = compare(a, b) if a and b else None
        return result if result is not None and result.n >= 2 else None

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
