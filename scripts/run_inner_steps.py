r"""D1 -- K inner steps per time step, and the iterated EKF (D139), on Mackey--Glass.

    python scripts/run_inner_steps.py --lr          # tune SGD and AdamW per (cell, K), first
    python scripts/run_inner_steps.py               # the cells themselves
    python scripts/run_inner_steps.py --report-only [--smoke]
    python scripts/run_inner_steps.py --lr --smoke  # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

## The question

The image task's runner states it in full (main's `scripts/run_inner_steps.py`):
every learner makes $K$ passes over each step's batch -- $K$ optimiser steps, or
$K$ Gauss--Newton relinearisations of the measurement update (the iterated EKF,
its covariance from the last linearisation, no damping) -- with one combine per
time step, so communication does not change. Predicted: extra passes help the
centralised learners and over-fit the distributed ones (local SGD's client drift).

## Named before the run (confirmatory under D118)

1. **Do the centralised learners gain from extra passes?** Settled RMSE at $K=8$
   minus $K=1$, per seed, for centralised SGD, AdamW and the centralised filter, in
   each cell; Holm across the nine. **Predicted negative.**
2. **Does decentralisation cost more as $K$ grows?** ATC and ATC AdamW against
   their centralised counterparts, local adapt and one-hop against the centralised
   filter, at $K=8$ minus at $K=1$, in each cell; Holm across the twelve.
   **Predicted positive**, largest under the $\tau$ spread, where the agents' laws
   differ.

## The design

$K\in\{2,5,8\}$, against $K=1$, crossed with three cells (D139): stationary, **linear** $\beta$
drift (MG's harder drift: +0.0044--0.0108 RMSE in M6 against ~0.005 under the
reflecting abrupt walk, D106) and M8's per-agent $\tau$ spread (`m_tau_spread`) --
at M6's scale: ER 0.3, $N=10$, $T=1500$, five seeds, evaluations every 25 steps,
settled over the last 20%, M4's $\bm R$ scale. **The filters carry M4's and M5's
selections at every $K$** (the X14 discipline, decided 2026-10-06). **The gradient
learners are re-tuned at every (cell, $K$)** on M3's paired grids, each extended
one step down -- $K$ steps at rate $\eta$ move about as far as one at $K\eta$.
M6's $\gamma=0.9995$ reference arms are left out: they are reference lines, not
methods, and every learner here costs up to eight times its M6 price.

**K = 1 is reused where it already exists** (the user, 2026-10-07): M6's
stationary and linear cells; the $	au$ spread's is run here, since M8 carried six
of the twelve learners. $K=10$ is a later item (schedule.md, D1b).
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
from run_m3_rates import ADAMW_FAMILY, SGD_FAMILY, settled  # noqa: E402
from run_m6_comparison import (  # noqa: E402
    GROUP_B_FILTERS,
    SMOKE_FILTER,
    load_settings,
    r_scale_override,
)

from dekf_bench.metrics.paired import differences, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: K = 1 is the reference every contrast is taken against; K = 10 was cut on
#: 2026-10-07 (the user: K in {2, 5, 8} now, K = 10 later -- schedule.md, D1b).
KS = [1, 2, 5, 8]
K_LOW, K_HIGH = 1, 8
#: label -> experiment config. Linear, not abrupt: MG's harder drift (D106, D139).
CELLS = {"stationary": "m_stationary", "linear": "m_linear", "tau_spread": "m_tau_spread"}
#: M3 selected rates per condition; the tau spread is stationary, so it tunes on
#: its own cells here exactly as the others do.
CONDITIONS: list[tuple[str, int]] = [(cell, k) for cell in CELLS for k in KS]

#: K = 1 cells that already exist as exactly this experiment, reused rather than re-run
#: (the user, 2026-10-07): M6's stationary and linear cells -- the same roster plus
#: M6's two gamma = 0.9995 reference arms, which share nothing with the other learners
#: but the data -- at T = 1500, five seeds, M3's rates and M4/M5's filters. M8's tau
#: spread cell carries six of the twelve learners, so the tau spread's K = 1 runs here.
REUSED: dict[tuple[str, int], dict[str, str]] = {
    ("stationary", 1): {"a": "m6_stationary_a", "b": "m6_stationary_b"},
    ("linear", 1): {"a": "m6_linear_a", "b": "m6_linear_b"},
}

CENTRAL = "centralized_ekf_walk"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
MEAN_ONLY = [LOCAL, ONEHOP]
GROUPS = ["a", "b"]

#: M3's paired grids, each extended one step down: index i runs SGD_RATES[i] and
#: ADAMW_RATES[i] in one cell, as M3 did.
SGD_RATES = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
ADAMW_RATES = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LR_SEEDS = [0, 1]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 25
SMOKE_HORIZON, SMOKE_EVAL = 60, 20
STATUS = ROOT / "results" / "ins_status.json"


def label(cell: str, k: int) -> str:
    return f"{cell}_k{k}"


def lr_run_name(cond: str, index: int, suffix: str = "") -> str:
    return f"ins_lr_{cond}_r{index}{suffix}"


def cell_name(cond: str, group: str, suffix: str = "") -> str:
    """The run that holds this (cell, K, group): a reused one at K = 1 where it exists.
    Smokes never reuse -- they prove this runner's own path."""
    cell, k = cond.rsplit("_k", 1)
    if not suffix and (cell, int(k)) in REUSED:
        return REUSED[(cell, int(k))][group]
    return f"ins_{cond}_{group}{suffix}"


def is_reused(cell: str, k: int, suffix: str = "") -> bool:
    return not suffix and (cell, k) in REUSED


def rate_of(learner: str, index: int) -> float:
    return (ADAMW_RATES if learner in ADAMW_FAMILY else SGD_RATES)[index]


def selected_rates(cond: str, suffix: str = "") -> dict[str, float] | None:
    """Each gradient learner's argmin in this (cell, K), or None when any is unswept."""
    rates: dict[str, float] = {}
    for learner in [*SGD_FAMILY, *ADAMW_FAMILY]:
        scored = [(settled(lr_run_name(cond, i, suffix), learner), i)
                  for i in range(len(SGD_RATES))]
        finite = [(v, i) for v, i in scored if v != float("inf")]
        if not finite:
            return None
        rates[learner] = rate_of(learner, min(finite)[1])
    return rates


def config_for(args, name: str, cell: str, learners: list[dict], seeds: list[int],
               model_override: dict):
    return load_config(
        CELLS[cell],
        overrides={
            "run": {"name": name, "horizon": args.horizon, "seeds": seeds,
                    "eval_every": SMOKE_EVAL if args.smoke else EVAL_EVERY,
                    "device": args.device, "dtype": args.dtype},
            **({"model": model_override} if model_override else {}),
            "learners": learners,
        },
    )


def entries(group: str, centralised: dict, diffusion: dict, rates: dict, k: int) -> list[dict]:
    """Assigned by name, never by position (the mis-assignment D106 caught in M6)."""
    if group == "b":
        learners = [{"name": n, **diffusion, "combine_exponent": 1.0} for n in GROUP_B_FILTERS]
    else:
        learners = [{"name": CENTRAL, **centralised}]
        learners += [{"name": n, **diffusion} for n in MEAN_ONLY]
        learners += [{"name": n, "lr": rates[n], **o} for n, o in SGD_FAMILY.items()]
        learners += [{"name": n, "lr": rates[n], **o} for n, o in ADAMW_FAMILY.items()]
    return [{**entry, "inner_steps": k} for entry in learners]


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, model_override: dict, suffix: str) -> int:
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    runs = [(cell, k, i) for cell, k in CONDITIONS if not is_reused(cell, k, suffix)
            for i in range(len(SGD_RATES))]
    print(f"D1 lr{' SMOKE' if suffix else ''}: {len(CONDITIONS)} (cell, K) x "
          f"{len(SGD_RATES)} paired rates at {len(seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (cell, k, i) in enumerate(runs, start=1):
        name = lr_run_name(label(cell, k), i, suffix)
        learners = [{"name": n, "lr": SGD_RATES[i], **o, "inner_steps": k}
                    for n, o in SGD_FAMILY.items()]
        learners += [{"name": n, "lr": ADAMW_RATES[i], **o, "inner_steps": k}
                     for n, o in ADAMW_FAMILY.items()]
        note = run_one(config_for(args, name, cell, learners, seeds, model_override),
                       None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(runs)}] {name:<32} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    names = [*SGD_FAMILY, *ADAMW_FAMILY]
    print(f"  {'cell':>16}" + "".join(f"{n:>24}" for n in names))
    for cell, k in CONDITIONS:
        rates = None if is_reused(cell, k, suffix) else selected_rates(label(cell, k), suffix)
        if rates:
            edge = [n for n in names if rates[n] in (rate_of(n, 0), rate_of(n, len(SGD_RATES) - 1))]
            print(f"  {label(cell, k):>16}" + "".join(f"{rates[n]:>24g}" for n in names)
                  + (f"   <- grid edge: {', '.join(edge)}" if edge else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="tune the SGD and AdamW baselines instead of running the cells")
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
    if args.lr:
        return tune(args, model_override, suffix)

    unswept = [lr_run_name(label(c, k), i, suffix) for c, k in CONDITIONS
               if not is_reused(c, k, suffix) for i in range(len(SGD_RATES))
               if not any((ROOT / "results" / lr_run_name(label(c, k), i, suffix) / m).exists()
                          for m in ("_complete", "_diverged"))]
    if unswept:
        print(f"  REFUSED: {len(unswept)} lr cell(s) never ran, e.g. {unswept[0]}.")
        print(f"  Run --lr{' --smoke' if suffix else ''} first; completed cells are cached.")
        return 1
    missing = [name for groups in REUSED.values() if not suffix for name in groups.values()
               if not (ROOT / "results" / name / "_complete").exists()]
    if missing:
        print(f"  REFUSED: the reused K = 1 run(s) {missing} are not complete on disk.")
        return 1
    centralised, diffusion, _m3 = load_settings(args.smoke)
    if args.smoke:
        centralised = diffusion = SMOKE_FILTER

    # Cheapest first, so a stopped queue leaves the most comparable cells on disk.
    cells = [(c, k, g) for k in KS for c in CELLS for g in GROUPS
             if not is_reused(c, k, suffix)]
    print(f"D1{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds, "
          f"T={args.horizon}, device {args.device}\n", flush=True)
    status = load_status()
    started = time.time()
    for index, (cell, k, group) in enumerate(cells, start=1):
        cond = label(cell, k)
        name = cell_name(cond, group, suffix)
        rates = selected_rates(cond, suffix)
        learners = entries(group, centralised, diffusion, rates, k)
        cell_started = time.time()
        note = run_one(config_for(args, name, cell, learners, args.seeds, model_override),
                       None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<30} {len(learners):>2} learners  {note:<14}"
              f" {(time.time() - cell_started) / 60:.1f} min this cell, "
              f"{(time.time() - started) / 60:.0f} total", flush=True)
    print(f"\nD1{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

ALPHA = 0.05
STAT_HEADER = f"{'diff':>9}  {'95% CI':^20}  {'t':>7}  {'p_holm':>7}"


def per_seed(cell: str, learner: str) -> dict[int, float]:
    """Settled RMSE per seed; a seed whose window holds any NaN is diverged (D126)
    and is reported as NaN, never averaged around."""
    import pandas as pd  # noqa: PLC0415

    out: dict[int, float] = {}
    for path in sorted((ROOT / "results" / cell).glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["learner", "metric", "evalset", "t", "value"])
        rows = frame[(frame["learner"] == learner) & (frame["metric"] == "rmse")
                     & (frame["evalset"] == "current")]
        if rows.empty:
            continue
        rows = rows[rows["t"] >= int(0.8 * rows["t"].max())]
        seed = int(path.stem.split("_")[1])
        out[seed] = float("nan") if rows["value"].isna().any() else float(rows["value"].mean())
    return out


def _cell_of(learner: str, cond: str, suffix: str) -> str:
    return cell_name(cond, "b" if learner in GROUP_B_FILTERS else "a", suffix)


def _finite(values: dict[int, float]) -> dict[int, float]:
    return {s: v for s, v in values.items() if math.isfinite(v)}


def _table(title: str, lines: list[str], rows: list[tuple[str, object]]) -> None:
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
    width = max((len(r[0]) for r in rows), default=0) + 2
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


def report(suffix: str = "") -> None:
    seeds_of = lambda learner, cond: per_seed(_cell_of(learner, cond, suffix), learner)  # noqa: E731
    if suffix:
        print("  SMOKE: a short horizon at one seed. These numbers mean nothing; the")
        print("  point is that every code path below ran.\n")

    every = [CENTRAL, *MEAN_ONLY, *GROUP_B_FILTERS, *SGD_FAMILY, *ADAMW_FAMILY]
    for cell in CELLS:
        print(f"\n  settled RMSE, {cell} (last 20%, current set), by K\n")
        print(f"    {'learner':<36}" + "".join(f"{'K=' + str(k):>9}" for k in KS))
        for learner in every:
            cells = []
            for k in KS:
                values = seeds_of(learner, label(cell, k))
                if not values:
                    cells.append(f"{'-':>9}")
                elif any(not math.isfinite(v) for v in values.values()):
                    cells.append(f"{'DIVERGED':>9}")
                else:
                    cells.append(f"{sum(values.values()) / len(values):>9.4f}")
            print(f"    {learner:<36}" + "".join(cells))

    def gain(learner: str, cell: str, k: int = K_HIGH) -> object:
        high = _finite(seeds_of(learner, label(cell, k)))
        low = _finite(seeds_of(learner, label(cell, K_LOW)))
        result = compare(high, low) if high and low else None
        return result if result is not None and result.n >= 2 else None

    def gap_change(first: str, second: str, cell: str, k: int = K_HIGH) -> object:
        high = differences(_finite(seeds_of(first, label(cell, k))),
                           _finite(seeds_of(second, label(cell, k))))
        low = differences(_finite(seeds_of(first, label(cell, K_LOW))),
                          _finite(seeds_of(second, label(cell, K_LOW))))
        result = compare(high, low) if high and low else None
        return result if result is not None and result.n >= 2 else None

    centralised = [CENTRAL, "centralized_sgd", "centralized_adamw"]
    _table("Q1 [confirmatory]: centralised learners, K=8 minus K=1",
           ["negative = extra passes help. Predicted negative."],
           [(f"{c} {cell}", gain(c, cell)) for c in centralised for cell in CELLS])
    pairs = [("diffusion_sgd_atc", "centralized_sgd"),
             ("diffusion_atc_adamw", "centralized_adamw"), (LOCAL, CENTRAL), (ONEHOP, CENTRAL)]
    _table("Q2 [confirmatory]: gap to the family's centralised learner, K=8 minus K=1",
           ["positive = decentralisation costs more with more passes. Predicted positive,",
            "largest under the tau spread."],
           [(f"{d} {cell}", gap_change(d, c, cell)) for d, c in pairs for cell in CELLS])

    # ---- exploratory -------------------------------------------------------------
    for k in KS[1:-1]:
        _table(f"exploratory: gap to the family's centralised learner, K={k} minus K=1",
               ["the intermediate K: where the curve bends."],
               [(f"{d} {cell}", gap_change(d, c, cell, k)) for d, c in pairs for cell in CELLS])
    _table("exploratory: local learners and full sharing, K=8 minus K=1",
           ["no combine at all, and the covariance-sharing filters."],
           [(f"{d} {cell}", gain(d, cell))
            for d in ["local_only", "local_adamw", *GROUP_B_FILTERS] for cell in CELLS])
    _table("exploratory: one-hop minus ATC AdamW at matched K",
           ["negative = the filter wins at the same number of passes."],
           [(label(cell, k), (lambda cell=cell, k=k: (
               compare(_finite(seeds_of(ONEHOP, label(cell, k))),
                       _finite(seeds_of("diffusion_atc_adamw", label(cell, k))))
               if seeds_of(ONEHOP, label(cell, k)) else None))())
            for cell in CELLS for k in KS])
    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
