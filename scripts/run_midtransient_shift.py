r"""P5.23 -- an adversarial probe: a second shift placed mid-transient.

    python scripts/run_midtransient_shift.py --lr          # tune SGD and AdamW on `late`
    python scripts/run_midtransient_shift.py               # the two placements
    python scripts/run_midtransient_shift.py --report-only
    python scripts/run_midtransient_shift.py --lr --smoke  # then --smoke

`--lr` must run first and the main pass refuses without it (D77).

## The question (phase5_plan.md P5.23)

`piecewise` is the only schedule whose change points are *specified*, so it is the
one that can put a shift exactly where a filter is most exposed: mid-transient,
while the covariance is still inflated from the last shift, or just after it has
contracted again. The plan's worry is concrete -- X20's selected filter sits one
decade below a divergence cliff, and a well-placed shift is the kind of thing that
could push it over. "Worth one cell, not a grid."

**The transient was measured before choosing the placement.** In X20's recurring
cell (15 degrees every 25 steps, evaluated every 5) the error jumps at the shift
and is still falling 20 steps later -- the centralised filter from 0.110 to 0.087,
the one-hop filter from 0.125 to 0.097 -- so about 10 steps after a shift is the
middle of the recovery.

## The design

Two cells differing only in *when* the second of two identical 15-degree shifts
lands: **mid** at steps 750 and 760, **late** at 750 and 1000. Both settle for 750
steps first and both end at 30 degrees. $T=1250$ (50 000 of the 60 000 images),
evaluations every 5 steps so the transient is resolved, IID, ER 0.3, $N=10$, five
seeds. The baselines are tuned on **late**, by whole-run error, and those rates are
carried into **mid**: a learner tuned for normal operation, met by the adversarial
placement -- and the only way the two cells can differ in the drift alone, which
the report gates. The filters carry X20's selection. Cells a / b / adamw as the
other phase-5 runners, with D119's merge gate.

**The wound.** Per learner and seed, the error accumulated above that learner's own
recovered level at each rotation, integrated from the first shift to the end:

$$W=\sum_{t\ge750}\bigl(e(t)-f(\theta_t)\bigr)\,\Delta t,$$

$\Delta t=5$ the cadence, $\theta_t$ the rotation at step $t$, and $f$ the recovered
floor: $f(30^\circ)$ the cell's own mean error over its last 100 steps; $f(15^\circ)$
the **late** cell's mean over steps 950--995, the end of its long 15-degree plateau,
used for both cells because **mid** spends only ten steps there. Both cells face the
same shifts and end in the same state, so $W_{\text{mid}}-W_{\text{late}}$ isolates
the placement: positive means a shift landing mid-transient costs more than the two
shifts apart (compounding), negative means it costs less (the second shift rides on
a belief the first one already opened).

## Named before the run (confirmatory under D118)

1. **Does placing the second shift mid-transient change what the filters pay?**
   $W_{\text{mid}}-W_{\text{late}}$ per seed, for the centralised filter, local adapt
   and one-hop; Holm across the three. **No direction predicted**: the plan's
   divergence worry says positive, an inflated covariance absorbing the second shift
   says negative, and both are live.
2. **Does any filter diverge in either cell?** Reported as a count of diverged seeds
   per cell; not a test, and any divergence at all is the adversarial finding.

Exploratory: the same contrast for full sharing and every gradient baseline, the
filter-minus-ATC difference of the contrast, and the floors themselves.
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
from run_break_rate import whole_run  # noqa: E402
from run_diffusion_skew import BASELINES, CENTRALIZED, FILTER, LR_SEEDS  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import per_seed  # noqa: E402
from run_network_size import (  # noqa: E402
    ADAMW,
    ADAMW_GROUP,
    REPRODUCTION_ARM,
    REPRODUCTION_TOLERANCE,
    family_of,
    grid_for,
    tuned_baselines,
    tuned_learners,
)
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

JUMP = 15.0
FIRST = 750
#: The second shift: ten steps after the first (mid-transient), or 250 (settled).
PLACEMENTS = {"mid": 760, "late": 1000}
TUNED_ON = "late"
HORIZON, SEEDS, EVAL_EVERY = 1250, [0, 1, 2, 3, 4], 5
#: Floors: the cell's own last 100 steps at 30 degrees; the late cell's steps
#: 950-995 at 15 degrees -- the end of its plateau, before its second shift.
FLOOR_TAIL = 100
FLOOR_15 = (950, 1000)

CENTRAL = "centralized_ekf_gamma"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
MEAN_ONLY = [LOCAL, ONEHOP]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
FILTERS = [CENTRAL, *MEAN_ONLY]
ATC = "diffusion_sgd_atc"
GROUPS = ["a", "b", ADAMW_GROUP]

#: The smoke compresses the timeline tenfold; its numbers mean nothing.
SMOKE = {"horizon": 125, "first": 75, "placements": {"mid": 76, "late": 100},
         "floor_15": (95, 100), "tail": 10, "eval_every": 1}

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "mts_status.json"


def timeline(smoke: bool) -> dict:
    if smoke:
        return SMOKE
    return {"horizon": HORIZON, "first": FIRST, "placements": PLACEMENTS,
            "floor_15": FLOOR_15, "tail": FLOOR_TAIL, "eval_every": EVAL_EVERY}


def lr_run_name(rate: float, suffix: str = "", family: str = "sgd") -> str:
    stem = "mts_lr_adamw" if family == "adamw" else "mts_lr"
    return f"{stem}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(placement: str, group: str, suffix: str = "") -> str:
    return f"mts_{placement}_{group}{suffix}"


def selected_rates(suffix: str = "") -> dict[str, float] | None:
    rates: dict[str, float] = {}
    for name in tuned_learners():
        scored = [(whole_run(lr_run_name(r, suffix, family_of(name)), name), r)
                  for r in grid_for(name)]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, placement: str, entries: list[dict], smoke: bool,
               seeds: list[int] | None = None):
    line = timeline(smoke)
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": line["horizon"], "eval_every": line["eval_every"],
                    "seeds": seeds or args.seeds, "device": args.device, "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
            "env": {"dataset": args.dataset,
                    "drift": {"schedule": "piecewise", "jump_degrees": JUMP,
                              "change_points": [line["first"], line["placements"][placement]],
                              "total_degrees": 2 * JUMP}},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def entries_for(group: str, rates: dict[str, float]) -> list[dict]:
    if group == "b":
        return [{"name": n, **FILTER, "combine_exponent": 1.0} for n in FULL_SHARING]
    if group == ADAMW_GROUP:
        learners = [{"name": n, "lr": rates[n]} for n in ADAMW]
        learners.append({"name": REPRODUCTION_ARM, "lr": rates[REPRODUCTION_ARM],
                         **BASELINES[REPRODUCTION_ARM]})
        return learners
    learners = [{"name": CENTRAL, **CENTRALIZED}]
    learners += [{"name": n, **FILTER} for n in MEAN_ONLY]
    learners += [{"name": n, "lr": rates[n], **o} for n, o in tuned_baselines().items()]
    return learners


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, train, test, suffix: str = "") -> int:
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    grids = {family: sorted({r for b in tuned_learners() if family_of(b) == family
                             for r in grid_for(b)}, reverse=True)
             for family in ("sgd", "adamw")}
    cells = [(f, r) for f, grid in grids.items() for r in grid]
    print(f"P5.23 lr{' SMOKE' if suffix else ''}: the `{TUNED_ON}` placement at "
          f"({len(grids['sgd'])} SGD + {len(grids['adamw'])} AdamW) rates, {len(seeds)} seed(s)\n",
          flush=True)
    started = time.time()
    for index, (family, rate) in enumerate(cells, start=1):
        name = lr_run_name(rate, suffix, family)
        entries = [{"name": b, "lr": rate, **o} for b, o in tuned_learners().items()
                   if family_of(b) == family and rate in grid_for(b)]
        note = run_one(config_for(args, name, TUNED_ON, entries, bool(suffix), seeds=seeds),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<28} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    rates = selected_rates(suffix)
    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    for name, rate in (rates or {}).items():
        edge = rate in (grid_for(name)[0], grid_for(name)[-1])
        print(f"  {name:<28} {rate:g}{'   <- grid edge' if edge else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="tune the SGD and AdamW baselines on the late placement")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="a tenfold-compressed timeline at one seed, _smoke runs")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.seeds = [0]
    if args.report_only:
        report(suffix)
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)
    if args.lr:
        return tune(args, train, test, suffix)

    unswept = sorted({lr_run_name(r, suffix, family_of(b)) for b in tuned_learners()
                      for r in grid_for(b)
                      if not any((ROOT / "results" / lr_run_name(r, suffix, family_of(b)) / m)
                                 .exists() for m in ("_complete", "_diverged"))})
    if unswept:
        print(f"  REFUSED: {len(unswept)} lr cell(s) never ran, e.g. {unswept[0]}.")
        print(f"  Run --lr{' --smoke' if suffix else ''} first; completed cells are cached.")
        return 1
    rates = selected_rates(suffix)
    edges = [f"{b} at {r:g}" for b, r in rates.items() if r in (grid_for(b)[0], grid_for(b)[-1])]
    if edges:
        print("  ⚠ selected rates on their grid's edge: " + "; ".join(edges) + "\n")

    cells = [(placement, group) for group in GROUPS for placement in PLACEMENTS]
    print(f"P5.23{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds\n",
          flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (placement, group) in enumerate(cells, start=1):
        name = cell_name(placement, group, suffix)
        entries = entries_for(group, rates)
        note = run_one(config_for(args, name, placement, entries, bool(suffix)),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<22} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nP5.23{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _group_of(learner: str) -> str:
    if learner in FULL_SHARING:
        return "b"
    if learner in ADAMW:
        return ADAMW_GROUP
    return "a"


def _errors(run: str):
    """Per (learner, seed, t) error on the `current` set, counts-then-divide."""
    import glob  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from dekf_bench.metrics.breaks import error_by_step  # noqa: PLC0415

    files = sorted(glob.glob(str(ROOT / "results" / run / "seed_*.parquet")))
    if not files or not (ROOT / "results" / run / "_complete").exists():
        return None
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return error_by_step(frame, "current", by_seed=True)


def wounds(learner: str, suffix: str) -> tuple[dict[int, float], dict[int, float]] | None:
    """(W_mid, W_late) per seed, or None when a cell is missing."""
    line = timeline(bool(suffix))
    group = _group_of(learner)
    tables = {p: _errors(cell_name(p, group, suffix)) for p in PLACEMENTS}
    if any(t is None for t in tables.values()):
        return None
    late = tables["late"][tables["late"].learner == learner]
    low, high = line["floor_15"]
    floor_15 = late[(late.t >= low) & (late.t < high)].groupby("seed").error.mean()
    out = {}
    for placement, table in tables.items():
        rows = table[table.learner == learner]
        horizon = line["horizon"]
        floor_30 = rows[rows.t >= horizon - line["tail"]].groupby("seed").error.mean()
        second = line["placements"][placement]
        window = rows[rows.t >= line["first"]]
        values = {}
        for seed, series in window.groupby("seed"):
            floor = [floor_15[seed] if t < second else floor_30[seed] for t in series.t]
            values[int(seed)] = float(((series.error.to_numpy() - floor)
                                       * line["eval_every"]).sum())
        out[placement] = values
    return out["mid"], out["late"]


def report(suffix: str = "") -> None:
    from dekf_bench.metrics.breaks import BreakError, assert_paired_runs  # noqa: PLC0415

    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: a compressed timeline at one seed. These numbers mean nothing;")
        print("  the point is that every code path below ran.\n")

    print("  pairing gate: mid against late (only the drift -- the second shift's step --")
    print("  may differ)")
    for group in GROUPS:
        try:
            assert_paired_runs(ROOT / "results" / cell_name("mid", group, suffix),
                               ROOT / "results" / cell_name("late", group, suffix))
            print(f"    {group:<6} paired")
        except BreakError as failure:
            print(f"    {group:<6} NOT PAIRED: {str(failure)[:90]}")

    print("\n  merge gate: centralized_sgd in the AdamW cells against cell a, per seed")
    poolable = True
    for placement in PLACEMENTS:
        recorded = per_seed(cell_name(placement, "a", suffix), REPRODUCTION_ARM)
        rerun = per_seed(cell_name(placement, ADAMW_GROUP, suffix), REPRODUCTION_ARM)
        shared = sorted(set(recorded) & set(rerun))
        worst = max((abs(recorded[s] - rerun[s]) for s in shared), default=None)
        ok = worst is not None and worst <= REPRODUCTION_TOLERANCE
        poolable &= ok
        shown = f"{worst:.1e}" if worst is not None else "-"
        print(f"    {placement:<6} {len(shared)} seed(s)  max |diff| {shown:>8}   "
              f"{'reproduces' if ok else 'not run' if worst is None else 'DOES NOT REPRODUCE'}")

    print("\n  Q2 [descriptive]: divergence -- any diverged seed is the adversarial finding")
    status = load_status()
    for placement in PLACEMENTS:
        for group in GROUPS:
            name = cell_name(placement, group, suffix)
            note = status.get(name, "not run")
            print(f"    {name:<24} {note}")

    every = [*FILTERS, *FULL_SHARING, *tuned_baselines(), *ADAMW]
    table = {learner: wounds(learner, suffix) for learner in every}
    print("\n  the wound W (error-steps above the learner's own recovered floor, from the")
    print("  first shift on), per seed mean\n")
    print(f"    {'learner':<36}{'W mid':>10}{'W late':>10}{'mid-late':>10}")
    for learner, pair in table.items():
        if pair is None:
            print(f"    {learner:<36}{'-':>10}")
            continue
        mid, late = pair
        print(f"    {learner:<36}{mean(mid):>10.3f}{mean(late):>10.3f}"
              f"{mean(mid) - mean(late):>+10.3f}")

    def rows_for(names: list[str]) -> list[tuple[str, object]]:
        return [(n, compare(*table[n]) if table[n] is not None else None) for n in names]

    def show(title: str, lines: list[str], rows: list[tuple[str, object]]) -> None:
        print(f"\n  {title}")
        for line in lines:
            print(f"    {line}")
        print(f"\n    {'':<38}{STAT_HEADER}")
        live = [(lab, r) for lab, r in rows if r is not None]
        adjusted = dict(zip([lab for lab, _r in live], holm([r.p for _l, r in live]),
                            strict=True))
        for lab, result in rows:
            shown = stat_columns(result, adjusted[lab]) if result is not None else "  (missing)"
            print(f"    {lab:<38}{shown}")

    show("Q1 [confirmatory]: W_mid - W_late for the filters, paired per seed",
         ["positive = a shift mid-transient costs more; negative = less. No direction",
          "predicted. Holm across the three."], rows_for(FILTERS))
    show("exploratory: the same contrast for full sharing and the gradient baselines",
         ["Holm across these rows."], rows_for([*FULL_SHARING, *tuned_baselines(), *ADAMW]))

    if table[ONEHOP] is not None and table[ATC] is not None:
        def contrast(name: str) -> dict[int, float]:
            mid, late = table[name]
            return {s: mid[s] - late[s] for s in mid if s in late}
        show("exploratory: the placement contrast, filter minus ATC",
             ["negative = the filter is hurt less by the mid-transient placement than ATC."],
             [(f"{name} - ATC", compare(contrast(name), contrast(ATC)))
              for name in FILTERS if table[name] is not None])
    if not poolable:
        print("\n  ⚠ The AdamW cells fail (or have not reached) the merge gate.")
    print(f"\n  * = p_holm < {ALPHA} (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
