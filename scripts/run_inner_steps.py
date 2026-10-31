r"""D1 -- K inner steps per time step, and the iterated EKF (D139), on MNIST.

    python scripts/run_inner_steps.py --lr          # tune SGD and AdamW per (cell, K), first
    python scripts/run_inner_steps.py               # the cells themselves
    python scripts/run_inner_steps.py --report-only
    python scripts/run_inner_steps.py --lr --smoke  # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

## The question

Every learner so far makes one pass over each step's batch: one gradient step, or
one linearisation of the measurement update. Here every learner makes $K$:
the gradient learners take $K$ optimiser steps on the batch before the combine,
and the filters take $K$ Gauss--Newton relinearisations -- the iterated EKF, its
covariance from the last linearisation, no damping (D139). One combine per time
step either way, so communication does not change; only computation does.

**Where cell a diverges** (MNIST's three $K=8$ cells did, 2026-10-30), the cells pass
re-runs its centralised filter and tuned baselines without the decentralised filters,
as `ins_<cond>_central`, and the report reads them from there.

The supervisor's prediction, which is the user's: extra passes **help the
centralised learners**, which see the pooled $Nn = 40$ samples a step, and
**over-fit the distributed ones**, which see four. That is local SGD's *client
drift*: each agent moves towards its own batch's optimum before averaging. It
should be strongest under label skew, where the agents' optima differ most, and
`test_k_local_sgd_steps_break_the_atc_identity` already shows it in its purest
form -- even on a complete graph with plain SGD, two local steps break X0.

## Named before the run (confirmatory under D118)

1. **Do the centralised learners gain from extra passes?** Settled error at
   $K=8$ minus $K=1$, per seed, for centralised SGD, AdamW and the centralised
   filter, in each of the three cells; Holm across the nine. **Predicted negative.**
2. **Does decentralisation cost more as $K$ grows?** Each distributed learner's
   gap to its own family's centralised learner (ATC and ATC AdamW against
   centralised SGD and AdamW; local adapt and one-hop against the centralised
   filter), at $K=8$ minus at $K=1$, in each cell; Holm across the twelve.
   **Predicted positive**, largest under skew.

Everything else -- the intermediate $K$, `local_only`, full sharing, the filter
against the gradient methods at matched $K$ -- is exploratory.

## The design

$K\in\{2,5,8\}$, against $K=1$, crossed with three cells -- stationary IID, Dirichlet
label skew $\beta_{\mathrm{dir}}=0.1$, and abrupt rotation (X17's 15° every 25
steps, MNIST's harder drift, D67) -- at P5.4's scale: ER 0.3, $N=10$, $n=4$,
$T=750$, five seeds, evaluations every 10 steps, settled over the last 20%.
**The gradient learners are re-tuned at every (cell, $K$)**, so more steps is not
confounded with a larger effective step (D77); the SGD grid reaches down to
$5\times10^{-4}$ because $K$ steps at rate $\eta$ move about as far as one at
$K\eta$. **The filters carry X20's selection at every $K$** (decided 2026-10-06,
the X14 discipline), which the paper must state. Cells a / b / adamw with D119's
merge gate, as P5.4's.

**K = 1 is reused where it already exists** (the user, 2026-10-07): the stationary
cell's is P5.4's `lab_n4_p100_*`, the same experiment exactly; skew and abrupt have
no matching run, so theirs is run here. $K=10$, first planned, is a later item
(schedule.md, D1b): the curve is read at 2, 5 and 8 first.
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
from run_diffusion_skew import (  # noqa: E402
    BASELINES,
    CENTRALIZED,
    DRIFTS,
    FILTER,
    LEARNING_RATES,
    LR_SEEDS,
    settled,
)
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import per_seed  # noqa: E402
from run_network_size import (  # noqa: E402
    ADAMW,
    ADAMW_GROUP,
    REPRODUCTION_ARM,
    REPRODUCTION_TOLERANCE,
    family_of,
    tuned_baselines,
    tuned_learners,
)
from run_network_size import grid_for as size_grid  # noqa: E402
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import differences, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: K = 1 is the reference every contrast is taken against; K = 10 was cut on
#: 2026-10-07 (the user: K in {2, 5, 8} now, K = 10 later -- schedule.md, D1b).
KS = [1, 2, 5, 8]
K_LOW, K_HIGH = 1, 8
STATIONARY = {"schedule": "stationary", "total_degrees": 0.0}
#: label -> (partition, drift). Skew at X25's severe end; abrupt is X17's, IID.
CELLS: dict[str, tuple[dict, dict]] = {
    "still": ({"kind": "iid"}, STATIONARY),
    "skew": ({"kind": "dirichlet", "beta": 0.1}, STATIONARY),
    "abrupt": ({"kind": "iid"}, dict(DRIFTS["abrupt"])),
}
CONDITIONS: list[tuple[str, int]] = [(cell, k) for cell in CELLS for k in KS]

#: K = 1 cells that already exist as exactly this experiment, reused rather than
#: re-run (the user, 2026-10-07): P5.4's dense n = 4 cell is IID, stationary, ER 0.3,
#: N = 10, T = 750, five seeds, the same three groups with the same roster, its rates
#: tuned in that cell. Skew and abrupt have no such run (X25 is T = 1500 with an
#: older roster; N>10's abrupt is T = 500), so their K = 1 is run here.
REUSED: dict[tuple[str, int], dict[str, str]] = {
    ("still", 1): {"a": "lab_n4_p100_a", "b": "lab_n4_p100_b", "adamw": "lab_n4_p100_adamw"},
}

CENTRAL = "centralized_ekf_gamma"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
MEAN_ONLY = [LOCAL, ONEHOP]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
ATC, ATC_ADAMW = "diffusion_sgd_atc", "diffusion_atc_adamw"
GROUPS = ["a", "b", ADAMW_GROUP]
#: A diverged cell keeps nothing, so one filter leaving its trust region costs every
#: learner beside it its row. Wherever cell a diverges, the cells pass re-runs its
#: learners without the decentralised filters -- the centralised filter and the tuned
#: baselines -- and the report reads them from there (the user, 2026-10-30, after all
#: three of MNIST's K = 8 cells a diverged). A learner's numbers do not depend on its
#: cellmates: D119's merge gate is that fact, checked in every cell.
CENTRAL_GROUP = "central"

N_AGENTS, SAMPLES = 10, 4
MNIST_TRAIN = 60_000
HORIZON, SEEDS, EVAL_EVERY = 750, [0, 1, 2, 3, 4], 10

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "ins_status.json"

#: Below the shared grid: K steps at rate eta travel about as far as one at K*eta,
#: so the optimum at K = 8 should sit near an eighth of K = 1's.
SGD_LOW = [5e-4]


def grid_for(name: str) -> list[float]:
    """This learner's own grid, largest rate first."""
    if family_of(name) == "adamw":
        return size_grid(name)
    return sorted({*size_grid(name), *LEARNING_RATES, *SGD_LOW}, reverse=True)


def label(cell: str, k: int) -> str:
    return f"{cell}_k{k}"


def lr_run_name(cond: str, rate: float, suffix: str = "", family: str = "sgd") -> str:
    stem = "ins_lr_adamw" if family == "adamw" else "ins_lr"
    return f"{stem}_{cond}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(cond: str, group: str, suffix: str = "") -> str:
    """The run that holds this (cell, K, group): a reused one at K = 1 where it exists.
    Smokes never reuse -- they prove this runner's own path."""
    cell, k = cond.rsplit("_k", 1)
    if not suffix and (cell, int(k)) in REUSED:
        return REUSED[(cell, int(k))][group]
    return f"ins_{cond}_{group}{suffix}"


def is_reused(cell: str, k: int, suffix: str = "") -> bool:
    return not suffix and (cell, k) in REUSED


def diverged(name: str) -> bool:
    return (ROOT / "results" / name / "_diverged").exists()


def selected_rates(cond: str, suffix: str = "") -> dict[str, float] | None:
    """Each tuned learner's own argmin in this (cell, K), or None when any is unswept."""
    rates: dict[str, float] = {}
    for name in tuned_learners():
        family = family_of(name)
        scored = [(settled(lr_run_name(cond, r, suffix, family), name), r)
                  for r in grid_for(name)]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, cell: str, entries: list[dict],
               seeds: list[int] | None = None):
    partition, drift = CELLS[cell]
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
            "env": {"dataset": args.dataset, "samples_per_node_per_step": SAMPLES,
                    "partition": dict(partition), "drift": dict(drift)},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def entries_for(group: str, rates: dict[str, float], k: int) -> list[dict]:
    if group == "b":
        learners = [{"name": n, **FILTER, "combine_exponent": 1.0} for n in FULL_SHARING]
    elif group == CENTRAL_GROUP:
        learners = [{"name": CENTRAL, **CENTRALIZED}]
        learners += [{"name": n, "lr": rates[n], **o} for n, o in tuned_baselines().items()]
    elif group == ADAMW_GROUP:
        learners = [{"name": n, "lr": rates[n]} for n in ADAMW]
        learners.append({"name": REPRODUCTION_ARM, "lr": rates[REPRODUCTION_ARM],
                         **BASELINES[REPRODUCTION_ARM]})
    else:
        learners = [{"name": CENTRAL, **CENTRALIZED}]
        learners += [{"name": n, **FILTER} for n in MEAN_ONLY]
        learners += [{"name": n, "lr": rates[n], **o} for n, o in tuned_baselines().items()]
    return [{**entry, "inner_steps": k} for entry in learners]


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def preflight(args) -> bool:
    """The data budget N n T <= 60 000 (D5), the same in every cell."""
    used = N_AGENTS * SAMPLES * args.horizon
    print(f"  pre-flight: N n T = {used} of {MNIST_TRAIN}  "
          f"{'ok' if used <= MNIST_TRAIN else 'OVER BUDGET'}")
    return used <= MNIST_TRAIN


def tune(args, train, test, suffix: str = "") -> int:
    """SGD and AdamW per (cell, K), each on its own grid and names."""
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    grids = {family: sorted({r for b in tuned_learners() if family_of(b) == family
                             for r in grid_for(b)}, reverse=True)
             for family in ("sgd", "adamw")}
    runs = [(cell, k, f, r) for cell, k in CONDITIONS if not is_reused(cell, k, suffix)
            for f, grid in grids.items() for r in grid]
    print(f"D1 lr{' SMOKE' if suffix else ''}: {len(CONDITIONS)} (cell, K) x "
          f"({len(grids['sgd'])} SGD + {len(grids['adamw'])} AdamW) rates at "
          f"{len(seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (cell, k, family, rate) in enumerate(runs, start=1):
        name = lr_run_name(label(cell, k), rate, suffix, family)
        entries = [{"name": b, "lr": rate, **o, "inner_steps": k}
                   for b, o in tuned_learners().items()
                   if family_of(b) == family and rate in grid_for(b)]
        note = run_one(config_for(args, name, cell, entries, seeds=seeds),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(runs)}] {name:<40} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    names = list(tuned_learners())
    print(f"  {'cell':>11}" + "".join(f"{b:>26}" for b in names))
    for cell, k in CONDITIONS:
        rates = None if is_reused(cell, k, suffix) else selected_rates(label(cell, k), suffix)
        if rates:
            edge = [b for b in names if rates[b] in (grid_for(b)[0], grid_for(b)[-1])]
            print(f"  {label(cell, k):>11}" + "".join(f"{rates[b]:>26g}" for b in names)
                  + (f"   <- grid edge: {', '.join(edge)}" if edge else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="tune the SGD and AdamW baselines instead of running the cells")
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
    if not preflight(args):
        return 1
    if args.lr:
        return tune(args, train, test, suffix)

    unswept = sorted({lr_run_name(label(c, k), r, suffix, family_of(b))
                      for c, k in CONDITIONS if not is_reused(c, k, suffix)
                      for b in tuned_learners() for r in grid_for(b)
                      if not any((ROOT / "results" / lr_run_name(label(c, k), r, suffix,
                                                                 family_of(b)) / m).exists()
                                 for m in ("_complete", "_diverged"))})
    if unswept:
        print(f"  REFUSED: {len(unswept)} lr cell(s) never ran, e.g. {unswept[0]}.")
        print(f"  Run --lr{' --smoke' if suffix else ''} first; completed cells are cached.")
        return 1
    missing = [name for (c, k), groups in REUSED.items() if not suffix
               for name in groups.values() if not (ROOT / "results" / name / "_complete").exists()]
    if missing:
        print(f"  REFUSED: the reused K = 1 run(s) {missing} are not complete on disk.")
        return 1
    edges = [f"{label(c, k)}: {b} at {r:g}" for c, k in CONDITIONS
             if not is_reused(c, k, suffix)
             for b, r in selected_rates(label(c, k), suffix).items()
             if r in (grid_for(b)[0], grid_for(b)[-1])]
    if edges:
        print("  ⚠ selected rates on their grid's edge -- the optimum may lie outside it:")
        for edge in edges:
            print(f"    {edge}")
        print()

    # Cheapest first: K = 1 everywhere before any K = 8, so a stopped queue leaves
    # the most comparable cells on disk.
    cells = [(c, k, g) for k in KS for c in CELLS for g in GROUPS
             if not is_reused(c, k, suffix)]
    print(f"D1{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds, "
          f"T={args.horizon}\n", flush=True)
    status = load_status()
    started = time.time()

    def run_cell(tag: str, cell: str, k: int, group: str) -> None:
        cond = label(cell, k)
        name = cell_name(cond, group, suffix)
        entries = entries_for(group, selected_rates(cond, suffix), k)
        cell_started = time.time()
        note = run_one(config_for(args, name, cell, entries), train, test, args.fresh)
        status[name] = note
        save_status(status)
        elapsed = (time.time() - started) / 60
        print(f"[{tag}] {name:<26} {len(entries):>2} learners  {note:<26}"
              f" {(time.time() - cell_started) / 60:.1f} min this cell, {elapsed:.0f} total",
              flush=True)

    for index, (cell, k, group) in enumerate(cells, start=1):
        run_cell(f"{index}/{len(cells)}", cell, k, group)
    rescues = [(c, k) for k in KS for c in CELLS
               if diverged(cell_name(label(c, k), "a", suffix))]
    if rescues:
        print(f"\n  cell a diverged in {len(rescues)} condition(s); re-running its "
              "centralised filter and tuned baselines without the decentralised filters\n",
              flush=True)
    for index, (cell, k) in enumerate(rescues, start=1):
        run_cell(f"{CENTRAL_GROUP} {index}/{len(rescues)}", cell, k, CENTRAL_GROUP)
    print(f"\nD1{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _cell_of(learner: str, cond: str, suffix: str = "") -> str:
    if learner in FULL_SHARING:
        return cell_name(cond, "b", suffix)
    if learner in ADAMW:
        return cell_name(cond, ADAMW_GROUP, suffix)
    if learner not in MEAN_ONLY and diverged(cell_name(cond, "a", suffix)):
        return cell_name(cond, CENTRAL_GROUP, suffix)
    return cell_name(cond, "a", suffix)


def _table(title: str, lines: list[str], rows: list[tuple[str, object]]) -> None:
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
    width = max((len(r[0]) for r in rows), default=0) + 2
    print(f"\n    {'':<{width}}{STAT_HEADER}")
    live = [(lab, r) for lab, r in rows if r is not None]
    adjusted = dict(zip([lab for lab, _r in live], holm([r.p for _l, r in live]), strict=True))
    for lab, result in rows:
        if result is None:
            print(f"    {lab:<{width}}{'-':>9}   (cells missing)")
        else:
            print(f"    {lab:<{width}}{stat_columns(result, adjusted[lab])}")


def report(suffix: str = "") -> None:
    """The merge gate, the settled table, the two named questions, then exploratory."""
    seeds_of = lambda learner, cond: per_seed(_cell_of(learner, cond, suffix), learner)  # noqa: E731
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 rounds at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")

    rescued = [label(c, k) for c, k in CONDITIONS if diverged(cell_name(label(c, k), "a", suffix))]
    if rescued:
        print("  cell a diverged at " + ", ".join(rescued) + ": its centralised filter and")
        print(f"  tuned baselines are read from the `{CENTRAL_GROUP}` re-run there, which")
        print("  leaves the decentralised filters out; local adapt and one-hop have no row.\n")

    print("  merge gate: centralized_sgd in the AdamW cell against cell a, per seed")
    poolable = True
    for cell, k in CONDITIONS:
        cond = label(cell, k)
        recorded = per_seed(_cell_of(REPRODUCTION_ARM, cond, suffix), REPRODUCTION_ARM)
        rerun = per_seed(cell_name(cond, ADAMW_GROUP, suffix), REPRODUCTION_ARM)
        shared = sorted(set(recorded) & set(rerun))
        worst = max((abs(recorded[s] - rerun[s]) for s in shared), default=None)
        ok = worst is not None and worst <= REPRODUCTION_TOLERANCE
        poolable &= ok
        shown = f"{worst:.1e}" if worst is not None else "-"
        print(f"    {cond:<11} {len(shared)} seed(s)  max |diff| {shown:>8}   "
              f"{'reproduces' if ok else 'not run' if worst is None else 'DOES NOT REPRODUCE'}")

    every = [CENTRAL, *MEAN_ONLY, *FULL_SHARING, *tuned_baselines(), *ADAMW]
    for cell in CELLS:
        print(f"\n  settled error, {cell} (last 20%, current set), by K\n")
        print(f"    {'learner':<36}" + "".join(f"{'K=' + str(k):>9}" for k in KS))
        for learner in every:
            print(f"    {learner:<36}" + "".join(
                f"{mean(seeds_of(learner, label(cell, k))):>9.4f}" for k in KS))

    def gain(learner: str, cell: str) -> object:
        """K = 8 minus K = 1, per seed."""
        high, low = seeds_of(learner, label(cell, K_HIGH)), seeds_of(learner, label(cell, K_LOW))
        return compare(high, low) if high and low else None

    def gap_change(first: str, second: str, cell: str) -> object:
        """(first - second) at K = 8 minus at K = 1, per seed."""
        high = differences(seeds_of(first, label(cell, K_HIGH)),
                           seeds_of(second, label(cell, K_HIGH)))
        low = differences(seeds_of(first, label(cell, K_LOW)),
                          seeds_of(second, label(cell, K_LOW)))
        return compare(high, low) if high and low else None

    centralised = [CENTRAL, "centralized_sgd", "centralized_adamw"]
    _table("Q1 [confirmatory]: centralised learners, K=8 minus K=1",
           ["negative = extra passes help. Predicted negative."],
           [(f"{c} {cell}", gain(c, cell)) for c in centralised for cell in CELLS])
    pairs = [(ATC, "centralized_sgd"), (ATC_ADAMW, "centralized_adamw"),
             (LOCAL, CENTRAL), (ONEHOP, CENTRAL)]
    _table("Q2 [confirmatory]: gap to the family's centralised learner, K=8 minus K=1",
           ["positive = decentralisation costs more with more passes. Predicted positive,",
            "largest under skew."],
           [(f"{d} {cell}", gap_change(d, c, cell)) for d, c in pairs for cell in CELLS])

    # ---- exploratory -------------------------------------------------------------
    for k in KS[1:-1]:
        _table(f"exploratory: gap to the family's centralised learner, K={k} minus K=1",
               ["the intermediate K: where the curve bends."],
               [(f"{d} {cell}", (lambda d=d, c=c, cell=cell: compare(
                   differences(seeds_of(d, label(cell, k)), seeds_of(c, label(cell, k))),
                   differences(seeds_of(d, label(cell, K_LOW)), seeds_of(c, label(cell, K_LOW))))
                   if seeds_of(d, label(cell, k)) and seeds_of(d, label(cell, K_LOW)) else None)())
                for d, c in pairs for cell in CELLS])
    _table("exploratory: local_only and full sharing, K=8 minus K=1",
           ["no combine at all, and the covariance-sharing filters."],
           [(f"{d} {cell}", gain(d, cell)) for d in ["local_only", *FULL_SHARING]
            for cell in CELLS])
    if poolable:
        _table("exploratory: one-hop minus ATC AdamW at matched K (cross-cell)",
               ["negative = the filter wins; shown only once the merge gate holds."],
               [(label(cell, k), compare(seeds_of(ONEHOP, label(cell, k)),
                                         seeds_of(ATC_ADAMW, label(cell, k))))
                for cell in CELLS for k in KS])
    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
