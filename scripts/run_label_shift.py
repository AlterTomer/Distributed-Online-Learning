r"""P5.8 -- label shift: the other row where per-agent beliefs could beat a pooled one.

    python scripts/run_label_shift.py --lr          # tune SGD and AdamW on the shift, first
    python scripts/run_label_shift.py               # the shifted cells and their twins
    python scripts/run_label_shift.py --report-only
    python scripts/run_label_shift.py --lr --smoke  # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

## The question (phase5_plan.md P5.8, the X10 analogue)

Each agent's class distribution travels from uniform to its own Dirichlet(0.5)
draw, rotation held at zero, and every agent is scored on the mix it actually
faces (composition-matched evaluation). The centralised filter pools every agent's
batch and so tracks the *network's* mix, which no agent sees; a diffusion agent
could in principle hold a belief shaped by its own. That is why the plan lists
this beside P5.7 as the second place a distributed filter might **beat** the
pooled one. X10 found the SGD picture: label shift makes cooperation *more*
valuable, not less -- `local_only` pays 3.5x everyone else's damage and the
cooperation gap widens by +0.0445 -- because an agent alone loses the classes its
stream stops delivering while its neighbours still see them.

## Named before the run (confirmatory under D118)

1. **Does diffusion close its gap to the centralised filter under label shift?**
   Each diffusion variant's gap (diffusion - centralised), shifted minus twin, per
   seed; Holm across the four. **Predicted: it does not** -- P5.7 found diffusion
   agents never leave consensus, so they cannot personalise (D115), and the same
   mechanism should hold whichever channel the agents differ in. The plan's hope
   is recorded as the alternative this tests.
2. **Does cooperation pay more for the filter under label shift, as it did for
   SGD?** `local_only` minus each mean-only filter, shifted minus twin, per seed;
   Holm across the two. **Predicted positive** (X10: +0.0445 for ATC).

Everything else -- full sharing, the AdamW family, the damage table, the
centralised filter's own damage against centralised SGD's -- is exploratory.

## The design

X10's configs unchanged in what they drift: `mnist_prior_drift` (Dirichlet 0.5,
`total_shift` 1, rotation 0) against the twin at `total_shift` 0 -- the class-prior
machinery running but travelling nowhere, so the partition and the sampling are
the same mechanism in both (X10's own reason). $T=750$ is forced, not chosen: at
the usual $NnT=60\,000$ a non-uniform demand exceeds the class pools, and the
config's feasibility check refuses it. ER 0.3, $N=10$, five seeds, evaluations
every 10 steps, settled over the last 20%. **The baselines are tuned once, on the
shifted condition, and the same rates carried into the twin**: a shifted cell and
its twin must differ in the prior drift and nothing else, or the damage -- and the
cooperation contrast of question 2, which runs through `local_only` -- mixes the
label shift with a change of learning rate. The first smoke tuned per condition and
the pairing gate refused it (`local_only` at 0.2 against 0.05); P5.5 made the same
choice for the same reason. The filters carry X20's selection; cells a / b / adamw
with D119's merge gate; the report gates the pairing (`assert_paired_runs`).
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
from run_diffusion_skew import BASELINES, CENTRALIZED, FILTER, LR_SEEDS, settled  # noqa: E402
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
from dekf_bench.metrics.paired import differences, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: The shifted cell and its twin: X10's two configs, differing in total_shift alone.
CONDITIONS = {"shift": ("x10_prior_drift", 1.0), "control": ("x10_control", 0.0)}
#: The one condition the baselines are tuned on; the twin carries its rates.
TUNED_ON = "shift"

CENTRAL = "centralized_ekf_gamma"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
MEAN_ONLY = [LOCAL, ONEHOP]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
DIFFUSION = [*MEAN_ONLY, *FULL_SHARING]
GROUPS = ["a", "b", ADAMW_GROUP]

HORIZON, SEEDS, EVAL_EVERY = 750, [0, 1, 2, 3, 4], 10
DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "lsh_status.json"


def lr_run_name(condition: str, rate: float, suffix: str = "", family: str = "sgd") -> str:
    stem = "lsh_lr_adamw" if family == "adamw" else "lsh_lr"
    return f"{stem}_{condition}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(condition: str, group: str, suffix: str = "") -> str:
    return f"lsh_{condition}_{group}{suffix}"


def selected_rates(condition: str, suffix: str = "") -> dict[str, float] | None:
    rates: dict[str, float] = {}
    for name in tuned_learners():
        scored = [(settled(lr_run_name(condition, r, suffix, family_of(name)), name), r)
                  for r in grid_for(name)]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, condition: str, entries: list[dict],
               seeds: list[int] | None = None):
    experiment, shift = CONDITIONS[condition]
    return load_config(
        experiment,
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
            "env": {"dataset": args.dataset, "prior_drift": {"total_shift": shift}},
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
    """SGD and AdamW per condition, each on its own grid; the filters carry X20's."""
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    grids = {family: sorted({r for b in tuned_learners() if family_of(b) == family
                             for r in grid_for(b)}, reverse=True)
             for family in ("sgd", "adamw")}
    # Tuned on the shift alone: the twin carries the same rates (see the docstring).
    cells = [(c, f, r) for c in (TUNED_ON,) for f, grid in grids.items() for r in grid]
    print(f"P5.8 lr{' SMOKE' if suffix else ''}: the shifted condition x "
          f"({len(grids['sgd'])} SGD + {len(grids['adamw'])} AdamW) rates at "
          f"{len(seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (condition, family, rate) in enumerate(cells, start=1):
        name = lr_run_name(condition, rate, suffix, family)
        entries = [{"name": b, "lr": rate, **o} for b, o in tuned_learners().items()
                   if family_of(b) == family and rate in grid_for(b)]
        note = run_one(config_for(args, name, condition, entries, seeds=seeds),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<34} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    names = list(tuned_learners())
    print(f"  {'condition':>10}" + "".join(f"{n:>26}" for n in names))
    for condition in (TUNED_ON,):
        rates = selected_rates(condition, suffix)
        if rates:
            edge = [n for n in names if rates[n] in (grid_for(n)[0], grid_for(n)[-1])]
            print(f"  {condition:>10}" + "".join(f"{rates[n]:>26g}" for n in names)
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
    if args.lr:
        return tune(args, train, test, suffix)

    unswept = sorted({lr_run_name(c, r, suffix, family_of(b)) for c in (TUNED_ON,)
                      for b in tuned_learners() for r in grid_for(b)
                      if not any((ROOT / "results" / lr_run_name(c, r, suffix, family_of(b))
                                  / m).exists() for m in ("_complete", "_diverged"))})
    if unswept:
        print(f"  REFUSED: {len(unswept)} lr cell(s) never ran, e.g. {unswept[0]}.")
        print(f"  Run --lr{' --smoke' if suffix else ''} first; completed cells are cached.")
        return 1
    edges = [f"{c}: {b} at {r:g}" for c in (TUNED_ON,)
             for b, r in selected_rates(c, suffix).items()
             if r in (grid_for(b)[0], grid_for(b)[-1])]
    if edges:
        print("  ⚠ selected rates on their grid's edge: " + "; ".join(edges) + "\n")

    cells = [(condition, group) for group in GROUPS for condition in CONDITIONS]
    print(f"P5.8{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds, "
          f"T={args.horizon}\n", flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (condition, group) in enumerate(cells, start=1):
        name = cell_name(condition, group, suffix)
        entries = entries_for(group, selected_rates(TUNED_ON, suffix))
        note = run_one(config_for(args, name, condition, entries), train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<24} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nP5.8{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _cell_of(learner: str, condition: str, suffix: str = "") -> str:
    if learner in FULL_SHARING:
        return cell_name(condition, "b", suffix)
    if learner in ADAMW:
        return cell_name(condition, ADAMW_GROUP, suffix)
    return cell_name(condition, "a", suffix)


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
    from dekf_bench.metrics.breaks import BreakError, assert_paired_runs  # noqa: PLC0415

    seeds_of = lambda learner, c: per_seed(_cell_of(learner, c, suffix), learner)  # noqa: E731
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 rounds at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")

    print("  pairing gate: each shifted cell against its twin (only the prior drift may differ)")
    for group in GROUPS:
        try:
            assert_paired_runs(ROOT / "results" / cell_name("shift", group, suffix),
                               ROOT / "results" / cell_name("control", group, suffix))
            print(f"    {group:<6} paired")
        except BreakError as failure:
            print(f"    {group:<6} NOT PAIRED: {str(failure)[:90]}")

    print("\n  merge gate: centralized_sgd in the AdamW cells against cell a, per seed")
    poolable = True
    for condition in CONDITIONS:
        recorded = per_seed(cell_name(condition, "a", suffix), REPRODUCTION_ARM)
        rerun = per_seed(cell_name(condition, ADAMW_GROUP, suffix), REPRODUCTION_ARM)
        shared = sorted(set(recorded) & set(rerun))
        worst = max((abs(recorded[s] - rerun[s]) for s in shared), default=None)
        ok = worst is not None and worst <= REPRODUCTION_TOLERANCE
        poolable &= ok
        shown = f"{worst:.1e}" if worst is not None else "-"
        print(f"    {condition:<8} {len(shared)} seed(s)  max |diff| {shown:>8}   "
              f"{'reproduces' if ok else 'not run' if worst is None else 'DOES NOT REPRODUCE'}")

    every = [CENTRAL, *DIFFUSION, *tuned_baselines(), *ADAMW]
    print("\n  settled error, composition-matched current set; damage = shifted - twin\n")
    print(f"    {'learner':<36}{'twin':>9}{'shifted':>9}{'damage':>9}")
    for learner in every:
        twin, shifted = seeds_of(learner, "control"), seeds_of(learner, "shift")
        damage = differences(shifted, twin)
        print(f"    {learner:<36}{mean(twin):>9.4f}{mean(shifted):>9.4f}{mean(damage):>+9.4f}")

    def gap_change(first: str, second: str) -> object:
        """(first - second) shifted minus the same in the twin, per seed (D54)."""
        shifted = differences(seeds_of(first, "shift"), seeds_of(second, "shift"))
        twin = differences(seeds_of(first, "control"), seeds_of(second, "control"))
        return compare(shifted, twin) if shifted and twin else None

    _table("Q1 [confirmatory]: does diffusion close its gap to centralised under label shift?",
           ["(diffusion - centralised) shifted minus twin; negative = the gap closes.",
            "Predicted: it does not (D115: diffusion agents never leave consensus)."],
           [(name, gap_change(name, CENTRAL)) for name in DIFFUSION])
    _table("Q2 [confirmatory]: does cooperation pay more for the filter under label shift?",
           ["(local_only - filter) shifted minus twin; positive = cooperation pays more.",
            "Predicted positive, as X10 found for ATC (+0.0445)."],
           [(name, gap_change("local_only", name)) for name in MEAN_ONLY])

    # ---- exploratory -------------------------------------------------------------
    _table("exploratory: cooperation for the gradient methods, the X10 contrast",
           ["(local_only - learner) shifted minus twin."],
           [(name, gap_change("local_only", name))
            for name in ("diffusion_sgd_atc", "diffusion_sgd_atc_plain")])
    _table("exploratory: the centralised filter's damage against centralised SGD's",
           ["(filter - SGD) shifted minus twin; negative = the filter is hurt less."],
           [("centralised EKF - centralised SGD", gap_change(CENTRAL, "centralized_sgd"))])
    _table("exploratory: ATC AdamW's gap to centralised AdamW, shifted minus twin",
           ["within the AdamW cells."],
           [("ATC AdamW - centralised AdamW",
             gap_change("diffusion_atc_adamw", "centralized_adamw"))])
    if poolable:
        _table("exploratory: one-hop minus ATC AdamW, in the shifted cells",
               ["negative = the filter wins; cross-cell, so only once the gate holds."],
               [("one-hop - ATC AdamW",
                 compare(seeds_of(ONEHOP, "shift"), seeds_of("diffusion_atc_adamw", "shift")))])

    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
