r"""P5.4 -- sparse labels, n x pi_lab: what an unlabelled agent's belief does.

    python scripts/run_label_sparsity.py --lr          # tune SGD and AdamW per cell, first
    python scripts/run_label_sparsity.py               # the cells themselves
    python scripts/run_label_sparsity.py --report-only
    python scripts/run_label_sparsity.py --lr --smoke  # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

## The question (phase5_plan.md P5.4, the X4 analogue)

At $\pi_{\text{lab}}<1$ an agent has labels on only a fraction of steps. An
unlabelled agent passes its prediction through and still takes part in the
combine -- one of diffusion's more attractive properties, never tested for a
filter. And a *belief* between labels is not a parameter that has stopped: its
covariance keeps growing under the time update, so the next labelled step moves
it further. X4 found the sparse corner is where SGD's orderings bend (ATC even
beat pooling there, mostly an effective-step artefact, D-X4 in results.md §9).

**What sparsity does to one-hop, worked out rather than guessed.** One-hop
assimilates about $|\mathcal M_v|\approx3.9$ times local adapt's labelled data at
*any* $\pi_{\text{lab}}$ -- both scale by $\pi_{\text{lab}}$ -- so the information
ratio does not move. What moves is coverage: at $\pi_{\text{lab}}=0.25$ a
local-adapt agent updates on 25% of steps, a one-hop agent on
$1-0.75^{3.9}\approx67\%$ (some neighbour is labelled), so one-hop keeps updating
while local adapt mostly predicts and combines. Hence the prediction below:
one-hop's lead holds or widens as labels thin, not the reverse (the first sketch of
this said "shrinks"; the arithmetic above corrected it before anything ran).

## Named before the run (confirmatory under D118)

1. **Does one-hop's lead over local adapt change as labels thin?**
   `one-hop - local adapt`, the change from $\pi_{\text{lab}}=1$ to $0.25$, per seed,
   at each $n$; Holm across the three $n$. Predicted: negative or null -- the lead
   holds or widens.
2. **Does decentralisation cost more when labels are sparse?** Each mean-only
   filter's gap to the centralised filter, the change from $\pi_{\text{lab}}=1$ to
   $0.25$, at each $n$; Holm across the six (two filters x three $n$). No direction
   predicted: X4 found the pooled gap *falling* for ATC in the sparse corner, for
   reasons (effective step, implicit averaging) that need not carry over.
3. **Does a belief bear sparsity differently from a parameter?**
   `diff-EKF local adapt - ATC`, the change from $\pi_{\text{lab}}=1$ to $0.25$, at
   each $n$; Holm across the three. No direction predicted: the belief's growing
   covariance may make each labelled step count for more, or make the agent chase
   noise.

Everything else printed -- full sharing, the AdamW family, cooperation against
local-only, $\pi_{\text{lab}}=0.5$, and the belief's $\kappa^\star$ -- is exploratory.

## The design

$n\in\{1,2,4\}\times\pi_{\text{lab}}\in\{0.25,0.5,1\}$, nine cells per group
(decided with the user 2026-09-27). IID shards, ER 0.3, $N=10$, stationary,
$T=750$ -- X4's horizon -- five seeds, evaluations every 10 steps, settled over the
last 20%. The data budget is $NnT\pi_{\text{lab}}\le 60\,000$; the worst cell uses
30 000. The filters carry X20's selection (tuned at $n=4$, $\pi_{\text{lab}}=1$), and
that is a caveat rather than an oversight: re-tuning a filter per cell is what
this budget cannot afford, so any gap that opens in the sparse corner is read
beside the knowledge that the filter was tuned for the dense one. The gradient
baselines, SGD and AdamW, are re-tuned per cell -- X4 showed their optimal rate
moves by 4x across $\pi_{\text{lab}}$, which is exactly the artefact that must not
masquerade as a finding.

Cells as the calibration runner's: **a** -- the centralised filter, both mean-only
diffusion filters, the SGD baselines; **b** -- both full-sharing filters; **adamw**
-- the AdamW arms with `centralized_sgd` re-run as the merge gate (D119). Beliefs
are scored over the settled window (D120), for the exploratory $\kappa^\star$.
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
from run_belief_calibration import seed_means  # noqa: E402
from run_diffusion_skew import (  # noqa: E402
    BASELINES,
    CENTRALIZED,
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
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import differences, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

SAMPLES = [1, 2, 4]
AVAILABILITY = [0.25, 0.5, 1.0]
#: (label, n, pi_lab), dense to sparse within each n.
CONDITIONS: list[tuple[str, int, float]] = [
    (f"n{n}_p{int(round(pi * 100))}", n, pi) for n in SAMPLES for pi in reversed(AVAILABILITY)
]
DENSE, SPARSE = 1.0, 0.25

CENTRAL = "centralized_ekf_gamma"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
MEAN_ONLY = [LOCAL, ONEHOP]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
ATC = "diffusion_sgd_atc"
GROUPS = ["a", "b", ADAMW_GROUP]

N_AGENTS = 10
MNIST_TRAIN = 60_000
HORIZON, SEEDS, EVAL_EVERY = 750, [0, 1, 2, 3, 4], 10
SMOKE_SUBSET = 64

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "lab_status.json"

#: Every SGD learner is swept to 1.0 here, as X4's own grid was. An idle agent
#: contributes its unchanged theta to the combine, so ATC's effective step shrinks
#: by pi_lab and its optimum rises about 4x at pi_lab = 0.25 (results.md 9.1) --
#: a grid stopping at 0.2 would put that optimum on its edge. AdamW keeps N>10's.
SGD_EXTRA = [1.0, 0.5]


def grid_for(name: str) -> list[float]:
    """This learner's own grid, largest rate first."""
    if family_of(name) == "adamw":
        from run_network_size import grid_for as adamw_grid  # noqa: PLC0415
        return adamw_grid(name)
    return sorted({*LEARNING_RATES, *SGD_EXTRA}, reverse=True)


def label(n: int, pi: float) -> str:
    return f"n{n}_p{int(round(pi * 100))}"


def lr_run_name(cond: str, rate: float, suffix: str = "", family: str = "sgd") -> str:
    stem = "lab_lr_adamw" if family == "adamw" else "lab_lr"
    return f"{stem}_{cond}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(cond: str, group: str, suffix: str = "") -> str:
    return f"lab_{cond}_{group}{suffix}"


def selected_rates(cond: str, suffix: str = "") -> dict[str, float] | None:
    """Each tuned learner's own argmin in this cell, or None when any is unswept."""
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


def config_for(args, name: str, n: int, pi: float, entries: list[dict],
               seeds: list[int] | None = None, belief: bool = True, subset: int = 1000):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
            "env": {"dataset": args.dataset, "samples_per_node_per_step": n,
                    "label_availability": pi,
                    "drift": {"schedule": "stationary", "total_degrees": 0.0}},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"], "belief_calibration": belief,
                     "belief_subset": subset},
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


def preflight(args) -> bool:
    """The data budget at pi_lab = 1, the worst case of every n: N n T <= 60 000 (D5)."""
    ok = True
    print("  pre-flight: data budget N n T pi_lab (expected; at most N n T)")
    for cond, n, pi in CONDITIONS:
        worst = N_AGENTS * n * args.horizon
        within = worst <= MNIST_TRAIN
        ok &= within
        print(f"    {cond:<9} n={n} pi={pi:<4}  expected {N_AGENTS * n * args.horizon * pi:>7.0f}"
              f"  at most {worst:>6}  {'ok' if within else 'OVER BUDGET'}")
    return ok


def tune(args, train, test, suffix: str = "") -> int:
    """SGD and AdamW per cell, each on its own grid and names; the filters carry X20's."""
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    grids = {family: sorted({r for b in tuned_learners() if family_of(b) == family
                             for r in grid_for(b)}, reverse=True)
             for family in ("sgd", "adamw")}
    cells = [(cond, n, pi, f, r) for cond, n, pi in CONDITIONS
             for f, grid in grids.items() for r in grid]
    print(f"P5.4 lr{' SMOKE' if suffix else ''}: {len(CONDITIONS)} cells x "
          f"({len(grids['sgd'])} SGD + {len(grids['adamw'])} AdamW) rates at "
          f"{len(seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (cond, n, pi, family, rate) in enumerate(cells, start=1):
        name = lr_run_name(cond, rate, suffix, family)
        entries = [{"name": b, "lr": rate, **o} for b, o in tuned_learners().items()
                   if family_of(b) == family and rate in grid_for(b)]
        note = run_one(config_for(args, name, n, pi, entries, seeds=seeds, belief=False),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<38} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    names = list(tuned_learners())
    print(f"  {'cell':>10}" + "".join(f"{b:>26}" for b in names))
    for cond, _n, _p in CONDITIONS:
        rates = selected_rates(cond, suffix)
        if rates:
            edge = [b for b in names if rates[b] in (grid_for(b)[0], grid_for(b)[-1])]
            print(f"  {cond:>10}" + "".join(f"{rates[b]:>26g}" for b in names)
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

    unswept = sorted({lr_run_name(c, r, suffix, family_of(b)) for c, _n, _p in CONDITIONS
                      for b in tuned_learners() for r in grid_for(b)
                      if not any((ROOT / "results" / lr_run_name(c, r, suffix, family_of(b))
                                  / m).exists() for m in ("_complete", "_diverged"))})
    if unswept:
        print(f"  REFUSED: {len(unswept)} lr cell(s) never ran, e.g. {unswept[0]}.")
        print(f"  Run --lr{' --smoke' if suffix else ''} first; completed cells are cached.")
        return 1
    edges = [f"{c}: {b} at {r:g}" for c, _n, _p in CONDITIONS
             for b, r in selected_rates(c, suffix).items()
             if r in (grid_for(b)[0], grid_for(b)[-1])]
    if edges:
        print("  ⚠ selected rates on their grid's edge -- the optimum may lie outside it:")
        for edge in edges:
            print(f"    {edge}")
        print()

    subset = SMOKE_SUBSET if suffix else 1000
    cells = [(cond, n, pi, group) for cond, n, pi in CONDITIONS for group in GROUPS]
    print(f"P5.4{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds, "
          f"T={args.horizon}\n", flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (cond, n, pi, group) in enumerate(cells, start=1):
        name = cell_name(cond, group, suffix)
        entries = entries_for(group, selected_rates(cond, suffix))
        note = run_one(config_for(args, name, n, pi, entries, subset=subset),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<26} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nP5.4{' smoke' if suffix else ''} complete in "
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
    """The merge gate, the settled table, the three named questions, then exploratory."""
    seeds_of = lambda learner, cond: per_seed(_cell_of(learner, cond, suffix), learner)  # noqa: E731
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 rounds at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")

    print("  merge gate: centralized_sgd in the AdamW cell against cell a, per seed")
    poolable = True
    for cond, _n, _p in CONDITIONS:
        recorded = per_seed(cell_name(cond, "a", suffix), REPRODUCTION_ARM)
        rerun = per_seed(cell_name(cond, ADAMW_GROUP, suffix), REPRODUCTION_ARM)
        shared = sorted(set(recorded) & set(rerun))
        worst = max((abs(recorded[s] - rerun[s]) for s in shared), default=None)
        ok = worst is not None and worst <= REPRODUCTION_TOLERANCE
        poolable &= ok
        shown = f"{worst:.1e}" if worst is not None else "-"
        print(f"    {cond:<9} {len(shared)} seed(s)  max |diff| {shown:>8}   "
              f"{'reproduces' if ok else 'not run' if worst is None else 'DOES NOT REPRODUCE'}")

    labels = [cond for cond, _n, _p in CONDITIONS]
    every = [CENTRAL, *MEAN_ONLY, *FULL_SHARING, *tuned_baselines(), *ADAMW]
    print("\n  settled error (last 20%, current set)\n")
    print(f"    {'learner':<36}" + "".join(f"{c:>9}" for c in labels))
    for learner in every:
        print(f"    {learner:<36}" + "".join(
            f"{mean(seeds_of(learner, c)):>9.4f}" for c in labels))

    def change(first: str, second: str, n: int) -> object:
        """(first - second) at pi = 0.25 minus at pi = 1, per seed, at this n."""
        sparse = differences(seeds_of(first, label(n, SPARSE)), seeds_of(second, label(n, SPARSE)))
        dense = differences(seeds_of(first, label(n, DENSE)), seeds_of(second, label(n, DENSE)))
        return compare(sparse, dense) if sparse and dense else None

    _table("Q1 [confirmatory]: one-hop minus local adapt, change from pi_lab 1 to 0.25",
           ["negative = one-hop's lead widens as labels thin. Predicted negative or null."],
           [(f"n={n}", change(ONEHOP, LOCAL, n)) for n in SAMPLES])
    _table("Q2 [confirmatory]: gap to the centralised filter, change from pi_lab 1 to 0.25",
           ["positive = decentralisation costs more when labels are sparse. No prediction."],
           [(f"{f} n={n}", change(f, CENTRAL, n)) for f in MEAN_ONLY for n in SAMPLES])
    _table("Q3 [confirmatory]: diff-EKF local adapt minus ATC, change from pi_lab 1 to 0.25",
           ["negative = the belief bears sparsity better than the parameter. No prediction."],
           [(f"n={n}", change(LOCAL, ATC, n)) for n in SAMPLES])

    # ---- exploratory -------------------------------------------------------------
    _table("exploratory: full minus mean-only sharing, change from pi_lab 1 to 0.25",
           ["negative = sharing the covariance helps more when labels are sparse."],
           [(f"{full} n={n}", change(full, base, n)) for full, base
            in zip(FULL_SHARING, MEAN_ONLY, strict=True) for n in SAMPLES])
    _table("exploratory: ATC AdamW minus centralised AdamW, change from pi_lab 1 to 0.25",
           ["within the AdamW cell; positive = decentralisation costs more when sparse."],
           [(f"n={n}", change("diffusion_atc_adamw", "centralized_adamw", n)) for n in SAMPLES])
    if poolable:
        _table("exploratory: one-hop minus ATC AdamW, per cell (3 696 vs 8 724 scalars)",
               ["negative = the filter wins; cross-cell, so shown only once the gate holds."],
               [(c, compare(seeds_of(ONEHOP, c), seeds_of("diffusion_atc_adamw", c)))
                for c in labels])
    print("\n  exploratory: the belief's kappa* after combine (and before, for diffusion),")
    print("  geometric mean over agents and settled steps (D120); 0.0001 = floored at 0\n")
    print(f"    {'filter':<36}" + "".join(f"{c:>9}" for c in labels))
    for learner in [CENTRAL, *MEAN_ONLY]:
        for metric in (("kappa_star",) if learner == CENTRAL
                       else ("kappa_star", "pre_kappa_star")):
            values = [10 ** mean(seed_means(_cell_of(learner, c, suffix), learner, metric,
                                            log_kappa=True)[0]) for c in labels]
            tag = learner + (" (pre)" if metric.startswith("pre_") else "")
            print(f"    {tag:<36}" + "".join(f"{v:>9.3g}" for v in values))

    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
