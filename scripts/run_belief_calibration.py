r"""P5.11 / P5.14 -- is the filters' belief calibrated, and is lem:conservative tight?

    python scripts/run_belief_calibration.py --lr        # tune SGD and AdamW, first
    python scripts/run_belief_calibration.py             # the cells themselves
    python scripts/run_belief_calibration.py --report-only
    python scripts/run_belief_calibration.py --lr --smoke   # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

## The two questions (phase5_plan.md)

**P5.11 -- does the belief stay calibrated?** Every calibration number so far was
plug-in: softmax of the mean, which any point estimator has (D80). The filter's
claim is that its covariance knows what its mean does not, and this is the first
run that scores it, through the predictive $\int\operatorname{softmax}(\bm h)\,
\mathcal N(\bm h;\bm h(\bm m),\kappa\bm H\bm P\bm H^{\mathsf T})$ at $\kappa=1$, and
through $\kappa^\star$, the covariance scale that minimises its held-out NLL
(`metrics/belief.py`): 1 is calibrated, below 1 conservative, above 1
over-confident.

**P5.14 -- is the conservative bound conservative in practice?**
`lem:conservative` says $\sum_u a_{vu}\bm P^{\psi}_u$ bounds the combined error
covariance for any cross-correlation, with equality when the neighbours' errors
coincide. So each diffusion filter is scored twice at every settled evaluation:
**before combine**, on $(\bm\psi_v,\bm P^{\psi}_v)$, and **after**, on
$(\bm m_v,\bm P_v)$. If the agents' errors were independent, averaging would shrink
the error by about $|\mathcal M_v|$ ($\approx4$ here) while the covariance stayed put,
and $\kappa^\star$ would fall by that factor across the combine. If they coincide it
does not move. D88 answered this indirectly, and for full sharing only ($\beta$ is
inert under mean-only sharing, D85); this measures it directly, for all four
variants.

## Named before the run (confirmatory under D118)

1. **Does scoring the belief beat the plug-in prediction?** `belief_nll - plugin_nll`
   on the same images, per filter, per condition, Holm across the five filters.
   Predicted: negative for the centralised EKF. No direction is predicted for the
   diffusion filters: D80 found their plug-in already *under*-confident, and adding
   spread moves that the wrong way.
2. **Is the bound tight -- does $\kappa^\star$ stay put across the combine?**
   $\log_{10}\kappa^\star_{\text{pre}}-\log_{10}\kappa^\star_{\text{post}}$ per
   diffusion filter, per condition. Predicted: tight. A claim of sameness, so it is
   tested by TOST at $\pm$`TIGHTNESS_MARGIN` decades (a factor of 2) -- half the
   $\log_{10}|\mathcal M_v|\approx0.6$ that independent errors would give -- fixed
   here, before the run. Holm across the four.
3. **Is the pre-combine belief consistent -- the lemma's premise?**
   $\log_{10}\kappa^\star_{\text{pre}}$ against 0 per diffusion filter, and the
   centralised filter's own $\log_{10}\kappa^\star$ beside them; Holm across the
   five. No prediction.

**The degeneracy rule, fixed with the questions.** A covariance can only soften a
prediction, and every filter's plug-in mean is already under-confident here (by
0.016 to 0.040), so the raw $\kappa^\star$ may sit at 0 whatever $\bm P$ is. If it
does in more than `DEGENERATE_FRACTION` of a filter's scored evaluations, pooled
over seeds, in either stage a question reads, then questions 2 and 3 are
**undecidable for that filter on MNIST**: its row says so and spends no alpha,
and P5.14 rests on Mackey--Glass, whose Gaussian predictive has no such floor.
The tempered $\kappa^\star$ (`metrics/belief.py`) is printed beside it and is
exploratory throughout (decided 2026-09-27, D120).

Everything else printed -- ECE, the Monte Carlo check, the tempered $\kappa^\star$,
the plug-in table across the gradient baselines -- is exploratory.

## The design

IID shards, ER 0.3, $N=10$, $T=1500$, five seeds: the setting of X20 and of the
diffusion figures, stationary and X17's abrupt schedule (15 degrees every 25
steps). Evaluations every 10 steps, as N>10's -- every 25 would coincide with the
jumps and score only just-shifted states. The beliefs are scored at every full
evaluation over the settled window (the last 20%), on the first 1000 images of the
`current` set, every agent, every stage.

Cells: **a** -- the centralised filter, both mean-only diffusion filters, and the
SGD baselines; **b** -- both full-sharing filters, in their own process as ever;
**adamw** -- the AdamW arms, with `centralized_sgd` re-run as N>10's merge gate
(D119), since every figure the paper carries includes AdamW.
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
    BASELINES,
    CENTRALIZED,
    DRIFTS,
    FILTER,
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
    grid_for,
    tuned_baselines,
    tuned_learners,
)
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.belief import KAPPA_GRID  # noqa: E402
from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

CONDITIONS: dict[str, dict] = {
    "stationary": {"schedule": "stationary", "total_degrees": 0.0},
    "abrupt": dict(DRIFTS["abrupt"]),
}

CENTRAL = "centralized_ekf_gamma"
MEAN_ONLY = ["diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
DIFFUSION = [*MEAN_ONLY, *FULL_SHARING]
FILTERS = [CENTRAL, *DIFFUSION]

#: Question 2's equivalence margin, in decades of kappa*: a factor of 2 either way.
#: Half of log10 |M_v| ~ 0.6, what independent errors would give at mean degree 2.9.
TIGHTNESS_MARGIN = 0.3
#: kappa* = 0 (any spread hurts) has no logarithm. It is floored at the grid's
#: smallest positive point and counted, so a floored mean is visible as such.
KAPPA_FLOOR = min(k for k in KAPPA_GRID if k > 0.0)
#: Above this share of kappa* = 0, the raw reading says only that the mean is
#: under-confident, and questions 2 and 3 are undecidable for that filter.
DEGENERATE_FRACTION = 0.5

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 10
#: The smoke scores fewer images: the point is that every path runs, and the
#: Jacobians on the CPU dominate a twenty-step run otherwise.
SMOKE_SUBSET = 64

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "cal_status.json"


def lr_run_name(condition: str, rate: float, suffix: str = "", family: str = "sgd") -> str:
    stem = "cal_lr_adamw" if family == "adamw" else "cal_lr"
    return f"{stem}_{condition}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(condition: str, group: str, suffix: str = "") -> str:
    return f"cal_{condition}_{group}{suffix}"


GROUPS = ["a", "b", ADAMW_GROUP]


def selected_rates(condition: str, suffix: str = "") -> dict[str, float] | None:
    """Each tuned learner's own argmin in this condition, or None when any is unswept."""
    rates: dict[str, float] = {}
    for name in tuned_learners():
        family = family_of(name)
        scored = [(settled(lr_run_name(condition, r, suffix, family), name), r)
                  for r in grid_for(name)]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, condition: str, entries: list[dict],
               seeds: list[int] | None = None, belief: bool = True, subset: int = 1000):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
            "env": {"dataset": args.dataset, "drift": dict(CONDITIONS[condition])},
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


def tune(args, train, test, suffix: str = "") -> int:
    """SGD and AdamW, each on its own grid under its own names; the filters carry X20's."""
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    grids = {family: sorted({r for n in tuned_learners() if family_of(n) == family
                             for r in grid_for(n)}, reverse=True)
             for family in ("sgd", "adamw")}
    cells = [(c, f, r) for c in CONDITIONS for f, grid in grids.items() for r in grid]
    print(f"P5.11/14 lr{' SMOKE' if suffix else ''}: {len(CONDITIONS)} conditions x "
          f"({len(grids['sgd'])} SGD + {len(grids['adamw'])} AdamW) rates at "
          f"{len(seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (condition, family, rate) in enumerate(cells, start=1):
        name = lr_run_name(condition, rate, suffix, family)
        entries = [{"name": b, "lr": rate, **o} for b, o in tuned_learners().items()
                   if family_of(b) == family and rate in grid_for(b)]
        # No beliefs in the sweep: no filter is in it, and scoring is the costly part.
        note = run_one(config_for(args, name, condition, entries, seeds=seeds, belief=False),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<36} {note:<28} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    names = list(tuned_learners())
    print(f"  {'condition':>12}" + "".join(f"{n:>26}" for n in names))
    for condition in CONDITIONS:
        rates = selected_rates(condition, suffix)
        if rates:
            edge = [n for n in names if rates[n] in (grid_for(n)[0], grid_for(n)[-1])]
            print(f"  {condition:>12}" + "".join(f"{rates[n]:>26g}" for n in names)
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

    unswept = sorted({lr_run_name(c, r, suffix, family_of(b)) for c in CONDITIONS
                      for b in tuned_learners() for r in grid_for(b)
                      if not any((ROOT / "results" / lr_run_name(c, r, suffix, family_of(b))
                                  / m).exists() for m in ("_complete", "_diverged"))})
    if unswept:
        print(f"  REFUSED: {len(unswept)} lr cell(s) never ran, e.g. {unswept[0]}.")
        print(f"  Run --lr{' --smoke' if suffix else ''} first; completed cells are cached.")
        return 1
    edges = [f"{c}: {n} at {r:g}" for c in CONDITIONS
             for n, r in selected_rates(c, suffix).items()
             if r in (grid_for(n)[0], grid_for(n)[-1])]
    if edges:
        print("  ⚠ selected rates on their grid's edge -- the optimum may lie outside it:")
        for edge in edges:
            print(f"    {edge}")
        print()

    subset = SMOKE_SUBSET if suffix else 1000
    cells = [(condition, group) for condition in CONDITIONS for group in GROUPS]
    print(f"P5.11/14{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} "
          f"seeds, T={args.horizon}; beliefs scored on {subset} images over the last 20%\n",
          flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (condition, group) in enumerate(cells, start=1):
        name = cell_name(condition, group, suffix)
        entries = entries_for(group, selected_rates(condition, suffix))
        note = run_one(config_for(args, name, condition, entries, subset=subset),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<28} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nP5.11/14{' smoke' if suffix else ''} complete in "
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


def seed_means(run: str, learner: str, metric: str, log_kappa: bool = False
               ) -> tuple[dict[int, float], int, int]:
    """Per seed: the metric averaged over every agent and every scored step.

    ``log_kappa`` averages log10 kappa* instead -- a scale is averaged
    geometrically -- flooring kappa* = 0 at the grid's smallest positive point.
    Returns the per-seed means, how many values were floored, and how many there were.
    """
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    if not (directory / "_complete").exists():
        return {}, 0, 0
    means, floored, total = {}, 0, 0
    for path in sorted(directory.glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["learner", "metric", "evalset", "value"])
        values = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                       & (frame["evalset"] == "current")]["value"]
        if not len(values):
            continue
        if log_kappa:
            floored += int((values < KAPPA_FLOOR).sum())
            values = values.clip(lower=KAPPA_FLOOR).map(math.log10)
        total += len(values)
        means[int(path.stem.split("_")[1])] = float(values.mean())
    return means, floored, total


def _zeros(values: dict[int, float]) -> dict[int, float]:
    return {seed: 0.0 for seed in values}


def _table(title: str, lines: list[str], rows: list[tuple[str, object]]) -> None:
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
    print(f"\n    {'':<42}{STAT_HEADER}")
    live = [(label, result) for label, result in rows if result is not None]
    adjusted = dict(zip([label for label, _r in live], holm([r.p for _l, r in live]),
                        strict=True))
    for label, result in rows:
        if result is None:
            print(f"    {label:<42}{'-':>9}   (cells missing)")
        else:
            print(f"    {label:<42}{stat_columns(result, adjusted[label])}")


def report(suffix: str = "") -> None:
    """The merge gate, then the three named questions, then the exploratory tables."""
    cell = lambda learner, condition: _cell_of(learner, condition, suffix)  # noqa: E731
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 rounds at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")

    print("  merge gate: centralized_sgd in the AdamW cell against cell a, per seed")
    poolable = True
    for condition in CONDITIONS:
        recorded = per_seed(cell_name(condition, "a", suffix), REPRODUCTION_ARM)
        rerun = per_seed(cell_name(condition, ADAMW_GROUP, suffix), REPRODUCTION_ARM)
        shared = sorted(set(recorded) & set(rerun))
        worst = max((abs(recorded[s] - rerun[s]) for s in shared), default=None)
        ok = worst is not None and worst <= REPRODUCTION_TOLERANCE
        poolable &= ok
        shown = f"{worst:.1e}" if worst is not None else "-"
        print(f"    {condition:<11} {len(shared)} seed(s)  max |diff| {shown:>8}   "
              f"{'reproduces' if ok else 'not run' if worst is None else 'DOES NOT REPRODUCE'}")

    for condition in CONDITIONS:
        print(f"\n  ===== {condition} =====")

        # ---- plug-in, every learner: exploratory context ---------------------
        print("\n  plug-in calibration on the full `current` set, settled (exploratory)")
        print(f"    {'learner':<36}{'error':>9}{'NLL':>9}{'ECE':>9}{'overconf':>10}")
        learners = [*FILTERS, *tuned_baselines(), *ADAMW]
        for learner in learners:
            run = cell(learner, condition)
            values = [mean(per_seed(run, learner, metric))
                      for metric in ("error_rate", "nll", "ece", "overconfidence")]
            print(f"    {learner:<36}" + "".join(f"{v:>9.4f}" for v in values[:3])
                  + f"{values[3]:>+10.4f}")

        # ---- question 1 -------------------------------------------------------
        rows = []
        for learner in FILTERS:
            run = cell(learner, condition)
            belief = seed_means(run, learner, "belief_nll")[0]
            plugin = seed_means(run, learner, "plugin_nll")[0]
            rows.append((learner, compare(belief, plugin) if belief and plugin else None))
        _table("Q1 [confirmatory]: does scoring the belief beat the plug-in? belief - plugin NLL",
               ["same images, same agents, same steps; negative = the covariance helps.",
                "Predicted negative for the centralised filter; no prediction for diffusion."],
               rows)

        # ---- question 2 -------------------------------------------------------
        print("\n  Q2 [confirmatory]: is lem:conservative tight? log10 kappa*_pre - log10 kappa*_post")
        print("    TOST at +/-" f"{TIGHTNESS_MARGIN} decades (a factor of 2), fixed before the run;")
        print("    independent errors would give about +0.6. Holm across the four filters,")
        print("    on the difference test and, separately, on the TOST p.\n")
        print(f"    {'':<42}{'pre':>8}{'post':>8}{STAT_HEADER}   {'TOST p':>7}  verdict")
        results, undecidable = [], []
        for learner in DIFFUSION:
            run = cell(learner, condition)
            pre, pre_floored, pre_n = seed_means(run, learner, "pre_kappa_star", log_kappa=True)
            post, post_floored, post_n = seed_means(run, learner, "kappa_star", log_kappa=True)
            if not (pre and post):
                continue
            shares = (pre_floored / pre_n, post_floored / post_n)
            if max(shares) > DEGENERATE_FRACTION:
                undecidable.append((learner, shares))  # spends no alpha
                continue
            results.append((learner, pre, post, compare(pre, post)))
        p_diff = holm([r[3].p for r in results])
        p_tost = holm([r[3].tost_p(TIGHTNESS_MARGIN) for r in results])
        for (learner, pre, post, result), pd_, pt in zip(results, p_diff, p_tost, strict=True):
            verdict = ("TIGHT (shown within the margin)" if pt < ALPHA
                       else "LOOSE (the combine shrank the error)" if pd_ < ALPHA and result.mean > 0
                       else "neither shown")
            print(f"    {learner:<42}{mean(pre):>+8.2f}{mean(post):>+8.2f}"
                  f"{stat_columns(result, pd_)}   {pt:>7.3f}  {verdict}")
        for learner, (pre_share, post_share) in undecidable:
            print(f"    {learner:<42}UNDECIDABLE on MNIST: kappa* at 0 in {pre_share:.0%} (pre) "
                  f"and {post_share:.0%} (post) of evaluations")

        # ---- question 3 -------------------------------------------------------
        rows, skipped = [], []
        for learner in FILTERS:
            metric = "kappa_star" if learner == CENTRAL else "pre_kappa_star"
            values, floored, n = seed_means(cell(learner, condition), learner, metric,
                                            log_kappa=True)
            label = f"{learner} ({'post' if learner == CENTRAL else 'pre'})"
            if n and floored / n > DEGENERATE_FRACTION:
                skipped.append((label, floored / n))
                continue
            rows.append((label, compare(values, _zeros(values)) if values else None))
        _table("Q3 [confirmatory]: is the belief the lemma starts from consistent? log10 kappa*",
               ["against 0 (kappa* = 1); negative = conservative, positive = over-confident.",
                "The centralised filter has no combine, so its own belief stands in. No prediction."],
               rows)
        for label, share in skipped:
            print(f"    {label:<42}UNDECIDABLE on MNIST: kappa* at 0 in {share:.0%} of evaluations")

        # ---- exploratory ------------------------------------------------------
        print("\n  exploratory: the belief scored three ways, and kappa* in plain units")
        print(f"    {'filter':<36}{'plug-in':>9}{'probit':>9}{'MC':>9}{'ECE pl.':>9}"
              f"{'ECE pr.':>9}{'k* post':>9}{'k* pre':>9}")
        for learner in FILTERS:
            run = cell(learner, condition)
            nll = [mean(seed_means(run, learner, m)[0])
                   for m in ("plugin_nll", "belief_nll", "belief_nll_mc",
                             "plugin_ece", "belief_ece")]
            post = mean(seed_means(run, learner, "kappa_star", log_kappa=True)[0])
            pre = (mean(seed_means(run, learner, "pre_kappa_star", log_kappa=True)[0])
                   if learner != CENTRAL else float("nan"))
            print(f"    {learner:<36}" + "".join(f"{v:>9.4f}" for v in nll)
                  + f"{10 ** post:>9.3g}{10 ** pre:>9.3g}")

        print("\n  exploratory: the tempered reading -- tau* calibrates the plug-in mean first,")
        print("  then kappa* scales the spread on the tempered predictive (D120). tau* < 1")
        print("  sharpens an under-confident mean. Pre minus post in log10, paired per seed.\n")
        print(f"    {'filter':<36}{'tau*':>8}{'k~ post':>9}{'k~ pre':>9}{'log10 pre-post':>16}")
        for learner in FILTERS:
            run = cell(learner, condition)
            tau = mean(seed_means(run, learner, "temperature_star")[0])
            post = seed_means(run, learner, "tempered_kappa_star", log_kappa=True)[0]
            pre = (seed_means(run, learner, "pre_tempered_kappa_star", log_kappa=True)[0]
                   if learner != CENTRAL else {})
            change = compare(pre, post) if pre and post else None
            shown = (f"{change.mean:>+9.2f} (t {change.t:+.2f})" if change is not None
                     else f"{'-':>16}")
            print(f"    {learner:<36}{tau:>8.3f}{10 ** mean(post):>9.3g}"
                  f"{10 ** mean(pre) if pre else float('nan'):>9.3g}{shown:>16}")

    if not poolable:
        print("\n  ⚠ The AdamW cells fail (or have not reached) the merge gate: their plug-in")
        print("  rows above are within-cell only.")
    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113). kappa* is averaged")
    print("  geometrically over agents and settled steps, per seed, before any test.")


if __name__ == "__main__":
    raise SystemExit(main())
