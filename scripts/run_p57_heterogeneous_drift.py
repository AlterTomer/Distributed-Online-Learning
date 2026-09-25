r"""P5.7 -- heterogeneous drift: the row where per-agent beliefs could beat a pooled one.

    python scripts/run_p57_heterogeneous_drift.py --lr      # re-tune the baselines, first
    python scripts/run_p57_heterogeneous_drift.py           # the cells themselves
    python scripts/run_p57_heterogeneous_drift.py --report-only
    python scripts/run_p57_heterogeneous_drift.py --lr --smoke   # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

``--smoke`` runs a 20-step horizon at one seed and suffixes **every** run name with
`_smoke`. The suffix is the point, not the horizon: `run_one` caches by name, so a
smoke sharing the real names would be found complete by the real pass, which would
then select learning rates from a twenty-step run and never say so.

## The question

`phase5_plan.md` P5.7, the X8 analogue: agents drifting *differently* is the case
where per-agent beliefs should beat a pooled one. Under global drift every agent
faces the same rotation, so combining is pure variance reduction and diffusion can
essentially only help. Under per-node drift neighbours sit at different rotations,
so combining injects bias as well -- and the centralised filter, holding one shared
state, must average incompatible ones.

This is **one of only two rows in the plan where the diffusion filter could
plausibly beat the centralised reference** rather than approach it. Everywhere else
centralisation is the ceiling.

## The prediction, recorded before the run

**Diffusion closes or reverses its gap to centralised under per-node drift.** The
gap is $+0.0017$ to $+0.0021$ for one-hop on the image task's global-drift
conditions; if heterogeneity is what breaks a shared state, it should shrink here,
and the interesting outcome is a sign change.

There is now independent evidence for the mechanism from the *other* task. M8
([[D111]]) gave ten Mackey--Glass agents ten different delays and found the cost
falls on the filters and nothing else -- and the **centralised** filter, the one arm
that must pool incompatible states into a single vector, was hurt most (+0.0045,
$t=7.9$) while every gradient method paid nothing. That is P5.7's hypothesis,
measured on a different task, a different architecture and a different likelihood.

⚠ M8 also refuted the *reason* usually given for it. The cost there was **not**
inter-agent disagreement: `centralized_ekf_walk` holds one pooled vector, so its
`e_agree` is exactly zero, and it was still the most damaged arm. What moved was the
covariance -- error rose while coverage fell and predictive NLL worsened, so the
filter kept accumulating information at the same rate while its errors grew. If
P5.7 reproduces the effect, the same diagnostic applies: look at calibration, not at
consensus.

## The control matches the treatment's mean, not its cap

Under `per_node` each agent's rotation is scaled by a multiplier evenly spaced over
$[1-\text{spread}, 1]$, capped at 1 so the spread slows the laggards rather than
carrying the leaders past the 45-degree well-posedness cap (D21). At
`per_node_spread = 0.5` the multipliers average **0.75**, so the treatment's agents
end between 22.5 and 45 degrees with a mean of **33.75**.

The global control therefore runs at `total_degrees: 33.75`, not 45. X8 used 45 and
so compared heterogeneity *and* drift amount at once, a confound D53 caught only by
reading the results. **The tell was `frozen_atc`**: it stops adapting, so its error
tracks pure displacement, and it came out 0.10 *better* under per-node -- which no
story about heterogeneity explains and "less drift" explains exactly.

The match is checked twice, and neither check is "the t-test did not reject"
(D113):

1. **Exactly, before the run.** Mean-matching is arithmetic, so the pre-flight
   checks it as arithmetic: the per-node agents' mean rotation must equal the
   control's rotation at every evaluated step, or the run is refused.
2. **Empirically, after it, by equivalence test.** `frozen_atc` is carried because
   it never adapts, so its error is a read-out of displacement. It is read at the
   *same* rotation in both cells -- the per-node cell's `current_mean` against the
   global cell's `current` -- and must be shown equal within a stated margin
   (TOST). Reading the per-node agents at their own rotations instead would add a
   Jensen term: error is not linear in displacement, so agents spread around a
   mean average worse than one agent at it, even when the means match exactly.

## What is carried and what is re-tuned

The **filter** carries X20's selection unchanged, confirmed jointly by X23 -- the
X14 discipline, so a shortfall is attributable to the drift scope rather than
confounded with tuning. The **gradient baselines are re-tuned per scope**: under
per-node drift the agents move at different rates, and a rate carried across
conditions is the mistake D77 exists to record.

⚠ No AdamW arms here, deliberately. Adding AdamW to the image task is a horizontal
pass across every main cell with its own rate grid (the grid below is SGD-shaped);
doing it in one experiment first would leave the baseline set differing *between*
experiments, which is the asymmetry the pass exists to remove. See `schedule.md`.
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
from run_atc_plain import PLAIN  # noqa: E402
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

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import Paired, differences, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: The whole experiment: one treatment, one mean-matched control. `total_degrees`
#: differs *because* the scope does -- see the module docstring. Anything else
#: differing between these two rows is a bug.
PER_NODE_SPREAD = 0.5
FULL_ROTATION = 45.0
#: multipliers span [1 - spread, 1], so the mean is 1 - spread/2.
MATCHED_DEGREES = FULL_ROTATION * (1.0 - PER_NODE_SPREAD / 2.0)   # 33.75

SCOPES: list[tuple[str, str, float]] = [
    ("per_node", "per_node", FULL_ROTATION),
    ("global", "global", MATCHED_DEGREES),
]

#: Mean-only filters and every gradient baseline share a cell.
GROUP_A = ["diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
#: Full sharing gets its own process: X24 measured that variant at 3.43 GiB of 8.
GROUP_B = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]

#: Never adapts, so its error is pure displacement. The mean-matching gate.
DISPLACEMENT_TELL = "frozen_atc"

#: Family-wise, per table (Holm), and the size of each one-sided test in TOST.
ALPHA = 0.05

#: How far apart the two cells' `frozen_atc` may be, read at the same rotation,
#: and still count as matched. Half the smallest heterogeneity effect X8 resolved
#: (+0.0093 for diffusion_sgd_atc, +0.0095 for centralized_sgd): a mismatch below
#: it cannot manufacture or erase an effect of the size this design detects. Fixed
#: before the run, as an equivalence margin has to be; X8's own tell passes it.
GATE_MARGIN = 0.005

#: 25, the project default and what X8 used -- not the 5 this runner's template
#: carries. Under *global* drift a cadence of 5 costs 61 distinct rotations; under
#: per-node it costs 398, because each of the ten agents has its own. The template
#: was written for the cheap case.
HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 25

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "p57_status.json"


def lr_run_name(label: str, rate: float, suffix: str = "") -> str:
    return f"p57_lr_{label}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(label: str, group: str, suffix: str = "") -> str:
    return f"p57_{label}_{group}{suffix}"


def tuned_baselines() -> dict[str, dict]:
    """The three X25 baselines plus `atc_plain`, all re-tuned per scope here."""
    return {**BASELINES, PLAIN["name"]: {k: v for k, v in PLAIN.items() if k != "name"}}


def selected_rates(label: str, suffix: str = "") -> dict[str, float] | None:
    """Each baseline's own argmin at this scope, or None when it is unswept.

    ``suffix`` keeps a smoke run reading its own cells: selecting a rate from a
    twenty-step sweep would be invisible in every number that followed.
    """
    rates: dict[str, float] = {}
    for name in tuned_baselines():
        scored = [(settled(lr_run_name(label, r, suffix), name), r) for r in LEARNING_RATES]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, scope: str, degrees: float, entries: list[dict],
               seeds: list[int] | None = None):
    """One cell. The drift block and the scope move together and nothing else does."""
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
            "env": {
                "dataset": args.dataset,
                "drift_scope": scope,
                "drift": {"schedule": "linear", "total_degrees": degrees,
                          "per_node_spread": PER_NODE_SPREAD},
            },
            "learners": entries,
            # current-only: the question is tracking, not forgetting, and a canonical
            # set would score a distribution no agent has faced since step 0. Under
            # per-node drift `current_mean` is appended automatically, which separates
            # "this agent learned worse" from "this agent is further from the mean
            # rotation" -- so the treatment cell records one metric the control cannot.
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def entries_for(group: str, rates: dict[str, float]) -> list[dict]:
    if group == "b":
        return [{"name": n, **FILTER, "combine_exponent": 1.0} for n in GROUP_B]
    learners = [{"name": "centralized_ekf_gamma", **CENTRALIZED}]
    learners += [{"name": n, **FILTER} for n in GROUP_A]
    learners += [{"name": n, "lr": rates[n], **o} for n, o in tuned_baselines().items()]
    # Carried untuned on purpose: it never adapts, so a learning rate would change
    # nothing about it and a swept one would imply otherwise.
    learners.append({"name": DISPLACEMENT_TELL})
    return learners


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, train, test, suffix: str = "") -> int:
    """The gradient baselines only; the filter carries its own selection by design."""
    status = load_status()
    lr_seeds = [0] if suffix else LR_SEEDS
    total = len(SCOPES) * len(LEARNING_RATES)
    print(f"P5.7{' SMOKE' if suffix else ''} lr: {len(SCOPES)} scopes x "
          f"{len(LEARNING_RATES)} rates at {len(lr_seeds)} seeds\n", flush=True)
    started, index = time.time(), 0
    for label, scope, degrees in SCOPES:
        for rate in LEARNING_RATES:
            index += 1
            name = lr_run_name(label, rate, suffix)
            entries = [{"name": n, "lr": rate, **o} for n, o in tuned_baselines().items()]
            note = run_one(config_for(args, name, scope, degrees, entries, seeds=lr_seeds),
                           train, test, args.fresh)
            status[name] = note
            save_status(status)
            print(f"[{index}/{total}] {name:<34} {note:<28} "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'scope':>10}" + "".join(f"{n:>26}" for n in tuned_baselines()))
    for label, _s, _d in SCOPES:
        rates = selected_rates(label, suffix)
        if rates:
            print(f"  {label:>10}" + "".join(f"{rates[n]:>26g}" for n in tuned_baselines()))
    return 0


def preflight(args, test, with_filters: bool = True) -> bool:
    """Price the run before it starts. False refuses it.

    This exists because the first launch died out of memory halfway through its
    first seed, and every number needed to predict that was available beforehand.
    It is a cost predictor rather than a crash predictor now that the evalset
    cache is bounded: what it reports is how much recomputation the bound implies,
    and it refuses only when the part that *cannot* be evicted -- the covariances
    -- will not fit.

    The rotation count is measured, not multiplied, via the drift's own
    accounting. Linear rotations alias: the agent at multiplier 0.5 on step 2t
    sits at the same angle as the agent at 1.0 on step t, so the true count falls
    well below points x agents (398 rather than 610 at this cadence).
    """
    import torch

    from dekf_bench.env.drift import build_drift
    from dekf_bench.evaluation.evalsets import MAX_CACHE_BYTES

    rates = {name: LEARNING_RATES[0] for name in tuned_baselines()}
    degrees = {label: d for label, _scope, d in SCOPES}
    probe = config_for(args, "p57_preflight", "per_node", degrees["per_node"],
                       entries_for("a", rates))
    itemsize = 8 if probe.run.dtype == "float64" else 4
    n_nodes = probe.graph.n_nodes
    side = probe.model.input_size
    per_set = len(test) * side * side * itemsize

    drift = build_drift(probe)
    steps = list(range(0, args.horizon, EVAL_EVERY)) + [args.horizon - 1]
    # Counted on the *evaluation* grid -- `should_evaluate`'s: every eval_every
    # step plus the last one. The drift's own `distinct_rotations` walks
    # range(0, horizon + 1, every) instead, which includes a step never evaluated
    # and omits the final one; at T=20, every=25 that reports 2 rotations where
    # the run builds 11.
    #
    # Two families, because `current_mean` is appended under per-node drift: the
    # agents' own states, and the network mean that no agent occupies.
    agents = {
        round(drift.rotation_at(step, node), 9)
        for step in steps for node in range(n_nodes)
    }
    means = {
        round(sum(drift.rotation_at(step, node) for node in range(n_nodes)) / n_nodes, 9)
        for step in steps
    }
    wanted = len(agents | means)
    naive = len(steps) * n_nodes
    held = max(1, MAX_CACHE_BYTES // per_set)
    p = probe.model.num_params
    covariance = p * p * itemsize
    group_a = len(GROUP_A) * n_nodes * covariance
    group_b = len(GROUP_B) * 2 * n_nodes * covariance   # the combine holds two sets
    # The --lr pass sweeps the gradient baselines only. It builds the same rotated
    # sets -- it is the pass that met the OOM -- but carries no covariance at all,
    # so pricing it against the filters' footprint would refuse it for memory it
    # never asks for.
    resident = max(group_a, group_b) if with_filters else 0

    def gib(n: float) -> float:
        return n / 2**30

    print("  pre-flight")
    print(f"    p = {p}, N = {n_nodes}, {probe.run.dtype}, eval_every {EVAL_EVERY}")
    print(f"    distinct rotations   {wanted:>6}  ({len(agents)} per-agent + "
          f"{len(means - agents)} network-mean; {naive} naive, aliased down)")
    print(f"    one rotated set      {per_set / 2**20:>6.1f} MiB")
    print(f"    cache bound holds    {held:>6}  -> {max(0, wanted - held)} of them "
          f"rebuilt per seed")
    if with_filters:
        print(f"    covariances, group A {gib(group_a):>6.2f} GiB")
        print(f"    covariances, group B {gib(group_b):>6.2f} GiB  (two in combine)")
    else:
        print("    covariances            none  (--lr sweeps the baselines only)")

    # The exact half of the mean-matching gate (D53, D113): arithmetic, so checked
    # as arithmetic here rather than inferred from a tell after the compute is spent.
    control = build_drift(config_for(args, "p57_preflight", "global", degrees["global"],
                                     entries_for("a", rates)))
    gap = max(
        abs(sum(drift.rotation_at(step, node) for node in range(n_nodes)) / n_nodes
            - control.rotation_at(step))
        for step in steps
    )
    matched = gap < 1e-9
    print(f"    mean-matched         {'exact' if matched else 'NO':>6}  (largest per-step gap "
          f"{gap:.1e} deg over {len(steps)} evaluated steps)")
    if not matched:
        print("\n  REFUSED: the control does not sit at the treatment's mean rotation, so")
        print("  the cells would differ in drift amount as well as spread (D53).")
        return False

    if args.device == "cpu" or not torch.cuda.is_available():
        print("    host memory only: the cache has no device ceiling to breach")
        return True
    free, total = torch.cuda.mem_get_info()
    worst = resident + min(MAX_CACHE_BYTES, wanted * per_set)
    print(f"    CUDA free {gib(free):.1f} of {gib(total):.1f} GiB; worst cell "
          f"needs about {gib(worst):.2f} GiB")
    if resident > free:
        print("\n  REFUSED: the covariances alone exceed free device memory, and "
              "those\n  cannot be evicted. Use --device cpu (X8 ran per-node drift "
              "there;\n  it is about 1.7x slower) or free the card.")
        return False
    if worst > free:
        print("    tight: the bound will evict more than the estimate above.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="re-tune the gradient baselines instead of running the sweep")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="20 steps at one seed, into _smoke-suffixed runs")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.horizon, args.seeds = 20, [0]
    if args.report_only:
        # smoke passed through: `--report-only --smoke` must read the smoke cells
        # rather than silently reporting the real ones (M11 inherited this bug).
        report(suffix)
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)
    # Ahead of the --lr branch on purpose: that is the pass that met the OOM, and
    # it builds the same rotated sets as the main one.
    if not preflight(args, test, with_filters=not args.lr):
        return 1
    if args.lr:
        return tune(args, train, test, suffix)

    missing = [label for label, _s, _d in SCOPES if selected_rates(label, suffix) is None]
    if missing:
        print(f"no selected rates for {missing}: run --lr{' --smoke' if suffix else ''} "
              "first.\nA rate carried across conditions is the mistake D77 exists to "
              "record --\nit put a baseline at chance and inverted a damage ordering.")
        return 1

    cells = [(label, scope, degrees, group)
             for label, scope, degrees in SCOPES for group in ("a", "b")]
    print(f"\nP5.7{' SMOKE' if suffix else ''}: {len(cells)} cells at "
          f"{len(args.seeds)} seeds, T={args.horizon}")
    for label, _s, degrees, group in cells:
        print(f"  {cell_name(label, group, suffix):<30} group {group}   {degrees:g} deg")
    print(f"  control is mean-matched at {MATCHED_DEGREES:g}, not {FULL_ROTATION:g} "
          f"(D53)\n", flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (label, scope, degrees, group) in enumerate(cells, start=1):
        name = cell_name(label, group, suffix)
        entries = entries_for(group, selected_rates(label, suffix))
        note = run_one(config_for(args, name, scope, degrees, entries),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<30} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nP5.7{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


def _cell_of(learner: str, label: str, suffix: str = "") -> str:
    """Group B holds the full-sharing variants; everything else is in group A."""
    return cell_name(label, "b" if learner in GROUP_B else "a", suffix)


STAT_HEADER = f"{'diff':>9}  {'95% CI':^20}  {'t':>7}  {'p_holm':>7}   n"


def stat_columns(result: Paired, p_adjusted: float) -> str:
    """One row's inference: signed mean, 95% interval, signed t, Holm p, seeds."""
    lo, hi = result.ci(0.95)
    interval = f"[{lo:+.4f}, {hi:+.4f}]"   # padded: a NaN interval is narrower
    mark = "*" if p_adjusted < ALPHA else " "
    return (f"{result.mean:>+9.4f}  {interval:<20}  {result.t:>+7.2f}"
            f"  {p_adjusted:>7.3f}{mark}  {result.n}")


def report(suffix: str = "") -> None:
    """The gate first, then the headline, then the gap that P5.7 exists to move."""
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    cell = lambda learner, label: _cell_of(learner, label, suffix)  # noqa: E731
    labels = [label for label, _s, _d in SCOPES]
    every = ["centralized_ekf_gamma", *GROUP_A, *GROUP_B, *tuned_baselines(),
             DISPLACEMENT_TELL]
    if suffix:
        print("  SMOKE: 20 rounds at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")

    # ---- the gate -----------------------------------------------------------
    tell_name = DISPLACEMENT_TELL
    print("  GATE: is the control mean-matched?\n")
    print("  1. exactly: the pre-flight checked, as arithmetic, that the per-node")
    print("     agents' mean rotation equals the control's at every evaluated step.")
    print(f"  2. empirically: `{tell_name}` never adapts, so its error reads out")
    print("     displacement. It is compared at the SAME rotation in both cells --")
    print("     per-node `current_mean` against global `current` -- and must be SHOWN")
    print(f"     equal within +/-{GATE_MARGIN} by TOST. Failing to find a difference is")
    print("     not showing equality: at five seeds a real mismatch does that easily.\n")
    own = per_seed(cell(tell_name, "per_node"), tell_name)
    at_mean = per_seed(cell(tell_name, "per_node"), tell_name, evalset="current_mean")
    tell = compare(at_mean, per_seed(cell(tell_name, "global"), tell_name))
    # Four verdicts. Equivalence and difference are separate questions, so "neither
    # shown" is a real outcome -- and with one seed there is no standard error at all.
    equal = tell.equivalent(GATE_MARGIN, ALPHA)
    if equal is None:
        verdict = f"UNDECIDABLE -- {tell.n} seed(s), no standard error"
    elif equal:
        verdict = "PASS -- shown equal within the margin"
    elif tell.p < ALPHA:
        verdict = "FAIL -- the cells differ; read nothing below (D53)"
    else:
        verdict = ("INCONCLUSIVE -- neither shown equal nor shown different; "
                   "the exact check stands alone")
    lo, hi = tell.ci(1.0 - 2.0 * ALPHA)
    print(f"    at-mean minus global  {tell.mean:+.4f}   90% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"TOST p={tell.tost_p(GATE_MARGIN):.3f}   n={tell.n}")
    print(f"    {verdict}")
    jensen = compare(own, at_mean)
    print(f"    for scale only, not a gate -- agents at their own rotations minus at the mean:"
          f"\n    {jensen.mean:+.4f} (t={jensen.t:+.2f}). That is the Jensen term the gate avoids.\n")

    # ---- levels -------------------------------------------------------------
    print("  settled error by drift scope\n")
    print(f"    {'learner':<36}" + "".join(f"{label:>12}" for label in labels))
    for learner in every:
        row = f"    {learner:<36}"
        for label in labels:
            row += f"{mean(per_seed(cell(learner, label), learner)):>12.4f}"
        print(row)

    # ---- headline -----------------------------------------------------------
    print("\n  what heterogeneity costs each learner")
    print("    per_node minus global, paired per seed; positive = heterogeneity hurt.")
    print("    t is signed, on n-1 df; p_holm is adjusted across this table's rows and")
    print(f"    * marks p_holm < {ALPHA}.\n")
    print(f"    {'learner':<36}{STAT_HEADER}")
    costs = [(learner, compare(per_seed(cell(learner, "per_node"), learner),
                               per_seed(cell(learner, "global"), learner)))
             for learner in every]
    for (learner, result), p_adj in zip(costs, holm([r.p for _l, r in costs]), strict=True):
        print(f"    {learner:<36}{stat_columns(result, p_adj)}")

    # ---- the question -------------------------------------------------------
    central = "centralized_ekf_gamma"
    print("\n  THE QUESTION: does diffusion close its gap to centralised?")
    print("    gap = centralised minus diffusion; positive = diffusion is ahead.")
    print("    The test is the CHANGE in the gap, per_node minus global, taken seed by")
    print("    seed as a difference of differences (D54): two gaps printed side by side")
    print("    are not a test. P5.7 predicts it is positive; a per_node gap above zero")
    print("    would be the first time a distributed variant beats the pooled one.\n")
    print(f"    {'':<36}{'gap, global':>12}{'per_node':>10}   change{STAT_HEADER[9:]}")
    rows = []
    for diffuse in GROUP_A + GROUP_B:
        gaps = {label: differences(per_seed(cell(central, label), central),
                                   per_seed(cell(diffuse, label), diffuse))
                for label in labels}
        rows.append((diffuse, gaps, compare(gaps["per_node"], gaps["global"])))
    for (diffuse, gaps, change), p_adj in zip(rows, holm([c.p for *_r, c in rows]),
                                              strict=True):
        print(f"    {diffuse:<36}{mean(gaps['global']):>+12.4f}{mean(gaps['per_node']):>+10.4f}"
              f"{stat_columns(change, p_adj)}")

    print("\n  ⚠ If the effect appears, M8 says to look at calibration rather than")
    print("  consensus: there the cost fell on the filters, the most damaged arm had")
    print("  exactly zero inter-agent disagreement, and what moved was the covariance")
    print("  failing to expand as the error grew (D111).")


if __name__ == "__main__":
    raise SystemExit(main())
