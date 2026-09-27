r"""P5.5 -- the break rate: how fast can the world drift before diffusion stops tracking?

    python scripts/run_break_rate.py --lr          # tune SGD and AdamW on the ramp, first
    python scripts/run_break_rate.py               # the ramp and its stationary twin
    python scripts/run_break_rate.py --report-only
    python scripts/run_break_rate.py --lr --smoke  # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

## The question (phase5_plan.md P5.5, the X9/X16 analogue)

X9 swept the drift *rate* inside one run -- an accelerating ramp from 0 to 0.18
deg/step, ending at 45 degrees -- against a stationary twin, and located where each
method's damage takes off. The gradient methods break between 0.038 and 0.044
deg/step. X16 put the centralised filter on the same ramp. **Reproduced
2026-09-27 before this was written:** on a noise bar shared across every X9 and X16
learner the centralised filter breaks at **0.0639** deg/step against 0.0381--0.0399,
and at the matched rate of 0.10 deg/step its damage is **0.021** against ATC's
0.036. This is the only place the filter's advantage is a *rate*, and P5.5 asks how
much of it survives decentralisation.

## Named before the run (confirmatory under D118)

The reading is **damage at the matched rate, 0.10 deg/step** -- the drifting run's
error minus its twin's at the first evaluation where the schedule reaches that
speed, per seed. X9 found it the reading to prefer: a rate everyone faced, needing
no noise estimate and immune to a noisier method clearing a laxer bar. Five
contrasts, paired per seed, Holm across the five:

1. one-hop - ATC. Predicted negative: one-hop is less damaged.
2. local adapt - ATC. No prediction.
3. one-hop - centralised EKF. Predicted positive: decentralising costs some of it.
4. local adapt - centralised EKF. Predicted positive.
5. one-hop - ATC AdamW. Predicted negative. Cross-cell, so it is tested only once
   the merge gate holds; otherwise the family is the first four.

Exploratory: the break rates themselves (on one noise bar pooled across every
adapting learner, as X9 requires for comparability), the comparative break against
`frozen_atc`, full sharing, and the damage table.

## The design (decided with the user 2026-09-27)

**ER 0.3, every arm run fresh** -- the graph the diffusion filter was tuned on (X20)
and the standard since D52; X9's ring numbers stay as history. X9's ramp and
cadence unchanged: `mnist_rotating_ramp` (45 degrees, exponent 6, peak 0.18
deg/step at $T=1500$) against `mnist_stationary`, evaluations every 10 steps, five
seeds, IID, $N=10$. **The baselines are tuned on the ramp itself**, by whole-run
mean error -- the ramp is the condition, as D77 asks -- and the *same* rates are
carried into the twin, because the subtraction is exact only when the two runs
differ in the drift alone (`assert_paired_runs` checks it). The filters carry
X20's selection. `frozen_atc` runs at ATC's tuned rate, frozen at step 300 as in X9.

Cells: **a** -- centralised filter, both mean-only filters, the SGD baselines and
`frozen_atc`; **b** -- both full-sharing filters; **adamw** -- the AdamW arms with
`centralized_sgd` as D119's merge gate. Each in a ramp and a twin version.
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

#: The drifting run and its twin: X9's two configs, on ER 0.3 rather than ring.
CONDITIONS = {"ramp": "x9_rate_ramp", "control": "x9_control"}
MATCHED_RATE = 0.10
NOISE_MULTIPLE = 3.0
PERSISTENCE = 3

CENTRAL = "centralized_ekf_gamma"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
MEAN_ONLY = [LOCAL, ONEHOP]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
ATC, ATC_ADAMW = "diffusion_sgd_atc", "diffusion_atc_adamw"
FROZEN = "frozen_atc"
GROUPS = ["a", "b", ADAMW_GROUP]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 10
#: The smoke's horizon; `frozen_atc` freezes at its midpoint there rather than at 300,
#: so the comparative break has something to read. A ramp scales with T, so 40
#: steps still passes 0.10 deg/step.
SMOKE_HORIZON = 40
DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "brk_status.json"


def lr_run_name(rate: float, suffix: str = "", family: str = "sgd") -> str:
    stem = "brk_lr_adamw" if family == "adamw" else "brk_lr"
    return f"{stem}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(condition: str, group: str, suffix: str = "") -> str:
    return f"brk_{condition}_{group}{suffix}"


def whole_run(run: str, learner: str) -> float:
    """Mean `current` error over every evaluation of a run: the ramp tuning criterion.

    Not the settled window: on a ramp the last 20% is the fastest drift, and tuning
    there would pick the rate that suits 0.15 deg/step and nothing slower.
    """
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    if not (directory / "_complete").exists():
        return float("inf")
    values = []
    for path in sorted(directory.glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["learner", "metric", "evalset", "value"])
        rows = frame[(frame["learner"] == learner) & (frame["metric"] == "error_rate")
                     & (frame["evalset"] == "current")]
        if len(rows):
            values.append(float(rows["value"].mean()))
    return sum(values) / len(values) if values else float("inf")


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


def config_for(args, name: str, condition: str, entries: list[dict],
               seeds: list[int] | None = None):
    return load_config(
        CONDITIONS[condition],
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
            "env": {"dataset": args.dataset},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def entries_for(group: str, rates: dict[str, float], freeze_after: int | None = None
                ) -> list[dict]:
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
    # ATC's algorithm at ATC's tuned rate, frozen at X9's step 300 (its yaml).
    frozen = {"name": FROZEN, "lr": rates[ATC]}
    if freeze_after is not None:  # the smoke only: its horizon ends before step 300
        frozen["freeze_after"] = freeze_after
    learners.append(frozen)
    return learners


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, train, test, suffix: str = "") -> int:
    """SGD and AdamW on the ramp, whole-run error; the filters carry X20's."""
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    grids = {family: sorted({r for b in tuned_learners() if family_of(b) == family
                             for r in grid_for(b)}, reverse=True)
             for family in ("sgd", "adamw")}
    cells = [(f, r) for f, grid in grids.items() for r in grid]
    print(f"P5.5 lr{' SMOKE' if suffix else ''}: the ramp at ({len(grids['sgd'])} SGD + "
          f"{len(grids['adamw'])} AdamW) rates, {len(seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (family, rate) in enumerate(cells, start=1):
        name = lr_run_name(rate, suffix, family)
        entries = [{"name": b, "lr": rate, **o} for b, o in tuned_learners().items()
                   if family_of(b) == family and rate in grid_for(b)]
        note = run_one(config_for(args, name, "ramp", entries, seeds=seeds),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<28} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    rates = selected_rates(suffix)
    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    if rates:
        for name, rate in rates.items():
            edge = rate in (grid_for(name)[0], grid_for(name)[-1])
            print(f"  {name:<28} {rate:g}{'   <- grid edge' if edge else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="tune the SGD and AdamW baselines on the ramp")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="a short horizon at one seed, into _smoke-suffixed runs")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.horizon, args.seeds = SMOKE_HORIZON, [0]
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

    cells = [(condition, group) for group in GROUPS for condition in CONDITIONS]
    print(f"P5.5{' SMOKE' if suffix else ''}: {len(cells)} cells at {len(args.seeds)} seeds, "
          f"T={args.horizon}, ramp and twin on ER 0.3\n", flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (condition, group) in enumerate(cells, start=1):
        name = cell_name(condition, group, suffix)
        entries = entries_for(group, rates, SMOKE_HORIZON // 2 if suffix else None)
        note = run_one(config_for(args, name, condition, entries), train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<24} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nP5.5{' smoke' if suffix else ''} complete in "
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


def report(suffix: str = "") -> None:
    """The gates, the damage at the matched rate, then the break rates."""
    import glob  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from dekf_bench.env.drift import build_drift  # noqa: PLC0415
    from dekf_bench.metrics.breaks import (  # noqa: PLC0415
        BreakError,
        assert_paired_runs,
        comparative_break,
        error_by_step,
        excess_break,
        paired_excess,
        pooled_sem,
    )

    if suffix:
        print("  SMOKE: a short run at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")

    def load(run: str) -> pd.DataFrame:
        files = sorted(glob.glob(str(ROOT / "results" / run / "seed_*.parquet")))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files \
            else pd.DataFrame()

    # ---- the pairing gate: ramp and twin must differ in the drift alone ---------
    print("  pairing gate: each ramp cell against its twin (graph, model, learners, env,")
    print("  seeds, horizon, cadence identical; only the drift may differ)")
    frames, pair_ok = [], True
    for group in GROUPS:
        ramp, twin = cell_name("ramp", group, suffix), cell_name("control", group, suffix)
        try:
            assert_paired_runs(ROOT / "results" / ramp, ROOT / "results" / twin)
            frames.append(paired_excess(load(ramp), load(twin)))
            print(f"    {group:<6} paired")
        except (BreakError, KeyError, AttributeError) as failure:
            pair_ok = False
            print(f"    {group:<6} NOT PAIRED: {str(failure)[:90]}")
    if not frames:
        print("\n  nothing to report yet")
        return
    excess = pd.concat(frames, ignore_index=True)

    # ---- the merge gate (D119) --------------------------------------------------
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

    config = load_config(CONDITIONS["ramp"], overrides={"run": {"horizon": _horizon(suffix)}})
    drift, horizon = build_drift(config), config.run.horizon
    steps = sorted(excess.t.unique())
    reached = [s for s in steps if drift.schedule.rate_at(int(s)) >= MATCHED_RATE]
    if not reached:
        print(f"\n  the ramp never reaches {MATCHED_RATE} deg/step in this run")
        return
    matched_step = reached[0]

    def damage(learner: str) -> dict[int, float]:
        rows = excess[(excess.learner == learner) & (excess.t == matched_step)]
        return {int(s): float(v) for s, v in zip(rows.seed, rows.excess, strict=True)}

    learners = sorted(excess.learner.unique())
    print(f"\n  damage at the matched rate {MATCHED_RATE} deg/step (t={matched_step}), per seed"
          " mean; drifting minus twin\n")
    for learner in learners:
        values = damage(learner)
        print(f"    {learner:<36}{sum(values.values()) / len(values):>+9.4f}")

    # ---- the confirmatory family --------------------------------------------------
    contrasts = [("one-hop - ATC", ONEHOP, ATC), ("local adapt - ATC", LOCAL, ATC),
                 ("one-hop - centralised EKF", ONEHOP, CENTRAL),
                 ("local adapt - centralised EKF", LOCAL, CENTRAL)]
    if poolable:
        contrasts.append(("one-hop - ATC AdamW", ONEHOP, ATC_ADAMW))
    rows = [(label, compare(damage(a), damage(b))) for label, a, b in contrasts]
    live = [(label, r) for label, r in rows if r.n]
    print("\n  CONFIRMATORY (D123): damage at the matched rate, paired per seed; negative")
    print("  = the first is less damaged. Predicted: one-hop < ATC, one-hop > central,")
    print("  local > central, one-hop < ATC AdamW; no prediction for local vs ATC.")
    print("  Holm across the family" + ("" if poolable else " (ATC AdamW withheld: gate)")
          + ".\n")
    print(f"    {'':<32}{STAT_HEADER}")
    for (label, result), p_adj in zip(live, holm([r.p for _l, r in live]), strict=True):
        print(f"    {label:<32}{stat_columns(result, p_adj)}")

    # ---- exploratory: the break rates ---------------------------------------------
    adapting = [name for name in learners if name != FROZEN]
    noise = pooled_sem(excess, adapting)
    print("\n  exploratory: break rates on one bar pooled across every adapting learner")
    print(f"  ({NOISE_MULTIPLE:g} s.e.m., {PERSISTENCE} consecutive evaluations, as X9); and the")
    print(f"  comparative break against {FROZEN}, from its freeze point\n")
    errors = pd.concat([error_by_step(load(cell_name("ramp", g, suffix)))
                        for g in GROUPS if load(cell_name("ramp", g, suffix)).size],
                       ignore_index=True)
    # The freeze point the cell actually ran with, read from its own config.
    import yaml  # noqa: PLC0415

    recorded = yaml.safe_load((ROOT / "results" / cell_name("ramp", "a", suffix)
                               / "config.yaml").read_text(encoding="utf-8"))
    freeze_after = next((e.get("freeze_after") for e in recorded["learners"]
                         if e["name"] == FROZEN), 300)
    for learner in learners:
        absolute = excess_break(excess, learner, drift, horizon, NOISE_MULTIPLE,
                                PERSISTENCE, noise=noise)
        comparative = "--"
        if learner != FROZEN and _group_of(learner) == "a":
            try:
                point = comparative_break(errors[errors.learner.isin([learner, FROZEN])],
                                          learner, FROZEN, drift, horizon, PERSISTENCE,
                                          start_step=freeze_after or 300)
                comparative = (f"{point.rate_at_break:.4f}" if point.broke
                               else ("never ahead" if point.note else "no break"))
            except BreakError:
                comparative = "--"
        rate = f"{absolute.rate_at_break:.4f}" if absolute.broke else "no break"
        print(f"    {learner:<36} absolute {rate:>9}   comparative {comparative:>11}")

    if not pair_ok:
        print("\n  ⚠ A group failed the pairing gate: its excess is omitted above.")
    print(f"\n  * = p_holm < {ALPHA} (D113). Break rates on a pooled bar are exploratory;")
    print("  the damage at a matched rate is the reading X9 found to prefer.")


def _horizon(suffix: str) -> int:
    return SMOKE_HORIZON if suffix else HORIZON


if __name__ == "__main__":
    raise SystemExit(main())
