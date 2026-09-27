r"""M10 -- network size on Mackey--Glass at fixed connectivity: N = 10, 20, 30.

    python scripts/run_m10_network_size.py --lr --device cuda   # re-tune the baselines per N, first
    python scripts/run_m10_network_size.py --device cuda        # the cells
    python scripts/run_m10_network_size.py --abrupt ...         # also the abrupt condition
    python scripts/run_m10_network_size.py --full-sharing ...   # also full sharing at N = 20, 30
    python scripts/run_m10_network_size.py --report-only
    python scripts/run_m10_network_size.py --lr --smoke         # then --smoke: the path, not the numbers

`--lr` must run first and the main pass refuses without it (D77). Both are cumulative:
finished cells are cached by name, so a later `--abrupt` or `--full-sharing` runs only
what it adds.

`docs/mackey_glass_plan.md` M10, the N>10 analogue ([[D114]], [[D117]]). On MNIST it was
**established** that local-adapt diff-EKF falls behind the centralised filter as $N$
grows ($+0.007$, $+0.013$ from $N=10$ to 30), while one-hop's growth was not detected.
M10 asks whether that survives a second task and architecture.

## What is carried from N>10, and what is not

**Connectivity is held, as there.** $p$ per $N$ holds the ER mixing gap at its $N=10$
value, 0.119 ($p=0.3$, 0.224, 0.167), and each seed's draw is conditioned into
$0.119\pm0.02$ with at most three draws, the last kept regardless. `graph.py`'s band is
ported from main unchanged, and graphs derive from the seed alone, so **these are the
same graphs N>10 drew** -- the two tasks share their networks seed for seed.

**The horizon is not carried.** N>10 ran $T=500$ only because MNIST has 60 000 images
and $NnT$ must fit in them (D5). Mackey--Glass has no budget, and $T=500$ would measure a
transient: in M6 the centralised filter is still falling there (0.1432 against 0.1391
over the last 20% of 1500), and so is the very quantity M10 is about -- local adapt minus
centralised is $+0.0080$ over steps 400--500 against $+0.0052$ over 1200--1500. A larger
network pools more per step and settles sooner, so a short horizon would confound $N$
with time-to-settle. $T=1500$, as M6.

**Conditions.** Stationary by default; abrupt behind `--abrupt` (decided 2026-09-27: run
if time allows, scheduled at the end of the experiments). **Full sharing** at $N=10$ by
default and behind `--full-sharing` above it, as N>10 -- here for compute rather than
memory: $\boldsymbol P$ is 41 MB per agent, so even two full-sharing sets at $N=30$ are
2.5 GB.

## Built from M6's recorded cells

Each cell loads `results/m6_<condition>_a/config.yaml` -- law, sensor, $\boldsymbol R$ scale,
horizon, cadence, dtype and seeds -- and replaces the graph ($N$, $p$, band), the
learners and the name. Filters carry M6's entries (M4, M5), selected at $N=10$: the X14
discipline, and N>10 did the same with X20's. Every gradient baseline is re-tuned per
$(N, \text{condition})$ on M3's grids at M3's seeds (0--1): at $N=20,30$ even the pooled
learners see a different batch. As in N>10, each diffusion filter runs in its own
process, so a divergence costs one arm, not the cell.

## The gate

At $N=10$ the five graph-blind learners (centralised filter, SGD and AdamW, and both
local-only arms) see exactly M6's data, whatever graph was drawn. Their settled error must
match `m6_<condition>_a` per seed within $10^{-9}$. It checks that the recorded config
still reproduces, and that the conditioned graph stays out of the data path. The
confirmatory tables are withheld if it fails.

## Named before the run (confirmatory under D118), stationary

* **Q1 (family of two, Holm).** (a) The change in `local adapt - centralised filter` from
  $N=10$ to $N=30$, per seed: **predicted positive**, as on MNIST. (b) That change minus
  the same change for one-hop: **predicted positive** -- one-hop scales better, because
  it brings neighbours' fresh evidence into the adapt step, which is what larger $N$
  dilutes for local adapt.
* **Q2 (family of three, Holm).** `one-hop - ATC AdamW` at each $N$: **predicted
  negative** at every size (D119's question, on the task where AdamW is strongest).

Exploratory: the $N=20$ midpoints; ATC against centralised SGD and ATC AdamW against
centralised AdamW as $N$ grows; cooperation (`local_only - ATC`) by $N$; full sharing;
each seed's realised mixing gap; wall-clock per cell; and every abrupt row.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, ADAMW_RATES, SGD_FAMILY, SGD_RATES, settled  # noqa: E402
from run_m8_tau_heterogeneity import ALPHA, STAT_HEADER, _by_seed, _stat_columns  # noqa: E402

from dekf_bench.metrics.paired import differences, holm, summarise  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: (label, N, ER edge probability), small to large; p holds the mixing gap (D114).
SIZES: list[tuple[str, int, float]] = [("n10", 10, 0.3), ("n20", 20, 0.224), ("n30", 30, 0.167)]
TARGET_MIXING_GAP = 0.119
GAP_BAND = [TARGET_MIXING_GAP - 0.02, TARGET_MIXING_GAP + 0.02]
BAND_DRAWS = 3

CONDITIONS = ["stationary", "abrupt"]
DEFAULT_CONDITIONS = ["stationary"]

#: Group label -> the one diffusion filter its cell carries (N>10's layout).
MEAN_ONLY = {"local": "diffusion_ekf", "onehop": "diffusion_ekf_onehop_mean_receiver"}
FULL_SHARING = {"full": "diffusion_ekf_full", "onehopfull": "diffusion_ekf_onehop_receiver"}
FULL_SHARING_BY_DEFAULT = {"n10"}
CENTRAL_FILTER = "centralized_ekf_walk"
LOCAL, ONEHOP = MEAN_ONLY["local"], MEAN_ONLY["onehop"]
ATC_ADAMW = "diffusion_atc_adamw"

BASELINES = {**SGD_FAMILY, **ADAMW_FAMILY}
GRAPH_BLIND = [CENTRAL_FILTER, "centralized_sgd", "local_only", "centralized_adamw",
               "local_adamw"]
BLIND_TOLERANCE = 1e-9

LR_SEEDS = [0, 1]
SMOKE_HORIZON, SMOKE_EVAL_EVERY = 20, 10
STATUS = ROOT / "results" / "m10_status.json"


def source(condition: str) -> str:
    return f"m6_{condition}_a"


def lr_run_name(size: str, condition: str, index: int, suffix: str = "") -> str:
    return f"m10_lr_{size}_{condition}_r{index}{suffix}"


def cell_name(size: str, condition: str, group: str, suffix: str = "") -> str:
    return f"m10_{size}_{condition}_{group}{suffix}"


def groups_for(size: str, full_sharing: bool) -> list[str]:
    groups = ["a", *MEAN_ONLY]
    if full_sharing or size in FULL_SHARING_BY_DEFAULT:
        groups += list(FULL_SHARING)
    return groups


def recorded(condition: str) -> dict:
    return yaml.safe_load((ROOT / "results" / source(condition) / "config.yaml")
                          .read_text(encoding="utf-8"))


def rate_of(learner: str, index: int) -> float:
    return (ADAMW_RATES if learner in ADAMW_FAMILY else SGD_RATES)[index]


def config_for(condition: str, name: str, n: int, p: float, entries: list[dict], args,
               suffix: str, seeds: list[int] | None = None):
    """M6's recorded config with the graph, the learners and the name replaced."""
    run = {"name": name, "seeds": seeds or recorded(condition)["run"]["seeds"],
           "device": args.device}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "eval_every": SMOKE_EVAL_EVERY, "seeds": [0]})
    graph = {"topology": "erdos_renyi", "n_nodes": n,
             "params": {"p": p, "mixing_gap_band": list(GAP_BAND), "band_draws": BAND_DRAWS}}
    return load_config(ROOT / "results" / source(condition) / "config.yaml",
                       overrides={"run": run, "graph": graph, "learners": entries})


def m6_entries(condition: str) -> dict[str, dict]:
    return {e["name"]: dict(e) for e in recorded(condition)["learners"]}


def lr_entries(index: int) -> list[dict]:
    return [{"name": n, "lr": rate_of(n, index), **o} for n, o in BASELINES.items()]


def selected_rates(size: str, condition: str, suffix: str = "") -> dict[str, dict] | None:
    chosen = {}
    for learner in BASELINES:
        scored = [(settled(lr_run_name(size, condition, k, suffix), learner), k)
                  for k in range(len(SGD_RATES))]
        finite = [(v, k) for v, k in scored if v != float("inf")]
        if not finite:
            return None
        value, index = min(finite)
        chosen[learner] = {"lr": rate_of(learner, index), "settled_rmse": value,
                           "edge": index in (0, len(SGD_RATES) - 1)}
    return chosen


def entries_for(group: str, condition: str, rates: dict[str, dict]) -> list[dict]:
    """M6's recorded filter entries; the baselines at this cell's own rates."""
    recorded_entries = m6_entries(condition)
    if group in FULL_SHARING:
        return [m6_full_entry(FULL_SHARING[group], condition)]
    if group in MEAN_ONLY:
        return [recorded_entries[MEAN_ONLY[group]]]
    learners = [recorded_entries[CENTRAL_FILTER]]
    learners += [{"name": n, "lr": rates[n]["lr"], **o} for n, o in BASELINES.items()]
    return learners


def m6_full_entry(name: str, condition: str) -> dict:
    """Full sharing lives in M6's group-B cell."""
    config = yaml.safe_load((ROOT / "results" / f"m6_{condition}_b" / "config.yaml")
                            .read_text(encoding="utf-8"))
    return next(dict(e) for e in config["learners"] if e["name"] == name)


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def preflight(args, conditions: list[str], suffix: str) -> None:
    """Each seed's realised mixing gap per N; `!` marks a seed whose draws all missed."""
    from dekf_bench.env.graph import build_graphs  # noqa: PLC0415
    from dekf_bench.runner.seeding import Seeds  # noqa: PLC0415

    print(f"  pre-flight: mixing gap target {TARGET_MIXING_GAP}, band "
          f"{GAP_BAND[0]:.3f}-{GAP_BAND[1]:.3f}, at most {BAND_DRAWS} draws")
    graphs = {}
    for size, n, p in SIZES:
        config = config_for(conditions[0], "m10_preflight", n, p, lr_entries(0), args, suffix)
        gaps = [build_graphs(config, Seeds.from_master(s).torch_generator("graph")).comm.mixing_gap
                for s in config.run.seeds]
        graphs[size] = gaps
        marked = " ".join(f"{g:.3f}" + (" " if GAP_BAND[0] <= g <= GAP_BAND[1] else "!")
                          for g in gaps)
        print(f"    {size}: N={n:>2} p={p:<5}  {marked}  mean {sum(gaps) / len(gaps):.3f}")
    if not suffix:
        status = load_status()
        status["graphs"] = graphs
        save_status(status)
    print()


def tune(args, conditions: list[str], suffix: str) -> int:
    status = load_status()
    cells = [(s, n, p, c, k) for s, n, p in SIZES for c in conditions
             for k in range(len(SGD_RATES))]
    print(f"M10 lr{' SMOKE' if suffix else ''}: {len(cells)} cells, seven baselines each\n",
          flush=True)
    started = time.time()
    for index, (size, n, p, condition, k) in enumerate(cells, start=1):
        name = lr_run_name(size, condition, k, suffix)
        note = run_one(config_for(condition, name, n, p, lr_entries(k), args, suffix,
                                  seeds=LR_SEEDS), None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<28} sgd {SGD_RATES[k]:<7g} adamw "
              f"{ADAMW_RATES[k]:<6g} {note:<24} {(time.time() - started) / 60:.0f} min",
              flush=True)
    print()
    for size, _n, _p in SIZES:
        for condition in conditions:
            rates = selected_rates(size, condition, suffix)
            if rates:
                edge = [n for n, r in rates.items() if r["edge"]]
                print(f"  {size}/{condition:<10} " + "  ".join(
                    f"{n} {r['lr']:g}" for n, r in rates.items())
                    + (f"   <- grid edge: {', '.join(edge)}" if edge else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2, 3, 4])
    parser.add_argument("--lr", action="store_true", help="re-tune the baselines per (N, condition)")
    parser.add_argument("--abrupt", action="store_true", help="also the abrupt condition")
    parser.add_argument("--full-sharing", action="store_true",
                        help="also full covariance sharing at N = 20 and 30")
    parser.add_argument("--smoke", action="store_true", help="20 rounds, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    conditions = CONDITIONS if args.abrupt else DEFAULT_CONDITIONS
    if args.report_only:
        report(conditions, args.full_sharing, suffix)
        return 0
    missing = [source(c) for c in conditions
               if not (ROOT / "results" / source(c) / "_complete").exists()]
    if missing:
        print(f"  REFUSED: M10 is built from M6's recorded cells, and {missing} is not finished.")
        return 1
    preflight(args, conditions, suffix)
    if args.lr:
        return tune(args, conditions, suffix)

    unswept = [(s, c) for s, _n, _p in SIZES for c in conditions
               if selected_rates(s, c, suffix) is None]
    if unswept:
        print(f"  REFUSED: no selected rates for {unswept}. Run --lr first (D77).")
        return 1

    cells = [(s, n, p, c, g) for c in conditions for s, n, p in SIZES
             for g in groups_for(s, args.full_sharing)]
    print(f"M10{' SMOKE' if suffix else ''}: {len(cells)} cells, "
          f"{', '.join(conditions)}, device {args.device}\n", flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (size, n, p, condition, group) in enumerate(cells, start=1):
        name = cell_name(size, condition, group, suffix)
        learners = entries_for(group, condition, selected_rates(size, condition, suffix))
        cell_started = time.time()
        note = run_one(config_for(condition, name, n, p, learners, args, suffix),
                       None, None, args.fresh)
        status[name] = note
        if note != "cached":
            ran += 1
            status[f"{name}_minutes"] = round((time.time() - cell_started) / 60, 1)
        save_status(status)
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<32} {len(learners):>2} learners  {note:<14}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM10{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(conditions, args.full_sharing, suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _seeds(size: str, condition: str, learner: str, suffix: str) -> dict[int, float]:
    """Per-seed settled RMSE from whichever of this (size, condition)'s cells holds it."""
    for group in ["a", *MEAN_ONLY, *FULL_SHARING]:
        got = _by_seed(cell_name(size, condition, group, suffix), learner)
        if got:
            return got
    return {}


def _gap(size: str, condition: str, learner: str, centre: str, suffix: str) -> dict[int, float]:
    return differences(_seeds(size, condition, learner, suffix),
                       _seeds(size, condition, centre, suffix))


def _row(label: str, result, p_adj: float) -> str:
    if result.n < 2:
        return f"    {label:<44}{'-':>9}"
    return f"    {label:<44}{_stat_columns(result, p_adj)}"


def report(conditions: list[str], full_sharing: bool, suffix: str = "") -> None:
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    sizes = [s for s, _n, _p in SIZES]
    if suffix:
        print("  SMOKE: 20 rounds at one seed. The numbers mean nothing, and the gate")
        print("  cannot hold: M6's cells ran the full horizon.\n")
    status = load_status()
    for condition in conditions:
        print(f"\n  ===== {condition} =====\n")
        print("  realised mixing gap, per seed:")
        for size in sizes:
            gaps = status.get("graphs", {}).get(size)
            print(f"    {size}: " + (" ".join(f"{g:.3f}" for g in gaps) if gaps else "-"))
        print("\n  settled RMSE on the held-out current set")
        print(f"    {'learner':<40}" + "".join(f"{s:>10}" for s in sizes))
        every = [CENTRAL_FILTER, *MEAN_ONLY.values(), *FULL_SHARING.values(), *BASELINES]
        for learner in every:
            row = f"    {learner:<40}"
            for size in sizes:
                value = mean(_seeds(size, condition, learner, suffix))
                row += f"{value:>10.4f}" if value == value else f"{'-':>10}"
            print(row)

        print(f"\n  gate: at N=10 the graph-blind learners must match {source(condition)} "
              f"per seed within {BLIND_TOLERANCE:g}")
        gate = True
        for learner in GRAPH_BLIND:
            ours = _by_seed(cell_name("n10", condition, "a", suffix), learner)
            theirs = _by_seed(source(condition), learner)
            shared = sorted(set(ours) & set(theirs))
            worst = max((abs(ours[s] - theirs[s]) for s in shared), default=float("nan"))
            ok = bool(shared) and worst <= BLIND_TOLERANCE
            gate &= ok
            print(f"    {learner:<24} max |diff| {worst:.1e} over {len(shared)} seed(s)  "
                  f"{'ok' if ok else 'FAILS'}")
        if suffix:
            continue

        confirmatory = condition == "stationary"
        tag = "CONFIRMATORY (D129)" if confirmatory else "EXPLORATORY (abrupt: named for stationary only)"
        if not gate:
            print("\n  TABLES WITHHELD: N=10 does not reproduce M6, so nothing here is paired.")
            continue

        local_change = differences(_gap("n30", condition, LOCAL, CENTRAL_FILTER, ""),
                                   _gap("n10", condition, LOCAL, CENTRAL_FILTER, ""))
        onehop_change = differences(_gap("n30", condition, ONEHOP, CENTRAL_FILTER, ""),
                                    _gap("n10", condition, ONEHOP, CENTRAL_FILTER, ""))
        q1 = [("(a) local-adapt gap, N=30 minus N=10", summarise(local_change)),
              ("(b) that change minus one-hop's", compare(local_change, onehop_change))]
        print(f"\n  {tag}. Q1: does decentralising cost more as N grows? Distance to the")
        print("  centralised filter, per seed. Both predicted positive. Holm across two.")
        print(f"\n    {'':<44}{STAT_HEADER}")
        for (label, result), p_adj in zip(q1, holm([r.p for _l, r in q1]), strict=True):
            print(_row(label, result, p_adj))

        q2 = [(f"one-hop - ATC AdamW, {size}",
               compare(_seeds(size, condition, ONEHOP, ""), _seeds(size, condition, ATC_ADAMW, "")))
              for size in sizes]
        print(f"\n  {tag}. Q2: one-hop minus ATC AdamW at each N; negative = the filter")
        print("  better. Predicted negative at every N. Holm across three.")
        print(f"\n    {'':<44}{STAT_HEADER}")
        for (label, result), p_adj in zip(q2, holm([r.p for _l, r in q2]), strict=True):
            print(_row(label, result, p_adj))

        print("\n  EXPLORATORY. Distance to the centralised counterpart by N; positive = costs.")
        pairs = [(LOCAL, CENTRAL_FILTER), (ONEHOP, CENTRAL_FILTER),
                 ("diffusion_sgd_atc", "centralized_sgd"), (ATC_ADAMW, "centralized_adamw"),
                 ("local_only", "diffusion_sgd_atc"), ("local_adamw", ATC_ADAMW)]
        print(f"    {'':<44}" + "".join(f"{s:>10}" for s in sizes) + f"{'30-10':>10}{'t':>7}")
        for learner, centre in pairs:
            gaps = [_gap(size, condition, learner, centre, "") for size in sizes]
            change = compare(gaps[-1], gaps[0])
            row = f"    {learner + ' - ' + centre:<44}" + "".join(
                f"{mean(g):>+10.4f}" if g else f"{'-':>10}" for g in gaps)
            print(row + (f"{change.mean:>+10.4f}{change.t:>+7.2f}" if change.n >= 2 else ""))
        print("\n  full sharing minus mean-only, where it ran")
        for size in sizes:
            if size not in FULL_SHARING_BY_DEFAULT and not full_sharing:
                continue
            for full, base in ((FULL_SHARING["full"], LOCAL), (FULL_SHARING["onehopfull"], ONEHOP)):
                got = compare(_seeds(size, condition, full, ""), _seeds(size, condition, base, ""))
                if got.n:
                    print(f"    {size} {full:<34}{got.mean:>+9.4f}  t={got.t:>+6.2f}  n={got.n}")

    print("\n  wall-clock, minutes (cells run by this script)")
    for key, value in status.items():
        if key.endswith("_minutes") and (suffix in key if suffix else "_smoke" not in key):
            print(f"    {key[:-8]:<36}{value}")
    print(f"\n  * = p_holm < {ALPHA} (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
