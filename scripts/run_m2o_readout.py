r"""M2O -- many-to-one against many-to-many: why is the series task read out whole?

    python scripts/run_m2o_readout.py --lr --device cuda   # tune the baselines per readout
    python scripts/run_m2o_readout.py --device cuda        # the three readouts
    python scripts/run_m2o_readout.py --report-only
    python scripts/run_m2o_readout.py --lr --smoke         # then --smoke

## The question (schedule.md, "M2O"; first raised in the 2026-09-23 review)

Every Mackey--Glass result reads the Transformer out many-to-many: a block of $L=32$
samples gives 31 inputs and 31 one-step-ahead targets, all scored. A reviewer will
ask why not many-to-one -- predict the next sample from a full window, the usual
forecasting setup -- and the answer has to be measured, not asserted. Three arms,
on the **same series** (a shared `history_prefix` makes it identical sample for
sample; `tests/test_many_to_one.py` checks it):

* **m2m** -- many-to-many, the reference, with the prefix so it pairs exactly.
* **M2O-b** -- *supervision-matched*: every one of a block's 31 targets gets its own
  window of the 31 samples before it; only the last position is scored. 31 fresh
  targets a step, rank at most 31, the same Woodbury width $m=31$, no target used
  twice. The fair competitor.
* **M2O-c** -- *sample-matched*: one window per block, the block's own inputs,
  its last target alone. Rank one per block; kept as a labelled illustration of the
  starvation that motivated the design, not as a competitor.

M2O-a, "is the headline inflated by the short-context positions?", needs no arm: a
many-to-many run now records `rmse_last`, the last (full-context) position alone,
beside `rmse` and `rmse_full_context`.

## What it costs, stated before the run

With `Diff_EKF_Resource_Complexity.pdf` section 4.6: M2O-b costs the filter
$31(2C_F+C_B)\approx124\,C_F$ in $C_J$ against $2C_F+31\,C_B\approx64\,C_F$ (about
2x) and nothing extra in $C_W$ -- the last-position readout differentiates one
output per window, not 31 (`models/transformer.py`) -- and costs the gradient
baselines about 31x in forward and backward passes. So if M2O-b matches or beats
many-to-many on accuracy, the case for many-to-many is its cost.

## Named before the run (confirmatory under D118)

1. **Is M2O-b more accurate than many-to-many where both have a full context?**
   M2O-b's RMSE minus m2m's `rmse_last`, per seed, for the centralised filter,
   local adapt, one-hop, centralised AdamW and ATC AdamW; Holm across the five. **No
   direction predicted**: M2O-b devotes the whole model to the last position, m2m
   trains every position and gets 31x the gradient signal per window.

Exploratory: m2m's whole-block `rmse` against M2O-b (different targets' contexts,
so not a like-for-like), M2O-a's three readings of m2m, M2O-c, full sharing, the
SGD family, the calibration columns, and the cells' wall-clock.

## The design

Stationary, the main law, M6's filters (M4, M5) and R scale, five seeds, $T=1500$,
evaluations every 25; cells a (filters with the gradient baselines) and b (full
sharing), as M6. **R** is M4's scaled per-position profile for m2m and its *last*
entry alone for M2O -- the one position M2O scores. **The filters carry M4/M5's
selections, tuned many-to-many** -- a caveat in m2m's favour, stated rather than
fixed. **The gradient baselines are re-tuned per readout** on M3's grids (D77): a
readout is a different problem, and a rate carried across is the mistake D77
records. m2m reuses M3's own selections, whose condition it is. An abrupt cell is
added only if stationary separates the designs (schedule.md).
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
from run_ekf_generalization import run_one  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, ADAMW_RATES, SGD_FAMILY, SGD_RATES, settled  # noqa: E402
from run_m6_comparison import CONDITIONS, GROUP_B_FILTERS, load_settings  # noqa: E402

from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

L = 32
CONTEXT = L - 1
#: readout name -> (model.readout, window_stride). m2m carries the prefix too.
READOUTS = {"m2m": ("sequence", 1), "m2o_b": ("last", 1), "m2o_c": ("last", CONTEXT)}
TUNED = ("m2o_b", "m2o_c")  # m2m reuses M3's selections

CENTRAL = "centralized_ekf_walk"
MEAN_ONLY = ["diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
FILTERS = [CENTRAL, *MEAN_ONLY]
CONFIRMATORY = [*FILTERS, "centralized_adamw", "diffusion_atc_adamw"]
BASELINES = [*SGD_FAMILY, *ADAMW_FAMILY]

HORIZON, SEEDS, EVAL_EVERY, LR_SEEDS = 1500, [0, 1, 2, 3, 4], 25, [0, 1]
#: The SGD grid's extension for the many-to-one readouts (2026-10-02). A last-position
#: readout differentiates one output per window, not 31, so its gradients are smaller
#: and M3's grid -- built for many-to-many, where 1e-4 and 3e-4 diverged -- was outgrown:
#: every m2o_b SGD learner, and centralised SGD and ATC on m2o_c, chose its top, 3e-4,
#: still falling steeply. Recorded runs are kept; only these rates are new.
SGD_EXTRA = [1e-3, 3e-3]
#: The SGD-only cell's reproduction arm (D119): centralised AdamW at cell a's own rate,
#: which must match cell a to this tolerance before the SGD rows are pooled with it.
REPRODUCTION_ARM, REPRODUCTION_TOLERANCE = "centralized_adamw", 1e-9
STATUS = ROOT / "results" / "m2o_status.json"
M3_RATES = ROOT / "results" / "m3_rates.json"
M4_SELECTION = ROOT / "results" / "m4_selection.json"


def cell_name(readout: str, group: str, suffix: str = "") -> str:
    return f"m2o_{readout}_{group}{suffix}"


def lr_name(readout: str, index: int, suffix: str = "") -> str:
    return f"m2o_lr_{readout}_r{index}{suffix}"


def lr_extra_name(readout: str, index: int, suffix: str = "") -> str:
    """An SGD-only tuning cell at ``SGD_EXTRA[index]``."""
    return f"m2o_lr_{readout}_sgdx{index}{suffix}"


def sgd_cell_name(readout: str, suffix: str = "") -> str:
    """The SGD family at its extended-grid rates, beside cell a (D119's pattern)."""
    return cell_name(readout, "sgd", suffix)


def sgd_curve(readout: str, learner: str, suffix: str = "") -> list[tuple[float, float]]:
    """(rate, settled RMSE) over M3's grid and the extension; inf where not run."""
    base = [(rate, settled(lr_name(readout, k, suffix), learner))
            for k, rate in enumerate(SGD_RATES)]
    extra = [(rate, settled(lr_extra_name(readout, k, suffix), learner))
             for k, rate in enumerate(SGD_EXTRA)]
    return base + extra


def extended_sgd_rates(readout: str, suffix: str = "") -> dict[str, float] | None:
    """Each SGD learner's argmin over the extended grid; None until the extension ran."""
    if any(not (ROOT / "results" / lr_extra_name(readout, k, suffix) / "_complete").exists()
           for k in range(len(SGD_EXTRA))):
        return None
    rates = {}
    for learner in SGD_FAMILY:
        finite = [(v, r) for r, v in sgd_curve(readout, learner, suffix) if v != float("inf")]
        if finite:
            rates[learner] = min(finite)[1]
    return rates


def recorded_rates(run: str) -> dict[str, float]:
    """The learning rates a completed cell actually ran with, from its own config."""
    import yaml  # noqa: PLC0415

    path = ROOT / "results" / run / "config.yaml"
    if not path.exists():
        return {}
    learners = yaml.safe_load(path.read_text(encoding="utf-8")).get("learners", [])
    return {e["name"]: e["lr"] for e in learners if e.get("lr") is not None}


def r_profile(readout: str, smoke: bool) -> list[float]:
    """M4's scaled per-position R, or its last entry alone for a many-to-one readout."""
    base = list(load_config(CONDITIONS["stationary"]).model.observation_variances)
    if not smoke and M4_SELECTION.exists():
        scale = json.loads(M4_SELECTION.read_text(encoding="utf-8"))["r_scale"]
        base = [v * scale for v in base]
    return base if READOUTS[readout][0] == "sequence" else base[-1:]


def config_for(args, name: str, readout: str, learners: list[dict], smoke: bool,
               horizon: int, seeds: list[int]):
    model_readout, stride = READOUTS[readout]
    return load_config(CONDITIONS["stationary"], overrides={
        "run": {"name": name, "horizon": horizon, "seeds": seeds,
                "eval_every": 20 if smoke else EVAL_EVERY, "device": args.device,
                "dtype": args.dtype},
        "env": {"series": {"history_prefix": CONTEXT, "window_stride": stride}},
        "model": {"readout": model_readout,
                  "output_dim": 1 if model_readout == "last" else CONTEXT,
                  "observation_variances": r_profile(readout, smoke)},
        "learners": learners,
    })


def baseline_rates(readout: str, smoke: bool, suffix: str = "") -> dict[str, float] | None:
    """Per learner: M3's selection for m2m; this runner's own argmin for M2O."""
    if smoke:
        return {**{n: 3e-5 for n in SGD_FAMILY}, **{n: 3e-3 for n in ADAMW_FAMILY}}
    if readout == "m2m":
        chosen = json.loads(M3_RATES.read_text(encoding="utf-8"))["stationary"]
        return {n: chosen[n]["lr"] for n in BASELINES if n in chosen}
    rates = {}
    for learner in BASELINES:
        grid = ADAMW_RATES if learner in ADAMW_FAMILY else SGD_RATES
        scored = [(settled(lr_name(readout, k, suffix), learner), k) for k in range(len(grid))]
        finite = [(v, k) for v, k in scored if v != float("inf")]
        if not finite:
            return None
        rates[learner] = grid[min(finite)[1]]
    return rates


def entries(group: str, centralised: dict, diffusion: dict, rates: dict) -> list[dict]:
    if group == "b":
        return [{"name": n, **diffusion, "combine_exponent": 1.0} for n in GROUP_B_FILTERS]
    learners = [{"name": CENTRAL, **centralised}]
    learners += [{"name": n, **diffusion} for n in MEAN_ONLY]
    learners += [{"name": n, "lr": rates[n], **o} for n, o in SGD_FAMILY.items() if n in rates]
    learners += [{"name": n, "lr": rates[n], **o} for n, o in ADAMW_FAMILY.items() if n in rates]
    return learners


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def note_status(entries: dict) -> None:
    """Merge into the status file as it is now, so a concurrent sweep's entries survive."""
    status = load_status()
    status.update(entries)
    save_status(status)


def extend_sgd(args, suffix: str, horizon: int) -> int:
    """Tune SGD_EXTRA for the M2O readouts, then run each readout's SGD-only cell if needed.

    Nothing recorded is touched: the extra tuning cells and the SGD-only cells have names
    of their own. A readout gets its SGD cell only when the extended selection differs
    from the rates its cell a ran with; the cell carries the reproduction arm at cell a's
    rate, and the report pools its SGD rows with cell a only if that arm reproduces.
    """
    lr_seeds = [0] if suffix else LR_SEEDS
    seeds = [0] if suffix else args.seeds
    cells = [(readout, k) for readout in TUNED for k in range(len(SGD_EXTRA))]
    print(f"M2O SGD extension{' SMOKE' if suffix else ''}: {len(cells)} tuning cells at "
          f"{', '.join(f'{r:g}' for r in SGD_EXTRA)}, {len(lr_seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (readout, k) in enumerate(cells, start=1):
        learners = [{"name": n, "lr": SGD_EXTRA[k], **o} for n, o in SGD_FAMILY.items()]
        name = lr_extra_name(readout, k, suffix)
        note = run_one(config_for(args, name, readout, learners, bool(suffix), horizon,
                                  lr_seeds), None, None, args.fresh)
        note_status({name: note})
        print(f"[{index}/{len(cells)}] {name:<30} {note:<20} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    for readout in TUNED:
        rates = extended_sgd_rates(readout, suffix) or {}
        recorded = recorded_rates(cell_name(readout, "a", suffix))
        print(f"\n  {readout}: settled RMSE over the extended SGD grid")
        for learner in SGD_FAMILY:
            curve = sgd_curve(readout, learner, suffix)
            edge = rates.get(learner) == SGD_EXTRA[-1]
            print(f"    {learner:<26}" + "  ".join(f"{r:g}:{v:.4f}" for r, v in curve)
                  + f"   -> {rates.get(learner, float('nan')):g}"
                  + ("  EDGE: still the top of the grid" if edge else "")
                  + f"  (cell a ran {recorded.get(learner, float('nan')):g})")
        changed = {n: r for n, r in rates.items() if recorded.get(n) != r}
        if not recorded:
            print(f"  {readout}: cell a not recorded yet; run the main pass first, then this.")
            continue
        if not changed:
            print(f"  {readout}: the extension changes no selection; cell a stands.")
            continue
        arm = {"name": REPRODUCTION_ARM, "lr": recorded[REPRODUCTION_ARM],
               **ADAMW_FAMILY[REPRODUCTION_ARM]}
        learners = [{"name": n, "lr": rates[n], **o} for n, o in SGD_FAMILY.items()] + [arm]
        name = sgd_cell_name(readout, suffix)
        began = time.time()
        note = run_one(config_for(args, name, readout, learners, bool(suffix), horizon, seeds),
                       None, None, args.fresh)
        entries = {name: note}
        if note != "cached":
            entries[f"{name}__seconds"] = round(time.time() - began, 1)
        note_status(entries)
        print(f"  {readout}: {name} {note} ({(time.time() - began) / 60:.0f} min), "
              f"changed: " + ", ".join(f"{n} {r:g}" for n, r in changed.items()), flush=True)
    print()
    report(suffix)
    return 0


def tune(args, suffix: str, horizon: int) -> int:
    """M3's grids, once per many-to-one readout, both families per cell as M3 does."""
    status = load_status()
    seeds = [0] if suffix else LR_SEEDS
    cells = [(readout, k) for readout in TUNED for k in range(len(SGD_RATES))]
    print(f"M2O lr{' SMOKE' if suffix else ''}: {len(cells)} cells, {len(seeds)} seed(s)\n",
          flush=True)
    started = time.time()
    for index, (readout, k) in enumerate(cells, start=1):
        learners = [{"name": n, "lr": SGD_RATES[k], **o} for n, o in SGD_FAMILY.items()]
        learners += [{"name": n, "lr": ADAMW_RATES[k], **o} for n, o in ADAMW_FAMILY.items()]
        name = lr_name(readout, k, suffix)
        note = run_one(config_for(args, name, readout, learners, bool(suffix), horizon, seeds),
                       None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<28} {note:<20} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    for readout in TUNED:
        rates = baseline_rates(readout, False, suffix)
        print(f"\n  {readout}: " + ", ".join(f"{n} {r:g}" for n, r in (rates or {}).items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS, device="cpu")
    parser.add_argument("--lr", action="store_true", help="tune the baselines per M2O readout")
    parser.add_argument("--extend-sgd", action="store_true",
                        help="tune the SGD family at SGD_EXTRA, then run each readout's "
                             "SGD-only cell where the selection moved (after the main pass)")
    parser.add_argument("--smoke", action="store_true", help="short horizon, one seed")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0

    import torch  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    horizon = 40 if args.smoke else args.horizon
    seeds = [0] if args.smoke else args.seeds
    if args.lr:
        return tune(args, suffix, horizon)
    if args.extend_sgd:
        return extend_sgd(args, suffix, horizon)
    centralised, diffusion, _m6_rates = load_settings(args.smoke)
    if not args.smoke and not M3_RATES.exists():
        print("  REFUSED: m3_rates.json is missing; m2m reuses M3's selections.")
        return 1

    cells = [(readout, group) for readout in READOUTS for group in ("a", "b")]
    missing = [r for r in READOUTS if baseline_rates(r, args.smoke, suffix) is None]
    if missing:
        print(f"  REFUSED: no baseline rates for {missing}: run --lr first (D77).")
        return 1
    print(f"M2O{' SMOKE' if suffix else ''}: {len(cells)} cells x {len(seeds)} seeds, "
          f"T={horizon}, stationary, device {args.device}\n", flush=True)
    status = load_status()
    started = time.time()
    for index, (readout, group) in enumerate(cells, start=1):
        name = cell_name(readout, group, suffix)
        learners = entries(group, centralised, diffusion,
                           baseline_rates(readout, args.smoke, suffix))
        began = time.time()
        note = run_one(config_for(args, name, readout, learners, args.smoke, horizon, seeds),
                       None, None, args.fresh)
        status[name] = note
        if note != "cached":
            status[f"{name}__seconds"] = round(time.time() - began, 1)
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<24} {len(learners):>2} learners  {note:<14} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nM2O{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def seed_values(run: str, learner: str, metric: str) -> dict[int, float]:
    """Per seed: the metric on the current set, settled (last 20%), over agents."""
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    if not (directory / "_complete").exists():
        return {}
    out = {}
    for path in sorted(directory.glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["learner", "metric", "evalset", "t", "value"])
        rows = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                     & (frame["evalset"] == "current") & (frame["t"] >= 0.8 * frame["t"].max())]
        if len(rows):
            out[int(path.stem.split("_")[1])] = float(rows["value"].mean())
    return out


def report(suffix: str = "") -> None:
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    group = lambda learner: "b" if learner in GROUP_B_FILTERS else "a"  # noqa: E731
    if suffix:
        print("  SMOKE: a short run at one seed with placeholder settings. These numbers")
        print("  mean nothing; the point is that every code path below ran.\n")

    # ---- the SGD extension's merge gate (D119) ------------------------------------
    pooled = set()
    for readout in TUNED:
        extra = sgd_cell_name(readout, suffix)
        if not (ROOT / "results" / extra / "_complete").exists():
            continue
        recorded = seed_values(cell_name(readout, "a", suffix), REPRODUCTION_ARM, "rmse")
        rerun = seed_values(extra, REPRODUCTION_ARM, "rmse")
        shared = sorted(set(recorded) & set(rerun))
        worst = max((abs(recorded[s] - rerun[s]) for s in shared), default=None)
        ok = worst is not None and worst <= REPRODUCTION_TOLERANCE
        if ok:
            pooled.add(readout)
        shown = f"{worst:.1e}" if worst is not None else "-"
        print(f"  merge gate, {extra}: {REPRODUCTION_ARM} against cell a, {len(shared)} seed(s), "
              f"max |diff| {shown}  {'reproduces' if ok else 'DOES NOT REPRODUCE'}")
    for readout in TUNED:
        if readout not in pooled:
            print(f"  {readout}: SGD rows from cell a, on M3's grid -- EDGE-LIMITED "
                  f"(see the extension, 2026-10-02)")
    print()

    def cell(readout: str, learner: str) -> str:
        if learner in SGD_FAMILY and readout in pooled:
            return sgd_cell_name(readout, suffix)
        return cell_name(readout, group(learner), suffix)

    learners = [*FILTERS, *GROUP_B_FILTERS, *BASELINES]
    print("  settled RMSE, current set. m2m reads three ways (M2O-a): every position,")
    print("  the positions with a full delay of context, and the last position alone.\n")
    print(f"    {'learner':<36}{'m2m all':>9}{'m2m 16+':>9}{'m2m last':>9}"
          f"{'M2O-b':>9}{'M2O-c':>9}")
    for learner in learners:
        values = [mean(seed_values(cell("m2m", learner), learner, m))
                  for m in ("rmse", "rmse_full_context", "rmse_last")]
        values += [mean(seed_values(cell(r, learner), learner, "rmse")) for r in ("m2o_b", "m2o_c")]
        print(f"    {learner:<36}" + "".join(f"{v:>9.4f}" for v in values))

    rows = []
    for learner in CONFIRMATORY:
        b = seed_values(cell("m2o_b", learner), learner, "rmse")
        m = seed_values(cell("m2m", learner), learner, "rmse_last")
        rows.append((learner, compare(b, m) if b and m else None))
    live = [(lab, r) for lab, r in rows if r is not None]
    adjusted = dict(zip([lab for lab, _r in live], holm([r.p for _l, r in live]), strict=True))
    print("\n  CONFIRMATORY (D126): M2O-b RMSE minus m2m's last-position RMSE, paired per")
    print("  seed; negative = many-to-one is more accurate at full context. Holm across five.\n")
    for learner, result in rows:
        if result is None:
            print(f"    {learner:<36}(cells missing)")
            continue
        lo, hi = result.ci(0.95)
        mark = "*" if adjusted[learner] < 0.05 else " "
        print(f"    {learner:<36}{result.mean:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]  t {result.t:>+6.2f}"
              f"  p_holm {adjusted[learner]:.3f}{mark}  n {result.n}")

    print("\n  exploratory: the filters' calibration by readout (variance ratio at kappa = 1)\n")
    for learner in [*FILTERS, *GROUP_B_FILTERS]:
        values = [mean(seed_values(cell(r, learner), learner, "variance_ratio")) for r in READOUTS]
        print(f"    {learner:<36}" + "".join(f"{v:>9.3f}" for v in values))

    status = load_status()
    print("\n  exploratory: wall-clock per cell, seconds (cells hold several learners)\n")
    for readout in READOUTS:
        for grp in ("a", "b"):
            seconds = status.get(f"{cell_name(readout, grp, suffix)}__seconds")
            print(f"    {cell_name(readout, grp, suffix):<26}{seconds if seconds else '-':>10}")


if __name__ == "__main__":
    raise SystemExit(main())
