r"""M14 -- the break rate on Mackey--Glass: a beta ramp against its stationary twin.

    python scripts/run_m14_break_rate.py --lr --device cuda   # tune the baselines on the ramp, first
    python scripts/run_m14_break_rate.py --device cuda        # the ramp and its twin
    python scripts/run_m14_break_rate.py --report-only
    python scripts/run_m14_break_rate.py --lr --smoke         # then --smoke: the path, not the numbers

`--lr` must run first and the main pass refuses without it (D77).

`docs/mackey_glass_plan.md` M14, the P5.5 analogue ([[D123]]): on MNIST the centralised
filter breaks at 0.0639 deg/step against 0.038--0.040 for the gradient methods, the one
place its advantage is a *rate*. M14 asks the same of a law that drifts.

## The ramp, transposed

X9's ramp unchanged in schedule units -- 45 "degrees", exponent 6, peak 0.18 deg/step at
$T=1500$, evaluations every 10 -- on the $\beta$ channel, where 45 degrees is the
channel's full span (0.02): $\beta$ climbs from 0.22 to 0.24, the upper half of the
chaotic window (D96), as M6's linear condition does at a constant 0.03 deg/step. The
twin is M6's stationary law. Both are built from M6's recorded config, with the drift,
the cadence, the learners and the name replaced, so they differ in the drift alone
(`assert_paired_runs` is a gate in the report). ER 0.3, $N=10$, five seeds.

## The pilot that shaped it (M6's linear cell, measured 2026-09-27)

At 0.03 deg/step the filters' damage is the law getting harder and nothing more: the
centralised filter pays $+0.0050$ and one-hop $+0.0056$ by the end against a
difficulty shift of $+0.0054$ (M2's $e^\star(\beta)$ fit, slope 0.274). The SGD family
lags about $+0.005$ beyond it, AdamW about $+0.0015$. So:

* **Contrasts between learners are clean.** At a matched rate every learner sits at the
  same $\beta$, so the difficulty shift is common and cancels.
* **Absolute break rates are not.** Damage on a ramp grows with displacement even for a
  perfect tracker, and a pooled bar would fire on difficulty alone. The break-rate table
  therefore subtracts $e^\star(\beta_t)-e^\star(0.22)$, from M2's linear fit (D97: gaps
  are read against the fit, residual s.d. 0.0015), before locating anything.

## What is tuned

The seven baselines on the ramp itself, by **whole-run** mean RMSE -- the ramp's settled
window is its fastest drift and would pick a rate suited to nothing slower -- on M3's
grids at M3's seeds, and the same rates carried into the twin, as P5.5 did. The filters
carry M6's entries (M4, M5). `frozen_atc` runs at ATC's tuned rate, frozen at 300.

## Named before the run (confirmatory under D118)

**Damage at the matched rate, 0.10 deg/step** ($\beta$ moving $4.4\times10^{-5}$ per
step, reached at 89% of the run), drifting minus twin per seed, Holm across five --
P5.5's family verbatim, so the two tasks answer the same question:

1. one-hop - ATC. Predicted negative.
2. local adapt - ATC. No prediction.
3. one-hop - centralised filter. Predicted positive (the pilot: $+0.0006$ at 0.03).
4. local adapt - centralised filter. Predicted positive (the pilot: $+0.0024$).
5. one-hop - ATC AdamW. Predicted negative (the pilot: $-0.0013$). Same cell here, so
   no merge gate is needed.

Exploratory: difficulty-corrected break rates on one bar pooled across every adapting
learner, the comparative break against `frozen_atc`, full sharing, the damage table.
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

import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, ADAMW_RATES, SGD_FAMILY, SGD_RATES  # noqa: E402
from run_m8_tau_heterogeneity import ALPHA, STAT_HEADER, _stat_columns  # noqa: E402

from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

SOURCE = {"a": "m6_stationary_a", "b": "m6_stationary_b"}
RAMP = {"schedule": "ramp", "total_degrees": 45.0, "ramp_exponent": 6.0}
CONDITIONS = ["ramp", "control"]
GROUPS = ["a", "b"]
MATCHED_RATE = 0.10
NOISE_MULTIPLE = 3.0
PERSISTENCE = 3
EVAL_EVERY = 10
FREEZE_AFTER = 300

CENTRAL = "centralized_ekf_walk"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
ATC, ATC_ADAMW = "diffusion_sgd_atc", "diffusion_atc_adamw"
FROZEN = "frozen_atc"
BASELINES = {**SGD_FAMILY, **ADAMW_FAMILY}
#: M6's gamma = 0.9995 reference arms, left out here as in M8 and M9.
DROPPED = {"centralized_ekf_gamma", "diffusion_ekf_onehop_mean_receiver_gamma"}

LR_SEEDS = [0, 1]
#: A ramp scales with T, so 40 steps still passes 0.10 deg/step; the freeze moves to
#: the midpoint so the comparative break has something to read (the smoke only).
SMOKE_HORIZON = 40
STATUS = ROOT / "results" / "m14_status.json"
REFERENCES = ROOT / "results" / "m2_references.json"


def lr_run_name(index: int, suffix: str = "") -> str:
    return f"m14_lr_r{index}{suffix}"


def cell_name(condition: str, group: str, suffix: str = "") -> str:
    return f"m14_{condition}_{group}{suffix}"


def recorded(group: str) -> dict:
    return yaml.safe_load((ROOT / "results" / SOURCE[group] / "config.yaml")
                          .read_text(encoding="utf-8"))


def rate_of(learner: str, index: int) -> float:
    return (ADAMW_RATES if learner in ADAMW_FAMILY else SGD_RATES)[index]


def config_for(group: str, name: str, condition: str, entries: list[dict], args, suffix: str,
               seeds: list[int] | None = None):
    """M6's recorded stationary config; the ramp replaces only its drift block."""
    run = {"name": name, "seeds": seeds or recorded(group)["run"]["seeds"],
           "eval_every": EVAL_EVERY, "device": args.device}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "eval_every": 1, "seeds": [0]})
    overrides = {"run": run, "learners": entries}
    if condition == "ramp":
        overrides["env"] = {"drift": dict(RAMP)}
    return load_config(ROOT / "results" / SOURCE[group] / "config.yaml", overrides=overrides)


def whole_run(run: str, learner: str) -> float:
    """Mean `current` RMSE over every evaluation and seed: the ramp's tuning criterion."""
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    if not (directory / "_complete").exists():
        return float("inf")
    values = []
    for path in sorted(directory.glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["learner", "metric", "evalset", "value"])
        rows = frame[(frame["learner"] == learner) & (frame["metric"] == "rmse")
                     & (frame["evalset"] == "current")]
        if len(rows):
            values.append(float(rows["value"].mean()))
    value = sum(values) / len(values) if values else float("inf")
    return value if math.isfinite(value) else float("inf")


def selected_rates(suffix: str = "") -> dict[str, dict] | None:
    chosen = {}
    for learner in BASELINES:
        scored = [(whole_run(lr_run_name(k, suffix), learner), k) for k in range(len(SGD_RATES))]
        finite = [(v, k) for v, k in scored if v != float("inf")]
        if not finite:
            return None
        value, index = min(finite)
        chosen[learner] = {"lr": rate_of(learner, index), "whole_run_rmse": value,
                           "edge": index in (0, len(SGD_RATES) - 1)}
    return chosen


def entries_for(group: str, rates: dict[str, dict], suffix: str) -> list[dict]:
    """M6's recorded filter entries; the baselines at the ramp's rates; frozen ATC."""
    out = []
    for entry in recorded(group)["learners"]:
        if entry["name"] in DROPPED:
            continue
        entry = dict(entry)
        if entry["name"] in BASELINES:
            entry["lr"] = rates[entry["name"]]["lr"]
        out.append(entry)
    if group == "a":
        out.append({"name": FROZEN, "lr": rates[ATC]["lr"], **SGD_FAMILY[ATC],
                    "freeze_after": SMOKE_HORIZON // 2 if suffix else FREEZE_AFTER})
    return out


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, suffix: str) -> int:
    status = load_status()
    print(f"M14 lr{' SMOKE' if suffix else ''}: the ramp at {len(SGD_RATES)} rate pairs, "
          "seven baselines each, whole-run RMSE\n", flush=True)
    started = time.time()
    for k in range(len(SGD_RATES)):
        name = lr_run_name(k, suffix)
        entries = [{"name": n, "lr": rate_of(n, k), **o} for n, o in BASELINES.items()]
        note = run_one(config_for("a", name, "ramp", entries, args, suffix, seeds=LR_SEEDS),
                       None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{k + 1}/{len(SGD_RATES)}] {name:<18} sgd {SGD_RATES[k]:<7g} adamw "
              f"{ADAMW_RATES[k]:<6g} {note:<24} {(time.time() - started) / 60:.0f} min",
              flush=True)
    rates = selected_rates(suffix)
    print()
    for name, pick in (rates or {}).items():
        print(f"  {name:<28} {pick['lr']:<8g} whole-run {pick['whole_run_rmse']:.4f}"
              + ("   <- grid edge" if pick["edge"] else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2, 3, 4])
    parser.add_argument("--lr", action="store_true", help="tune the baselines on the ramp")
    parser.add_argument("--smoke", action="store_true", help="40 rounds, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0
    missing = [c for c in SOURCE.values() if not (ROOT / "results" / c / "_complete").exists()]
    if missing:
        print(f"  REFUSED: M14 is built from M6's recorded cells, and {missing} is not finished.")
        return 1
    if args.lr:
        return tune(args, suffix)
    rates = selected_rates(suffix)
    if rates is None:
        print("  REFUSED: no selected rates. Run --lr first (D77).")
        return 1
    edges = [n for n, p in rates.items() if p["edge"]]
    if edges:
        print(f"  WARNING: selected rates on their grid's edge: {', '.join(edges)}\n")

    cells = [(c, g) for g in GROUPS for c in CONDITIONS]
    print(f"M14{' SMOKE' if suffix else ''}: {len(cells)} cells, ramp and twin, device "
          f"{args.device}\n", flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (condition, group) in enumerate(cells, start=1):
        name = cell_name(condition, group, suffix)
        learners = entries_for(group, rates, suffix)
        cell_started = time.time()
        note = run_one(config_for(group, name, condition, learners, args, suffix),
                       None, None, args.fresh)
        status[name] = note
        if note != "cached":
            ran += 1
            status[f"{name}_minutes"] = round((time.time() - cell_started) / 60, 1)
        save_status(status)
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<22} {len(learners):>2} learners  {note:<14}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nM14{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def difficulty_fit() -> tuple[float, float]:
    """(intercept, slope) of M2's e*(beta), least squares over its thirteen levels (D97)."""
    import numpy as np  # noqa: PLC0415

    references = json.loads(REFERENCES.read_text(encoding="utf-8"))
    betas = np.array([entry["value"] for entry in references.values()])
    e_star = np.array([entry["rmse"] for entry in references.values()])
    slope, intercept = np.polyfit(betas, e_star, 1)
    return float(intercept), float(slope)


def report(suffix: str = "") -> None:
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
        print("  SMOKE: 40 rounds at one seed. The numbers mean nothing; the point is")
        print("  that every code path below ran.\n")

    def load(run: str) -> pd.DataFrame:
        files = sorted(glob.glob(str(ROOT / "results" / run / "seed_*.parquet")))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files \
            else pd.DataFrame()

    print("  pairing gate: each ramp cell against its twin (only the drift may differ)")
    frames, pair_ok = [], True
    for group in GROUPS:
        ramp, twin = cell_name("ramp", group, suffix), cell_name("control", group, suffix)
        try:
            assert_paired_runs(ROOT / "results" / ramp, ROOT / "results" / twin)
            frames.append(paired_excess(load(ramp), load(twin), metric="rmse"))
            print(f"    {group:<4} paired")
        except (BreakError, KeyError, AttributeError) as failure:
            pair_ok = False
            print(f"    {group:<4} NOT PAIRED: {str(failure)[:90]}")
    if not frames:
        print("\n  nothing to report yet")
        return
    excess = pd.concat(frames, ignore_index=True)

    ramp_config = load_config(ROOT / "results" / cell_name("ramp", "a", suffix) / "config.yaml")
    drift, horizon = build_drift(ramp_config), ramp_config.run.horizon
    series = ramp_config.env.series
    intercept, slope = difficulty_fit()

    def beta_at(step: int) -> float:
        return series.beta + series.span * drift.rotation_at(int(step)) / 45.0

    steps = sorted(excess.t.unique())
    reached = [s for s in steps if drift.schedule.rate_at(int(s)) >= MATCHED_RATE]
    if not reached:
        print(f"\n  the ramp never reaches {MATCHED_RATE} deg/step in this run")
        return
    matched = reached[0]
    shift = slope * (beta_at(matched) - series.beta)

    def damage(learner: str) -> dict[int, float]:
        rows = excess[(excess.learner == learner) & (excess.t == matched)]
        return {int(s): float(v) for s, v in zip(rows.seed, rows.excess, strict=True)}

    learners = sorted(excess.learner.unique())
    print(f"\n  damage at the matched rate {MATCHED_RATE} deg/step (t={matched}, beta "
          f"{beta_at(matched):.4f}), drifting minus twin, seed mean.")
    print(f"  The law's own difficulty shift there is {shift:+.4f} (M2 fit, slope "
          f"{slope:.3f}); a perfect tracker shows about that.\n")
    for learner in learners:
        values = damage(learner)
        mean = sum(values.values()) / len(values)
        print(f"    {learner:<40}{mean:>+9.4f}   beyond difficulty {mean - shift:>+9.4f}")

    contrasts = [("one-hop - ATC", ONEHOP, ATC), ("local adapt - ATC", LOCAL, ATC),
                 ("one-hop - centralised filter", ONEHOP, CENTRAL),
                 ("local adapt - centralised filter", LOCAL, CENTRAL),
                 ("one-hop - ATC AdamW", ONEHOP, ATC_ADAMW)]
    rows = [(label, compare(damage(a), damage(b))) for label, a, b in contrasts]
    live = [(label, r) for label, r in rows if r.n >= 2]
    print("\n  CONFIRMATORY (D130): damage at the matched rate, paired per seed; negative")
    print("  = the first is less damaged. Predicted: one-hop < ATC, one-hop > central,")
    print("  local > central, one-hop < ATC AdamW; none for local vs ATC. Holm across five.")
    print("  The difficulty shift is common to every learner at one step, so it cancels.\n")
    print(f"    {'':<36}{STAT_HEADER}")
    for (label, result), p_adj in zip(live, holm([r.p for _l, r in live]), strict=True):
        print(f"    {label:<36}{_stat_columns(result, p_adj)}")

    corrected = excess.copy()
    corrected["excess"] = corrected.excess - [slope * (beta_at(t) - series.beta)
                                              for t in corrected.t]
    adapting = [n for n in learners if n != FROZEN]
    noise = pooled_sem(corrected, adapting)
    errors = pd.concat([error_by_step(load(cell_name("ramp", "a", suffix)), metric="rmse")],
                       ignore_index=True)
    recorded_a = yaml.safe_load((ROOT / "results" / cell_name("ramp", "a", suffix)
                                 / "config.yaml").read_text(encoding="utf-8"))
    freeze_after = next((e.get("freeze_after") for e in recorded_a["learners"]
                         if e["name"] == FROZEN), FREEZE_AFTER)
    print("\n  EXPLORATORY: break rates on the difficulty-corrected damage, one bar pooled")
    print(f"  across every adapting learner ({NOISE_MULTIPLE:g} s.e.m., {PERSISTENCE} consecutive"
          " evaluations, as X9);")
    print(f"  and the comparative break against {FROZEN} from its freeze point ({freeze_after}).")
    print(f"  Peak rate probed {drift.schedule.peak_rate(horizon):.3f} deg/step.\n")
    for learner in learners:
        absolute = excess_break(corrected, learner, drift, horizon, NOISE_MULTIPLE,
                                PERSISTENCE, noise=noise)
        comparative = "--"
        if learner not in (FROZEN, *FULL_SHARING):
            try:
                point = comparative_break(errors[errors.learner.isin([learner, FROZEN])],
                                          learner, FROZEN, drift, horizon, PERSISTENCE,
                                          start_step=freeze_after)
                comparative = (f"{point.rate_at_break:.4f}" if point.broke
                               else ("never ahead" if point.note else "no break"))
            except BreakError:
                comparative = "--"
        rate = f"{absolute.rate_at_break:.4f}" if absolute.broke else "no break"
        print(f"    {learner:<40} absolute {rate:>9}   comparative {comparative:>11}")

    if not pair_ok:
        print("\n  WARNING: a group failed the pairing gate; its rows are omitted above.")
    print(f"\n  * = p_holm < {ALPHA} (D113). Break rates are exploratory; the damage at a")
    print("  matched rate is the reading X9 found to prefer.")


if __name__ == "__main__":
    raise SystemExit(main())
