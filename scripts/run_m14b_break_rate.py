r"""M14b -- where the Mackey--Glass learners break: a triangle wave, two channels.

    python scripts/run_m14b_break_rate.py --pilot --device cuda    # 10 cells, ~7 h
    python scripts/run_m14b_break_rate.py --pilot --report-only
    python scripts/run_m14b_break_rate.py --pilot --smoke          # the path, not the numbers

`docs/mackey_glass_plan.md` M14b, [[D140]]. M14 (D130) ramped $\beta$ to 0.18 deg/step
and nothing broke: the ramp ran out of window before it ran out of rate. M14b raises
the rate without leaving the band, by turning back at its edges.

## The schedule

`triangle` ([[D140]]): back and forth across the band at a constant speed, so "damage
at rate $r$" is the damage at $r$ -- not a cycle's average, as a sinusoid's would be.
A full cycle is a path of $4\times45 = 180$ degrees, so period $P$ means $180/P$
deg/step. Two arms, each across one noise $\sigma$ either side of rest (D108's
scaling, so the arms differ in what moves and not by how much):

* **beta**: $\beta \in [0.20, 0.24]$, the chaotic window (D96), span 0.02.
* **gain**: $g \in [0.9, 1.1]$, the sensor, span 0.1 (M11's).

Each arm has its own stationary twin, the rest law under that arm's channel, because
the pairing gate (`assert_paired_runs`) compares the whole `env` block.

## The pilot (this script, `--pilot`)

Four constant speeds, multiples of M14's peak: **1x, 5x, 25x, 100x** of 0.18 deg/step,
periods 1000, 200, 40 and 10. Two seeds, **100 and 101**, kept apart from the main
run's 0--9 so that pilot and confirmation never share a draw.

**Evaluations every 9 steps, not 10.** The held-out set is built at the channel's value
at the evaluated step. A cadence that divides the period scores every evaluation at
one phase: at period 10 with a cadence of 10 that phase is the rest value, the twin's
own, and every damage reads zero. Nine is coprime with all four periods, so the
evaluations walk through every phase.

Learners: M6's centralised filter, local adapt and one-hop (M4, M5); ATC, ATC AdamW
and local only at **M14's tuned rates** (the pilot is not confirmatory, and those
rates were tuned on the nearest drift we have); `frozen_atc` at ATC's rate, frozen at
300. Full sharing waits for the main run.

## What the pilot reads (all exploratory, D118)

Per arm and speed, over the settled window (the last 20% of evaluations):

1. **damage**, drifting minus twin, per learner;
2. **margin over frozen ATC**, the learner minus `frozen_atc` in the same cell --
   twin-free, and the direct reading of "is adapting still buying anything here";
3. **the comparative break** against `frozen_atc` from its freeze point.

The main sweep's range is chosen from these so the break falls mid-run.
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
from run_m3_rates import SGD_FAMILY  # noqa: E402
from run_m14_break_rate import (  # noqa: E402
    ATC,
    ATC_ADAMW,
    CENTRAL,
    FREEZE_AFTER,
    FROZEN,
    LOCAL,
    ONEHOP,
    PERSISTENCE,
    SOURCE,
    recorded,
    selected_rates,
)

from dekf_bench.utils.config import load_config  # noqa: E402

M14_PEAK = 0.18
#: One full triangle cycle is 0 -> +45 -> -45 -> 0: a path of 180 schedule degrees.
CYCLE_DEGREES = 4 * 45.0
ARMS = {"beta": {"channel": "beta", "span": 0.02}, "gain": {"channel": "gain", "span": 0.1}}
PILOT_MULTIPLES = [1, 5, 25, 100]
PILOT_SEEDS = [100, 101]
PILOT_EVAL_EVERY = 9
SETTLED = 0.8
LOCAL_ONLY = "local_only"
PILOT_LEARNERS = [CENTRAL, LOCAL, ONEHOP, ATC, ATC_ADAMW, LOCAL_ONLY]

SMOKE_HORIZON = 40
#: A report column: a seed mean and the two per-seed values.
WIDTH = 30
SMOKE_EVAL_EVERY = 3
STATUS = ROOT / "results" / "m14b_status.json"


def period_of(multiple: float) -> int:
    period = CYCLE_DEGREES / (M14_PEAK * multiple)
    if abs(period - round(period)) > 1e-9:
        raise ValueError(f"{multiple}x M14's peak is not a whole period ({period})")
    return int(round(period))


def pilot_cell(arm: str, multiple: int | None, suffix: str = "") -> str:
    return f"m14b_pilot_{arm}_{'twin' if multiple is None else f'x{multiple}'}{suffix}"


def pilot_entries(rates: dict[str, dict], suffix: str) -> list[dict]:
    """M6's recorded filters; the baselines at M14's tuned rates; frozen ATC."""
    out = []
    for entry in recorded("a")["learners"]:
        if entry["name"] not in PILOT_LEARNERS:
            continue
        entry = dict(entry)
        if entry["name"] in rates:
            entry["lr"] = rates[entry["name"]]["lr"]
        out.append(entry)
    out.append({"name": FROZEN, "lr": rates[ATC]["lr"], **SGD_FAMILY[ATC],
                "freeze_after": SMOKE_HORIZON // 2 if suffix else FREEZE_AFTER})
    return out


def pilot_config(arm: str, multiple: int | None, entries: list[dict], args, suffix: str):
    """M6's recorded stationary config (group a); the arm's channel, and the triangle."""
    run = {"name": pilot_cell(arm, multiple, suffix), "seeds": PILOT_SEEDS,
           "eval_every": PILOT_EVAL_EVERY, "device": args.device}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "eval_every": SMOKE_EVAL_EVERY,
                    "seeds": PILOT_SEEDS[:1]})
    env: dict = {"series": dict(ARMS[arm])}
    if multiple is not None:
        env["drift"] = {"schedule": "triangle", "amplitude_degrees": 45.0,
                        "period": period_of(multiple)}
    return load_config(ROOT / "results" / SOURCE["a"] / "config.yaml",
                       overrides={"run": run, "learners": entries, "env": env})


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def announce(arm: str, entries: list[dict], args) -> None:
    """The band each arm actually spans, read from the channel map rather than assumed."""
    from dekf_bench.env.series import channel_value  # noqa: PLC0415

    series = pilot_config(arm, 1, entries, args, "").env.series
    low, rest, high = (float(channel_value(series, d)) for d in (-45.0, 0.0, 45.0))
    print(f"  {arm:<5} {series.channel} {low:.4f} <-> {high:.4f}, rest {rest:.4f}")


def pilot(args, suffix: str) -> int:
    rates = selected_rates()
    if rates is None:
        print("  REFUSED: M14's tuned rates are not on disk (run M14's --lr first).")
        return 1
    cells = [(arm, m) for arm in ARMS for m in [None, *PILOT_MULTIPLES]]
    entries = pilot_entries(rates, suffix)
    print(f"M14b pilot{' SMOKE' if suffix else ''}: {len(cells)} cells, device {args.device}")
    for arm in ARMS:
        announce(arm, entries, args)
    print("  speeds " + ", ".join(f"{m}x = {M14_PEAK * m:g} deg/step (period "
                                   f"{period_of(m)})" for m in PILOT_MULTIPLES) + "\n",
          flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (arm, multiple) in enumerate(cells, start=1):
        name = pilot_cell(arm, multiple, suffix)
        cell_started = time.time()
        note = run_one(pilot_config(arm, multiple, entries, args, suffix), None, None,
                       args.fresh)
        status[name] = note
        if note != "cached":
            ran += 1
            status[f"{name}_minutes"] = round((time.time() - cell_started) / 60, 1)
        save_status(status)
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<26} {len(entries)} learners  {note:<14}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nM14b pilot{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report_pilot(suffix)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=PILOT_SEEDS)
    parser.add_argument("--pilot", action="store_true", help="the four constant speeds")
    parser.add_argument("--smoke", action="store_true", help="40 rounds, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if not args.pilot:
        print("  The main sweep is set from the pilot's results ([[D140]]) and is not built "
              "yet. Run --pilot.")
        return 1
    if args.report_only:
        report_pilot(suffix)
        return 0
    missing = [c for c in SOURCE.values() if not (ROOT / "results" / c / "_complete").exists()]
    if missing:
        print(f"  REFUSED: M14b is built from M6's recorded cells, and {missing} is not "
              "finished.")
        return 1
    return pilot(args, suffix)


# =========================================================================== #
# report
# =========================================================================== #

def report_pilot(suffix: str = "") -> None:
    import glob  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from dekf_bench.env.drift import build_drift  # noqa: PLC0415
    from dekf_bench.metrics.breaks import (  # noqa: PLC0415
        BreakError,
        assert_paired_runs,
        comparative_break,
        error_by_step,
        paired_excess,
    )

    if suffix:
        print("  SMOKE: 40 rounds at one seed. The numbers mean nothing; the point is")
        print("  that every code path below ran.\n")

    def load(run: str) -> pd.DataFrame:
        files = sorted(glob.glob(str(ROOT / "results" / run / "seed_*.parquet")))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files \
            else pd.DataFrame()

    def done(run: str) -> bool:
        return (ROOT / "results" / run / "_complete").exists()

    for arm in ARMS:
        twin = pilot_cell(arm, None, suffix)
        if not done(twin):
            print(f"  {arm}: the twin is not finished\n")
            continue
        twin_frame = load(twin)
        damage, margin, breaks, order = {}, {}, {}, []
        for multiple in PILOT_MULTIPLES:
            cell = pilot_cell(arm, multiple, suffix)
            if not done(cell):
                continue
            try:
                assert_paired_runs(ROOT / "results" / cell, ROOT / "results" / twin)
            except BreakError as failure:
                print(f"  {cell}: NOT PAIRED, skipped: {str(failure)[:90]}")
                continue
            order.append(multiple)
            frame = load(cell)
            config = load_config(ROOT / "results" / cell / "config.yaml")
            horizon = config.run.horizon
            settled_from = SETTLED * horizon

            excess = paired_excess(frame, twin_frame, metric="rmse")
            late = excess[excess.t >= settled_from]
            damage[multiple] = late.groupby(["learner", "seed"]).excess.mean()

            per_seed = error_by_step(frame, metric="rmse", by_seed=True)
            settled = (per_seed[per_seed.t >= settled_from]
                       .groupby(["learner", "seed"]).error.mean())
            frozen = settled.xs(FROZEN, level="learner")
            margin[multiple] = settled.drop(FROZEN, level="learner").sub(frozen, level="seed")

            drift = build_drift(config)
            errors = error_by_step(frame, metric="rmse")
            freeze = next(e.freeze_after for e in config.learners if e.name == FROZEN)
            for learner in PILOT_LEARNERS:
                try:
                    point = comparative_break(errors[errors.learner.isin([learner, FROZEN])],
                                              learner, FROZEN, drift, horizon, PERSISTENCE,
                                              start_step=freeze)
                    breaks[(learner, multiple)] = (
                        f"t={point.step}" if point.broke
                        else ("never ahead" if point.note else "holds"))
                except BreakError:
                    breaks[(learner, multiple)] = "--"
        if not order:
            print(f"  {arm}: no drifting cell finished\n")
            continue

        header = "".join(f"{f'{m}x ({M14_PEAK * m:g} deg/step)':>{WIDTH}}" for m in order)
        print(f"  {arm.upper()} arm, settled window (t >= {SETTLED:.0%} of T), seed mean "
              "[per-seed values]\n")
        for title, table in (("damage, drifting - twin (RMSE)", damage),
                             ("margin over frozen ATC, same cell (negative = adapting pays)",
                              margin)):
            print(f"    {title}")
            print(f"    {'':<40}{header}")
            for learner in PILOT_LEARNERS + ([FROZEN] if table is damage else []):
                cells = []
                for m in order:
                    try:
                        values = table[m].xs(learner, level="learner")
                    except KeyError:
                        cells.append(f"{'--':>{WIDTH}}")
                        continue
                    spread = ",".join(f"{v:+.4f}" for v in values)
                    cells.append(f"{values.mean():+.4f} [{spread}]".rjust(WIDTH))
                print(f"    {learner:<40}" + "".join(cells))
            print()
        print(f"    comparative break against {FROZEN} (first step it stops being beaten,"
              f" {PERSISTENCE} evaluations)")
        print(f"    {'':<40}" + "".join(f"{f'{m}x':>{WIDTH}}" for m in order))
        for learner in PILOT_LEARNERS:
            print(f"    {learner:<40}"
                  + "".join(f"{breaks.get((learner, m), '--'):>{WIDTH}}" for m in order))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
