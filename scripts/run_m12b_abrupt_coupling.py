r"""M12b -- the coupling contrast on the abrupt schedule, so `independent` can be read.

    python scripts/run_m12b_abrupt_coupling.py --smoke        # the path, not the numbers
    python scripts/run_m12b_abrupt_coupling.py --device cuda  # six cells, ~9 h
    python scripts/run_m12b_abrupt_coupling.py --report-only

## Why this exists

M12 ([[D109]]) found that two drift channels interact when and only when they contend
for the same observable: $\beta$+gain, both moving the signal's spread, cancel when
opposed. That rests on `anti - correlated`, a cell-minus-cell contrast in which the
twin and both single-channel damages cancel -- but only on the **linear** schedule.

Its three `independent` cells ran **abrupt**, because a linear ramp has one path however
it is seeded and cannot express independence. So nothing could be subtracted from them:
their only instrument was the additivity residual, which keeps one twin term and takes
the single-channel damages as inputs. For the two bias pairs that input included
D(bias, abrupt), which D108 withdrew, and both residuals were withdrawn with it
(D109). `independent` has had no clean reading.

## The design

This run adds the missing comparators on the **same** schedule: `correlated` and `anti`
for each of the three pairs on `recurring` -- six cells. Each is its pair's
`independent` config with `env.series.coupling` overridden and nothing else, so
channels, spans, jump size and cadence, horizon, learners, rates and seeds are
identical by construction. Two contrasts follow, paired per seed:

* **independent - correlated**: what independence does against alignment. Only the
  secondary's path differs: the primary's jumps come from their own seed sub-stream
  (`series.py`: `"jumps"`, keyed by run seed alone) under every coupling, while
  `independent` draws the secondary from a separate one (`"jumps2"`).
* **anti - correlated (abrupt)**: whether M12's cancellation holds for abrupt shifts,
  the schedule on which it has never been measured.

Neither contains the twin or any single-channel damage, so nothing D108 withdrew can
reach them. The report **checks** the shared primary path per seed -- the recorded
`drift_state` must be identical across the three couplings -- rather than assuming it.

## The risk, stated before the run

Abrupt has returned nulls five times at five seeds (D109), because a reflecting
schedule sends each seed on its own excursion and the twin cancels only 0.54-0.69 of
the variance. These contrasts do not use the twin, and the primary path is shared per
seed, so the cancellation should be much better -- but that is a prediction, and a null
here would say "not detectable at five seeds", not "absent".

**Predicted:** $\beta$+gain anti - correlated negative (the cancellation holds);
independent - correlated for $\beta$+gain between the two, since independent opposes
the channels about half the time; every bias pair null on both contrasts.

Carries M12's settings exactly: M4's and M5's filter selections, M3's **abrupt** rates
(the ones M12's independent cells used), the R scale, the twin-free design.
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
from run_m3_rates import settled  # noqa: E402
from run_m11_sensor_drift import BASELINES, FILTERS  # noqa: E402
from run_m12_combined_drift import (  # noqa: E402
    CONDITIONS,
    EVAL_EVERY,
    HORIZON,
    PAIRS,
    SEEDS,
    _cached,
    entries,
    load_settings,
    r_scale_override,
)
from run_m12_combined_drift import cell_name as m12_cell  # noqa: E402

from dekf_bench.metrics.paired import Paired, holm, paired  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: The two couplings this run adds on the abrupt schedule; `independent` is M12's.
COUPLINGS = ("correlated", "anti")
STATUS = ROOT / "results" / "m12b_status.json"
ALPHA = 0.05


def cell_name(pair: str, coupling: str, suffix: str = "") -> str:
    return f"m12b_{pair}_{coupling}{suffix}"


def independent_cell(pair: str, suffix: str = "") -> str:
    """M12's independent cell for this pair -- the third coupling, already run."""
    return m12_cell(f"{pair}_independent") + suffix


def config_for(pair: str, coupling: str, name: str, horizon: int, seeds: list[int],
               eval_every: int, learners: list[dict], model: dict, args):
    """The pair's `independent` config with the coupling overridden and nothing else."""
    base = CONDITIONS[f"{pair}_independent"]
    return load_config(base, overrides={
        "run": {"name": name, "horizon": horizon, "seeds": seeds,
                "eval_every": eval_every, "device": args.device, "dtype": args.dtype},
        **({"model": model} if model else {}),
        "env": {"series": {"coupling": coupling}},
        "learners": learners,
    })


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--smoke", action="store_true", help="short horizon, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0

    centralised, diffusion, rates = load_settings(args.smoke)
    horizon = 60 if args.smoke else args.horizon
    seeds = [0] if args.smoke else args.seeds
    eval_every = 20 if args.smoke else EVAL_EVERY
    model = r_scale_override(args.smoke)
    cells = [(f"{a}_{b}", c) for a, b in PAIRS for c in COUPLINGS]

    missing = [independent_cell(p, suffix) for p in {pair for pair, _c in cells}
               if not (ROOT / "results" / independent_cell(p, suffix) / "_complete").exists()]
    if missing:
        print(f"  ⚠ {sorted(missing)} not on disk: the independent - correlated contrast\n"
              "    needs M12's independent cells. The run proceeds; that half of the report\n"
              "    will be empty until they exist.\n")

    print(f"M12b{' SMOKE' if args.smoke else ''}: {len(cells)} cells x {len(seeds)} seeds, "
          f"T={horizon}, abrupt (recurring) schedule, device {args.device}\n", flush=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, (pair, coupling) in enumerate(cells, start=1):
        name = cell_name(pair, coupling, suffix)
        # Rates by the independent cell's name: M12 assigns abrupt rates to it, and
        # these cells share its schedule, so they must carry the same ones.
        learners = entries(f"{pair}_independent", centralised, diffusion, rates)
        config = config_for(pair, coupling, name, horizon, seeds, eval_every,
                            learners, model, args)
        assert config.env.drift.schedule == "recurring", "M12b must run on the abrupt schedule"
        assert config.env.series.coupling == coupling
        note = run_one(config, None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<34} {len(learners):>2} learners  "
              f"{note:<14} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM12b{' smoke' if args.smoke else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


def _primary_paths(cell: str) -> dict[int, tuple]:
    """Per seed, the recorded (t, drift_state) sequence -- the primary channel's path."""
    import pandas as pd  # noqa: PLC0415

    out: dict[int, tuple] = {}
    for path in sorted((ROOT / "results" / cell).glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["t", "evalset", "drift_state"])
        rows = frame[frame["evalset"] == "prequential"].drop_duplicates("t").sort_values("t")
        out[int(path.stem.split("_")[1])] = tuple(
            round(float(v), 9) for v in rows["drift_state"].to_numpy())
    return out


def _columns(result: Paired, p_adjusted: float) -> str:
    lo, hi = result.ci(0.95)
    interval = f"[{lo:+.4f}, {hi:+.4f}]"
    mark = "*" if p_adjusted < ALPHA else " "
    return (f"{result.mean:>+9.4f}  {interval:<20}  {result.t:>+7.2f}"
            f"  {p_adjusted:>7.3f}{mark}")


def report(suffix: str = "") -> None:
    arms = [*FILTERS, *BASELINES]
    pairs = [f"{a}_{b}" for a, b in PAIRS]

    print("  settled RMSE on the held-out current set, abrupt schedule\n")
    header = "".join(f"{p + '/' + c[:4]:>18}" for p in pairs
                     for c in ("corr", "anti", "indp"))
    print(f"    {'learner':<38}{header}")
    for learner in arms:
        row = f"    {learner:<38}"
        for pair in pairs:
            for cell in (cell_name(pair, "correlated", suffix), cell_name(pair, "anti", suffix),
                         independent_cell(pair, suffix)):
                value = settled(cell, learner)
                row += f"{value:>18.4f}" if value != float("inf") else f"{'-':>18}"
        print(row)

    if suffix:
        print("\n  Smoke numbers mean nothing: 60 rounds, one seed, placeholder settings.")

    print("\n  GATE: is the primary channel's path shared across the three couplings?")
    print("  The contrasts below are clean only if it is -- only the secondary may differ.\n")
    for pair in pairs:
        paths = {c: _primary_paths(cell_name(pair, c, suffix)) for c in COUPLINGS}
        paths["independent"] = _primary_paths(independent_cell(pair, suffix))
        seeds = sorted(set.intersection(*[set(p) for p in paths.values()])) if all(
            paths.values()) else []
        if not seeds:
            print(f"    {pair:<12} not checkable: a cell is missing")
            continue
        same = all(paths["correlated"][s] == paths[c][s] for s in seeds for c in paths)
        print(f"    {pair:<12} {len(seeds)} seed(s): "
              f"{'identical primary path in all three couplings' if same else 'DIFFERS -- read nothing below'}")

    for title, left, right in (
        ("anti - correlated, abrupt: does M12's cancellation hold for abrupt shifts?",
         lambda p: cell_name(p, "anti", suffix), lambda p: cell_name(p, "correlated", suffix)),
        ("independent - correlated, abrupt: what independence does against alignment",
         lambda p: independent_cell(p, suffix), lambda p: cell_name(p, "correlated", suffix)),
    ):
        print(f"\n  {title}")
        print("  paired per seed; negative = the first coupling is cheaper. Signed t on n-1 df;")
        print(f"  p_holm adjusted across this table's rows; * marks p_holm < {ALPHA} (D113).\n")
        print(f"    {'pair':<12}{'learner':<38}{'diff':>9}  {'95% CI':^20}  {'t':>7}"
              f"  {'p_holm':>7}")
        rows = []
        for pair in pairs:
            for learner in arms:
                result = paired(_cached(left(pair), learner), _cached(right(pair), learner))
                if result.n:
                    rows.append((pair, learner, result))
        adjusted = holm([r.p for *_x, r in rows])
        for (pair, learner, result), p_adj in zip(rows, adjusted, strict=True):
            print(f"    {pair:<12}{learner:<38}{_columns(result, p_adj)}")
        if not rows:
            print("    nothing to compare yet")

    print("\n  PREDICTED, before the run: beta+gain anti - correlated negative; beta+gain")
    print("  independent - correlated between that and zero, since independent opposes the")
    print("  channels about half the time; every bias pair null on both contrasts.")


if __name__ == "__main__":
    raise SystemExit(main())
