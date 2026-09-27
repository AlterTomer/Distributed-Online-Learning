r"""M12b -- the coupling contrast on the abrupt schedule, so `independent` can be read.

    python scripts/run_m12b_abrupt_coupling.py --smoke        # the path, not the numbers
    python scripts/run_m12b_abrupt_coupling.py --device cuda  # six cells, ~9 h
    python scripts/run_m12b_abrupt_coupling.py --report-only
    python scripts/run_m12b_abrupt_coupling.py --topup --device cuda  # seeds 5-9, nine cells

`--topup` is the one extension D121 allows, fixed after the five-seed look: seeds
5-9 for every cell the two contrasts read, M12's three independent cells included,
under `_s5to9` names and pooled by seed. The confirmatory families are then read at
alpha 0.025 -- Bonferroni over the two looks -- and never extended again.

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

#: The one extension (D121), fixed after the five-seed look and before these seeds
#: ran: seeds 5-9 in separately named cells -- a finished cell asked for more seeds
#: returns its cached answer (D101) -- pooled by seed with the first five. It
#: covers M12's three independent cells too, which the second contrast reads.
TOPUP_SEEDS = [5, 6, 7, 8, 9]
TOPUP = "_s5to9"
#: The confirmatory families are looked at twice, so the final look pays for the
#: first: Bonferroni over the two looks. No further extension, whatever it shows.
FINAL_ALPHA = ALPHA / 2
#: D118's confirmatory families: beta+gain, both contrasts, Holm across the six.
CONFIRMATORY_PAIR = "beta_gain"


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
    parser.add_argument("--topup", action="store_true",
                        help="seeds 5-9 for every cell both contrasts read, M12's "
                             "independent cells included (D121)")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0
    if args.topup:
        return topup(args, suffix)

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


def topup(args, suffix: str = "") -> int:
    """Seeds 5-9 for all nine cells the two contrasts read, under their own names."""
    centralised, diffusion, rates = load_settings(args.smoke)
    horizon = 60 if args.smoke else args.horizon
    seeds = [TOPUP_SEEDS[0]] if args.smoke else TOPUP_SEEDS
    eval_every = 20 if args.smoke else EVAL_EVERY
    model = r_scale_override(args.smoke)
    cells = [(f"{a}_{b}", c) for a, b in PAIRS for c in (*COUPLINGS, "independent")]
    print(f"M12b top-up{' SMOKE' if suffix else ''}: {len(cells)} cells x seeds {seeds}, "
          f"T={horizon}, device {args.device} -- the one extension D121 allows\n", flush=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, (pair, coupling) in enumerate(cells, start=1):
        base = (independent_cell(pair) if coupling == "independent"
                else cell_name(pair, coupling))
        name = base + TOPUP + suffix
        learners = entries(f"{pair}_independent", centralised, diffusion, rates)
        # `independent` is the base config's own coupling, so this rebuilds M12's
        # cell exactly -- M12 passed run, model and learners and nothing else.
        config = config_for(pair, coupling, name, horizon, seeds, eval_every,
                            learners, model, args)
        assert config.env.drift.schedule == "recurring"
        assert config.env.series.coupling == coupling
        note = run_one(config, None, None, args.fresh)
        status[name] = note
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<42} {note:<14} {elapsed:.0f} min, "
              f"~{remaining:.0f} left", flush=True)
    print(f"\nM12b top-up complete in {(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


def _pooled(cell: str, learner: str, suffix: str = "") -> dict[int, float]:
    """Per-seed settled RMSE from the cell and its top-up, which hold disjoint seeds."""
    first = _cached(cell + suffix, learner)
    second = _cached(cell + TOPUP + suffix, learner)
    if set(first) & set(second):
        raise SystemExit(f"{cell}: seeds {sorted(set(first) & set(second))} in both halves")
    return {**first, **second}


def _primary_paths(cell: str, suffix: str = "") -> dict[int, tuple]:
    """Per seed, the recorded (t, drift_state) sequence -- the primary channel's path."""
    import pandas as pd  # noqa: PLC0415

    out: dict[int, tuple] = {}
    for directory in (cell + suffix, cell + TOPUP + suffix):
        for path in sorted((ROOT / "results" / directory).glob("seed_*.parquet")):
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
    """Every cell pooled with its top-up where one exists (D121)."""
    arms = [*FILTERS, *BASELINES]
    pairs = [f"{a}_{b}" for a, b in PAIRS]
    seeds_on_disk = sorted(_pooled(cell_name(pairs[0], "correlated"), arms[0], suffix))

    print(f"  settled RMSE on the held-out current set, abrupt schedule, seeds {seeds_on_disk}\n")
    header = "".join(f"{p + '/' + c[:4]:>18}" for p in pairs
                     for c in ("corr", "anti", "indp"))
    print(f"    {'learner':<38}{header}")
    for learner in arms:
        row = f"    {learner:<38}"
        for pair in pairs:
            for cell in (cell_name(pair, "correlated"), cell_name(pair, "anti"),
                         independent_cell(pair)):
                values = _pooled(cell, learner, suffix)
                row += (f"{sum(values.values()) / len(values):>18.4f}" if values
                        else f"{'-':>18}")
        print(row)

    if suffix:
        print("\n  Smoke numbers mean nothing: 60 rounds, one seed, placeholder settings.")

    print("\n  GATE: is the primary channel's path shared across the three couplings?")
    print("  The contrasts below are clean only if it is -- only the secondary may differ.\n")
    for pair in pairs:
        paths = {c: _primary_paths(cell_name(pair, c), suffix) for c in COUPLINGS}
        paths["independent"] = _primary_paths(independent_cell(pair), suffix)
        seeds = sorted(set.intersection(*[set(p) for p in paths.values()])) if all(
            paths.values()) else []
        if not seeds:
            print(f"    {pair:<12} not checkable: a cell is missing")
            continue
        same = all(paths["correlated"][s] == paths[c][s] for s in seeds for c in paths)
        print(f"    {pair:<12} {len(seeds)} seed(s): "
              f"{'identical primary path in all three couplings' if same else 'DIFFERS -- read nothing below'}")

    contrasts = (
        ("anti - correlated", lambda p: cell_name(p, "anti"),
         lambda p: cell_name(p, "correlated")),
        ("independent - correlated", independent_cell, lambda p: cell_name(p, "correlated")),
    )
    for title, left, right in contrasts:
        print(f"\n  {title}, abrupt [exploratory: Holm across all 18 rows]")
        print("  paired per seed; negative = the first coupling is cheaper. Signed t on n-1 df;")
        print(f"  * marks p_holm < {ALPHA} (D113).\n")
        print(f"    {'pair':<12}{'learner':<38}{'diff':>9}  {'95% CI':^20}  {'t':>7}"
              f"  {'p_holm':>7}   n")
        rows = []
        for pair in pairs:
            for learner in arms:
                result = paired(_pooled(left(pair), learner, suffix),
                                _pooled(right(pair), learner, suffix))
                if result.n:
                    rows.append((pair, learner, result))
        adjusted = holm([r.p for *_x, r in rows])
        for (pair, learner, result), p_adj in zip(rows, adjusted, strict=True):
            print(f"    {pair:<12}{learner:<38}{_columns(result, p_adj)}  {result.n}")
        if not rows:
            print("    nothing to compare yet")

    # ---- the confirmatory families (D118), at the final look's alpha (D121) ----
    final = len(seeds_on_disk) >= len(SEEDS) + len(TOPUP_SEEDS)
    alpha = FINAL_ALPHA if final else ALPHA
    look = ("the final look, ten seeds: alpha 0.025, Bonferroni over the two looks"
            if final else f"the first look, {len(seeds_on_disk)} seed(s): alpha {ALPHA}")
    print(f"\n  CONFIRMATORY (D118, D121): {CONFIRMATORY_PAIR}, Holm across the six learners")
    print(f"  per contrast -- {look}.\n")
    for title, left, right in contrasts:
        rows = [(learner, paired(_pooled(left(CONFIRMATORY_PAIR), learner, suffix),
                                 _pooled(right(CONFIRMATORY_PAIR), learner, suffix)))
                for learner in arms]
        rows = [(learner, result) for learner, result in rows if result.n]
        print(f"    {title}")
        for (learner, result), p_adj in zip(rows, holm([r.p for _l, r in rows]), strict=True):
            lo, hi = result.ci(0.95)
            verdict = "ESTABLISHED" if p_adj < alpha else "not detected"
            print(f"      {learner:<38}{result.mean:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]"
                  f"  t {result.t:>+6.2f}  p_holm {p_adj:.3f}  n {result.n}   {verdict}")

    print("\n  PREDICTED, before the run: beta+gain anti - correlated negative; beta+gain")
    print("  independent - correlated between that and zero, since independent opposes the")
    print("  channels about half the time; every bias pair null on both contrasts.")


if __name__ == "__main__":
    raise SystemExit(main())
