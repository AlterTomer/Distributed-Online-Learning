r"""Where each method breaks, under both definitions.

Run this file directly -- edit the constants below and go.

    python scripts/report_breaks.py
    python scripts/report_breaks.py x9_rate_ramp x9_control

Takes a drifting run and its **paired control**: a run identical in seeds,
horizon, evaluation cadence and learners, with the drift switched off. The
absolute break is measured on $e_\text{drift}(t) - e_\text{control}(t)$, which
cancels the convergence trend and the online-versus-offline penalty exactly,
and needs no reference table (design note D50).

Two definitions are reported side by side because they disagree in both
directions, and each disagreement means something:

* absolute broke, comparative did not -- adaptation is helping and still not
  keeping up;
* comparative broke, absolute did not -- the drift was slow enough that
  standing still was fine, and the online updates are adding more variance than
  they remove.

**The absolute test is per step, not against one threshold.** Each step's mean
excess is tested against the seed standard error *at that step*. A threshold
fixed in advance would need a quiet window to calibrate on, and an exact
pairing does not provide one: while the schedule has barely moved the two runs
are the *same run* seed for seed, so the excess is identically zero and has no
spread at all to measure.

**The comparative test starts at the baseline's freeze point.** Before it,
`frozen_atc` is still running the algorithm it is the baseline for -- identical
parameters, identical predictions -- so the margin is zero and every learner
would be reported as breaking at step 0.

A learner that was **never ahead** of the baseline is reported as such rather
than as breaking at the first eligible step: it cannot stop leading if it never
led, and calling that a break would blame the drift for a gap that predates it.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dekf_bench.env.drift import build_drift  # noqa: E402
from dekf_bench.metrics.breaks import (  # noqa: E402
    BreakError,
    comparative_break,
    error_by_step,
    excess_break,
    paired_excess,
)
from dekf_bench.utils.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
DRIFTING = "x9_rate_ramp"
CONTROL = "x9_control"
BASELINE = "frozen_atc"

#: Standard errors of the seed mean the excess must clear. 3 is the usual
#: "clear of the noise" convention.
NOISE_MULTIPLE = 3.0
#: Consecutive evaluations the condition must hold for. One crossing is noise.
PERSISTENCE = 3


def load(experiment: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "results" / experiment / "seed_*.parquet")))
    if not files:
        raise SystemExit(
            f"no results for {experiment!r}. Run it first:\n"
            f"    python scripts/run_experiment.py {experiment}"
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def cell(point) -> tuple[str, str]:
    if point is None:
        return "--", "--"
    if not point.broke:
        return ("never ahead" if point.note else "no break"), "--"
    return f"t={point.step}", f"{point.rate_at_break:.4f}"


def main() -> int:
    drifting_name = sys.argv[1] if len(sys.argv) > 1 else DRIFTING
    control_name = sys.argv[2] if len(sys.argv) > 2 else CONTROL

    drifting = load(drifting_name)
    control = load(control_name)
    config = load_config(drifting_name)
    drift = build_drift(config)
    horizon = config.run.horizon

    excess = paired_excess(drifting, control)
    errors = error_by_step(drifting)

    # The baseline is the learner it is a baseline *for* until it freezes, so
    # the comparative test cannot start before that point.
    freeze_after = next(
        (entry.freeze_after for entry in config.learners if entry.name == BASELINE), None
    )
    if freeze_after is None:
        print(f"note: {BASELINE!r} is not in this run; no comparative break will be reported\n")

    schedule = drift.schedule
    n_seeds = int(excess.seed.nunique())
    print(f"{drifting_name}  vs  {control_name}")
    print(f"  schedule      {schedule.name}, {n_seeds} seeds, T={horizon}")
    print(
        f"  rate          mean {schedule.mean_rate(horizon):.4f}"
        f"  peak {schedule.peak_rate(horizon):.4f} deg/step"
    )
    print(
        f"  absolute      excess > {NOISE_MULTIPLE:g}x its seed s.e.m. at that step,"
        f" for {PERSISTENCE} consecutive evaluations"
    )
    print(f"  comparative   vs {BASELINE}, from its freeze point t={freeze_after}\n")

    header = (
        f"{'learner':26s} {'absolute':>11} {'rate':>8} | {'comparative':>12} {'rate':>8} | "
        f"{'excess@end':>10}"
    )
    print(header)
    print("-" * len(header))

    final = excess[excess.t >= 0.9 * horizon].groupby("learner").excess.mean()
    for learner in sorted(excess.learner.unique()):
        absolute = excess_break(excess, learner, drift, horizon, NOISE_MULTIPLE, PERSISTENCE)
        comparative = None
        if learner != BASELINE and freeze_after is not None:
            try:
                comparative = comparative_break(
                    errors,
                    learner,
                    BASELINE,
                    drift,
                    horizon,
                    PERSISTENCE,
                    start_step=freeze_after,
                )
            except BreakError:
                comparative = None

        a_step, a_rate = cell(absolute)
        c_step, c_rate = cell(comparative)
        print(
            f"{learner:26s} {a_step:>11} {a_rate:>8} | {c_step:>12} {c_rate:>8} | "
            f"{final.get(learner, float('nan')):>10.4f}"
        )

    print(
        f"\n'no break' means the condition never held for {PERSISTENCE} consecutive "
        f"evaluations up to {schedule.peak_rate(horizon):.4f} deg/step, which is the "
        "fastest this run went -- not that no rate would break it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
