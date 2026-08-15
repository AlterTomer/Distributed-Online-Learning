r"""Where each method breaks, under both definitions.

Run this file directly — edit the constants below and go.

    python scripts/report_breaks.py
    python scripts/report_breaks.py x11_every50_jump15 x1_stationary

Takes a drifting run and its **paired control**: a run identical in seeds,
horizon, evaluation cadence and learners, with the drift switched off. The
absolute break is then measured on $e_\text{drift}(t) - e_\text{control}(t)$,
which cancels the convergence trend and the online-versus-offline penalty
exactly, and needs no reference table (design note D50).

Two definitions are reported side by side because they disagree in both
directions, and each disagreement means something:

* absolute broke, comparative did not — adaptation is helping and still not
  keeping up;
* comparative broke, absolute did not — the drift was slow enough that standing
  still was fine, and the online updates are adding more variance than they
  remove.

**The threshold is derived, not chosen.** It is a multiple of the seed noise of
the excess, estimated over the opening quarter of the run where an accelerating
schedule has barely moved — so the excess there is zero by construction and its
spread is noise and nothing else.
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
    threshold_from_seed_noise,
)
from dekf_bench.utils.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
DRIFTING = "x9_rate_ramp"
CONTROL = "x9_control"
BASELINE = "frozen_atc"

#: Multiple of the seed noise. 3 is the usual "clear of the noise" convention.
NOISE_MULTIPLE = 3.0
#: `sem` thresholds the seed mean, which is the quantity being estimated.
#: `sd` asks the more conservative question of whether any single run shows it.
NOISE_STATISTIC = "sem"
#: Opening fraction of the run used to estimate noise. Under a ramp this window
#: is nearly stationary, so the excess in it is zero by construction.
CONTROL_FRACTION = 0.25
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
    threshold = threshold_from_seed_noise(
        excess,
        horizon,
        multiple=NOISE_MULTIPLE,
        control_fraction=CONTROL_FRACTION,
        statistic=NOISE_STATISTIC,
    )

    schedule = drift.schedule
    n_seeds = int(excess.seed.nunique())
    print(f"{drifting_name}  vs  {control_name}")
    print(f"  schedule      {schedule.name}, {n_seeds} seeds, T={horizon}")
    print(
        f"  rate          mean {schedule.mean_rate(horizon):.4f}"
        f"  peak {schedule.peak_rate(horizon):.4f} deg/step"
    )
    print(
        f"  threshold     {threshold:.4f}"
        f"  ({NOISE_MULTIPLE:g}x {NOISE_STATISTIC} of the excess over the"
        f" opening {CONTROL_FRACTION:.0%})"
    )
    print(f"  persistence   {PERSISTENCE} consecutive evaluations\n")

    header = (
        f"{'learner':26s} {'absolute':>10} {'rate':>8} | {'comparative':>12} {'rate':>8} | "
        f"{'excess@end':>10}"
    )
    print(header)
    print("-" * len(header))

    final = excess[excess.t >= 0.9 * horizon].groupby("learner").excess.mean()
    for learner in sorted(excess.learner.unique()):
        absolute = excess_break(excess, learner, drift, horizon, threshold, PERSISTENCE)
        try:
            comparative = (
                comparative_break(errors, learner, BASELINE, drift, horizon, PERSISTENCE)
                if learner != BASELINE
                else None
            )
        except BreakError:
            comparative = None

        def cell(point) -> tuple[str, str]:
            if point is None:
                return "--", "--"
            if not point.broke:
                return "no break", "--"
            return f"t={point.step}", f"{point.rate_at_break:.4f}"

        a_step, a_rate = cell(absolute)
        c_step, c_rate = cell(comparative)
        print(
            f"{learner:26s} {a_step:>10} {a_rate:>8} | {c_step:>12} {c_rate:>8} | "
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
