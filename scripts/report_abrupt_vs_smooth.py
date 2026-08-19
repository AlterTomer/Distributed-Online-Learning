r"""Does abrupt drift cost more than smooth drift at the same average speed?

Run this file directly.

    python scripts/report_abrupt_vs_smooth.py
    python scripts/report_abrupt_vs_smooth.py --learner local_only

The question phase 5 rests on. A filter should help most where a gradient
method's fixed step size is the binding constraint, and an abrupt shift is
exactly that case -- so it matters whether abruptness costs anything *beyond*
the speed it implies.

**Both sides are damage, not error.** Each regime is subtracted from a run
identical to it except for the drift, so what remains is what the drift cost
and not what the method would have cost anyway (design note D50).

**Both sides are read over the same step window.** This is the part that needs
care. Rotation is capped at 45 degrees, so smooth drift at 0.15 deg/step runs
out of band after 300 steps while an abrupt cell at the same average speed runs
1500 indefinitely -- they cannot be matched on speed *and* duration. Comparing
X12's second half against X11's second half would compare a learner 200 steps
in against one 1000 steps in, and the difference in maturity would be read as a
difference between the regimes. So the window is the second half of the
*shorter* run, applied to both.

That asymmetry is not an inconvenience to be corrected away. Repeated bounded
shifts are the only regime that can sustain a high average speed in a bounded
state space, which is an argument for the recurring schedule as the phase-5
benchmark rather than a defect in it.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dekf_bench.metrics.breaks import assert_paired_runs, paired_excess  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
LEARNER = "diffusion_sgd_atc"

#: speed -> (smooth run, its control, horizon)
SMOOTH = {
    0.025: ("x12_linear_a0p025", "x12_control_a0p025", 1500),
    0.05: ("x12_linear_a0p05", "x12_control_a0p05", 900),
    0.10: ("x12_linear_a0p1", "x12_control_a0p1", 450),
    0.15: ("x12_linear_a0p15", "x12_control_a0p15", 300),
}

#: speed -> the X11 cells that deliver it, as (jump_every, jump_degrees)
ABRUPT = {
    0.025: [(200, 5.0)],
    0.05: [(100, 5.0)],
    0.10: [(50, 5.0)],
    0.15: [(100, 15.0), (200, 30.0)],
}
ABRUPT_CONTROL = "x11_control"


def load(name: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(ROOT / "results" / name / "seed_*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def damage_in_window(run: str, control: str, learner: str, lo: int, hi: int) -> float | None:
    """Mean drift damage over ``[lo, hi)``, or None if either run is missing."""
    drifting, twin = load(run), load(control)
    if drifting is None or twin is None:
        return None
    assert_paired_runs(ROOT / "results" / run, ROOT / "results" / control)
    excess = paired_excess(drifting, twin)
    rows = excess[(excess.learner == learner) & (excess.t >= lo) & (excess.t < hi)]
    return float(rows.excess.mean()) if len(rows) else None


def main() -> int:
    learner = LEARNER
    if "--learner" in sys.argv:
        learner = sys.argv[sys.argv.index("--learner") + 1]

    print(f"Abrupt against smooth at matched average speed -- {learner}")
    print("damage = error minus a stationary twin; both read over the same window\n")
    header = (
        f"{'speed':>7} {'window':>11} {'smooth':>8} {'abrupt cell':>13} "
        f"{'abrupt':>8} {'ratio':>7} {'shifts':>7}"
    )
    print(header)
    print("-" * len(header))

    thin = []
    for speed in sorted(SMOOTH):
        run, control, horizon = SMOOTH[speed]
        lo, hi = horizon // 2, horizon
        smooth = damage_in_window(run, control, learner, lo, hi)
        if smooth is None:
            print(f"{speed:>7.3f}  {run} not run yet")
            continue

        for index, (jump_every, jump_degrees) in enumerate(ABRUPT.get(speed, [])):
            cell = f"x11_every{jump_every}_jump{jump_degrees:g}"
            abrupt = damage_in_window(cell, ABRUPT_CONTROL, learner, lo, hi)
            if abrupt is None:
                continue
            # How many shifts the window actually contains. A ratio computed
            # over one transient is a different kind of evidence from one
            # averaged over four, and the table should not hide which it is.
            shifts = sum(1 for s in range(jump_every, hi, jump_every) if lo <= s < hi)
            if shifts < 2:
                thin.append((speed, jump_every, jump_degrees, shifts))
            ratio = abrupt / smooth if smooth else float("nan")
            print(
                f"{speed if index == 0 else '':>7} {f'{lo}-{hi}' if index == 0 else '':>11} "
                f"{f'{smooth:.4f}' if index == 0 else '':>8} "
                f"{f'{jump_every}/{jump_degrees:g}':>13} {abrupt:>8.4f} "
                f"{ratio:>6.2f}x {shifts:>7}"
            )

    print(
        "\nThe window is the second half of the *smooth* run, because the 45-degree cap\n"
        "makes it the shorter one. Comparing each regime's own second half would compare\n"
        "learners of different maturity and read that as a difference between regimes."
    )
    if thin:
        print("\nRead these rows with care -- the window holds too few shifts to average:")
        for speed, jump_every, jump_degrees, shifts in thin:
            print(
                f"  speed {speed:g}, cell {jump_every}/{jump_degrees:g}: {shifts} shift(s) "
                "in the window, so the number is one transient and depends on where it "
                "falls, not an average over many"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
