r"""X11 — recovery under repeated abrupt shifts, as a grid.

Run this file directly.

    python scripts/report_recurring.py
    python scripts/report_recurring.py --learner local_only

**No paired control, and none is needed.** The absolute break (design note D50)
subtracts a stationary twin because the gap to $e^\star$ is dominated by the
learner still converging. Here each jump supplies its own baseline — the error
immediately *before* it — so the transient is measured within the run and the
convergence trend cancels over the few tens of steps a single jump spans.
(`x1_stationary` could not serve as a control anyway: it runs at `eval_every`
25 against X11's 5 and carries no `frozen_atc`.)

Three numbers per cell:

``rise``
    How far the error jumps above its pre-shift level, averaged over every
    shift in the run and every seed. The size of the wound.
``recovered``
    The fraction of shifts after which the error returns to its pre-shift level
    *before the next shift arrives*. This is the headline: it answers "can this
    method keep up with shifts at this frequency" without any threshold on how
    good it has to be.
``standing``
    Mean error over the second half of the run. Where repeated un-recovered
    shifts accumulate, this is what they accumulate into.

**Read the matched-speed cells against each other.** $J/t'$ is only the average
speed, so $(t'{=}50, J{=}15)$ and $(t'{=}100, J{=}30)$ both average
0.300 deg/step while posing different problems — frequent-small versus
rare-large. A difference between them is the part of the story a
one-dimensional sweep along average speed could not see.
"""

from __future__ import annotations

import glob
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dekf_bench.metrics.breaks import error_by_step  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
JUMP_EVERY = [25, 50, 100, 200]
JUMP_DEGREES = [5.0, 15.0, 30.0]
LEARNER = "diffusion_sgd_atc"
#: A shift counts as recovered when the error comes back to within this many
#: seed standard errors of its pre-shift level. Derived from the data rather
#: than chosen, for the same reason the break threshold is (design note D50).
RECOVERY_TOLERANCE_SEM = 1.0


def cell_name(jump_every: int, jump_degrees: float) -> str:
    return f"x11_every{jump_every}_jump{jump_degrees:g}"


def load(experiment: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(ROOT / "results" / experiment / "seed_*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def jump_profile(
    frame: pd.DataFrame, learner: str, jump_every: int
) -> tuple[float, float, float, float]:
    """(rise, recovered fraction, standing error, tolerance) for one cell.

    Aligns every evaluation to the most recent shift, so all the shifts in a
    run are pooled into one transient rather than being averaged away by a
    single run-long mean.
    """
    per_seed = error_by_step(frame, "current", by_seed=True)
    per_seed = per_seed[per_seed.learner == learner]
    if per_seed.empty:
        return (np.nan,) * 4

    horizon = int(per_seed.t.max()) + 1
    tolerance = RECOVERY_TOLERANCE_SEM * float(
        per_seed.groupby("t").error.std().mean() / max(per_seed.seed.nunique(), 1) ** 0.5
    )

    rises: list[float] = []
    recoveries: list[bool] = []
    for _seed, part in per_seed.groupby("seed"):
        series = part.set_index("t").error.sort_index()
        steps = series.index.to_numpy()
        for jump in range(jump_every, horizon, jump_every):
            before = steps[steps < jump]
            after = steps[(steps >= jump) & (steps < jump + jump_every)]
            if not len(before) or not len(after):
                continue
            baseline = float(series.loc[before[-1]])
            window = series.loc[after].to_numpy()
            rises.append(float(window.max() - baseline))
            recoveries.append(bool((window[1:] <= baseline + tolerance).any()))

    standing = float(per_seed[per_seed.t >= horizon // 2].error.mean())
    if not rises:
        return (np.nan, np.nan, standing, tolerance)
    return (float(np.mean(rises)), float(np.mean(recoveries)), standing, tolerance)


def main() -> int:
    learner = LEARNER
    if "--learner" in sys.argv:
        learner = sys.argv[sys.argv.index("--learner") + 1]

    results: dict[tuple[int, float], tuple[float, float, float, float]] = {}
    missing: list[str] = []
    for jump_every, jump_degrees in itertools.product(JUMP_EVERY, JUMP_DEGREES):
        frame = load(cell_name(jump_every, jump_degrees))
        if frame is None:
            missing.append(cell_name(jump_every, jump_degrees))
            continue
        results[(jump_every, jump_degrees)] = jump_profile(frame, learner, jump_every)

    if not results:
        raise SystemExit(
            "no X11 results yet. Run them first:\n    python scripts/run_recurring_sweep.py"
        )
    if missing:
        print(f"note: {len(missing)} cell(s) not yet run: {', '.join(missing)}\n")

    print(f"X11 recovery under repeated shifts — {learner}\n")
    for title, index in (
        ("rise above pre-shift error", 0),
        ("fraction recovered before next shift", 1),
        ("standing error, second half", 2),
    ):
        print(f"  {title}")
        header = "    t'\\J  " + "".join(f"{value:>10g}" for value in JUMP_DEGREES)
        print(header)
        for jump_every in JUMP_EVERY:
            cells = ""
            for jump_degrees in JUMP_DEGREES:
                value = results.get((jump_every, jump_degrees))
                cells += "         -" if value is None else f"{value[index]:>10.4f}"
            print(f"    {jump_every:<5}{cells}")
        print()

    print("  matched average speed (deg/step), read these against each other")
    speeds: dict[float, list[tuple[int, float]]] = {}
    for jump_every, jump_degrees in results:
        speeds.setdefault(round(jump_degrees / jump_every, 4), []).append(
            (jump_every, jump_degrees)
        )
    for speed, cells in sorted(speeds.items()):
        if len(cells) < 2:
            continue
        parts = []
        for jump_every, jump_degrees in sorted(cells):
            rise, recovered, _standing, _tol = results[(jump_every, jump_degrees)]
            parts.append(
                f"t'={jump_every} J={jump_degrees:g}: rise {rise:.4f}, recovered {recovered:.2f}"
            )
        print(f"    {speed:.4f}  " + " | ".join(parts))

    print(
        "\n  'recovered' counts a shift as survived when the error returns to within "
        f"{RECOVERY_TOLERANCE_SEM:g} seed s.e.m. of its pre-shift level before the next "
        "shift lands. A value near 0 at high frequency is the compounding this "
        "experiment exists to find."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
