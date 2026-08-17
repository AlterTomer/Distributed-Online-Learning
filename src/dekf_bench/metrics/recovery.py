r"""Recovery from abrupt shifts, aligned to the shifts themselves.

X5 measures one transient. Under a ``recurring`` schedule there are tens of
them, and a run-long mean would average the recovery away entirely -- the error
spends part of each interval elevated and part of it settled, and the mean of
those is a number describing neither.

So every evaluation is expressed as an **offset from the most recent shift**,
and the shifts are pooled. That turns $N$ noisy transients into one, which is
the only way a 5-seed run says anything about a shape.

**Each shift is its own control.** The rise is measured against the error
immediately *before* that shift, not against a global baseline or a stationary
twin, so the learner's ongoing convergence cancels over the few tens of steps
one interval spans. This is why X11 needs no paired control run while the
absolute break does (design note D50).

**"Recovered" is the headline, and it needs no threshold on quality.** It asks
whether the error came back to its pre-shift level before the next shift
arrived. A method can be bad in absolute terms and still recover fully; a
method can be good and never catch up. Those are different failures and the
fraction recovered separates them from the standing error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class RecoveryError(ValueError):
    """Raised when a recovery profile cannot be built from what was recorded."""


#: A shift counts as recovered when the error returns to within this many seed
#: standard errors of its pre-shift level. Derived from the run rather than
#: chosen, for the same reason the break threshold is.
DEFAULT_TOLERANCE_SEM = 1.0

#: An offset is kept only if this fraction of the typical number of shifts
#: reached it. Half is generous -- the offsets this excludes carry a fiftieth of
#: the samples, not half of them.
MIN_OFFSET_SHARE = 0.5


@dataclass(frozen=True)
class RecoveryProfile:
    """What repeated shifts cost one learner in one cell.

    Attributes:
        rise: mean peak error above the pre-shift level, over every shift.
        recovered: fraction of shifts the error returned from before the next.
        standing: mean error over the second half of the run, where
            un-recovered shifts accumulate.
        tolerance: the seed-derived margin "returned" was judged against.
        n_shifts: how many shifts were pooled. A profile from three transients
            is not the same evidence as one from sixty.
    """

    rise: float
    recovered: float
    standing: float
    tolerance: float
    n_shifts: int

    def as_row(self) -> dict[str, Any]:
        return {
            "rise": self.rise,
            "recovered": self.recovered,
            "standing": self.standing,
            "tolerance": self.tolerance,
            "n_shifts": self.n_shifts,
        }


def _per_seed_errors(errors: pd.DataFrame, learner: str) -> pd.DataFrame:
    rows = errors[errors.learner == learner]
    if rows.empty:
        raise RecoveryError(f"no rows for learner {learner!r}")
    if "seed" not in rows.columns:
        raise RecoveryError(
            "recovery needs per-seed rows; call error_by_step(..., by_seed=True). Pooling "
            "the seeds first would discard the spread the tolerance is derived from."
        )
    return rows


def tolerance_from_seeds(
    errors: pd.DataFrame, learner: str, multiple: float = DEFAULT_TOLERANCE_SEM
) -> float:
    """How close counts as "back to where it was", from the seed spread."""
    rows = _per_seed_errors(errors, learner)
    n_seeds = max(int(rows.seed.nunique()), 1)
    spread = float(rows.groupby("t").error.std().mean())
    return multiple * spread / n_seeds**0.5


def aligned_profile(
    errors: pd.DataFrame,
    learner: str,
    jump_every: int,
    horizon: int,
) -> pd.DataFrame:
    """Mean error rise by offset from the most recent shift, pooled over shifts.

    The shape of the transient. Offset 0 is the first evaluation at or after a
    shift; the rise is relative to the last evaluation *before* it.
    """
    rows = _per_seed_errors(errors, learner)
    records: list[dict[str, float]] = []

    for seed, part in rows.groupby("seed"):
        series = part.set_index("t").error.sort_index()
        steps = series.index.to_numpy()
        for shift in range(jump_every, horizon, jump_every):
            before = steps[steps < shift]
            after = steps[(steps >= shift) & (steps < shift + jump_every)]
            if not len(before) or not len(after):
                continue
            baseline = float(series.loc[before[-1]])
            for step in after:
                records.append(
                    {
                        "seed": int(seed),
                        "shift": int(shift),
                        "offset": int(step - shift),
                        "rise": float(series.loc[step]) - baseline,
                    }
                )

    if not records:
        raise RecoveryError(
            f"no shift produced both a before and an after evaluation at jump_every="
            f"{jump_every}. The evaluation cadence is too coarse to see a transient: "
            "lower run.eval_every."
        )
    frame = pd.DataFrame(records)
    profile = frame.groupby("offset").rise.agg(["mean", "std", "count"]).reset_index()

    # Drop offsets almost no shift reaches. The final evaluation is forced onto
    # `horizon - 1`, which is not on the cadence, so it lands at an offset the
    # other shifts never produce -- measured, 5 samples against a median of 295.
    # Left in, those points are pure noise and appear as a spike at the right
    # edge of every transient curve, exactly where a reader looks to judge
    # whether the error came back.
    typical = float(profile["count"].median())
    return profile[profile["count"] >= MIN_OFFSET_SHARE * typical].reset_index(drop=True)


def recovery_profile(
    errors: pd.DataFrame,
    learner: str,
    jump_every: int,
    horizon: int,
    multiple: float = DEFAULT_TOLERANCE_SEM,
) -> RecoveryProfile:
    """Summarise one cell: how big the wound, how often it healed, what remains."""
    rows = _per_seed_errors(errors, learner)
    tolerance = tolerance_from_seeds(errors, learner, multiple)

    rises: list[float] = []
    recoveries: list[bool] = []
    for _seed, part in rows.groupby("seed"):
        series = part.set_index("t").error.sort_index()
        steps = series.index.to_numpy()
        for shift in range(jump_every, horizon, jump_every):
            before = steps[steps < shift]
            after = steps[(steps >= shift) & (steps < shift + jump_every)]
            if not len(before) or len(after) < 2:
                continue
            baseline = float(series.loc[before[-1]])
            window = series.loc[after].to_numpy()
            rises.append(float(window.max() - baseline))
            # Skip the first point: at offset 0 the shift may not yet have been
            # felt, and counting it would score a shift as recovered before it
            # had a chance to do damage.
            recoveries.append(bool((window[1:] <= baseline + tolerance).any()))

    standing = float(rows[rows.t >= horizon // 2].error.mean())
    if not rises:
        raise RecoveryError(
            f"no shift at jump_every={jump_every} had two evaluations after it, so no "
            "recovery could be judged. Lower run.eval_every."
        )
    return RecoveryProfile(
        rise=float(np.mean(rises)),
        recovered=float(np.mean(recoveries)),
        standing=standing,
        tolerance=tolerance,
        n_shifts=len(rises),
    )
