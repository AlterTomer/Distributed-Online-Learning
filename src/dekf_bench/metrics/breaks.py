r"""Where a tracking method breaks, under two definitions that can disagree.

**The threshold is on a rate, not on a step count.** Under a constant drift rate
a tracker settles to a steady-state lag and then stops degrading, so "it broke
at step 900" is only meaningful when the rate is changing -- which is what the
``ramp`` schedule is for (design note D47). A located step is therefore
immediately converted to the rate the schedule was moving at, and that rate is
the answer.

Two definitions, deliberately both:

**Absolute.** The tracking gap $e_v(t) - e^\star(\varphi_v(t))$ exceeds a
threshold. Measured against the per-rotation reference rather than against raw
error, because raw error rises with rotation even for an oracle -- without
normalising, every run "breaks" at large angle and what is being measured is
task difficulty.

**Comparative.** The learner stops beating a baseline that saw the same data
with the same optimizer up to a warmup point and then simply stopped adapting.
This asks whether *continuing* to adapt pays, and it needs no threshold at all.

They answer different questions and are expected to disagree. A method can
track badly in absolute terms while still comfortably beating a frozen model --
that is a regime where adaptation helps but is not keeping up. The reverse,
losing to a frozen model while the absolute gap still looks acceptable, means
the drift is slow enough that not adapting was fine and the online updates are
adding more variance than they remove. Recording both is what makes those two
distinguishable.

**A break must persist.** A single evaluation crossing a threshold is sampling
noise; the locator requires the condition to hold for a run of consecutive
evaluations, and reports the *first* step of that run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BreakError(ValueError):
    """Raised when a break cannot be located from what was recorded."""


#: How many consecutive evaluation points must satisfy the condition before it
#: counts. Three at the default cadence is 75 steps -- long enough that a noise
#: excursion is unlikely, short enough to locate a rate finely.
DEFAULT_PERSISTENCE = 3


@dataclass(frozen=True)
class BreakPoint:
    """Where, and at what drift rate, a method stopped tracking.

    ``step`` is ``None`` when the condition never held for long enough -- which
    is a result, not a failure. "Did not break within the range this run
    reached" is the honest reading, and it must not be recorded as a break at
    the end of the run.
    """

    learner: str
    definition: str
    step: int | None
    rate_at_break: float | None
    value_at_break: float | None
    #: The largest rate the run actually reached. What "did not break" is
    #: relative to -- without it the null result is uninterpretable.
    max_rate_probed: float

    @property
    def broke(self) -> bool:
        return self.step is not None

    def as_row(self) -> dict[str, Any]:
        return {
            "learner": self.learner,
            "definition": self.definition,
            "broke": self.broke,
            "step": self.step,
            "rate_at_break": self.rate_at_break,
            "value_at_break": self.value_at_break,
            "max_rate_probed": self.max_rate_probed,
        }


def error_by_step(frame: pd.DataFrame, evalset: str = "current") -> pd.DataFrame:
    """Counts-then-divide error per learner and step, pooled over agents and seeds.

    Never a mean of per-agent rates: an agent that saw two samples would weigh
    the same as one that saw eight, quietly reweighting the result toward
    whichever agent happened to idle.
    """
    rows = frame[(frame.evalset == evalset) & (frame.metric == "error_rate")]
    if rows.empty:
        raise BreakError(
            f"no {evalset!r} error_rate rows to locate a break in. The run must record "
            f"{evalset!r}; check eval.evalsets."
        )
    grouped = rows.groupby(["learner", "t"])[["n_correct", "n_samples"]].sum()
    grouped["error"] = 1.0 - grouped.n_correct / grouped.n_samples
    return grouped.reset_index()


def tracking_gap(
    frame: pd.DataFrame,
    reference: Any,
    drift: Any,
    evalset: str = "current",
) -> pd.DataFrame:
    r"""Error minus $e^\star$ at the rotation each step actually carried.

    The normalisation that makes the absolute definition mean anything. Under
    rotation the error rises with angle even for a model retrained at that
    angle, so an unnormalised curve conflates "the learner fell behind" with
    "the task got harder".
    """
    errors = error_by_step(frame, evalset)
    errors["rotation"] = [float(drift.rotation_at(int(step))) for step in errors.t]
    errors["e_star"] = [reference.at(rotation) for rotation in errors.rotation]
    errors["gap"] = errors.error - errors.e_star
    return errors


def _first_persistent(
    steps: np.ndarray, condition: np.ndarray, persistence: int
) -> tuple[int, int] | None:
    """First index where ``condition`` holds for ``persistence`` in a row."""
    if persistence < 1:
        raise BreakError(f"persistence must be >= 1, got {persistence}")
    run = 0
    for index, holds in enumerate(condition):
        run = run + 1 if holds else 0
        if run >= persistence:
            start = index - persistence + 1
            return int(steps[start]), start
    return None


def _rate_summary(drift: Any, horizon: int) -> float:
    return float(drift.schedule.peak_rate(horizon))


def absolute_break(
    gaps: pd.DataFrame,
    learner: str,
    drift: Any,
    horizon: int,
    threshold: float,
    persistence: int = DEFAULT_PERSISTENCE,
) -> BreakPoint:
    r"""First step where the gap to $e^\star$ stays above ``threshold``."""
    series = gaps[gaps.learner == learner].sort_values("t")
    if series.empty:
        raise BreakError(f"no rows for learner {learner!r}")
    found = _first_persistent(series.t.to_numpy(), (series.gap.to_numpy() > threshold), persistence)
    max_rate = _rate_summary(drift, horizon)
    if found is None:
        return BreakPoint(learner, "absolute", None, None, None, max_rate)
    step, index = found
    return BreakPoint(
        learner=learner,
        definition="absolute",
        step=step,
        rate_at_break=float(drift.schedule.rate_at(step)),
        value_at_break=float(series.gap.to_numpy()[index]),
        max_rate_probed=max_rate,
    )


def comparative_break(
    errors: pd.DataFrame,
    learner: str,
    baseline: str,
    drift: Any,
    horizon: int,
    persistence: int = DEFAULT_PERSISTENCE,
) -> BreakPoint:
    """First step where ``learner`` stops beating the non-adapting ``baseline``.

    Threshold-free, which is its advantage: it asks whether continuing to adapt
    is buying anything, and the answer does not depend on a number anyone chose.
    """
    wide = errors.pivot_table(index="t", columns="learner", values="error")
    for name in (learner, baseline):
        if name not in wide.columns:
            raise BreakError(
                f"{name!r} is not in the recorded learners ({sorted(wide.columns)}). The "
                "comparative break needs the frozen baseline in the same run, so the "
                "comparison is paired by construction rather than across runs."
            )
    steps = wide.index.to_numpy()
    margin = (wide[learner] - wide[baseline]).to_numpy()
    found = _first_persistent(steps, margin >= 0.0, persistence)
    max_rate = _rate_summary(drift, horizon)
    if found is None:
        return BreakPoint(learner, "comparative", None, None, None, max_rate)
    step, index = found
    return BreakPoint(
        learner=learner,
        definition="comparative",
        step=step,
        rate_at_break=float(drift.schedule.rate_at(step)),
        value_at_break=float(margin[index]),
        max_rate_probed=max_rate,
    )


def locate_breaks(
    frame: pd.DataFrame,
    reference: Any,
    drift: Any,
    horizon: int,
    threshold: float,
    baseline: str = "frozen_atc",
    evalset: str = "current",
    persistence: int = DEFAULT_PERSISTENCE,
) -> pd.DataFrame:
    """Both definitions, for every learner in the frame.

    The baseline is excluded from the comparative rows: asking when a learner
    stops beating itself would report step 0 and mean nothing.
    """
    gaps = tracking_gap(frame, reference, drift, evalset)
    points: list[BreakPoint] = []
    for learner in sorted(gaps.learner.unique()):
        points.append(absolute_break(gaps, learner, drift, horizon, threshold, persistence))
        if learner != baseline and baseline in set(gaps.learner):
            points.append(comparative_break(gaps, learner, baseline, drift, horizon, persistence))
    return pd.DataFrame([point.as_row() for point in points])
