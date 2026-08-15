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


def error_by_step(
    frame: pd.DataFrame, evalset: str = "current", by_seed: bool = False
) -> pd.DataFrame:
    """Counts-then-divide error per learner and step, pooled over agents.

    Never a mean of per-agent rates: an agent that saw two samples would weigh
    the same as one that saw eight, quietly reweighting the result toward
    whichever agent happened to idle.

    With ``by_seed`` the seeds are kept apart, which is what the noise estimate
    and the paired control both need.
    """
    rows = frame[(frame.evalset == evalset) & (frame.metric == "error_rate")]
    if rows.empty:
        raise BreakError(
            f"no {evalset!r} error_rate rows to locate a break in. The run must record "
            f"{evalset!r}; check eval.evalsets."
        )
    keys = ["learner", "seed", "t"] if by_seed else ["learner", "t"]
    grouped = rows.groupby(keys)[["n_correct", "n_samples"]].sum()
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


def paired_excess(
    drifting: pd.DataFrame,
    control: pd.DataFrame,
    evalset: str = "current",
) -> pd.DataFrame:
    r"""Drift damage: the drifting run's error minus its stationary twin's.

    **Why a control run rather than a threshold on the gap itself.** The gap to
    $e^\star$ is dominated by the learner still converging, not by drift --
    measured on the shipped runs, ATC's gap *falls* from 0.076 to 0.046 over
    x2, while drift's cost at that rate is about 0.015. Thresholding the raw
    gap would fire at step 0 for every method, for reasons unrelated to
    tracking.

    Subtracting a run identical except for the drift pairs by seed *and* by
    step, so the convergence trend and the online-versus-offline penalty cancel
    exactly. $e^\star$ drops out of the subtraction too, which is why this needs
    no reference table.

    Returns one row per (learner, seed, step) so the seed spread survives -- it
    is what the threshold is derived from.
    """
    left = error_by_step(drifting, evalset, by_seed=True)
    right = error_by_step(control, evalset, by_seed=True)
    merged = left.merge(right, on=["learner", "seed", "t"], suffixes=("", "_control"), how="inner")
    if merged.empty:
        raise BreakError(
            "the drifting run and its control share no (learner, seed, step) rows. The "
            "control must match on seeds, horizon, evaluation cadence and learner list, "
            "or the pairing that makes this subtraction exact does not hold."
        )
    for column, label in (("learner", "learners"), ("seed", "seeds"), ("t", "steps")):
        missing = set(left[column]) - set(right[column])
        if missing:
            raise BreakError(
                f"the control run is missing {label} {sorted(missing)!r} that the drifting "
                "run has. Unpaired rows would be dropped silently and the excess would be "
                "computed over a different population than it claims."
            )
    merged["excess"] = merged.error - merged.error_control
    return merged


def threshold_from_seed_noise(
    excess: pd.DataFrame,
    horizon: int,
    multiple: float = 3.0,
    control_fraction: float = 0.25,
    statistic: str = "sem",
) -> float:
    """A threshold derived from the data rather than chosen.

    Estimated over the opening ``control_fraction`` of the run, where an
    accelerating schedule has barely moved: the excess there is essentially
    zero by construction, so its spread is noise and nothing else.

    ``sem`` is the noise of the *seed mean*, which is the quantity actually
    thresholded; ``sd`` is the spread of a single run, which asks the more
    conservative question of whether any given run would show it.
    """
    if statistic not in ("sem", "sd"):
        raise BreakError(f"statistic must be 'sem' or 'sd', got {statistic!r}")
    window = excess[excess.t <= control_fraction * horizon]
    if window.empty:
        raise BreakError(
            f"no evaluation steps in the opening {control_fraction:.0%} of the run to "
            "estimate noise from"
        )
    n_seeds = int(window.seed.nunique())
    if n_seeds < 2:
        raise BreakError(
            f"noise cannot be estimated from {n_seeds} seed. Derive the threshold from a "
            "run with several seeds, or pass one explicitly."
        )
    spread = float(window.groupby(["learner", "t"]).excess.std().mean())
    if statistic == "sem":
        spread /= n_seeds**0.5
    return multiple * spread


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


def excess_break(
    excess: pd.DataFrame,
    learner: str,
    drift: Any,
    horizon: int,
    threshold: float,
    persistence: int = DEFAULT_PERSISTENCE,
) -> BreakPoint:
    """The absolute break, measured on drift damage against a paired control.

    This is the definition to use when a control run exists. ``absolute_break``
    thresholds the raw gap to $e^\\star$ and is kept for the case where no
    control was run -- but on the shipped runs that gap is dominated by
    convergence, so prefer this.
    """
    series = excess[excess.learner == learner].groupby("t").excess.mean().sort_index()
    if series.empty:
        raise BreakError(f"no rows for learner {learner!r}")
    found = _first_persistent(series.index.to_numpy(), (series.to_numpy() > threshold), persistence)
    max_rate = _rate_summary(drift, horizon)
    if found is None:
        return BreakPoint(learner, "absolute", None, None, None, max_rate)
    step, index = found
    return BreakPoint(
        learner=learner,
        definition="absolute",
        step=step,
        rate_at_break=float(drift.schedule.rate_at(step)),
        value_at_break=float(series.to_numpy()[index]),
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
