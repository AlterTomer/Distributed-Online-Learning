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
    #: Why there is no break, when the reason is not "it kept tracking". A
    #: learner that was never ahead of the baseline cannot stop being ahead of
    #: it, and recording that as a break at the first eligible step would blame
    #: the drift for a gap that was there before any drift happened.
    note: str | None = None

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
            "note": self.note,
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


#: Everything that must match between a drifting run and its control for the
#: subtraction to be exact. `env` is compared field by field *except* the drift
#: blocks, which are the one thing that is supposed to differ.
PAIRING_KEYS = ("graph", "model", "learners")
ENV_KEYS_EXEMPT = ("drift", "prior_drift", "drift_scope")


def assert_paired_runs(drifting_dir: Any, control_dir: Any) -> None:
    r"""Refuse a pairing whose two runs were not configured identically.

    The subtraction in :func:`paired_excess` is only exact when the two runs
    differ in the drift and nothing else. Nothing in the *data* reveals a
    mismatch: a control run at a different learning rate produces perfectly
    well-formed rows, and the excess would then be part drift damage and part
    hyperparameter difference with no way to separate them afterwards.

    This is not hypothetical. Re-running a pair after a config fix leaves the
    old control on disk until its stage starts, so for a window of hours the
    new drifting run and the *previous* control are both present and pair
    without complaint.
    """
    from pathlib import Path

    import yaml

    configs = {}
    for label, directory in (("drifting", Path(drifting_dir)), ("control", Path(control_dir))):
        path = directory / "config.yaml"
        if not path.exists():
            raise BreakError(
                f"the {label} run at {directory} has no config.yaml, so the pairing cannot "
                "be verified. Re-run it, or pair by hand knowing the risk."
            )
        configs[label] = yaml.safe_load(path.read_text(encoding="utf-8"))

    differences: list[str] = []
    left, right = configs["drifting"], configs["control"]
    for key in PAIRING_KEYS:
        if left.get(key) != right.get(key):
            differences.append(key)
    for key in set(left.get("env", {})) | set(right.get("env", {})):
        if key in ENV_KEYS_EXEMPT:
            continue
        if left.get("env", {}).get(key) != right.get("env", {}).get(key):
            differences.append(f"env.{key}")
    for key in ("seeds", "horizon", "eval_every"):
        if left.get("run", {}).get(key) != right.get("run", {}).get(key):
            differences.append(f"run.{key}")

    if differences:
        raise BreakError(
            f"the drifting run and its control differ in {sorted(differences)}, so their "
            "subtraction would mix drift damage with a configuration difference and no "
            "later step could separate the two. Re-run the control against the current "
            "configuration."
        )


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

    **Degenerate under an exact pairing, and it refuses rather than returning
    zero.** When the drifting run and its control are still *identical* -- which
    is what an accelerating schedule's opening quarter gives, since it has
    barely rotated -- the excess is zero seed for seed and its spread is zero
    too. A threshold of 0 would then call every later step a break. Prefer
    :func:`excess_break`, which tests each step against the seed spread *at that
    step* and needs no quiet window at all.
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
    if spread <= 0.0:
        raise BreakError(
            "the excess has no spread over the estimation window: the drifting run and "
            "its control are still identical there, so there is no noise to measure and "
            "a threshold of 0 would call every later step a break. Use excess_break, "
            "which tests each step against the seed spread at that step."
        )
    if statistic == "sem":
        spread /= n_seeds**0.5
    return multiple * spread


def pooled_sem(excess: pd.DataFrame, learners: list[str] | None = None) -> pd.Series:
    r"""One noise estimate per step, pooled across learners.

    **Why cross-learner comparison needs this.** Testing each learner against
    its own seed spread answers "is *this* method damaged", correctly. It does
    not give comparable break rates, because a noisier method clears a laxer
    bar: measured on x9, ``atc_plain`` is 2.7x noisier than momentum ATC in the
    break region (0.0024 against 0.0009), so it needed 2.7x the damage to
    trigger and appeared to survive longer while in fact being *more* damaged at
    every step.

    Pooling gives every method the same bar, so the ordering of break rates
    means something.
    """
    rows = excess if learners is None else excess[excess.learner.isin(learners)]
    if rows.empty:
        raise BreakError("no rows to pool a noise estimate from")
    counts = rows.groupby(["learner", "t"]).seed.nunique()
    spread = rows.groupby(["learner", "t"]).excess.std() / counts.pow(0.5)
    return spread.groupby("t").mean().fillna(0.0)


def damage_at_rate(
    excess: pd.DataFrame,
    drift: Any,
    horizon: int,
    rate: float,
) -> pd.Series:
    """Mean excess at the step where the schedule first reaches ``rate``.

    The threshold-free companion to the break rate: it asks how damaged each
    method is at a drift speed everyone faced, which needs no noise estimate
    and cannot be gamed by variance. Where the two orderings disagree, the
    disagreement is itself worth reporting.
    """
    steps = sorted(excess.t.unique())
    reached = [step for step in steps if drift.schedule.rate_at(int(step)) >= rate]
    if not reached:
        raise BreakError(
            f"the schedule never reaches {rate:.4f} deg/step (peak is "
            f"{drift.schedule.peak_rate(horizon):.4f}), so there is no matched point to "
            "compare at. Pick a rate inside the range the run actually probed."
        )
    return excess[excess.t == reached[0]].groupby("learner").excess.mean()


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
    multiple: float = 3.0,
    persistence: int = DEFAULT_PERSISTENCE,
    noise: pd.Series | None = None,
) -> BreakPoint:
    r"""The absolute break: drift damage significantly above zero.

    ``noise`` supplies a per-step standard error to test against. Pass
    :func:`pooled_sem` to give every learner the same bar, which is what makes
    break *rates* comparable across methods; leave it ``None`` to use the
    learner's own spread, which answers whether that one method is damaged.

    **Tested step by step against the seed spread at that step**, not against
    one threshold fixed in advance. A global threshold needs a quiet window to
    calibrate on, and an exact pairing does not provide one: while the schedule
    has barely moved, the drifting run and its control are the *same run* seed
    for seed, so the excess is identically zero and has no spread to measure.
    The step-wise test needs no such window -- where the runs are identical the
    mean is zero and cannot exceed anything, which is the right answer for the
    right reason rather than by luck.

    The null is "no drift damage", which the paired design makes exactly zero,
    so this is a one-sided test of the seed mean against ``multiple`` standard
    errors of that mean.
    """
    rows = excess[excess.learner == learner]
    if rows.empty:
        raise BreakError(f"no rows for learner {learner!r}")

    grouped = rows.groupby("t").excess
    mean = grouped.mean().sort_index()
    if noise is None:
        n_seeds = rows.groupby("t").seed.nunique().sort_index()
        sem = (grouped.std().sort_index() / n_seeds.pow(0.5)).fillna(0.0)
    else:
        sem = noise.reindex(mean.index).fillna(0.0)

    values = mean.to_numpy()
    condition = (values > multiple * sem.to_numpy()) & (values > 0.0)
    found = _first_persistent(mean.index.to_numpy(), condition, persistence)
    max_rate = _rate_summary(drift, horizon)
    if found is None:
        return BreakPoint(learner, "absolute", None, None, None, max_rate)
    step, index = found
    return BreakPoint(
        learner=learner,
        definition="absolute",
        step=step,
        rate_at_break=float(drift.schedule.rate_at(step)),
        value_at_break=float(values[index]),
        max_rate_probed=max_rate,
    )


def comparative_break(
    errors: pd.DataFrame,
    learner: str,
    baseline: str,
    drift: Any,
    horizon: int,
    persistence: int = DEFAULT_PERSISTENCE,
    start_step: int = 0,
) -> BreakPoint:
    """First step where ``learner`` stops beating the non-adapting ``baseline``.

    Threshold-free, which is its advantage: it asks whether continuing to adapt
    is buying anything, and the answer does not depend on a number anyone chose.

    **``start_step`` must be the baseline's freeze point.** Before it, the
    baseline is running the very algorithm it is the baseline for -- identical
    parameters, identical predictions -- so the margin is zero and "the learner
    stopped beating it" fires immediately and means nothing. Passing 0 against a
    baseline that freezes later reports a break at step 0 for every learner,
    which is how this was found.
    """
    if start_step > 0:
        errors = errors[errors.t >= start_step]
        if errors.empty:
            raise BreakError(
                f"no evaluations at or after the baseline's freeze point ({start_step}). "
                "The run ends before the baseline stops adapting, so there is nothing to "
                "compare against."
            )
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
    if index == 0:
        # Behind from the first eligible evaluation: it never led, so it cannot
        # have stopped leading. Calling this a break would blame the drift for a
        # gap that predates it -- `local_only` is simply worse than an ATC model
        # frozen mid-run, and always was.
        return BreakPoint(
            learner,
            "comparative",
            None,
            None,
            None,
            max_rate,
            note="never ahead of the baseline",
        )
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
