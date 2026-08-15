r"""Locating the break, under both definitions.

These run on synthesised curves rather than on a real run, so the answers are
known in advance. What is being tested is the *locator*: that it converts a step
to a rate, that it refuses to call a noise excursion a break, and that "did not
break" survives as a distinct outcome rather than collapsing to the end of the
run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dekf_bench.env.drift import Drift, Linear, Ramp
from dekf_bench.metrics.breaks import (
    BreakError,
    absolute_break,
    comparative_break,
    error_by_step,
    locate_breaks,
    tracking_gap,
)

HORIZON = 300
STEPS = list(range(0, HORIZON, 10))


class FlatReference:
    """An oracle whose error does not depend on rotation.

    Deliberately flat so these tests isolate the locator. A real $e^\\star$
    rises with rotation, which is exactly why the gap is taken at all.
    """

    def __init__(self, value: float = 0.1) -> None:
        self.value = value

    def at(self, rotation: float) -> float:
        return self.value


def drift_of(schedule=None) -> Drift:
    return Drift(schedule=schedule or Linear(total_degrees=45.0, horizon=HORIZON))


def frame_of(curves: dict[str, list[float]], n_samples: int = 1000) -> pd.DataFrame:
    """Recorded rows for given error curves, in counts rather than rates."""
    rows = []
    for learner, errors in curves.items():
        for step, error in zip(STEPS, errors, strict=True):
            rows.append(
                {
                    "learner": learner,
                    "t": step,
                    "evalset": "current",
                    "metric": "error_rate",
                    "n_samples": n_samples,
                    "n_correct": round(n_samples * (1.0 - error)),
                }
            )
    return pd.DataFrame(rows)


def ramp_curve(start: float, end: float) -> list[float]:
    span = len(STEPS) - 1
    return [start + (end - start) * index / span for index in range(len(STEPS))]


# =========================================================================== #
# 1. the error series
# =========================================================================== #


def test_error_is_counts_then_divide() -> None:
    """Not a mean of per-agent rates: an agent that saw two samples would weigh
    the same as one that saw eight."""
    frame = pd.DataFrame(
        [
            {
                "learner": "a",
                "t": 0,
                "evalset": "current",
                "metric": "error_rate",
                "n_samples": 2,
                "n_correct": 0,
            },
            {
                "learner": "a",
                "t": 0,
                "evalset": "current",
                "metric": "error_rate",
                "n_samples": 8,
                "n_correct": 8,
            },
        ]
    )
    assert float(error_by_step(frame).error.iloc[0]) == pytest.approx(0.2)


def test_a_missing_evalset_is_an_error_not_an_empty_answer() -> None:
    frame = frame_of({"a": ramp_curve(0.1, 0.2)})
    with pytest.raises(BreakError, match="no 'backward' error_rate rows"):
        error_by_step(frame, "backward")


# =========================================================================== #
# 2. the absolute definition
# =========================================================================== #


def test_the_gap_is_taken_against_the_reference_at_that_rotation() -> None:
    frame = frame_of({"a": [0.3] * len(STEPS)})
    gaps = tracking_gap(frame, FlatReference(0.1), drift_of())
    assert gaps.gap.round(6).eq(0.2).all()


def test_an_absolute_break_reports_the_rate_not_only_the_step() -> None:
    """The rate is the answer; the step is how it was found."""
    frame = frame_of({"a": ramp_curve(0.1, 0.5)})
    gaps = tracking_gap(frame, FlatReference(0.1), drift_of())
    point = absolute_break(gaps, "a", drift_of(), HORIZON, threshold=0.1)
    assert point.broke
    assert point.rate_at_break == pytest.approx(0.15)  # 45 deg / 300 steps


def test_a_method_that_tracks_does_not_break() -> None:
    """`None` must survive as an outcome. Reporting the end of the run would
    turn 'did not break in the range probed' into a measured threshold."""
    frame = frame_of({"a": [0.11] * len(STEPS)})
    gaps = tracking_gap(frame, FlatReference(0.1), drift_of())
    point = absolute_break(gaps, "a", drift_of(), HORIZON, threshold=0.1)
    assert not point.broke
    assert point.step is None
    assert point.max_rate_probed == pytest.approx(0.15)


def test_a_single_excursion_is_not_a_break() -> None:
    """One evaluation over the line is sampling noise."""
    curve = [0.11] * len(STEPS)
    curve[5] = 0.9
    frame = frame_of({"a": curve})
    gaps = tracking_gap(frame, FlatReference(0.1), drift_of())
    assert not absolute_break(gaps, "a", drift_of(), HORIZON, threshold=0.1).broke


def test_persistence_of_one_would_have_called_it() -> None:
    """The positive control for the test above: without the run-length
    requirement the excursion *is* located, so the guard is doing work."""
    curve = [0.11] * len(STEPS)
    curve[5] = 0.9
    frame = frame_of({"a": curve})
    gaps = tracking_gap(frame, FlatReference(0.1), drift_of())
    point = absolute_break(gaps, "a", drift_of(), HORIZON, threshold=0.1, persistence=1)
    assert point.step == STEPS[5]


def test_a_ramp_reports_the_instantaneous_rate_at_the_crossing() -> None:
    """The whole reason the ramp exists: the located step names a rate, and
    under acceleration that rate is not the run's average."""
    schedule = Ramp(total_degrees=45.0, horizon=HORIZON, exponent=4.0)
    drift = drift_of(schedule)
    frame = frame_of({"a": ramp_curve(0.1, 0.6)})
    gaps = tracking_gap(frame, FlatReference(0.1), drift)
    point = absolute_break(gaps, "a", drift, HORIZON, threshold=0.2)
    assert point.broke
    assert point.rate_at_break == pytest.approx(schedule.rate_at(point.step))
    assert point.rate_at_break != pytest.approx(schedule.mean_rate(HORIZON))


# =========================================================================== #
# 3. the comparative definition
# =========================================================================== #


def test_a_learner_that_stays_ahead_of_frozen_does_not_break() -> None:
    errors = error_by_step(
        frame_of({"a": [0.15] * len(STEPS), "frozen_atc": ramp_curve(0.15, 0.6)})
    )
    point = comparative_break(errors, "a", "frozen_atc", drift_of(), HORIZON)
    assert not point.broke


def test_a_learner_overtaken_by_frozen_breaks_where_they_cross() -> None:
    adapting = ramp_curve(0.10, 0.50)
    frozen = ramp_curve(0.30, 0.32)
    errors = error_by_step(frame_of({"a": adapting, "frozen_atc": frozen}))
    point = comparative_break(errors, "a", "frozen_atc", drift_of(), HORIZON)
    assert point.broke
    crossing = next(s for s, x, f in zip(STEPS, adapting, frozen, strict=True) if x >= f)
    assert point.step == crossing


def test_a_missing_baseline_is_refused_rather_than_guessed() -> None:
    """The baseline has to be in the same run, so the comparison is paired by
    construction rather than across runs with different seeds."""
    errors = error_by_step(frame_of({"a": ramp_curve(0.1, 0.5)}))
    with pytest.raises(BreakError, match="not in the recorded learners"):
        comparative_break(errors, "a", "frozen_atc", drift_of(), HORIZON)


# =========================================================================== #
# 4. the two definitions together
# =========================================================================== #


def test_the_definitions_can_disagree_and_both_are_reported() -> None:
    """The reason for having both. Here the learner tracks poorly in absolute
    terms -- well above the oracle -- while still beating a frozen model that
    degrades faster. That is a real regime: adaptation is helping, and is also
    not keeping up. One number could not say both."""
    drift = drift_of()
    frame = frame_of({"a": ramp_curve(0.15, 0.45), "frozen_atc": ramp_curve(0.15, 0.8)})
    table = locate_breaks(frame, FlatReference(0.1), drift, HORIZON, threshold=0.1)

    absolute = table[(table.learner == "a") & (table.definition == "absolute")].iloc[0]
    comparative = table[(table.learner == "a") & (table.definition == "comparative")].iloc[0]
    assert bool(absolute.broke)
    assert not bool(comparative.broke)


def test_the_baseline_is_not_compared_against_itself() -> None:
    """It would report step 0 and mean nothing."""
    frame = frame_of({"a": ramp_curve(0.15, 0.45), "frozen_atc": ramp_curve(0.15, 0.8)})
    table = locate_breaks(frame, FlatReference(0.1), drift_of(), HORIZON, threshold=0.1)
    rows = table[(table.learner == "frozen_atc") & (table.definition == "comparative")]
    assert rows.empty
