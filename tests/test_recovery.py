r"""Recovery from repeated abrupt shifts.

Synthesised transients, so the answers are known. What is under test is the
*alignment*: that shifts are pooled by offset rather than averaged away, that
each shift is scored against its own pre-shift level, and that "recovered" and
"still elevated" are told apart.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dekf_bench.metrics.breaks import error_by_step
from dekf_bench.metrics.recovery import (
    RecoveryError,
    aligned_profile,
    recovery_profile,
    tolerance_from_seeds,
)

HORIZON = 400
JUMP_EVERY = 50
CADENCE = 5
STEPS = list(range(0, HORIZON, CADENCE))
SEEDS = (0, 1, 2, 3, 4)


def frame_from(shape, jitter: float = 0.002) -> pd.DataFrame:
    """Per-seed rows for an error curve given as a function of the step."""
    rows = []
    for index, seed in enumerate(SEEDS):
        offset = jitter * (index - (len(SEEDS) - 1) / 2)
        for step in STEPS:
            value = min(max(shape(step) + offset, 0.0), 1.0)
            rows.append(
                {
                    "learner": "a",
                    "seed": seed,
                    "t": step,
                    "evalset": "current",
                    "metric": "error_rate",
                    "n_samples": 2000,
                    "n_correct": round(2000 * (1.0 - value)),
                }
            )
    return error_by_step(pd.DataFrame(rows), by_seed=True)


def sawtooth(base: float, spike: float, decay: int):
    """Jumps by `spike` at each shift, then decays linearly back over `decay` steps."""

    def shape(step: int) -> float:
        since = step % JUMP_EVERY
        if since >= decay:
            return base
        return base + spike * (1.0 - since / decay)

    return shape


def never_recovers(base: float, spike: float):
    """Each shift adds `spike` and none of it is ever recovered.

    Compounding, not a constant elevated level: if the error simply sat high
    the *pre-shift* baseline would be high too, so there would be no rise to
    recover from and the profile would -- correctly -- report full recovery.
    Accumulation is what un-recovered shifts actually look like.
    """
    return lambda step: base + spike * (step // JUMP_EVERY)


def delayed_then_stuck(base: float, spike: float):
    """Unchanged at the first evaluation after a shift, then up and stuck.

    Isolates the offset-0 skip: without it, that first evaluation sits at the
    pre-shift level and would score the shift as recovered before the damage
    had appeared.
    """

    def shape(step: int) -> float:
        shifts, since = divmod(step, JUMP_EVERY)
        felt = shifts if since > 0 else shifts - 1
        return base + spike * max(felt, 0)

    return shape


# =========================================================================== #
# 1. alignment
# =========================================================================== #


def test_the_transient_shape_survives_pooling() -> None:
    """A run-long mean would flatten this to one number describing neither the
    elevated part nor the settled part."""
    errors = frame_from(sawtooth(0.20, 0.10, decay=25), jitter=0.0)
    profile = aligned_profile(errors, "a", JUMP_EVERY, HORIZON)

    assert profile.offset.min() == 0
    early = profile[profile.offset == 0]["mean"].iloc[0]
    late = profile[profile.offset == 45]["mean"].iloc[0]
    assert early > 0.05, "the shift should show as a rise at offset 0"
    assert abs(late) < 0.02, "and should have decayed by the end of the interval"


def test_every_shift_is_pooled_not_just_the_first() -> None:
    """With T=400 and shifts every 50 there are seven, times five seeds."""
    errors = frame_from(sawtooth(0.20, 0.10, decay=25))
    profile = recovery_profile(errors, "a", JUMP_EVERY, HORIZON)
    assert profile.n_shifts == 7 * len(SEEDS)


def test_offsets_almost_no_shift_reaches_are_dropped() -> None:
    """The final evaluation is forced onto horizon-1, which is off-cadence, so
    it lands at an offset the other shifts never produce. Measured on X11 that
    was 5 samples against a median of 295, and it showed as a spike at the right
    edge of every transient curve -- exactly where a reader looks to judge
    whether the error came back."""
    rows = []
    for index, seed in enumerate(SEEDS):
        offset = 0.002 * index
        # The cadence, plus one off-cadence evaluation at the very end.
        for step in [*STEPS, HORIZON - 1]:
            value = sawtooth(0.20, 0.10, decay=25)(step) + offset
            rows.append(
                {
                    "learner": "a",
                    "seed": seed,
                    "t": step,
                    "evalset": "current",
                    "metric": "error_rate",
                    "n_samples": 2000,
                    "n_correct": round(2000 * (1.0 - value)),
                }
            )
    errors = error_by_step(pd.DataFrame(rows), by_seed=True)
    profile = aligned_profile(errors, "a", JUMP_EVERY, HORIZON)

    odd = (HORIZON - 1) % JUMP_EVERY
    assert odd not in set(profile.offset), "the off-cadence offset must not survive"
    assert profile["count"].min() >= 0.5 * profile["count"].median()


def test_a_cadence_too_coarse_to_see_a_transient_is_refused() -> None:
    """One evaluation per interval cannot show a recovery, and reporting 0 or 1
    would be an artefact of the cadence rather than a property of the learner."""
    errors = frame_from(sawtooth(0.20, 0.10, decay=25))
    with pytest.raises(RecoveryError, match="too coarse|two evaluations"):
        recovery_profile(errors, "a", jump_every=CADENCE, horizon=HORIZON)


# =========================================================================== #
# 2. recovered versus still elevated
# =========================================================================== #


def test_a_learner_that_decays_back_is_scored_as_recovered() -> None:
    errors = frame_from(sawtooth(0.20, 0.10, decay=20))
    profile = recovery_profile(errors, "a", JUMP_EVERY, HORIZON)
    assert profile.recovered == pytest.approx(1.0)
    assert profile.rise > 0.05


def test_a_learner_that_stays_elevated_is_not() -> None:
    """The compounding case: each shift lands and none of it is recovered, so
    the run accumulates a standing error."""
    errors = frame_from(never_recovers(0.20, 0.03))
    profile = recovery_profile(errors, "a", JUMP_EVERY, HORIZON)
    assert profile.recovered == pytest.approx(0.0)
    assert profile.rise == pytest.approx(0.03, abs=0.005)
    assert profile.standing > 0.20, "un-recovered shifts must show up as standing error"


def test_recovery_is_judged_against_each_shifts_own_baseline() -> None:
    """Not a global level. A learner still converging has a falling baseline, and
    scoring against a fixed one would read that fall as recovery from shifts it
    never recovered from."""

    def converging_and_never_recovering(step: int) -> float:
        trend = 0.40 - 0.30 * step / HORIZON
        return trend + 0.10

    errors = frame_from(converging_and_never_recovering)
    profile = recovery_profile(errors, "a", JUMP_EVERY, HORIZON)
    # The curve only ever falls, so there is no rise above the pre-shift level
    # and nothing to recover from -- which is the honest reading.
    assert profile.rise <= 0.001


def test_offset_zero_cannot_score_a_shift_as_recovered() -> None:
    """At offset 0 the shift may not have been felt yet, and counting it would
    credit a learner with recovering before any damage appeared.

    `delayed_then_stuck` is built so that first evaluation sits exactly at the
    pre-shift level while the error afterwards never comes back -- so including
    offset 0 would score every shift as recovered and this asserts it does not.
    """
    errors = frame_from(delayed_then_stuck(0.20, 0.03))
    profile = recovery_profile(errors, "a", JUMP_EVERY, HORIZON)
    assert profile.recovered == pytest.approx(0.0)

    aligned = aligned_profile(errors, "a", JUMP_EVERY, HORIZON)
    at_zero = aligned[aligned.offset == 0]["mean"].iloc[0]
    later = aligned[aligned.offset == JUMP_EVERY - CADENCE]["mean"].iloc[0]
    assert abs(at_zero) < 0.005, "offset 0 is at the pre-shift level, as constructed"
    assert later > 0.02, "and the damage is plainly there afterwards"


# =========================================================================== #
# 3. the tolerance
# =========================================================================== #


def test_the_tolerance_comes_from_the_seed_spread() -> None:
    """Derived, not chosen -- and it must shrink as the seeds agree more."""
    noisy = tolerance_from_seeds(frame_from(sawtooth(0.2, 0.1, 20), jitter=0.04), "a")
    quiet = tolerance_from_seeds(frame_from(sawtooth(0.2, 0.1, 20), jitter=0.002), "a")
    assert noisy > quiet > 0


def test_pooled_seeds_are_refused() -> None:
    """The tolerance is derived from the seed spread, so pooling first would
    silently leave nothing to derive it from."""
    rows = frame_from(sawtooth(0.2, 0.1, 20))
    pooled = rows.groupby(["learner", "t"], as_index=False).error.mean()
    with pytest.raises(RecoveryError, match="per-seed rows"):
        recovery_profile(pooled, "a", JUMP_EVERY, HORIZON)
