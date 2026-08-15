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
    assert_paired_runs,
    comparative_break,
    damage_at_rate,
    error_by_step,
    excess_break,
    locate_breaks,
    paired_excess,
    pooled_sem,
    threshold_from_seed_noise,
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


# =========================================================================== #
# 5. the paired control, and the threshold derived from it
# =========================================================================== #


def seeded_frame(curves: dict[str, list[float]], seeds=(0, 1, 2, 3, 4), jitter=0.004):
    """Recorded rows with a per-seed offset, so a seed spread exists to measure."""
    rows = []
    for index, seed in enumerate(seeds):
        offset = jitter * (index - (len(seeds) - 1) / 2)
        for learner, errors in curves.items():
            for step, error in zip(STEPS, errors, strict=True):
                value = min(max(error + offset, 0.0), 1.0)
                rows.append(
                    {
                        "learner": learner,
                        "seed": seed,
                        "t": step,
                        "evalset": "current",
                        "metric": "error_rate",
                        "n_samples": 1000,
                        "n_correct": round(1000 * (1.0 - value)),
                    }
                )
    return pd.DataFrame(rows)


def test_the_control_cancels_a_trend_shared_by_both_runs() -> None:
    """The reason for pairing. Both runs are converging hard; only the drifting
    one is also being damaged, and only that difference should survive."""
    converging = ramp_curve(0.30, 0.05)
    damaged = [value + 0.02 for value in converging]
    excess = paired_excess(seeded_frame({"a": damaged}), seeded_frame({"a": converging}))
    assert excess.excess.mean() == pytest.approx(0.02, abs=0.002)


def test_a_control_missing_a_seed_is_refused() -> None:
    """Unpaired rows would be dropped silently, and the excess would then be
    computed over a different population than it claims."""
    drifting = seeded_frame({"a": ramp_curve(0.3, 0.1)})
    control = seeded_frame({"a": ramp_curve(0.3, 0.1)}, seeds=(0, 1, 2))
    with pytest.raises(BreakError, match="missing seeds"):
        paired_excess(drifting, control)


def test_the_threshold_comes_from_the_quiet_opening_of_the_run() -> None:
    """Where an accelerating schedule has barely moved, so the excess is zero by
    construction and its spread is noise and nothing else."""
    flat = ramp_curve(0.2, 0.2)
    excess = paired_excess(seeded_frame({"a": flat}), seeded_frame({"a": flat}, jitter=0.0))
    threshold = threshold_from_seed_noise(excess, HORIZON, multiple=3.0)
    assert threshold > 0
    # sem is the sd shrunk by sqrt(5), so it must be the tighter of the two.
    conservative = threshold_from_seed_noise(excess, HORIZON, multiple=3.0, statistic="sd")
    assert threshold < conservative
    assert conservative / threshold == pytest.approx(5**0.5, rel=0.01)


def test_an_exactly_paired_opening_refuses_to_produce_a_threshold() -> None:
    """Found by running it. A ramp has barely rotated over its first quarter,
    so the drifting run and its control are the *same run* seed for seed: the
    excess is identically zero and has no spread. Returning 0 would have made
    every later step a break, which is what the first x9 report did."""
    flat = ramp_curve(0.2, 0.2)
    identical = paired_excess(seeded_frame({"a": flat}), seeded_frame({"a": flat}))
    with pytest.raises(BreakError, match="no spread"):
        threshold_from_seed_noise(identical, HORIZON)


def test_the_step_wise_test_needs_no_quiet_window() -> None:
    """The replacement: where the runs are identical the mean excess is zero
    and cannot exceed anything, so it reports no break for the right reason
    rather than dividing by a spread that does not exist."""
    flat = ramp_curve(0.2, 0.2)
    identical = paired_excess(seeded_frame({"a": flat}), seeded_frame({"a": flat}))
    assert not excess_break(identical, "a", drift_of(), HORIZON).broke


def test_one_seed_cannot_produce_a_threshold() -> None:
    """Better to refuse than to return zero and call everything a break."""
    flat = ramp_curve(0.2, 0.2)
    excess = paired_excess(
        seeded_frame({"a": flat}, seeds=(0,)), seeded_frame({"a": flat}, seeds=(0,))
    )
    with pytest.raises(BreakError, match="cannot be estimated"):
        threshold_from_seed_noise(excess, HORIZON)


def test_the_excess_break_fires_on_damage_not_on_convergence() -> None:
    """The failure the control exists to prevent: a learner whose raw gap is
    large and falling is *not* breaking, and must not be reported as such."""
    converging = ramp_curve(0.40, 0.05)
    excess = paired_excess(seeded_frame({"a": converging}), seeded_frame({"a": converging}))
    point = excess_break(excess, "a", drift_of(), HORIZON)
    assert not point.broke

    damaged = [value + 0.05 * index / (len(STEPS) - 1) for index, value in enumerate(converging)]
    hurt = paired_excess(seeded_frame({"a": damaged}), seeded_frame({"a": converging}))
    assert excess_break(hurt, "a", drift_of(), HORIZON).broke


# =========================================================================== #
# 6. the baseline is only a baseline once it has frozen
# =========================================================================== #


def test_the_comparison_must_start_at_the_freeze_point() -> None:
    """Found by running it. Before freezing, `frozen_atc` *is* the learner it is
    the baseline for: identical parameters, identical predictions, margin zero.

    The failure that causes is subtler than it first looked. The zero-margin
    prefix satisfies "no longer ahead" at the very first evaluation, which the
    never-ahead guard then reads as "it never led" -- so a real break later in
    the run is *hidden*, not merely mislocated. Only starting the window at the
    freeze point gets both halves right.
    """
    freeze_index = len(STEPS) // 3
    tail = len(STEPS) - freeze_index
    # Identical until the freeze. Then the learner starts ahead and degrades
    # past the frozen baseline, which is a genuine break that must be found.
    learner = [0.30] * freeze_index + [0.20 + 0.20 * i / (tail - 1) for i in range(tail)]
    baseline = [0.30] * freeze_index + [0.25] * tail
    errors = error_by_step(frame_of({"a": learner, "frozen_atc": baseline}))

    naive = comparative_break(errors, "a", "frozen_atc", drift_of(), HORIZON)
    assert not naive.broke
    assert naive.note == "never ahead of the baseline", "the shared prefix masks the real break"

    guarded = comparative_break(
        errors, "a", "frozen_atc", drift_of(), HORIZON, start_step=STEPS[freeze_index]
    )
    assert guarded.broke
    assert guarded.step > STEPS[freeze_index]


def write_run(directory, **overrides):
    """A results directory with just the config.yaml the pairing check reads."""
    import yaml

    config = {
        "graph": {"topology": "ring", "n_nodes": 10},
        "model": {"name": "mlp_small"},
        "learners": [{"name": "diffusion_sgd_atc", "lr": 0.01}],
        "env": {"samples_per_node_per_step": 4, "drift": {"schedule": "ramp"}},
        "run": {"seeds": [0, 1, 2], "horizon": 1500, "eval_every": 10},
    }
    config.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return directory


def test_a_control_differing_only_in_drift_is_a_valid_pairing(tmp_path) -> None:
    """The one thing that is supposed to differ."""
    drifting = write_run(tmp_path / "ramp")
    control = write_run(
        tmp_path / "control",
        env={"samples_per_node_per_step": 4, "drift": {"schedule": "stationary"}},
    )
    assert_paired_runs(drifting, control)


def test_a_control_with_different_learners_is_refused(tmp_path) -> None:
    """Found live: re-running a pair after a config fix leaves the old control
    on disk until its stage starts, so for hours the new drifting run and the
    *previous* control both exist and pair without complaint. Nothing in the
    data reveals it -- the excess would just quietly be part drift damage and
    part learning-rate difference."""
    drifting = write_run(tmp_path / "ramp")
    control = write_run(tmp_path / "control", learners=[{"name": "diffusion_sgd_atc", "lr": 0.05}])
    with pytest.raises(BreakError, match="learners"):
        assert_paired_runs(drifting, control)


def test_a_control_with_a_different_seed_set_is_refused(tmp_path) -> None:
    """Pairing is by seed, so a different set silently changes the population."""
    drifting = write_run(tmp_path / "ramp")
    control = write_run(
        tmp_path / "control", run={"seeds": [0, 1], "horizon": 1500, "eval_every": 10}
    )
    with pytest.raises(BreakError, match="run.seeds"):
        assert_paired_runs(drifting, control)


def test_a_missing_config_is_refused_rather_than_assumed_fine(tmp_path) -> None:
    drifting = write_run(tmp_path / "ramp")
    (tmp_path / "control").mkdir()
    with pytest.raises(BreakError, match="no config.yaml"):
        assert_paired_runs(drifting, tmp_path / "control")


def test_a_pooled_bar_stops_a_noisy_learner_looking_robust() -> None:
    """The x9 artifact, pinned. Two learners with the *same* damage but very
    different seed spread: tested against their own noise the noisy one appears
    to survive longer, and against a pooled bar they break together."""
    damaged = [0.20 + 0.10 * i / (len(STEPS) - 1) for i in range(len(STEPS))]
    flat = [0.20] * len(STEPS)
    quiet = paired_excess(seeded_frame({"q": damaged}, jitter=0.001), seeded_frame({"q": flat}))
    noisy = paired_excess(seeded_frame({"n": damaged}, jitter=0.030), seeded_frame({"n": flat}))
    both = pd.concat([quiet, noisy], ignore_index=True)

    own = {name: excess_break(both, name, drift_of(), HORIZON).step for name in ("q", "n")}
    assert own["n"] > own["q"], "against its own noise the noisy learner survives longer"

    pooled = pooled_sem(both)
    together = {
        name: excess_break(both, name, drift_of(), HORIZON, noise=pooled).step
        for name in ("q", "n")
    }
    assert together["n"] == together["q"], "against one bar, equal damage breaks equally"


def test_damage_at_a_matched_rate_needs_no_noise_estimate() -> None:
    """The threshold-free companion: it compares methods at a speed both faced,
    so it cannot be gamed by one being noisier than the other."""
    worse = [0.30] * len(STEPS)
    better = [0.20] * len(STEPS)
    flat = [0.20] * len(STEPS)
    excess = paired_excess(
        seeded_frame({"worse": worse, "better": better}),
        seeded_frame({"worse": flat, "better": flat}),
    )
    at_rate = damage_at_rate(excess, drift_of(), HORIZON, rate=0.15)
    assert at_rate["worse"] > at_rate["better"]


def test_a_rate_outside_the_probed_range_is_refused() -> None:
    """Returning the closest step would silently answer a different question."""
    flat = [0.2] * len(STEPS)
    excess = paired_excess(seeded_frame({"a": flat}), seeded_frame({"a": flat}, jitter=0.0))
    with pytest.raises(BreakError, match="never reaches"):
        damage_at_rate(excess, drift_of(), HORIZON, rate=99.0)


def test_a_learner_never_ahead_is_reported_as_such_not_as_a_break() -> None:
    """`local_only` is simply worse than an ATC model frozen mid-run, and always
    was. Recording that as a break would blame the drift for a gap that predates
    it."""
    errors = error_by_step(
        frame_of({"a": ramp_curve(0.40, 0.45), "frozen_atc": ramp_curve(0.10, 0.20)})
    )
    point = comparative_break(errors, "a", "frozen_atc", drift_of(), HORIZON)
    assert not point.broke
    assert point.note == "never ahead of the baseline"


def test_the_baseline_is_not_compared_against_itself() -> None:
    """It would report step 0 and mean nothing."""
    frame = frame_of({"a": ramp_curve(0.15, 0.45), "frozen_atc": ramp_curve(0.15, 0.8)})
    table = locate_breaks(frame, FlatReference(0.1), drift_of(), HORIZON, threshold=0.1)
    rows = table[(table.learner == "frozen_atc") & (table.definition == "comparative")]
    assert rows.empty
