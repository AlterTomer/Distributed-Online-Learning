r"""Class-prior drift: the label-shift channel.

The properties that matter are that the *served* composition tracks the planned
prior, that it does so without weakening any guarantee the stream already made
-- disjointness, exactly-once, equal shard sizes -- and that an infeasible plan
is refused before a run starts rather than discovered at step 1400.

There is a positive control here for the same reason ``test_exactness`` has one:
a test that the composition "moves" would pass on a stream that ignored the
plan entirely, so the tests below check *where* it moves to.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.env.drift import Linear, Ramp
from dekf_bench.env.environment import build_environment
from dekf_bench.env.partition import build_partition
from dekf_bench.env.priors import (
    ClassPriors,
    PriorDriftError,
    build_class_plan,
    build_class_priors,
    check_plan_is_feasible,
)
from dekf_bench.env.stream import build_stream
from dekf_bench.utils.config import ConfigError, deep_merge, load_config

N_NODES = 6
N_CLASSES = 10
HORIZON = 400
SAMPLES = 2


def labels_of(n: int = 30000) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int64) % N_CLASSES


def split(n: int = 30000) -> MnistSplit:
    generator = torch.Generator().manual_seed(0)
    return MnistSplit(
        images=torch.rand(n, 1, 28, 28, generator=generator),
        labels=labels_of(n),
        split="synthetic",
    )


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def priors_of(total_shift: float = 1.0, beta: float = 0.3) -> ClassPriors:
    return build_class_priors(
        n_nodes=N_NODES,
        n_classes=N_CLASSES,
        beta=beta,
        total_shift=total_shift,
        generator=generator(),
    )


def plan_of(total_shift: float = 1.0, beta: float = 0.3, horizon: int = HORIZON):
    return build_class_plan(
        priors=priors_of(total_shift, beta),
        schedule=Linear(total_degrees=45.0, horizon=horizon),
        horizon=horizon,
        samples_per_step=SAMPLES,
        generator=generator(1),
    )


# =========================================================================== #
# 1. the priors themselves
# =========================================================================== #


def test_a_uniform_start_means_progress_zero_is_the_ordinary_experiment() -> None:
    """So that anything measured is the drift, not a different starting point."""
    priors = priors_of()
    assert torch.allclose(
        priors.at(0.0), torch.full((N_NODES, N_CLASSES), 0.1, dtype=torch.float64)
    )


def test_the_path_stays_on_the_simplex() -> None:
    priors = priors_of()
    for progress in (0.0, 0.25, 0.5, 1.0, 2.0):
        table = priors.at(progress)
        assert torch.all(table >= 0)
        assert torch.allclose(table.sum(dim=1), torch.ones(N_NODES, dtype=torch.float64))


def test_progress_past_one_does_not_overshoot_the_endpoint() -> None:
    """A ramp reaches progress 1 at the horizon, but `total_travel` and float
    error can nudge past it, and extrapolating would ask for negative mass."""
    priors = priors_of()
    assert torch.allclose(priors.at(1.0), priors.at(4.0))


def test_negative_progress_is_clamped_rather_than_extrapolated() -> None:
    """Sinusoidal progress goes negative; there is no distribution before the
    start to travel toward, so the run simply sits at the start."""
    priors = priors_of()
    assert torch.allclose(priors.at(-0.5), priors.at(0.0))


def test_total_shift_scales_how_far_the_prior_travels() -> None:
    """The magnitude knob. Half the shift should move half the distance."""
    full = priors_of(total_shift=1.0).travel()
    half = priors_of(total_shift=0.5).travel()
    assert torch.allclose(half, full / 2, atol=1e-9)


def test_a_shift_past_one_is_rejected() -> None:
    with pytest.raises(PriorDriftError, match="leaves the simplex"):
        ClassPriors(
            start=torch.full((2, 2), 0.5, dtype=torch.float64),
            end=torch.full((2, 2), 0.5, dtype=torch.float64),
            total_shift=1.5,
        )


def test_rows_that_are_not_distributions_are_rejected() -> None:
    with pytest.raises(PriorDriftError, match="sum to 1"):
        ClassPriors(
            start=torch.full((2, 2), 0.4, dtype=torch.float64),
            end=torch.full((2, 2), 0.5, dtype=torch.float64),
        )


# =========================================================================== #
# 2. the plan
# =========================================================================== #


def test_the_plan_covers_every_agent_and_step() -> None:
    plan = plan_of()
    assert plan.classes.shape == (N_NODES, HORIZON, SAMPLES)
    assert int(plan.classes.min()) >= 0
    assert int(plan.classes.max()) < N_CLASSES


def test_demand_totals_the_whole_plan() -> None:
    """The partition is sized from this, so an undercount would exhaust a shard."""
    plan = plan_of()
    assert int(plan.demand().sum()) == N_NODES * HORIZON * SAMPLES


def test_the_planned_composition_tracks_the_prior_not_just_moves() -> None:
    """The positive control. A stream that drifted *somewhere* would pass a test
    that only asked whether the composition changed; this pins where it goes."""
    priors = priors_of()
    plan = plan_of()
    late = plan.classes[:, -HORIZON // 4 :, :]
    for node in range(N_NODES):
        observed = torch.bincount(late[node].reshape(-1), minlength=N_CLASSES).to(torch.float64)
        observed = observed / observed.sum()
        intended = priors.at(1.0)[node]
        # Multinomial noise at n=2 per step is large, so this is a loose bound;
        # what it rules out is the composition drifting to the *wrong* place.
        assert float(0.5 * (observed - intended).abs().sum()) < 0.2


def test_a_ramp_delays_the_prior_shift_the_way_it_delays_rotation() -> None:
    """Both channels ride the same schedule, so an accelerating run must be
    late in the prior too -- otherwise the two would drift out of step."""
    ramp = build_class_plan(
        priors=priors_of(),
        schedule=Ramp(total_degrees=45.0, horizon=HORIZON, exponent=4.0),
        horizon=HORIZON,
        samples_per_step=SAMPLES,
        generator=generator(1),
    )
    linear = plan_of()
    half = HORIZON // 2
    start = priors_of().at(0.0)[0]

    def distance(plan, window) -> float:
        counts = torch.bincount(plan.classes[0][window].reshape(-1), minlength=N_CLASSES)
        share = counts.to(torch.float64) / counts.sum()
        return float(0.5 * (share - start).abs().sum())

    assert distance(ramp, slice(0, half)) < distance(linear, slice(0, half))


# =========================================================================== #
# 3. feasibility
# =========================================================================== #


def test_an_oversubscribed_plan_is_refused_up_front() -> None:
    """The shards compete for one disjoint pool per class, so this has no valid
    partition at all -- there is nothing to degrade gracefully into."""
    plan = build_class_plan(
        priors=build_class_priors(
            n_nodes=N_NODES,
            n_classes=N_CLASSES,
            beta=0.01,
            generator=generator(),
        ),
        schedule=Linear(total_degrees=45.0, horizon=4000),
        horizon=4000,
        samples_per_step=8,
        generator=generator(1),
    )
    with pytest.raises(PriorDriftError, match="oversubscribes"):
        check_plan_is_feasible(plan, labels_of(3000))


def test_a_feasible_plan_passes_the_check() -> None:
    check_plan_is_feasible(plan_of(), labels_of())


# =========================================================================== #
# 4. partition and stream together
# =========================================================================== #


def test_the_shard_holds_exactly_what_the_plan_demands() -> None:
    """Sized from the plan rather than drawn and hoped to fit."""
    plan = plan_of()
    labels = labels_of()
    partition = build_partition(
        labels, N_NODES, n_classes=N_CLASSES, generator=generator(2), demand=plan.demand()
    )
    held = partition.class_counts(labels)
    assert torch.all(held >= plan.demand())


def test_prior_drift_keeps_shard_sizes_equal_and_the_cover_complete() -> None:
    """Neither guarantee is negotiable: equal sizes keep X6 measuring one thing,
    and full coverage is what the exactly-once accounting rests on."""
    plan = plan_of()
    labels = labels_of()
    partition = build_partition(
        labels, N_NODES, n_classes=N_CLASSES, generator=generator(2), demand=plan.demand()
    )
    assert int(partition.sizes.max()) - int(partition.sizes.min()) <= 1
    assert partition.total == int(labels.numel())


def test_the_unserved_filler_does_not_invent_skew() -> None:
    """The filler never reaches a learner, but it lands in `partition.skew()`,
    and a lumpy remainder would report a skew the experiment does not have."""
    plan = plan_of()
    labels = labels_of()
    partition = build_partition(
        labels, N_NODES, n_classes=N_CLASSES, generator=generator(2), demand=plan.demand()
    )
    consumed = plan.demand().sum(dim=1)[0]
    assert consumed < partition.sizes[0]  # there is filler to get wrong
    assert partition.skew(labels) < 0.25


def test_the_stream_serves_the_planned_classes() -> None:
    """The whole point: prior drift reaches the learner through the serving
    order, so what comes out of `indices_at` must have the planned labels."""
    plan = plan_of()
    labels = labels_of()
    partition = build_partition(
        labels, N_NODES, n_classes=N_CLASSES, generator=generator(2), demand=plan.demand()
    )
    stream = build_stream(
        partition,
        horizon=HORIZON,
        samples_per_step=SAMPLES,
        generator=generator(3),
        class_plan=plan,
        labels=labels,
    )
    for node in (0, N_NODES - 1):
        for step in (0, HORIZON // 2, HORIZON - 1):
            served = labels[stream.indices_at(node, step)]
            assert sorted(served.tolist()) == sorted(plan.classes[node, step].tolist())


def test_prior_drift_preserves_exactly_once() -> None:
    """Assumption 5: a sample counted twice makes the filter's covariance shrink
    faster than the evidence justifies."""
    plan = plan_of()
    labels = labels_of()
    partition = build_partition(
        labels, N_NODES, n_classes=N_CLASSES, generator=generator(2), demand=plan.demand()
    )
    stream = build_stream(
        partition,
        horizon=HORIZON,
        samples_per_step=SAMPLES,
        generator=generator(3),
        class_plan=plan,
        labels=labels,
    )
    served = torch.cat([stream.consumed_by(node) for node in range(N_NODES)])
    assert served.numel() == N_NODES * HORIZON * SAMPLES
    assert served.unique().numel() == served.numel()


# =========================================================================== #
# 5. through the config
# =========================================================================== #


def environment_with(**prior_drift):
    config = load_config(
        "x2_rotating",
        overrides=deep_merge(
            {},
            {
                "run": {"horizon": 300},
                "graph": {"n_nodes": N_NODES},
                "env": {"prior_drift": {"enabled": True, **prior_drift}},
            },
        ),
    )
    return build_environment(config, 0, split()), config


def test_the_channel_is_off_by_default() -> None:
    """A run without prior drift must be the same experiment it was before this
    channel existed -- so the plan is absent, not uniform. A uniform plan would
    still reorder every shard, which is not the same thing as doing nothing."""
    config = load_config(
        "x2_rotating",
        overrides={"run": {"horizon": 300}, "graph": {"n_nodes": N_NODES}},
    )
    assert config.env.prior_drift.enabled is False
    environment = build_environment(config, 0, split())
    assert environment.partition.kind != "prior_drift"


def test_enabling_it_moves_the_served_composition() -> None:
    environment, _ = environment_with(beta=0.3, total_shift=1.0)
    labels = environment.train.labels
    horizon = environment.horizon

    def composition(steps) -> torch.Tensor:
        counts = torch.zeros(N_CLASSES, dtype=torch.int64)
        for step in steps:
            counts += torch.bincount(
                labels[environment.stream.indices_at(0, step)], minlength=N_CLASSES
            )
        return counts.to(torch.float64) / counts.sum()

    early = composition(range(horizon // 5))
    late = composition(range(4 * horizon // 5, horizon))
    assert float(0.5 * (late - early).abs().sum()) > 0.25


def test_zero_shift_leaves_the_composition_alone() -> None:
    """The negative control: the channel wired in but asked to travel nowhere."""
    environment, _ = environment_with(beta=0.3, total_shift=0.0)
    labels = environment.train.labels
    horizon = environment.horizon
    counts = torch.zeros(N_CLASSES, dtype=torch.int64)
    for step in range(4 * horizon // 5, horizon):
        counts += torch.bincount(
            labels[environment.stream.indices_at(0, step)], minlength=N_CLASSES
        )
    share = counts.to(torch.float64) / counts.sum()
    assert float(0.5 * (share - 0.1).abs().sum()) < 0.15


def test_a_negative_shift_is_rejected_by_the_config() -> None:
    with pytest.raises(ConfigError, match="total_shift"):
        load_config(
            "x2_rotating",
            overrides={"env": {"prior_drift": {"enabled": True, "total_shift": -0.5}}},
        )
