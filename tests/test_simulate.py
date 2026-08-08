r"""The runner: the loop, the ordering, and what every learner shares.

The gate `IMPLEMENTATION.md` §13.7 names. What matters here is not that the loop
produces numbers -- it is that the *comparison* is fair. Two learners in one run
must see the same stream, the same graph and the same order of operations, so a
difference in the curves is a difference in the method.

The exactness identity itself lives in ``test_exactness.py``; this file tests
the machinery that identity is measured through.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.env.environment import build_environment
from dekf_bench.evaluation.evalsets import build_evalsets
from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.runner import simulate
from dekf_bench.runner.simulate import REFERENCE_LEARNER, SimulationError, pool
from dekf_bench.utils.config import load_config

STEPS = 12


# =========================================================================== #
# fixtures
# =========================================================================== #


@pytest.fixture(scope="module")
def data() -> tuple[MnistSplit, MnistSplit]:
    """Synthetic, so the suite does not need the MNIST cache. Shapes and dtypes
    match the real split; the runner cannot tell the difference."""
    generator = torch.Generator().manual_seed(0)
    return (
        MnistSplit(
            images=torch.rand(4000, 1, 28, 28, generator=generator),
            labels=torch.randint(0, 10, (4000,), generator=generator),
            split="train",
        ),
        MnistSplit(
            images=torch.rand(400, 1, 28, 28, generator=generator),
            labels=torch.randint(0, 10, (400,), generator=generator),
            split="test",
        ),
    )


def setup(
    experiment: str = "x1_stationary",
    seed: int = 0,
    data: tuple[MnistSplit, MnistSplit] | None = None,
    **overrides: Any,
):
    """A run's worth of objects, wired exactly as ``run_experiment.py`` wires
    them -- so a test that passes here is testing the real path."""
    assert data is not None
    train, test = data
    # A short horizon so the suite runs in seconds. Legitimate here in a way it
    # is not for X0: alpha is derived as total_degrees/T, so shortening T changes
    # the drift *rate* -- which these tests do not depend on, and the exactness
    # check does. That is why simulate.run grew `stop_after` rather than letting
    # the X0 path reconfigure the horizon.
    overrides.setdefault("run", {}).setdefault("horizon", STEPS)
    config = load_config(experiment, overrides=overrides)
    environment = build_environment(config, seed, train)
    model = build_model_from_config(config)
    likelihood = Categorical(10)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    return (
        config,
        environment,
        learners,
        build_evalsets(config, environment, test),
        likelihood,
        theta0,
    )


def execute(*args: Any, **kwargs: Any):
    config, environment, learners, evalsets, likelihood, theta0 = args
    return simulate.run(config, environment, learners, evalsets, likelihood, theta0, **kwargs)


# =========================================================================== #
# 1. the loop
# =========================================================================== #


def test_a_run_produces_records_for_every_learner(data) -> None:
    pieces = setup(data=data)
    records = execute(*pieces, stop_after=STEPS - 1)
    assert {record.learner for record in records} == set(pieces[2])


def test_stop_after_is_inclusive(data) -> None:
    """The checkpoint stores the last *completed* step, so an off-by-one here
    would make a resumed run skip or repeat one."""
    records = execute(*setup(data=data), stop_after=STEPS - 1)
    assert max(record.step for record in records) == STEPS - 1


def test_every_learner_starts_from_the_same_parameters(data) -> None:
    """Otherwise a curve separates for a reason that has nothing to do with
    the method."""
    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)

    # run() is what calls init(), so the check is on what each learner holds at
    # the moment it first adapts -- every agent of every method at exactly
    # theta0. Before run() the learners hold no state at all, which is itself
    # part of the guarantee: there is no way to start one from parameters the
    # runner did not hand it.
    at_first_adapt: dict[str, list[torch.Tensor]] = {}
    for name, learner in learners.items():
        at_first_adapt[name] = []
        if name == REFERENCE_LEARNER:
            original_pooled = learner.adapt_pooled

            def pooled(x, y, _n=name, _l=learner, _o=original_pooled):
                at_first_adapt[_n].append(_l.flat_params(0).clone())
                return _o(x, y)

            learner.adapt_pooled = pooled  # type: ignore[method-assign]
        else:
            original = learner.adapt

            def capture(node, observation, _n=name, _l=learner, _o=original):
                at_first_adapt[_n].append(_l.flat_params(node).clone())
                return _o(node, observation)

            learner.adapt = capture  # type: ignore[method-assign]

    simulate.run(config, environment, learners, evalsets, likelihood, theta0, stop_after=0)

    for name, captured in at_first_adapt.items():
        assert captured, f"{name} never adapted"
        assert all(torch.equal(theta, theta0) for theta in captured), name


def test_the_environment_is_stepped_once_per_step_regardless_of_learner_count(
    data,
) -> None:
    """The property the whole comparison rests on. If the runner called
    ``env.step(t)`` inside the learner loop, four learners would each see a
    *different* draw and the run would compare methods on different data.

    Checked by counting calls, not by assuming: this is exactly the kind of
    refactor that gets broken later while every curve still looks reasonable.
    """
    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)
    assert len(learners) > 1

    calls: list[int] = []
    original = environment.step

    def counting(step: int):
        calls.append(step)
        return original(step)

    object.__setattr__(environment, "step", counting)  # Environment is frozen
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, stop_after=STEPS - 1)
    assert calls == list(range(STEPS))


def test_all_learners_see_the_same_observations(data) -> None:
    """The other half of the same guarantee: one draw, shared by reference."""
    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)
    seen: dict[int, list[int]] = {}
    original = environment.step

    def recording(step: int):
        observations = original(step)
        seen[step] = sorted(int(obs.n_samples) for obs in observations.values() if obs.has_label)
        return observations

    object.__setattr__(environment, "step", recording)
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, stop_after=STEPS - 1)
    assert len(seen) == STEPS


def test_two_runs_of_one_config_agree(data) -> None:
    """Determinism given a seed. Without it nothing downstream is measurable:
    a difference between two configs cannot be separated from run-to-run
    noise."""
    first = execute(*setup(data=data), stop_after=STEPS - 1)
    second = execute(*setup(data=data), stop_after=STEPS - 1)
    assert [(r.step, r.learner, r.rows) for r in first] == [
        (r.step, r.learner, r.rows) for r in second
    ]


@pytest.mark.parametrize(
    "name",
    ["centralized_sgd", "diffusion_sgd_atc", "diffusion_sgd_atc_plain", "local_only"],
)
def test_a_learner_is_unaffected_by_its_companions(name: str, data) -> None:
    """`IMPLEMENTATION.md` §13.7: a one-learner config must reproduce the same
    trajectory as that learner run alongside the others.

    The learners share the environment, the graph and the observation objects,
    so a leak between them is entirely possible -- and it would be invisible,
    because every curve would still look reasonable. It also means X1 can be
    re-run with one learner added without invalidating the others.
    """
    together = execute(*setup(data=data), stop_after=STEPS - 1)

    pieces = setup(data=data)
    config, environment, learners, evalsets, likelihood, theta0 = pieces
    alone = simulate.run(
        config,
        environment,
        {name: learners[name]},
        evalsets,
        likelihood,
        theta0,
        stop_after=STEPS - 1,
    )

    def rows(records: list[Any]) -> list[Any]:
        # e_cent is excluded: it is the distance to the centralized reference,
        # so it is *structurally* absent when a diffusion learner runs alone.
        # That is correct behaviour rather than a leak -- the runner omits the
        # row instead of logging a placeholder that would pool as a real value.
        return [
            (record.step, row)
            for record in sorted(records, key=lambda r: r.step)
            for row in record.rows
            if record.learner == name and row["metric"] != "e_cent"
        ]

    assert rows(alone) == rows(together)


def test_e_cent_needs_the_reference_and_is_omitted_without_it(data) -> None:
    """The control for the exclusion above: absent alone, present together."""
    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)
    alone = simulate.run(
        config,
        environment,
        {"diffusion_sgd_atc": learners["diffusion_sgd_atc"]},
        evalsets,
        likelihood,
        theta0,
        stop_after=STEPS - 1,
    )
    assert not [r for rec in alone for r in rec.rows if r["metric"] == "e_cent"]

    together = execute(*setup(data=data), stop_after=STEPS - 1)
    assert [
        r
        for rec in together
        for r in rec.rows
        if r["metric"] == "e_cent" and rec.learner == "diffusion_sgd_atc"
    ]


def test_different_seeds_give_different_runs(data) -> None:
    """The control: if the seed did nothing, the test above would pass on a
    runner that ignored randomness entirely."""
    first = execute(*setup(seed=0, data=data), stop_after=STEPS - 1)
    second = execute(*setup(seed=1, data=data), stop_after=STEPS - 1)
    assert [r.rows for r in first] != [r.rows for r in second]


# =========================================================================== #
# 2. prequential ordering
# =========================================================================== #


def test_the_incoming_batch_is_scored_before_it_is_learned_from(data) -> None:
    """Test-then-train. Scoring after the update leaks the label into the
    prediction and every method reports an error rate that is too low -- by an
    amount that grows with the learning rate, so it looks like a *result*.

    Verified structurally: the parameters used for the prequential score must be
    the ones held *before* adapt, so a learner frozen at theta0 for the scoring
    call must reproduce the logged value.
    """
    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)
    name = "diffusion_sgd_atc"
    learners = {name: learners[name]}

    captured: list[torch.Tensor] = []
    learner = learners[name]
    original_adapt = learner.adapt

    def capturing(node: int, observation: Any):
        captured.append(learner.flat_params(node).clone())
        return original_adapt(node, observation)

    learner.adapt = capturing  # type: ignore[method-assign]
    simulate.run(config, environment, learners, evalsets, likelihood, theta0, stop_after=0)

    # At step 0 every agent must still be at theta0 when adapt is called: the
    # prequential score has already been taken, and nothing has moved.
    assert captured, "adapt was never called"
    assert all(torch.equal(theta, theta0) for theta in captured)


def test_prequential_rows_carry_the_batch_size(data) -> None:
    """Aggregation is counts-then-divide, so a row without ``n_samples`` cannot
    be pooled across steps or agents (design note on rates vs counts)."""
    records = execute(*setup(data=data), stop_after=STEPS - 1)
    rows = [
        row
        for record in records
        for row in record.rows
        if row.get("evalset") == "prequential" and row["metric"] == "error_rate"
    ]
    assert rows
    assert all(row["n_samples"] > 0 for row in rows)
    assert all(0 <= row["n_correct"] <= row["n_samples"] for row in rows)


def test_an_idle_agent_logs_no_prequential_row(data) -> None:
    """There is nothing to score. A zero would be indistinguishable from a
    correct prediction and would drag the pooled rate toward zero."""
    records = execute(
        *setup(data=data, **{"env": {"label_availability": 0.4}}), stop_after=STEPS - 1
    )
    rows = [
        row
        for record in records
        for row in record.rows
        if row.get("evalset") == "prequential" and row["metric"] == "error_rate"
    ]
    assert rows, "no agent was ever labelled"
    assert all(row["n_samples"] > 0 for row in rows)


# =========================================================================== #
# 3. the pooled batch
# =========================================================================== #


def test_pooling_is_the_union_of_the_agent_batches(data) -> None:
    _, environment, *_ = setup(data=data)
    observations = environment.step(0)
    x, y = pool(observations)
    expected = sum(obs.n_samples for obs in observations.values() if obs.has_label)
    assert x.shape[0] == y.shape[0] == expected


def test_pooling_preserves_every_sample(data) -> None:
    """Not just the count. A pool that dropped or duplicated a row would leave
    the shapes right and X0's residual small but non-zero."""
    _, environment, *_ = setup(data=data)
    observations = environment.step(0)
    x, _ = pool(observations)
    for obs in observations.values():
        if not obs.has_label:
            continue
        for sample in obs.x:
            assert bool((x == sample).all(dim=(1, 2, 3)).any()), "a sample was lost in pooling"


def test_pooling_an_idle_network_gives_an_empty_batch(data) -> None:
    _, environment, *_ = setup(data=data)
    observations = environment.step(0)
    idle = {
        node: type(obs)(
            x=obs.x[:0],
            y=None,
            has_label=False,
            n_samples=0,
            node=obs.node,
            step=obs.step,
            rotation_degrees=obs.rotation_degrees,
        )
        for node, obs in observations.items()
    }
    x, y = pool(idle)
    assert x.shape[0] == 0 and y.shape[0] == 0
    assert x.shape[1:] == observations[0].x.shape[1:]  # shape survives, for the model


# =========================================================================== #
# 4. the communication ledger
# =========================================================================== #


def ledger(records: list[Any], name: str, column: str) -> list[int]:
    """The cumulative ledger for one learner, in step order.

    Communication rides on *every* row as ``cum_scalars_tx``/``cum_rounds``
    rather than being its own metric, so F2 can plot any metric against cost
    without a join.
    """
    return [
        record.rows[0][column]
        for record in sorted(records, key=lambda r: r.step)
        if record.learner == name and record.rows
    ]


def test_local_only_transmits_nothing(data) -> None:
    records = execute(*setup(data=data), stop_after=STEPS - 1)
    assert ledger(records, "local_only", "cum_scalars_tx") == [0] * STEPS
    assert ledger(records, "local_only", "cum_rounds") == [0] * STEPS


def test_the_centralized_reference_is_off_the_diffusion_axis(data) -> None:
    """It communicates -- it pools every sample -- but not on the axis F2
    measures, so it is logged as zero there and drawn as a horizontal line
    (design note D30)."""
    records = execute(*setup(data=data), stop_after=STEPS - 1)
    assert ledger(records, REFERENCE_LEARNER, "cum_scalars_tx") == [0] * STEPS


def test_the_ledger_accumulates_one_round_per_step(data) -> None:
    records = execute(*setup(data=data), stop_after=STEPS - 1)
    assert ledger(records, "diffusion_sgd_atc", "cum_rounds") == list(range(1, STEPS + 1))


def test_the_ledger_is_monotone_and_linear(data) -> None:
    """A per-step cost that varied would mean the payload depends on the data,
    which none of these methods do -- and would make F2's x-axis unreadable."""
    scalars = ledger(
        records := execute(*setup(data=data), stop_after=STEPS - 1),
        "diffusion_sgd_atc",
        "cum_scalars_tx",
    )
    assert records
    increments = {b - a for a, b in zip(scalars, scalars[1:], strict=False)}
    assert len(increments) == 1 and increments.pop() > 0


def test_the_momentum_variant_costs_twice_the_plain_one(data) -> None:
    """The pairing the phase-5 claim rests on, measured through the runner
    rather than asserted from the learner (design note D29)."""
    records = execute(*setup(data=data), stop_after=STEPS - 1)
    momentum = ledger(records, "diffusion_sgd_atc", "cum_scalars_tx")
    plain = ledger(records, "diffusion_sgd_atc_plain", "cum_scalars_tx")
    assert momentum == [2 * value for value in plain]


# =========================================================================== #
# 5. disagreement
# =========================================================================== #


def test_the_centralized_reference_never_disagrees(data) -> None:
    """Every agent holds the same theta by construction, so a non-zero
    E_agree there means the runner is reading the wrong state."""
    records = execute(*setup(data=data), stop_after=STEPS - 1)
    rows = [
        row
        for record in records
        for row in record.rows
        if record.learner == REFERENCE_LEARNER and row["metric"] == "e_agree"
    ]
    assert rows and all(row["value"] == pytest.approx(0.0, abs=1e-30) for row in rows)


def test_local_only_disagrees_and_diffusion_less_so(data) -> None:
    """The value of cooperation, in one assertion: with no communication the
    agents drift apart, and the combine step is what holds them together."""
    records = execute(*setup(data=data), stop_after=STEPS - 1)

    def final(name: str) -> float:
        values = [
            row["value"]
            for record in sorted(records, key=lambda r: r.step)
            for row in record.rows
            if record.learner == name and row["metric"] == "e_agree"
        ]
        return values[-1]

    assert final("local_only") > final("diffusion_sgd_atc")


# =========================================================================== #
# 6. preconditions and failure modes
# =========================================================================== #


def test_x0_preconditions_pass_on_the_x0_config(data) -> None:
    config, environment, *_ = setup("x0_exactness", data=data)
    simulate.check_exactness_preconditions(config)  # must not raise


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"graph": {"topology": "ring"}}, "complete"),
        ({"graph": {"weights": "metropolis"}}, "a_vu = 1/N"),
        ({"env": {"label_availability": 0.5}}, "label_availability"),
        ({"run": {"dtype": "float32"}}, "float32"),
    ],
)
def test_a_drifted_x0_config_is_refused_by_name(override, expected: str) -> None:
    """Each of these produces a *small, plausible* residual rather than an
    obvious break -- which invites loosening the tolerance until it passes. The
    check names the offending field so the drift is fixed instead."""
    config = load_config("x0_exactness", overrides=override)
    with pytest.raises(SimulationError, match=expected):
        simulate.check_exactness_preconditions(config)


def test_a_wrongly_sized_theta0_is_refused(data) -> None:
    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)
    with pytest.raises(Exception, match="entries but the model has"):
        simulate.run(
            config,
            environment,
            learners,
            evalsets,
            likelihood,
            torch.zeros(theta0.numel() + 1),
            stop_after=0,
        )


def test_stop_after_beyond_the_horizon_stops_at_the_horizon(data) -> None:
    config, *rest = setup(data=data)
    records = simulate.run(config, *rest, stop_after=config.run.horizon + 50)
    assert max(record.step for record in records) == config.run.horizon - 1


def test_verification_catches_an_observation_mutated_in_place(data) -> None:
    """One draw is shared by every learner, so a learner that normalises or
    augments its input in place corrupts the data the *later* learners see --
    and the run still trains, on quietly different data per method.

    A frozen dataclass stops a field being rebound but cannot stop
    ``obs.x.add_(1)``, so the check compares fingerprints instead. Simulated
    here by a learner that mutates what it was handed.
    """
    from dekf_bench.env.environment import EnvironmentError as EnvError

    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)
    learner = learners["diffusion_sgd_atc"]
    original_adapt = learner.adapt

    def mutating(node: int, observation: Any):
        if observation.has_label:
            observation.x.add_(1.0)  # the exact hazard the guard exists for
        return original_adapt(node, observation)

    learner.adapt = mutating  # type: ignore[method-assign]
    with pytest.raises(EnvError, match="modified in place"):
        simulate.run(
            config,
            environment,
            learners,
            evalsets,
            likelihood,
            theta0,
            stop_after=0,
            verify_observations=True,
        )


def test_verification_can_be_switched_off(data) -> None:
    """It recomputes every observation, so it runs on evaluation steps only and
    stays optional for a long sweep."""
    config, environment, learners, evalsets, likelihood, theta0 = setup(data=data)
    learner = learners["diffusion_sgd_atc"]
    original_adapt = learner.adapt

    def mutating(node: int, observation: Any):
        if observation.has_label:
            observation.x.add_(1.0)
        return original_adapt(node, observation)

    learner.adapt = mutating  # type: ignore[method-assign]
    simulate.run(
        config,
        environment,
        learners,
        evalsets,
        likelihood,
        theta0,
        stop_after=0,
        verify_observations=False,
    )  # must not raise
