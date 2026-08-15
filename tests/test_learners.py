r"""The four methods: state, the combine step, and what each transmits.

The exactness identity is tested separately in ``test_exactness.py``. What is
checked here is everything that identity *assumes*: that combine is a genuine
averaging operator, that an idle agent behaves, that the methods differ only
where they are supposed to.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.env.environment import Observation
from dekf_bench.learners.base import Intermediate, LearnerError, LearnerState, loss_gradient
from dekf_bench.learners.optim_state import (
    MIX_POLICIES,
    OPTIMIZER_STATE,
    Optimizer,
    OptimStateError,
    mixed_entries,
)
from dekf_bench.learners.registry import DIFFUSING, build_learner, build_learners
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.models.mlp import MLP
from dekf_bench.utils.config import load_config

N_NODES = 6
DTYPE = torch.float64


@pytest.fixture(scope="module")
def model() -> MLP:
    return MLP(input_size=8, hidden=(5,), output_dim=4, dtype=DTYPE)


@pytest.fixture(scope="module")
def likelihood() -> Categorical:
    return Categorical(4)


def observation(model: MLP, node: int = 0, n: int = 2, labelled: bool = True) -> Observation:
    generator = torch.Generator().manual_seed(node)
    if not labelled:
        return Observation(
            x=torch.empty(0, 1, model.input_size, model.input_size, dtype=DTYPE),
            y=None,
            has_label=False,
            n_samples=0,
            node=node,
            step=0,
            rotation_degrees=0.0,
        )
    return Observation(
        x=torch.rand(n, 1, model.input_size, model.input_size, generator=generator, dtype=DTYPE),
        y=torch.arange(n, dtype=torch.int64) % 4,
        has_label=True,
        n_samples=n,
        node=node,
        step=0,
        rotation_degrees=0.0,
    )


def make(name: str, model: MLP, likelihood: Categorical, **optimizer):
    from dekf_bench.learners.registry import BUILDERS

    settings = {"kind": "sgd", "lr": 0.05, "momentum": 0.0}
    settings.update(optimizer)
    mix = settings.pop("mix_policy", "none")
    freeze_after = settings.pop("freeze_after", None)
    learner = BUILDERS[name](
        name=name,
        model=model,
        likelihood=likelihood,
        optimizer=Optimizer(**settings),
        n_nodes=N_NODES,
        mix_policy=mix,
        freeze_after=freeze_after,
    )
    learner.init(model.flatten(model.init_params(torch.Generator().manual_seed(0))))
    return learner


def uniform_weights(n: int = N_NODES) -> torch.Tensor:
    return torch.full((n, n), 1.0 / n, dtype=DTYPE)


DIFFUSION_NAMES = ["diffusion_sgd_atc", "diffusion_sgd_atc_plain", "diffusion_sgd_cta"]
ALL_NAMES = [*DIFFUSION_NAMES, "local_only"]


# =========================================================================== #
# the non-adapting baseline
# =========================================================================== #


def at_step(model: MLP, step: int) -> Observation:
    import dataclasses

    return dataclasses.replace(observation(model), step=step)


def run_steps(learner, model: MLP, steps: range) -> torch.Tensor:
    """Drive the learner over exactly ``steps`` and return agent 0's parameters.

    A range rather than a step count: replaying from 0 would re-run the warmup
    every time and a frozen learner would appear to keep moving.
    """
    for step in steps:
        nodes = range(N_NODES)
        learner.combine(
            {v: learner.adapt(v, at_step(model, step)) for v in nodes}, uniform_weights()
        )
    return learner.flat_params(0).clone()


def test_a_frozen_learner_learns_first_and_then_stops(model: MLP, likelihood: Categorical) -> None:
    """Both halves matter. Freezing from the start would leave it at its random
    initialisation, which measures nothing about whether adaptation pays; never
    freezing would make it the very learner it is the baseline for."""
    learner = make("frozen_atc", model, likelihood, freeze_after=5)
    start = learner.flat_params(0).clone()
    at_freeze = run_steps(learner, model, range(0, 5))
    later = run_steps(learner, model, range(5, 13))

    assert float((at_freeze - start).abs().max()) > 0, "it must learn during warmup"
    assert float((later - at_freeze).abs().max()) == 0.0, "it must not move once frozen"


def test_an_unfrozen_learner_is_unaffected(model: MLP, likelihood: Categorical) -> None:
    """The negative control: `freeze_after=None` must change nothing at all, so
    the same schedule of steps keeps moving it."""
    learner = make("diffusion_sgd_atc", model, likelihood)
    early = run_steps(learner, model, range(0, 5))
    late = run_steps(learner, model, range(5, 13))
    assert float((late - early).abs().max()) > 0


def test_a_frozen_learner_stops_paying_bandwidth(model: MLP, likelihood: Categorical) -> None:
    """It stops transmitting as well as stepping. Averaging identical estimates
    would be a numerical no-op but would still be counted by the ledger, and a
    baseline paying for silence would distort every plot against cost."""
    learner = make("frozen_atc", model, likelihood, freeze_after=5)
    run_steps(learner, model, range(0, 4))
    during_warmup = learner.comm_scalars_per_step(10)
    run_steps(learner, model, range(4, 10))
    assert during_warmup > 0
    assert learner.comm_scalars_per_step(10) == 0


# =========================================================================== #
# 1. state
# =========================================================================== #


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_agent_starts_from_the_same_parameters(
    name: str, model: MLP, likelihood: Categorical
) -> None:
    """Required for the filter -- independently initialised agents do not
    represent one Bayesian model -- and it removes a confound from the SGD
    comparison."""
    learner = make(name, model, likelihood)
    first = learner.flat_params(0)
    assert all(torch.equal(learner.flat_params(v), first) for v in range(N_NODES))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_agents_hold_independent_copies(name: str, model: MLP, likelihood: Categorical) -> None:
    """Sharing one tensor would make the first in-place update change every
    agent at once, and the run would show perfect consensus for a reason
    unconnected to the combine step."""
    learner = make(name, model, likelihood)
    learner.state(0).theta.add_(1.0)
    assert not torch.equal(learner.flat_params(0), learner.flat_params(1))


def test_stepping_before_init_is_an_error(model: MLP, likelihood: Categorical) -> None:
    from dekf_bench.learners.sgd import DiffusionSGDATC

    learner = DiffusionSGDATC(
        name="atc", model=model, likelihood=likelihood, optimizer=Optimizer(), n_nodes=N_NODES
    )
    with pytest.raises(LearnerError, match="call init"):
        learner.adapt(0, observation(model))


def test_a_wrongly_sized_theta0_is_rejected(model: MLP, likelihood: Categorical) -> None:
    from dekf_bench.learners.sgd import DiffusionSGDATC

    learner = DiffusionSGDATC(
        name="atc", model=model, likelihood=likelihood, optimizer=Optimizer(), n_nodes=N_NODES
    )
    with pytest.raises(LearnerError, match="entries but the model has"):
        learner.init(torch.zeros(model.num_params + 1, dtype=DTYPE))


def test_state_is_dict_like(model: MLP, likelihood: Categorical) -> None:
    """So phase 5 can add a covariance without touching simulate.py."""
    learner = make("diffusion_sgd_atc", model, likelihood, kind="sgd_momentum", momentum=0.9)
    state = learner.state(0)
    assert "theta" in state and "momentum" in state
    assert state["theta"].shape == (model.num_params,)
    assert set(state.keys()) == {"theta", "momentum"}


def test_an_unknown_state_entry_names_what_exists(model: MLP, likelihood: Categorical) -> None:
    learner = make("diffusion_sgd_atc", model, likelihood)
    with pytest.raises(KeyError, match="no state entry"):
        learner.state(0)["covariance"]


def test_learner_state_clones_deeply() -> None:
    state = LearnerState(theta=torch.ones(3), extras={"momentum": torch.ones(3)})
    copy = state.clone()
    copy.theta.add_(1.0)
    copy.extras["momentum"].add_(1.0)
    assert float(state.theta[0]) == 1.0
    assert float(state.extras["momentum"][0]) == 1.0


# =========================================================================== #
# 2. the combine step
# =========================================================================== #


def test_combine_is_a_weighted_average(model: MLP, likelihood: Categorical) -> None:
    learner = make("diffusion_sgd_atc", model, likelihood)
    intermediates = {
        v: Intermediate(node=v, psi=torch.full((model.num_params,), float(v), dtype=DTYPE))
        for v in range(N_NODES)
    }
    learner.combine(intermediates, uniform_weights())
    expected = sum(range(N_NODES)) / N_NODES
    for v in range(N_NODES):
        assert torch.allclose(
            learner.flat_params(v), torch.full_like(learner.flat_params(v), expected)
        )


def test_combine_reads_every_message_before_writing_any(
    model: MLP, likelihood: Categorical
) -> None:
    """A sequential in-place loop would let agent 1 combine agent 0's *updated*
    parameters, making the result depend on node ordering -- and breaking X0
    while still producing a plausible curve."""
    learner = make("diffusion_sgd_atc", model, likelihood)
    intermediates = {
        v: Intermediate(node=v, psi=torch.full((model.num_params,), float(v), dtype=DTYPE))
        for v in range(N_NODES)
    }
    learner.combine(dict(reversed(intermediates.items())), uniform_weights())
    forward = learner.flat_params(0).clone()

    learner = make("diffusion_sgd_atc", model, likelihood)
    learner.combine(intermediates, uniform_weights())
    assert torch.equal(learner.flat_params(0), forward)


def test_combine_preserves_the_convex_hull(model: MLP, likelihood: Categorical) -> None:
    """Lemma 1 of the research note: nothing an agent receives can push its
    estimate outside the range its neighbours collectively propose."""
    from dekf_bench.env.graph import build_graph

    graph = build_graph("ring", N_NODES, "metropolis")
    learner = make("diffusion_sgd_atc", model, likelihood)
    generator = torch.Generator().manual_seed(3)
    psis = {
        v: torch.randn(model.num_params, generator=generator, dtype=DTYPE) for v in range(N_NODES)
    }
    learner.combine({v: Intermediate(node=v, psi=psis[v]) for v in range(N_NODES)}, graph.weights)

    for v in range(N_NODES):
        neighbourhood = torch.stack([psis[int(u)] for u in graph.closed_neighbourhood(v)])
        combined = learner.flat_params(v)
        assert bool((combined >= neighbourhood.min(dim=0).values - 1e-12).all())
        assert bool((combined <= neighbourhood.max(dim=0).values + 1e-12).all())


def test_consensus_is_a_fixed_point(model: MLP, likelihood: Categorical) -> None:
    """If the agents already agree, combining must not move anything -- which is
    what makes the X0 identity hold inductively.

    **To machine precision, not exactly.** Not because the weights are wrong --
    $\\sum_u a_{vu}$ is exactly 1.0 in float64 at every $N$ tested, $N{=}10$
    included. The residual comes from the *accumulation order* inside the
    matmul: the partial sums are not representable even when the summands and
    the total are, so combining $N$ identical vectors returns them perturbed by
    about one ulp.

    This is the floor on X0's residual. At $N{=}10$ the identity cannot do
    better than a few ulps however correct the algebra, which is why the target
    is 1e-12 rather than zero.
    """
    learner = make("diffusion_sgd_atc", model, likelihood)
    shared = learner.flat_params(0).clone()
    learner.combine(
        {v: Intermediate(node=v, psi=shared.clone()) for v in range(N_NODES)}, uniform_weights()
    )
    for v in range(N_NODES):
        assert torch.allclose(learner.flat_params(v), shared, rtol=0.0, atol=1e-14)


def test_a_tiny_network_reaches_consensus_exactly(model: MLP, likelihood: Categorical) -> None:
    """The control for the test above.

    At $N{=}2$ the accumulation is a single addition of two identical halves,
    with no unrepresentable partial sum, and the fixed point is exact. That
    pins the residual at larger $N$ on floating-point accumulation rather than
    on a defect in combine -- if this one drifted too, the averaging itself
    would be wrong.
    """
    from dekf_bench.learners.registry import BUILDERS

    learner = BUILDERS["diffusion_sgd_atc"](
        name="diffusion_sgd_atc",
        model=model,
        likelihood=likelihood,
        optimizer=Optimizer(kind="sgd"),
        n_nodes=2,
        mix_policy="none",
    )
    learner.init(model.flatten(model.init_params(torch.Generator().manual_seed(0))))
    shared = learner.flat_params(0).clone()
    learner.combine(
        {v: Intermediate(node=v, psi=shared.clone()) for v in range(2)}, uniform_weights(2)
    )
    assert all(torch.equal(learner.flat_params(v), shared) for v in range(2))


def test_combine_rejects_a_missing_agent(model: MLP, likelihood: Categorical) -> None:
    learner = make("diffusion_sgd_atc", model, likelihood)
    partial = {v: Intermediate(node=v, psi=learner.flat_params(v)) for v in range(N_NODES - 1)}
    with pytest.raises(LearnerError, match="but holds state for"):
        learner.combine(partial, uniform_weights())


def test_local_only_never_moves_toward_its_neighbours(model: MLP, likelihood: Categorical) -> None:
    learner = make("local_only", model, likelihood)
    before = [learner.flat_params(v).clone() for v in range(N_NODES)]
    learner.combine(
        {
            v: Intermediate(node=v, psi=torch.zeros(model.num_params, dtype=DTYPE))
            for v in range(N_NODES)
        },
        uniform_weights(),
    )
    assert all(torch.equal(learner.flat_params(v), before[v]) for v in range(N_NODES))


# =========================================================================== #
# 3. adapt
# =========================================================================== #


@pytest.mark.parametrize("name", DIFFUSION_NAMES)
def test_adapt_changes_the_parameters(name: str, model: MLP, likelihood: Categorical) -> None:
    learner = make(name, model, likelihood)
    before = learner.flat_params(0).clone()
    intermediate = learner.adapt(0, observation(model))
    # CTA emits theta unchanged and defers the step to combine, so check the
    # message differs from theta only for ATC.
    if name.startswith("diffusion_sgd_atc"):
        assert not torch.equal(intermediate.psi, before)
    else:
        learner.combine(
            {
                0: intermediate,
                **{v: learner.adapt(v, observation(model, v)) for v in range(1, N_NODES)},
            },
            uniform_weights(),
        )
        assert not torch.equal(learner.flat_params(0), before)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_an_idle_agent_takes_no_step(name: str, model: MLP, likelihood: Categorical) -> None:
    """With no label there is no gradient. Its estimate passes through unchanged
    and it still benefits from the combine step, which is one of diffusion's
    more attractive properties."""
    learner = make(name, model, likelihood)
    before = learner.flat_params(0).clone()
    intermediate = learner.adapt(0, observation(model, labelled=False))
    assert torch.equal(intermediate.psi, before)


def test_centralized_refuses_a_per_agent_adapt(model: MLP, likelihood: Categorical) -> None:
    """It consumes the pooled batch; a per-agent adapt would be a different
    method wearing the same name."""
    learner = make("centralized_sgd", model, likelihood)
    with pytest.raises(LearnerError, match="pooled batch"):
        learner.adapt(0, observation(model))


def test_centralized_keeps_every_agent_identical(model: MLP, likelihood: Categorical) -> None:
    learner = make("centralized_sgd", model, likelihood)
    generator = torch.Generator().manual_seed(1)
    x = torch.rand(12, 1, 8, 8, generator=generator, dtype=DTYPE)
    learner.adapt_pooled(x, torch.arange(12, dtype=torch.int64) % 4)
    first = learner.flat_params(0)
    assert all(torch.equal(learner.flat_params(v), first) for v in range(N_NODES))


def test_centralized_ignores_an_empty_pooled_batch(model: MLP, likelihood: Categorical) -> None:
    learner = make("centralized_sgd", model, likelihood)
    before = learner.flat_params(0).clone()
    learner.adapt_pooled(torch.empty(0, 1, 8, 8, dtype=DTYPE), torch.empty(0, dtype=torch.int64))
    assert torch.equal(learner.flat_params(0), before)


# =========================================================================== #
# 4. the shared gradient
# =========================================================================== #


def test_the_gradient_matches_autograd(model: MLP, likelihood: Categorical) -> None:
    """One implementation, used by every learner: if each computed its own, an
    X0 failure could mean a diffusion bug or two gradients disagreeing about
    reduction."""
    theta = model.flatten(model.init_params(torch.Generator().manual_seed(0)))
    obs = observation(model, n=4)

    params = {k: v.clone().requires_grad_(True) for k, v in model.unflatten(theta).items()}
    likelihood.nll(model.forward(params, obs.x), obs.y, reduction="mean").backward()
    expected = model.flatten({k: v.grad for k, v in params.items()})

    assert torch.allclose(
        loss_gradient(model, theta, obs.x, obs.y, likelihood), expected, atol=1e-12
    )


def test_the_gradient_uses_mean_reduction(model: MLP, likelihood: Categorical) -> None:
    """A precondition of X0: sum here and mean there gives a residual of order N
    that looks like a scaling bug rather than a reduction bug."""
    theta = model.flatten(model.init_params(torch.Generator().manual_seed(0)))
    small, large = observation(model, n=2), observation(model, n=8)

    # Doubling the batch of *identical* samples must not double the gradient.
    doubled = Observation(
        x=torch.cat([small.x, small.x]),
        y=torch.cat([small.y, small.y]),
        has_label=True,
        n_samples=4,
        node=0,
        step=0,
        rotation_degrees=0.0,
    )
    assert torch.allclose(
        loss_gradient(model, theta, small.x, small.y, likelihood),
        loss_gradient(model, theta, doubled.x, doubled.y, likelihood),
        atol=1e-12,
    )
    assert large.n_samples == 8  # the other fixture is used elsewhere


def test_an_empty_batch_gives_a_zero_gradient(model: MLP, likelihood: Categorical) -> None:
    theta = model.flatten(model.init_params(torch.Generator().manual_seed(0)))
    empty = observation(model, labelled=False)
    assert torch.all(loss_gradient(model, theta, empty.x, empty.y, likelihood) == 0)


# =========================================================================== #
# 5. optimizer state
# =========================================================================== #


def test_plain_sgd_carries_no_state() -> None:
    optimizer = Optimizer(kind="sgd")
    assert optimizer.state_names == ()
    assert not optimizer.is_stateful


def test_momentum_on_plain_sgd_is_rejected() -> None:
    """A value that silently does nothing is worse than an error, and X0 depends
    on plain SGD being plain."""
    with pytest.raises(OptimStateError, match="carries no momentum buffer"):
        Optimizer(kind="sgd", momentum=0.9)


def test_momentum_accumulates() -> None:
    optimizer = Optimizer(kind="sgd_momentum", lr=0.1, momentum=0.9)
    state = optimizer.init_state(3, DTYPE)
    theta = torch.zeros(3, dtype=DTYPE)
    gradient = torch.ones(3, dtype=DTYPE)

    theta = optimizer.step(theta, gradient, state)
    assert torch.allclose(state["momentum"], torch.ones(3, dtype=DTYPE))
    optimizer.step(theta, gradient, state)
    assert torch.allclose(state["momentum"], torch.full((3,), 1.9, dtype=DTYPE))


def test_the_step_returns_rather_than_mutating_theta() -> None:
    """ATC needs psi as a separate object: the agent's own parameters must
    survive until the combine step replaces them."""
    optimizer = Optimizer(kind="sgd", lr=0.1)
    theta = torch.zeros(3, dtype=DTYPE)
    psi = optimizer.step(theta, torch.ones(3, dtype=DTYPE), {})
    assert torch.all(theta == 0)
    assert torch.allclose(psi, torch.full((3,), -0.1, dtype=DTYPE))


@pytest.mark.parametrize("policy", sorted(MIX_POLICIES))
def test_mixed_entries_intersects_what_exists(policy: str) -> None:
    """`all` on plain SGD is a no-op rather than an error: there is nothing to
    mix."""
    assert mixed_entries(Optimizer(kind="sgd"), policy) == ()


def test_momentum_is_mixed_only_when_asked() -> None:
    optimizer = Optimizer(kind="sgd_momentum", momentum=0.9)
    assert mixed_entries(optimizer, "none") == ()
    assert mixed_entries(optimizer, "momentum") == ("momentum",)


def test_mixing_averages_the_momentum_too(model: MLP, likelihood: Categorical) -> None:
    """One operator on the whole learner state, so every property of A covers
    all of it rather than theta alone."""
    learner = make(
        "diffusion_sgd_atc",
        model,
        likelihood,
        kind="sgd_momentum",
        momentum=0.9,
        mix_policy="momentum",
    )
    for v in range(N_NODES):
        learner.state(v).extras["momentum"] = torch.full((model.num_params,), float(v), dtype=DTYPE)
    intermediates = {v: learner.adapt(v, observation(model, v)) for v in range(N_NODES)}
    learner.combine(intermediates, uniform_weights())

    mixed = learner.state(0).extras["momentum"]
    assert all(torch.equal(learner.state(v).extras["momentum"], mixed) for v in range(N_NODES))


def test_an_unmixed_moment_stays_local(model: MLP, likelihood: Categorical) -> None:
    learner = make(
        "diffusion_sgd_atc",
        model,
        likelihood,
        kind="sgd_momentum",
        momentum=0.9,
        mix_policy="none",
    )
    intermediates = {v: learner.adapt(v, observation(model, v)) for v in range(N_NODES)}
    learner.combine(intermediates, uniform_weights())
    moments = [learner.state(v).extras["momentum"] for v in range(N_NODES)]
    assert not all(torch.equal(m, moments[0]) for m in moments[1:])


def test_every_optimizer_declares_its_state() -> None:
    for kind, names in OPTIMIZER_STATE.items():
        optimizer = Optimizer(kind=kind, momentum=0.9 if kind != "sgd" else 0.0)
        assert set(optimizer.init_state(3, DTYPE)) == set(names)


# =========================================================================== #
# 6. what each learner transmits
# =========================================================================== #


def test_local_only_transmits_nothing(model: MLP, likelihood: Categorical) -> None:
    assert make("local_only", model, likelihood).comm_scalars_per_step(10) == 0


def test_centralized_transmits_nothing_on_the_diffusion_axis(
    model: MLP, likelihood: Categorical
) -> None:
    assert make("centralized_sgd", model, likelihood).comm_scalars_per_step(10) == 0


def test_plain_atc_sends_one_vector_per_link(model: MLP, likelihood: Categorical) -> None:
    learner = make("diffusion_sgd_atc_plain", model, likelihood)
    assert learner.comm_scalars_per_step(10) == model.num_params * 2 * 10


def test_mixing_momentum_doubles_the_payload(model: MLP, likelihood: Categorical) -> None:
    """A neighbour cannot mix a buffer it was never sent (design note D29)."""
    plain = make("diffusion_sgd_atc", model, likelihood)
    with_momentum = make(
        "diffusion_sgd_atc",
        model,
        likelihood,
        kind="sgd_momentum",
        momentum=0.9,
        mix_policy="momentum",
    )
    assert with_momentum.comm_scalars_per_step(10) == 2 * plain.comm_scalars_per_step(10)


def test_the_intermediate_reports_its_own_payload(model: MLP, likelihood: Categorical) -> None:
    learner = make(
        "diffusion_sgd_atc",
        model,
        likelihood,
        kind="sgd_momentum",
        momentum=0.9,
        mix_policy="momentum",
    )
    assert learner.adapt(0, observation(model)).payload_vectors == 2


def test_cta_costs_the_same_as_atc(model: MLP, likelihood: Categorical) -> None:
    """They differ in ordering, not in cost -- which is what makes X1b a clean
    comparison."""
    atc = make("diffusion_sgd_atc", model, likelihood)
    cta = make("diffusion_sgd_cta", model, likelihood)
    assert atc.comm_scalars_per_step(10) == cta.comm_scalars_per_step(10)


# =========================================================================== #
# 7. N = 1
# =========================================================================== #


def test_every_method_coincides_at_one_agent(model: MLP, likelihood: Categorical) -> None:
    """WORKPLAN section 7.2. With one agent there are no neighbours, so combine
    is the identity and every method reduces to plain online SGD."""
    from dekf_bench.learners.registry import BUILDERS

    obs = observation(model, n=3)
    weights = torch.ones(1, 1, dtype=DTYPE)
    finals = {}

    for name in ALL_NAMES:
        learner = BUILDERS[name](
            name=name,
            model=model,
            likelihood=likelihood,
            optimizer=Optimizer(kind="sgd", lr=0.05),
            n_nodes=1,
            mix_policy="none",
        )
        learner.init(model.flatten(model.init_params(torch.Generator().manual_seed(0))))
        learner.combine({0: learner.adapt(0, obs)}, weights)
        finals[name] = learner.flat_params(0)

    reference = finals["diffusion_sgd_atc"]
    for name, theta in finals.items():
        assert torch.allclose(theta, reference, atol=1e-12), f"{name} differs at N=1"


# =========================================================================== #
# 8. the registry
# =========================================================================== #


def test_every_configured_learner_builds() -> None:
    config = load_config("x1_stationary")
    model = MLP(input_size=14, hidden=(14,), output_dim=10)
    learners = build_learners(config, model, Categorical(10))
    assert set(learners) == {entry.name for entry in config.learners}


def test_the_diffusing_set_matches_the_costs(model: MLP, likelihood: Categorical) -> None:
    for name in ALL_NAMES:
        learner = make(name, model, likelihood)
        transmits = learner.comm_scalars_per_step(10) > 0
        assert transmits == (name in DIFFUSING), name


def test_an_unknown_learner_lists_the_available_ones(model: MLP, likelihood: Categorical) -> None:
    config = load_config("x1_stationary").learners[0]
    object.__setattr__(config, "name", "diffusion_sgd_xyz")
    with pytest.raises(LearnerError, match="unknown learner"):
        build_learner(config, model, likelihood, N_NODES)


def test_the_filter_stub_raises_pointing_at_phase_five() -> None:
    """It exists so the adapt/combine split is exercised by the type checker
    before the filter is written."""
    from dekf_bench.learners.registry import DiffusionEKF

    stub = DiffusionEKF()
    assert stub.name == "diffusion_ekf"
    with pytest.raises(NotImplementedError, match="phase 5"):
        stub.init(torch.zeros(3))
    with pytest.raises(NotImplementedError, match="13.5"):
        stub.adapt(0, None)
