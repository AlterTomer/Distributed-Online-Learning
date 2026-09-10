r"""The exactness check -- the highest-value test in the suite.

On a complete graph with uniform weights and plain SGD, ATC diffusion is
*algebraically identical* to centralized SGD on the pooled batch:

.. math::
    \sum_v \tfrac1N\Bigl(\bm\theta_{t-1}-\eta\nabla L(\bm\theta_{t-1};\mathcal D^v_t)\Bigr)
    = \bm\theta_{t-1}-\eta\,\tfrac1N\sum_v \nabla L(\bm\theta_{t-1};\mathcal D^v_t)

given a common $\bm\theta_{t-1}$, which the identity then preserves inductively.

This single check catches weight-normalisation errors, initialisation
mismatches, batch-partition errors and loss-reduction (mean vs sum) errors --
every one of which otherwise produces a plausible-looking but wrong curve. It is
also the direct analogue of the Diff-EKF's complete-graph exactness proposition,
so this harness is reused in phase 5.

The tests below do not only assert that the identity holds. They also **break
each precondition in turn** and assert that it stops holding, because an
exactness test that would pass regardless of the preconditions is not testing
anything.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.data.mnist import ImageSplit
from dekf_bench.env.environment import build_environment, pool
from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.runner.simulate import SimulationError, check_exactness_preconditions
from dekf_bench.utils.config import deep_merge, load_config

TOLERANCE = 1e-12
STEPS = 30


def split(n: int = 4000, seed: int = 0) -> ImageSplit:
    generator = torch.Generator().manual_seed(seed)
    return ImageSplit(
        images=torch.rand(n, 1, 28, 28, generator=generator),
        labels=torch.arange(n, dtype=torch.int64) % 10,
        split="synthetic",
    )


def run_pair(**overrides) -> float:
    """Run centralized and ATC on one environment; return the worst divergence."""
    config = load_config(
        "x0_exactness", overrides=deep_merge({"run": {"horizon": STEPS}}, overrides)
    )
    train = split()
    environment = build_environment(config, 0, train)
    model = build_model_from_config(config)
    likelihood = Categorical(config.model.output_dim)
    learners = build_learners(config, model, likelihood)

    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    for learner in learners.values():
        learner.init(theta0)

    centralized = learners["centralized_sgd"]
    atc = learners["diffusion_sgd_atc"]
    weights = environment.graph.weights
    nodes = list(range(environment.n_nodes))

    worst = 0.0
    for step in range(environment.horizon):
        observations = environment.step(step)
        pooled_x, pooled_y = pool(observations)
        centralized.adapt_pooled(pooled_x, pooled_y)
        atc.combine({v: atc.adapt(v, observations[v]) for v in nodes}, weights)
        worst = max(
            worst,
            max(
                float((atc.flat_params(v) - centralized.flat_params(0)).abs().max()) for v in nodes
            ),
        )
    return worst


# =========================================================================== #
# 1. the identity
# =========================================================================== #


def test_atc_reproduces_centralized_sgd_exactly() -> None:
    """The gate. Agreement to 1e-12 in float64 over the whole run."""
    assert run_pair() < TOLERANCE


def test_the_agents_also_agree_with_each_other() -> None:
    """A complete graph reaches full consensus in one combine step, so
    E_agree is zero at every step -- which is what makes the identity hold
    inductively rather than only at t=0."""
    config = load_config("x0_exactness", overrides={"run": {"horizon": STEPS}})
    environment = build_environment(config, 0, split())
    model = build_model_from_config(config)
    likelihood = Categorical(10)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    for learner in learners.values():
        learner.init(theta0)

    atc = learners["diffusion_sgd_atc"]
    nodes = list(range(environment.n_nodes))
    for step in range(environment.horizon):
        observations = environment.step(step)
        atc.combine({v: atc.adapt(v, observations[v]) for v in nodes}, environment.graph.weights)
        first = atc.flat_params(0)
        assert all(float((atc.flat_params(v) - first).abs().max()) < TOLERANCE for v in nodes)


def test_the_residual_stays_at_float64_accumulation() -> None:
    """Not merely under tolerance: it should be near machine epsilon times the
    step count, which is what a correct implementation in float64 gives."""
    assert run_pair() < 1e-13


# =========================================================================== #
# 2. break each precondition, and check the identity breaks
# =========================================================================== #


def test_float32_breaks_the_1e12_tolerance() -> None:
    """Not a bug -- float32 simply cannot carry 1e-12. Asserted so that the
    dtype requirement is a measured need rather than a stated one."""
    assert run_pair(run={"dtype": "float32"}) > TOLERANCE


def test_a_ring_breaks_the_identity() -> None:
    """One combine step on a ring mixes only one hop, so the agents do not
    reach the pooled estimate."""
    residual = run_pair(graph={"topology": "ring"})
    assert residual > 1e-6


def test_metropolis_weights_on_a_complete_graph_still_work() -> None:
    """A regular graph makes every rule coincide, so this must still pass --
    otherwise the uniform requirement would be hiding a different bug."""
    assert run_pair(graph={"weights": "metropolis"}) < TOLERANCE


def test_heavy_ball_momentum_preserves_the_identity() -> None:
    r"""It does **not** break -- and the reason is worth pinning.

    The identity survives any optimizer whose update is *linear* in the
    gradients, because averaging commutes with linear maps. Heavy-ball momentum
    qualifies: $\bm m \leftarrow \beta\bm m + \bm g$ then
    $\bm\theta \leftarrow \bm\theta - \eta\bm m$, so

    .. math::
        \tfrac1N\sum_v \bm m_v = \beta\,\tfrac1N\sum_v \bm m_v^{\text{old}}
        + \tfrac1N\sum_v \bm g_v

    is exactly the centralized momentum recursion. On a complete graph every
    agent evaluates its gradient at the same point, so the average trajectory
    matches centralized regardless of whether the buffers are mixed.

    So X0 tests more than plain SGD: it tests the whole class of linear update
    rules (design note D35).
    """
    residual = run_pair(
        learners=[
            {
                "name": "centralized_sgd",
                "optimizer": "sgd_momentum",
                "momentum": 0.9,
                "mix_optimizer_state": "momentum",
            },
            {
                "name": "diffusion_sgd_atc",
                "optimizer": "sgd_momentum",
                "momentum": 0.9,
                "mix_optimizer_state": "momentum",
            },
        ]
    )
    assert residual < TOLERANCE


def test_adamw_does_break_the_identity() -> None:
    """The positive control, and the reason the previous test is not vacuous.

    Adam's second moment carries $\\bm g^2$, which is *not* linear, so averaging
    no longer commutes and the two trajectories genuinely separate. Without a
    case that fails, an exactness test that always passes would be indis-
    tinguishable from one that never checks anything.

    **The threshold is stated relative to the exactness tolerance, not as an
    absolute.** How far the two trajectories separate depends on the learning
    rate, which is a tuned quantity: at lr 0.05 the residual was 5.24, at the
    tuned lr 0.01 it is 0.76. A literal bound tracks the tuning rather than the
    property, and broke the first time the lr moved. What the control actually
    asserts is that the failure is many orders of magnitude clear of the 1e-12
    the identity is checked at, which holds at any sane lr.
    """
    residual = run_pair(
        learners=[
            {
                "name": "centralized_sgd",
                "optimizer": "adamw",
                "momentum": 0.9,
                "mix_optimizer_state": "all",
            },
            {
                "name": "diffusion_sgd_atc",
                "optimizer": "adamw",
                "momentum": 0.9,
                "mix_optimizer_state": "all",
            },
        ]
    )
    assert residual > 1e6 * TOLERANCE


def test_unequal_batch_sizes_break_the_identity() -> None:
    """The failure mode the check exists for: pi_lab < 1 gives a *small,
    plausible* residual rather than an obvious break, because the average of
    per-agent means stops equalling the pooled mean."""
    residual = run_pair(env={"label_availability": 0.5})
    assert residual > TOLERANCE


# =========================================================================== #
# 3. the runtime guard
# =========================================================================== #


def test_the_shipped_x0_config_passes_the_precondition_check() -> None:
    check_exactness_preconditions(load_config("x0_exactness"))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"run": {"dtype": "float32"}}, "run.dtype"),
        ({"graph": {"topology": "ring"}}, "graph.topology"),
        ({"graph": {"weights": "metropolis"}}, "graph.weights"),
        ({"env": {"label_availability": 0.5}}, "env.label_availability"),
    ],
)
def test_a_drifted_precondition_refuses_to_run(overrides: dict, expected: str) -> None:
    """Named, so the message says which field is wrong rather than only that
    something is."""
    with pytest.raises(SimulationError, match=expected):
        check_exactness_preconditions(load_config("x0_exactness", overrides=overrides))


def test_momentum_is_caught_by_the_guard() -> None:
    config = load_config(
        "x0_exactness",
        overrides={
            "learners": [{"name": "centralized_sgd", "optimizer": "sgd_momentum", "momentum": 0.9}]
        },
    )
    with pytest.raises(SimulationError, match="momentum"):
        check_exactness_preconditions(config)


def test_the_guard_explains_why_it_is_not_a_preference() -> None:
    """The message has to say *why*, or the natural response to a failure is to
    relax the tolerance until it passes."""
    with pytest.raises(SimulationError, match="small, plausible, non-zero"):
        check_exactness_preconditions(
            load_config("x0_exactness", overrides={"run": {"dtype": "float32"}})
        )


# =========================================================================== #
# 3. the filter's analogue, through the runner's own dispatch
# =========================================================================== #
#
# The unit tests in `test_learners.py` drive `adapt`/`combine` by hand and prove
# the *algorithm* is right. These prove the *harness runs that algorithm*: the
# config resolves to the variant we meant, `_advance` dispatches a per-agent
# filter down the per-agent path and a pooled one down `adapt_pooled`, and the
# environment hands every agent the slice the identity assumes. That gap is
# exactly what X0 closes for SGD, and it is not hypothetical -- dispatch here is
# by set membership in `POOLING`, and a diffusion filter wrongly listed there
# would pool the batch and still produce a plausible curve.


FILTER = {
    "transition": "scalar",
    "gamma": 1.0,
    "process_noise_q": 0.0,
    "lambda_forget": 1.0,
    "prior_scale": 0.01,
}


def run_filters(diffusion_name: str, steps: int = 8) -> float:
    """Drive both filters through `_advance`; return the worst disagreement.

    Q = 0 and gamma = 1 deliberately: the identity is about the *update*, and a
    process model that both filters apply identically would only mask a mismatch
    by adding the same thing to each.
    """
    from dekf_bench.runner import simulate

    config = load_config(
        "x0_exactness",
        overrides={
            "run": {"horizon": steps, "dtype": "float64"},
            "learners": [
                {"name": "centralized_ekf_walk", **FILTER},
                {"name": diffusion_name, **FILTER},
            ],
        },
    )
    environment = build_environment(config, 0, split())
    model = build_model_from_config(config)
    likelihood = Categorical(config.model.output_dim)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    for learner in learners.values():
        learner.init(theta0)

    centralized = learners["centralized_ekf_walk"]
    diffusion = learners[diffusion_name]
    nodes = list(range(environment.n_nodes))

    worst = 0.0
    for step in range(environment.horizon):
        observations = environment.step(step)
        pooled_x, pooled_y = pool(observations)
        for name, learner in learners.items():
            simulate._advance(
                learner, name, observations, nodes, environment.graph.weights,
                pooled_x, pooled_y,
            )
        reference = centralized.flat_params(0)
        worst = max(
            worst,
            max(float((diffusion.flat_params(v) - reference).abs().max()) for v in nodes),
        )
    return worst


def test_the_diffusion_filter_reproduces_the_centralized_one_through_the_runner() -> None:
    """prop:complete_graph, end to end. The filter's X0.

    Held at the same 1e-12 as SGD's identity. The residual is measured at
    **4.4e-16** -- machine epsilon, and comparable to X0's own 1.7e-15 -- which is
    worth stating because it was not the expectation: the filter's path to each
    parameter runs through a Cholesky solve and a p x p subtraction, so a larger
    accumulation than a weighted mean of gradients would have been reasonable. It
    does not appear, because both filters run the *same* `woodbury_update` on the
    same concatenated information, in the same order.
    """
    assert run_filters("diffusion_ekf_onehop") < TOLERANCE


def test_a_local_adapt_breaks_the_identity_through_the_runner() -> None:
    """The negative half, so the positive one cannot pass vacuously.

    A local adapt has each agent update on its own data and the combine average
    covariances, which is not the same as summing information -- so the identity
    must fail even on a complete graph. If this ever passed, the test above would
    have stopped being evidence that one-hop is what makes the identity hold.
    """
    assert run_filters("diffusion_ekf_full") > 1e-6


def test_the_mean_only_variant_also_breaks_it() -> None:
    """And for a second, independent reason: it does not mix covariances at all."""
    assert run_filters("diffusion_ekf") > 1e-6


def test_the_filter_variants_are_not_dispatched_as_pooled() -> None:
    """The dispatch bug this section exists to catch, asserted directly.

    A per-agent learner listed in POOLING would consume the union of every
    agent's batch, agree with the centralized filter for the wrong reason, and
    produce a curve nobody could tell from a correct one.
    """
    from dekf_bench.learners.registry import DIFFUSION_EKF_VARIANTS, POOLING

    assert not (set(DIFFUSION_EKF_VARIANTS) & POOLING)
