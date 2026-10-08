"""The coupled-network GNN (D3c): its two forward passes, causality, and both Jacobians."""

from __future__ import annotations

import pytest
import torch

from dekf_bench.models.gnn import CoupledGNN, off_diagonal

N, B = 4, 3
DTYPE = torch.float64


def ring_weights(n: int = N) -> torch.Tensor:
    w = torch.zeros(n, n, dtype=DTYPE)
    for v in range(n):
        w[v, v] = w[v, (v + 1) % n] = w[v, (v - 1) % n] = 1.0 / 3.0
    return w


@pytest.fixture(scope="module")
def small() -> CoupledGNN:
    return CoupledGNN(context=7, d_model=6, n_heads=2, d_ff=8, layers=2, dtype=DTYPE)


def params_of(model: CoupledGNN, seed: int = 0):
    return model.init_params(torch.Generator().manual_seed(seed))


def windows(model: CoupledGNN, seed: int = 1) -> torch.Tensor:
    return torch.randn(B, N, model.context, generator=torch.Generator().manual_seed(seed),
                       dtype=DTYPE)


def test_the_default_gnn_is_about_twenty_thousand_parameters():
    assert CoupledGNN(dtype=DTYPE).num_params == 20_001
    assert CoupledGNN(layers=1, dtype=DTYPE).num_params < 20_001


def test_the_distributed_pass_is_the_centralised_one_under_a_shared_theta(small):
    """Every node with the same parameters: the per-node pass, built from what each
    node receives, must equal the one-snapshot network pass."""
    params = params_of(small)
    distributed, packed = small.network_forward(params, windows(small), ring_weights())
    shared = small.network_predict(params, windows(small), ring_weights())
    assert torch.allclose(distributed, shared, atol=1e-12)
    assert packed.shape == (B, N, small.context, small.packed_width)


def test_different_parameters_give_different_messages(small):
    a, b = params_of(small, 0), params_of(small, 1)
    one, _ = small.network_forward([a] * N, windows(small), ring_weights())
    mixed, _ = small.network_forward([a, b, a, a], windows(small), ring_weights())
    # Node 1's neighbours (0 and 2) receive node 1's different embeddings.
    assert not torch.allclose(one[:, 0], mixed[:, 0])
    assert not torch.allclose(one[:, 2], mixed[:, 2])


@pytest.mark.parametrize("node", [0, 1])
def test_position_t_never_sees_the_future_of_any_node(small, node):
    """Perturb positions > t of one node's window: every node's predictions at positions
    <= t are unchanged -- causality across nodes as well as across time."""
    params, x = params_of(small), windows(small)
    t = 3
    before = small.network_predict(params, x, ring_weights())
    later = x.clone()
    later[:, node, t + 1:] += 5.0
    after = small.network_predict(params, later, ring_weights())
    assert torch.allclose(before[..., : t + 1], after[..., : t + 1], atol=1e-12)
    assert not torch.allclose(before[..., t + 1:], after[..., t + 1:])


def test_an_isolated_node_reads_zero_aggregates(small):
    params, x = params_of(small), windows(small)
    alone = torch.eye(N, dtype=DTYPE)
    predictions, packed = small.network_forward(params, x, alone)
    assert torch.equal(packed[..., 1:], torch.zeros_like(packed[..., 1:]))
    assert torch.allclose(predictions[:, 2], small.forward(params, small.pack(x[:, 2], [])),
                          atol=1e-12)


def test_the_exact_jacobian_matches_finite_differences(small):
    params, x = params_of(small), windows(small)[:1]
    flat = small.flatten(params)
    jac = small.network_jacobian(params, x, ring_weights())[0]
    direction = torch.randn(flat.shape, generator=torch.Generator().manual_seed(9), dtype=DTYPE)
    eps = 1e-6
    plus = small.network_predict(small.unflatten(flat + eps * direction), x, ring_weights())
    minus = small.network_predict(small.unflatten(flat - eps * direction), x, ring_weights())
    numeric = ((plus - minus) / (2 * eps)).reshape(-1)
    assert torch.allclose(jac @ direction, numeric, atol=1e-7)


def test_truncation_drops_only_the_paths_through_neighbours(small):
    """On an edgeless graph nothing is received, so truncated and exact agree; with
    edges they differ, by the paths through the neighbours' computation."""
    params, x = params_of(small), windows(small)[:1]
    for weights, equal in ((torch.eye(N, dtype=DTYPE), True), (ring_weights(), False)):
        exact = small.network_jacobian(params, x, weights)[0].reshape(N, small.context, -1)
        _pred, packed = small.network_forward(params, x, weights)
        truncated = torch.stack([small.per_sample_jacobian(params, packed[:, v])[0]
                                 for v in range(N)])
        assert torch.allclose(exact, truncated, atol=1e-10) == equal


def test_embedding_zero_is_the_encoders_features(small):
    params, x = params_of(small), windows(small)[:, 0]
    sent = small.embedding(params, small.pack(x, []), 0)
    assert sent.shape == (B, small.context, small.d_model)
    assert torch.allclose(sent, features_of(small, params, x), atol=1e-12)


def features_of(model: CoupledGNN, params, x: torch.Tensor) -> torch.Tensor:
    """The encoder's features at ``params`` through a functional call of its method."""
    from torch import nn
    from torch.func import functional_call

    class Features(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, window):
            return self.block.features(window)

    wrapper = Features(model._module.encoder)
    sub = {f"block.{k[len('encoder.'):]}": v for k, v in params.items() if k.startswith("encoder.")}
    return functional_call(wrapper, sub, (x,))


def test_off_diagonal_drops_the_self_weight():
    w = ring_weights()
    off = off_diagonal(w)
    assert torch.equal(off.diagonal(), torch.zeros(N, dtype=DTYPE))
    assert torch.equal(off + torch.diag(w.diagonal()), w)


# ---- the GNN in a run (D3c step 3) -------------------------------------------------

TINY_GNN = {"name": "coupled_gnn", "likelihood": "gaussian", "context": 31, "output_dim": 31,
            "d_model": 6, "n_heads": 2, "d_ff": 8, "graph_layers": 2,
            "observation_variance": 0.03, "observation_variances": []}
FILTER = {"transition": "scalar", "gamma": 1.0, "process_noise_q": 0.0,
          "lambda_forget": 1.0, "prior_scale": 0.01, "forgetting": "process_noise"}


def coupled_config(topology: str = "complete", learners=None, horizon: int = 4):
    from dekf_bench.utils.config import load_config

    return load_config("m_coupled", overrides={
        "run": {"horizon": horizon, "dtype": "float64", "device": "cpu", "seeds": [0],
                "eval_every": 2},
        "graph": {"topology": topology, **({"params": {"p": 0.5}} if topology == "erdos_renyi"
                                            else {})},
        "env": {"series": {"burn_in": 50.0, "eval_blocks": 4, "eval_trajectories": 2}},
        "model": TINY_GNN,
        "learners": learners or ["centralized_sgd"],
    })


def built(config):
    from dekf_bench.env.series import build_series_environment
    from dekf_bench.learners.registry import build_learners
    from dekf_bench.likelihoods.registry import build_likelihood
    from dekf_bench.models.registry import build_model_from_config

    environment = build_series_environment(config, 0)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    for learner in learners.values():
        learner.init(theta0)
    return environment, model, likelihood, learners


def test_packing_never_touches_the_shared_observations():
    from dekf_bench.runner.network import pack_observations

    environment, _model, _lik, learners = built(coupled_config())
    observations = environment.step(0)
    before = {v: o.x.clone() for v, o in observations.items()}
    seen = pack_observations(learners["centralized_sgd"], observations, environment.graph.weights)
    assert all(torch.equal(observations[v].x, before[v]) for v in observations)
    assert seen[0].x.shape[-1] == 1 + 2 * TINY_GNN["d_model"]


@pytest.mark.parametrize("onehop", ["diffusion_ekf_onehop_mean_receiver",
                                    "diffusion_ekf_onehop_receiver"])
def test_the_gnn_one_hop_reproduces_the_truncated_centralised_fixture(onehop):
    """The GNN one-hop's identity (decided 2026-10-07): on K_n every agent packs the same
    inputs and pools every packed batch at a common mean, which is the centralised
    filter linearising each node *truncated* -- the test-only fixture. The results'
    centralised filter is exact and must NOT match (the positive control)."""
    from dekf_bench.env.series import pool
    from dekf_bench.runner.network import NetworkModel, pack_observations

    learners = [{"name": "centralized_ekf_walk", **FILTER},
                {"name": "centralized_ekf_gamma", **FILTER, "gamma": 1.0},
                {"name": onehop, **FILTER}]
    environment, model, _lik, built_learners = built(coupled_config(learners=learners))
    fixture = built_learners["centralized_ekf_walk"]          # truncated: packed, pooled
    exact = built_learners["centralized_ekf_gamma"]           # exact: the network model
    exact.network_model = NetworkModel(model, environment.graph.weights, environment.n_nodes)
    diffusion = built_learners[onehop]
    weights = environment.graph.weights
    nodes = list(range(environment.n_nodes))
    for step in range(3):
        observations = environment.step(step)
        packed = pack_observations(fixture, observations, weights)
        fixture.adapt_pooled(*pool(packed))
        exact.adapt_pooled(*pool(observations))
        seen = pack_observations(diffusion, observations, weights)
        diffusion.combine({v: diffusion.adapt(v, seen[v]) for v in nodes}, weights)
        for v in nodes:
            assert torch.allclose(diffusion.flat_params(v), fixture.flat_params(0), atol=1e-10)
    assert not torch.allclose(exact.flat_params(0), fixture.flat_params(0), atol=1e-8)


def test_the_network_models_gradient_is_its_jacobians():
    from dekf_bench.env.series import pool
    from dekf_bench.runner.network import NetworkModel

    environment, model, _lik, _l = built(coupled_config())
    x, _y = pool(environment.step(0))
    net = NetworkModel(model, environment.graph.weights, environment.n_nodes)
    params = model.init_params(torch.Generator().manual_seed(3))
    cotangent = torch.randn(net.forward(params, x).shape, dtype=DTYPE,
                            generator=torch.Generator().manual_seed(4))
    jac = net.per_sample_jacobian(params, x)
    assert torch.allclose(net.vjp(params, x, cotangent),
                          torch.einsum("nqp,nq->p", jac, cotangent), atol=1e-10)


def test_every_learner_kind_runs_on_the_coupled_network():
    """A short run through the simulator on an ER graph: filters on neuron blocks,
    one-hop, full sharing, and the gradient baselines -- every path, finite throughout."""
    from dekf_bench.evaluation.series_evalsets import build_series_evalsets
    from dekf_bench.runner import simulate

    blocks = {"covariance_blocks": "neuron"}
    learners = [
        {"name": "centralized_ekf_walk", **FILTER, **blocks},
        {"name": "diffusion_ekf", **FILTER, **blocks},
        {"name": "diffusion_ekf_onehop_mean_receiver", **FILTER, **blocks},
        {"name": "diffusion_ekf_full", **FILTER, **blocks},
        # 1e-6: a Gaussian likelihood at R = 0.03 scales gradients by 1/R, and at 1e-4
        # every SGD learner here diverges within six steps -- centralised SGD with the
        # exact network gradient included -- so the rate, not the truncation (measured
        # 2026-10-08). MG's tuned rates sit at 3e-6 to 3e-4 for the same reason.
        {"name": "centralized_sgd", "optimizer": "sgd_momentum", "momentum": 0.9, "lr": 1e-6},
        {"name": "diffusion_sgd_atc", "optimizer": "sgd_momentum", "momentum": 0.9, "lr": 1e-6,
         "mix_optimizer_state": "momentum"},
        {"name": "local_adamw", "lr": 1e-3},
    ]
    config = coupled_config(topology="erdos_renyi", learners=learners)
    environment, model, likelihood, built_learners = built(config)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    records = simulate.run(config, environment, built_learners,
                           build_series_evalsets(config, environment), likelihood, theta0)
    import math

    rmse = [(r.learner, r.step, row["node_id"], row["value"]) for r in records for row in r.rows
            if row["metric"] == "rmse" and row["evalset"] == "current"]
    # Finite is too weak a test in float64 -- a diverging learner reaches 1e119 and is
    # still finite -- so the scores are also bounded.
    bad = [entry for entry in rmse if not (math.isfinite(entry[3]) and entry[3] < 100.0)]
    assert rmse and not bad, bad[:5]
    seen = {r.learner for r in records}
    assert seen == {e["name"] for e in learners}
    # The forward pass is on the ledger, on its own columns: nothing for the pooled
    # learner, layers x 2|E| x context x d per step for every distributed one.
    last = {r.learner: r.rows[-1] for r in records if r.step == config.run.horizon - 1}
    per_step = 2 * 2 * environment.graph.n_edges * 31 * TINY_GNN["d_model"]
    assert last["centralized_ekf_walk"]["cum_forward_scalars_tx"] == 0
    for name in ("diffusion_ekf", "diffusion_sgd_atc", "local_adamw"):
        assert last[name]["cum_forward_scalars_tx"] == per_step * config.run.horizon
        assert last[name]["cum_forward_bits_tx"] == 64 * per_step * config.run.horizon
