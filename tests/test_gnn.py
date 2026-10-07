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
