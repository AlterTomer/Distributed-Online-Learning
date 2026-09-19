"""The Mackey--Glass models: size, causality, and Jacobians the filter can trust."""

from __future__ import annotations

import pytest
import torch

from dekf_bench.models.linear_ar import LinearAR
from dekf_bench.models.transformer import CausalTransformer

DTYPE = torch.float64


@pytest.fixture(scope="module")
def transformer() -> CausalTransformer:
    return CausalTransformer(dtype=DTYPE)


@pytest.fixture(scope="module")
def params(transformer: CausalTransformer):
    return transformer.init_params(torch.Generator().manual_seed(0))


def test_the_design_parameter_count(transformer: CausalTransformer) -> None:
    """2 273: the figure the design doc budgets the dense covariance against."""
    assert transformer.num_params == 2273


def test_outputs_are_one_per_position(transformer, params) -> None:
    x = torch.randn(4, 31, dtype=DTYPE)
    assert transformer.forward(params, x).shape == (4, 31)


def test_the_model_is_causal(transformer, params) -> None:
    """Output i depends on inputs 0..i only."""
    x = torch.randn(1, 31, dtype=DTYPE)
    base = transformer.forward(params, x)
    for j in (0, 10, 30):
        moved = x.clone()
        moved[0, j] += 1.0
        change = (transformer.forward(params, moved) - base).abs()[0]
        assert change[:j].max().item() == 0.0 if j else True
        assert change[j:].max().item() > 0.0


def test_blocks_are_independent(transformer, params) -> None:
    """A batch is a stack of blocks, never mixed: no attention across the batch."""
    x = torch.randn(3, 31, dtype=DTYPE)
    together = transformer.forward(params, x)
    for row in range(3):
        assert torch.allclose(transformer.forward(params, x[row : row + 1])[0], together[row])


def test_per_sample_jacobian_matches_the_reference(transformer, params) -> None:
    """The vmapped Jacobian the filter uses equals the loop-of-VJPs reference."""
    x = torch.randn(2, 31, dtype=DTYPE)
    fast = transformer.per_sample_jacobian(params, x)
    assert fast.shape == (2, 31, 2273)
    for row in range(2):
        slow = transformer.jacobian(params, x[row : row + 1])[0]
        assert torch.allclose(fast[row], slow, atol=1e-12)


def test_the_jacobian_is_the_derivative(transformer, params) -> None:
    """A finite-difference check along a random direction, in float64."""
    x = torch.randn(1, 31, dtype=DTYPE)
    theta = transformer.flatten(params)
    direction = torch.randn_like(theta)
    step = 1e-6
    ahead = transformer.forward(transformer.unflatten(theta + step * direction), x)
    behind = transformer.forward(transformer.unflatten(theta - step * direction), x)
    numeric = (ahead - behind) / (2 * step)
    analytic = transformer.jvp(params, x, direction)
    assert torch.allclose(numeric, analytic, atol=1e-7)


def test_init_is_reproducible(transformer) -> None:
    first = transformer.flatten(transformer.init_params(torch.Generator().manual_seed(5)))
    second = transformer.flatten(transformer.init_params(torch.Generator().manual_seed(5)))
    assert torch.equal(first, second)


def test_flatten_round_trips(transformer, params) -> None:
    theta = transformer.flatten(params)
    assert torch.equal(transformer.flatten(transformer.unflatten(theta)), theta)


@pytest.mark.parametrize("name", ["causal_transformer", "linear_ar"])
def test_the_config_counts_what_the_registry_builds(name: str) -> None:
    """ModelConfig.num_params is analytic; it must agree with the built model."""
    from dekf_bench.models.registry import build_model
    from dekf_bench.utils.config import ModelConfig

    config = ModelConfig(name=name, likelihood="gaussian", output_dim=31)
    assert build_model(config, DTYPE).num_params == config.num_params


def test_linear_ar_size_and_causality() -> None:
    model = LinearAR(dtype=DTYPE)
    assert model.num_params == 32
    params = model.init_params(torch.Generator().manual_seed(1))
    x = torch.randn(1, 31, dtype=DTYPE)
    base = model.forward(params, x)
    moved = x.clone()
    moved[0, 12] += 1.0
    change = (model.forward(params, moved) - base).abs()[0]
    assert change[:12].max().item() == 0.0 and change[12].item() > 0.0


def test_linear_ar_weight_j_reads_the_sample_j_back() -> None:
    """weight[0] is the latest sample: a unit weight there is persistence."""
    model = LinearAR(dtype=DTYPE)
    params = {"weight": torch.zeros(31, dtype=DTYPE), "bias": torch.zeros(1, dtype=DTYPE)}
    params["weight"][0] = 1.0
    x = torch.randn(2, 31, dtype=DTYPE)
    assert torch.equal(model.forward(params, x), x)


def test_linear_ar_is_linear_in_theta() -> None:
    """h(a t1 + b t2) = a h(t1) + b h(t2): no term free of theta, so the EKF is exact."""
    model = LinearAR(dtype=DTYPE)
    x = torch.randn(3, 31, dtype=DTYPE)
    t1 = torch.randn(32, dtype=DTYPE)
    t2 = torch.randn(32, dtype=DTYPE)
    combined = model.forward(model.unflatten(2.0 * t1 - 0.5 * t2), x)
    separate = 2.0 * model.forward(model.unflatten(t1), x) - 0.5 * model.forward(
        model.unflatten(t2), x
    )
    assert torch.allclose(combined, separate, atol=1e-12)
    assert model.is_linear
