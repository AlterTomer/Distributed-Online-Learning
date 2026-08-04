"""Models: shapes, functional evaluation, and Jacobian products.

Most of what is checked here is unused by phase-1 SGD. It is checked now because
the phase-5 filter is built on it, and a `vjp` that disagrees with the true
Jacobian produces a filter that converges to the wrong place without ever
raising (IMPLEMENTATION.md §13.3).
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.models.base import ModelError
from dekf_bench.models.functional import FunctionalError
from dekf_bench.models.mlp import ACTIVATIONS, MLP, SMOOTH_ACTIVATIONS
from dekf_bench.models.registry import build_model, build_model_from_config
from dekf_bench.utils.config import load_config

BATCH = 5


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def inputs(model: MLP, batch: int = BATCH, seed: int = 0) -> torch.Tensor:
    return torch.rand(
        batch, 1, model.input_size, model.input_size, generator=generator(seed), dtype=model.dtype
    )


@pytest.fixture
def model() -> MLP:
    return MLP(input_size=8, hidden=(6, 5), output_dim=4, dtype=torch.float64)


@pytest.fixture
def probe() -> MLP:
    return MLP(input_size=8, hidden=(), output_dim=4, dtype=torch.float64)


# =========================================================================== #
# 1. shape
# =========================================================================== #


def test_num_params_matches_the_flat_vector(model: MLP) -> None:
    params = model.init_params(generator())
    assert model.flatten(params).numel() == model.num_params


def test_num_params_matches_the_hand_count() -> None:
    """196*14 + 14 + 14*10 + 10 = 2908, the budget D1 committed to."""
    assert MLP(input_size=14, hidden=(14,), output_dim=10).num_params == 2908


def test_forward_shape(model: MLP) -> None:
    params = model.init_params(generator())
    assert model.forward(params, inputs(model)).shape == (BATCH, 4)


def test_input_dim_is_the_flattened_image(model: MLP) -> None:
    assert model.input_dim == 64 == 8 * 8


def test_a_probe_has_no_hidden_layer(probe: MLP) -> None:
    assert probe.is_linear
    assert probe.num_params == 64 * 4 + 4


def test_an_mlp_is_not_linear(model: MLP) -> None:
    assert not model.is_linear


# =========================================================================== #
# 2. flatten / unflatten
# =========================================================================== #


def test_flatten_unflatten_round_trips_exactly(model: MLP) -> None:
    """Exact, not approximate: the filter's state *is* the flat vector, so any
    loss here is loss in the estimate itself."""
    params = model.init_params(generator())
    restored = model.unflatten(model.flatten(params))
    assert set(restored) == set(params)
    for name, tensor in params.items():
        assert torch.equal(restored[name], tensor)


def test_unflatten_flatten_round_trips(model: MLP) -> None:
    vector = torch.randn(model.num_params, generator=generator(), dtype=model.dtype)
    assert torch.equal(model.flatten(model.unflatten(vector)), vector)


def test_flatten_order_is_stable(model: MLP) -> None:
    params = model.init_params(generator())
    assert torch.equal(model.flatten(params), model.flatten(dict(reversed(params.items()))))


def test_unflatten_rejects_the_wrong_length(model: MLP) -> None:
    with pytest.raises(FunctionalError, match="entries but the model has"):
        model.unflatten(torch.zeros(model.num_params + 1))


def test_unflatten_rejects_a_matrix(model: MLP) -> None:
    with pytest.raises(FunctionalError, match="expected a flat vector"):
        model.unflatten(torch.zeros(2, model.num_params))


def test_flatten_rejects_missing_parameters(model: MLP) -> None:
    params = model.init_params(generator())
    del params[next(iter(params))]
    with pytest.raises(FunctionalError, match="parameters missing"):
        model.flatten(params)


# =========================================================================== #
# 3. initialization
# =========================================================================== #


def test_init_is_deterministic_from_the_generator(model: MLP) -> None:
    """Every agent must be able to start from the *same* theta_0: Diff-EKF
    agents need a common prior, and it removes a confound from the SGD
    comparison."""
    first = model.flatten(model.init_params(generator(7)))
    second = model.flatten(model.init_params(generator(7)))
    assert torch.equal(first, second)


def test_different_seeds_give_different_initialisations(model: MLP) -> None:
    first = model.flatten(model.init_params(generator(1)))
    second = model.flatten(model.init_params(generator(2)))
    assert not torch.equal(first, second)


def test_init_does_not_touch_the_global_rng(model: MLP) -> None:
    """A global draw elsewhere must not change theta_0 (design note D9)."""
    before = model.flatten(model.init_params(generator(0)))
    torch.manual_seed(999)
    _ = torch.randn(1000)
    assert torch.equal(model.flatten(model.init_params(generator(0))), before)


def test_biases_start_at_zero(model: MLP) -> None:
    params = model.init_params(generator())
    for name, tensor in params.items():
        if name.endswith("bias"):
            assert torch.all(tensor == 0)


def test_weights_are_not_all_zero(model: MLP) -> None:
    params = model.init_params(generator())
    for name, tensor in params.items():
        if name.endswith("weight"):
            assert float(tensor.abs().max()) > 0


def test_init_respects_the_dtype() -> None:
    model = MLP(input_size=8, hidden=(4,), output_dim=3, dtype=torch.float64)
    assert all(t.dtype == torch.float64 for t in model.init_params(generator()).values())


# =========================================================================== #
# 4. functional evaluation
# =========================================================================== #


def test_forward_uses_the_parameters_it_is_given(model: MLP) -> None:
    x = inputs(model)
    a = model.forward(model.init_params(generator(1)), x)
    b = model.forward(model.init_params(generator(2)), x)
    assert not torch.allclose(a, b)


def test_forward_leaves_the_template_untouched(model: MLP) -> None:
    """The module is a template; ten agents share it while holding ten different
    parameter vectors."""
    before = [p.clone() for p in model._module.parameters()]
    model.forward(model.init_params(generator()), inputs(model))
    after = list(model._module.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_the_same_parameters_give_the_same_output(model: MLP) -> None:
    params = model.init_params(generator())
    x = inputs(model)
    assert torch.equal(model.forward(params, x), model.forward(params, x))


def test_forward_accepts_a_flat_round_trip(model: MLP) -> None:
    """What the filter does: hold theta as a vector, evaluate the network there."""
    params = model.init_params(generator())
    x = inputs(model)
    revived = model.unflatten(model.flatten(params))
    assert torch.equal(model.forward(revived, x), model.forward(params, x))


# =========================================================================== #
# 5. Jacobian products
# =========================================================================== #


def test_jacobian_shape(model: MLP) -> None:
    params = model.init_params(generator())
    assert model.jacobian(params, inputs(model)).shape == (BATCH, 4, model.num_params)


def test_vjp_matches_the_materialised_jacobian(model: MLP) -> None:
    """The check IMPLEMENTATION.md §13.3 asks for. A vjp that disagrees with H
    gives a filter that converges to the wrong place, silently."""
    params = model.init_params(generator())
    x = inputs(model)
    jac = model.jacobian(params, x)

    cotangent = torch.randn(BATCH, 4, generator=generator(3), dtype=model.dtype)
    expected = torch.einsum("bqp,bq->p", jac, cotangent)
    assert torch.allclose(model.vjp(params, x, cotangent), expected, atol=1e-10)


def test_jvp_matches_the_materialised_jacobian(model: MLP) -> None:
    params = model.init_params(generator())
    x = inputs(model)
    jac = model.jacobian(params, x)

    tangent = torch.randn(model.num_params, generator=generator(4), dtype=model.dtype)
    expected = torch.einsum("bqp,p->bq", jac, tangent)
    assert torch.allclose(model.jvp(params, x, tangent), expected, atol=1e-10)


def test_vjp_matches_autograd(model: MLP) -> None:
    """Against a second, independent route to the same gradient."""
    params = {k: v.clone().requires_grad_(True) for k, v in model.init_params(generator()).items()}
    x = inputs(model)
    cotangent = torch.randn(BATCH, 4, generator=generator(5), dtype=model.dtype)

    output = model.forward(params, x)
    (output * cotangent).sum().backward()
    expected = model.flatten({name: p.grad for name, p in params.items()})  # type: ignore[misc]

    detached = {k: v.detach() for k, v in params.items()}
    assert torch.allclose(model.vjp(detached, x, cotangent), expected, atol=1e-10)


def test_vjp_is_linear_in_the_cotangent(model: MLP) -> None:
    params = model.init_params(generator())
    x = inputs(model)
    u = torch.randn(BATCH, 4, generator=generator(6), dtype=model.dtype)
    v = torch.randn(BATCH, 4, generator=generator(7), dtype=model.dtype)
    combined = model.vjp(params, x, 2.0 * u + 3.0 * v)
    separate = 2.0 * model.vjp(params, x, u) + 3.0 * model.vjp(params, x, v)
    assert torch.allclose(combined, separate, atol=1e-10)


def test_vjp_rejects_a_mismatched_cotangent(model: MLP) -> None:
    params = model.init_params(generator())
    with pytest.raises(FunctionalError, match="does not match the output"):
        model.vjp(params, inputs(model), torch.zeros(BATCH, 99, dtype=model.dtype))


# =========================================================================== #
# 6. the linear probe, where the EKF is exact
# =========================================================================== #


def test_a_probes_jacobian_does_not_depend_on_its_parameters(probe: MLP) -> None:
    """The property that makes the EKF an *exact* KF for the probe: with
    theta -> h linear, the linearisation is not an approximation at all, so
    complete-graph exactness holds with no remainder (research note Ex. 1)."""
    x = inputs(probe)
    first = probe.jacobian(probe.init_params(generator(1)), x)
    second = probe.jacobian(probe.init_params(generator(2)), x)
    assert torch.allclose(first, second, atol=1e-12)


def test_an_mlps_jacobian_does_depend_on_its_parameters(model: MLP) -> None:
    """Otherwise the previous test would be vacuous."""
    x = inputs(model)
    first = model.jacobian(model.init_params(generator(1)), x)
    second = model.jacobian(model.init_params(generator(2)), x)
    assert not torch.allclose(first, second)


def test_a_probe_is_linear_in_its_parameters(probe: MLP) -> None:
    """h(a*theta1 + b*theta2) = a*h(theta1) + b*h(theta2), exactly."""
    x = inputs(probe)
    one = probe.flatten(probe.init_params(generator(1)))
    two = probe.flatten(probe.init_params(generator(2)))
    a, b = 0.3, 0.7

    combined = probe.forward(probe.unflatten(a * one + b * two), x)
    separate = a * probe.forward(probe.unflatten(one), x) + b * probe.forward(
        probe.unflatten(two), x
    )
    assert torch.allclose(combined, separate, atol=1e-12)


def test_the_first_order_expansion_is_exact_for_a_probe(probe: MLP) -> None:
    """h(theta) = h(m) + H (theta - m), with no remainder. This is the whole
    content of 'no linearisation error'."""
    x = inputs(probe)
    m = probe.flatten(probe.init_params(generator(1)))
    theta = probe.flatten(probe.init_params(generator(2)))

    base = probe.forward(probe.unflatten(m), x)
    predicted = base + probe.jvp(probe.unflatten(m), x, theta - m)
    assert torch.allclose(predicted, probe.forward(probe.unflatten(theta), x), atol=1e-10)


def test_the_expansion_is_not_exact_for_an_mlp(model: MLP) -> None:
    x = inputs(model)
    m = model.flatten(model.init_params(generator(1)))
    theta = model.flatten(model.init_params(generator(2)))

    base = model.forward(model.unflatten(m), x)
    predicted = base + model.jvp(model.unflatten(m), x, theta - m)
    assert not torch.allclose(predicted, model.forward(model.unflatten(theta), x), atol=1e-4)


# =========================================================================== #
# 7. parameter groups
# =========================================================================== #


def test_groups_tile_the_flat_vector(model: MLP) -> None:
    groups = model.param_groups()
    assert groups[0].start == 0
    assert groups[-1].stop == model.num_params
    for previous, current in zip(groups, groups[1:], strict=False):
        assert previous.stop == current.start, "blocks must be contiguous"


def test_one_group_per_layer(model: MLP) -> None:
    assert len(model.param_groups()) == len(model.hidden) + 1 == 3


def test_groups_carry_the_kronecker_shapes(model: MLP) -> None:
    """Phase 5 factors each block as an input covariance against an output
    covariance; having the widths here means it need not re-derive them."""
    groups = model.param_groups()
    assert (groups[0].fan_in, groups[0].fan_out) == (model.input_dim, 6)
    assert (groups[-1].fan_in, groups[-1].fan_out) == (5, 4)


def test_group_size_matches_the_layer(model: MLP) -> None:
    for group in model.param_groups():
        assert group.size == group.fan_in * group.fan_out + group.fan_out


def test_last_layer_is_the_readout(model: MLP) -> None:
    """The subspace Bayesian-last-layer filtering runs on."""
    last = model.last_layer()
    assert last.fan_out == model.output_dim
    assert last.stop == model.num_params
    assert last.size == 5 * 4 + 4


def test_a_probe_has_a_single_group(probe: MLP) -> None:
    groups = probe.param_groups()
    assert len(groups) == 1
    assert groups[0].size == probe.num_params


# =========================================================================== #
# 8. activations
# =========================================================================== #


def test_gelu_is_the_default() -> None:
    """The research note's remainder bound fails for ReLU at its kink set, and
    D1 commits phases 1-4 to the model phase 5 will filter."""
    assert MLP().activation == "gelu"
    assert load_config("x1_stationary").model.activation == "gelu"


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_every_activation_builds_and_runs(activation: str) -> None:
    model = MLP(input_size=8, hidden=(5,), output_dim=3, activation=activation)
    output = model.forward(model.init_params(generator()), inputs(model))
    assert output.shape == (BATCH, 3)


def test_smoothness_is_reported(model: MLP) -> None:
    assert model.summary()["smooth"] is True
    rough = MLP(input_size=8, hidden=(5,), output_dim=3, activation="relu")
    assert rough.summary()["smooth"] is False
    assert "relu" not in SMOOTH_ACTIVATIONS


def test_unknown_activation_is_rejected() -> None:
    with pytest.raises(ModelError, match="unknown activation"):
        MLP(activation="sigmoid")


def test_the_config_rejects_an_unknown_activation() -> None:
    from dekf_bench.utils.config import ConfigError

    with pytest.raises(ConfigError, match="model.activation"):
        load_config("x1_stationary", overrides={"model": {"activation": "sigmoid"}})


# =========================================================================== #
# 9. rejected models
# =========================================================================== #


def test_too_few_classes_is_rejected() -> None:
    with pytest.raises(ModelError, match="output_dim must be >= 2"):
        MLP(output_dim=1)


def test_zero_width_hidden_layer_is_rejected() -> None:
    with pytest.raises(ModelError, match="hidden widths must be >= 1"):
        MLP(hidden=(0,))


def test_zero_input_size_is_rejected() -> None:
    with pytest.raises(ModelError, match="input_size must be >= 1"):
        MLP(input_size=0)


def test_model_is_frozen(model: MLP) -> None:
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(model, "output_dim", 7)  # noqa: B010


# =========================================================================== #
# 10. registry and config
# =========================================================================== #


def test_registry_builds_the_configured_model() -> None:
    model = build_model_from_config(load_config("x1_stationary"))
    assert model.num_params == 2908
    assert model.input_size == 14


def test_exactness_run_builds_a_float64_model() -> None:
    model = build_model_from_config(load_config("x0_exactness"))
    assert model.dtype == torch.float64
    assert all(t.dtype == torch.float64 for t in model.init_params(generator()).values())


def test_full_size_model_is_two_orders_larger() -> None:
    model = build_model_from_config(
        load_config("x1_stationary", overrides={"include": {"model": "mlp"}})
    )
    assert model.num_params == 101_770


def test_the_probe_config_builds_a_linear_model() -> None:
    model = build_model_from_config(
        load_config("x1_stationary", overrides={"include": {"model": "linear_probe"}})
    )
    assert model.is_linear
    assert model.num_params == 196 * 10 + 10


def test_a_probe_with_hidden_layers_is_rejected() -> None:
    """The probe exists because theta -> h is linear; a hidden layer removes
    exactly that property, so accepting it would silently lose the guarantee."""
    config = load_config(
        "x1_stationary",
        overrides={"include": {"model": "linear_probe"}, "model": {"hidden": [8]}},
    )
    with pytest.raises(ModelError, match="must have no hidden layers"):
        build_model_from_config(config)


def test_unknown_model_lists_the_available_ones() -> None:
    config = load_config("x1_stationary").model
    object.__setattr__(config, "name", "transformer")
    with pytest.raises(ModelError, match="unknown model 'transformer'"):
        build_model(config)


def test_dense_covariance_budget_is_reported() -> None:
    """The number that decided the architecture: p^2 entries."""
    model = build_model_from_config(load_config("x1_stationary"))
    assert model.summary()["dense_covariance_entries"] == 2908**2
    assert model.summary()["dense_covariance_entries"] < 10**7
