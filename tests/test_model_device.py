"""Where a model's tensors live -- buffers, and the parameters `init_params` returns.

Two faults, one after the other, both recorded in D104.

**The first.** Models here are functional: parameters arrive as an argument and carry
their own placement, so a model owning no tensors runs correctly wherever it was
built. The causal Transformer owns two registered *buffers* -- sinusoidal positions
and a causal mask -- which travel with the module while ``functional_call``
substitutes parameters without touching them. Built on the CPU and handed cuda
parameters, ``forward`` raised.

**The second, caused by fixing the first.** Once the module was placed on a GPU,
``init_params`` split: ``*_like`` constructions followed the module to cuda while the
generator-driven weights stayed on the CPU, and ``flatten``'s ``cat`` failed. The
original version of this file hid that by moving every parameter to cuda itself --
a test that passed for the wrong reason.

So the contract asserted here is deliberately narrow: **buffers live where the model
was built; `init_params` always returns CPU tensors, whatever the module's device.**
The second half is what keeps $\\bm\\theta_0$ bit-identical across devices, because a
cuda tensor would need a cuda generator and a different random stream (D9, D18). The
runner moves the flat vector once, in ``run_one``.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.models.mlp import MLP
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.models.transformer import CausalTransformer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device here")

SMALL = {"context": 8, "d_model": 4, "n_heads": 2, "d_ff": 8, "dtype": torch.float64}


def _buffer_devices(model: object) -> set[str]:
    return {b.device.type for _n, b in model._module.named_buffers()}  # noqa: SLF001


def _param_devices(params: dict[str, torch.Tensor]) -> set[str]:
    return {t.device.type for t in params.values()}


# --- the premise, so the rest of the file is known to be testing something ----- #


def test_transformer_owns_buffers():
    model = CausalTransformer(**SMALL)
    names = {name for name, _ in model._module.named_buffers()}  # noqa: SLF001
    assert names == {"positions", "causal"}


def test_mlp_owns_no_buffers():
    model = MLP(input_size=4, hidden=(3,), output_dim=2, dtype=torch.float64)
    assert list(model._module.named_buffers()) == []  # noqa: SLF001


# --- init_params is CPU-only, whatever the module's device --------------------- #


def test_init_params_is_cpu_on_a_cpu_model():
    assert _param_devices(CausalTransformer(**SMALL, device="cpu").init_params()) == {"cpu"}


def test_mlp_init_params_is_internally_consistent():
    """The bias branch and the weight branch must agree; they are cat-ed together."""
    model = MLP(input_size=4, hidden=(3,), output_dim=2, dtype=torch.float64)
    params = model.init_params(torch.Generator().manual_seed(0))
    assert _param_devices(params) == {"cpu"}
    assert model.flatten(params).numel() == model.num_params


@CUDA
def test_init_params_is_cpu_even_when_the_module_is_on_cuda():
    """The regression: `*_like` used to follow the module and split the dict."""
    model = CausalTransformer(**SMALL, device="cuda")
    assert _buffer_devices(model) == {"cuda"}
    assert _param_devices(model.init_params()) == {"cpu"}


@CUDA
def test_mlp_init_params_is_cpu_even_when_the_module_is_on_cuda():
    model = MLP(input_size=4, hidden=(3,), output_dim=2, dtype=torch.float64)
    model._module.to("cuda")  # noqa: SLF001 -- what build_model now does
    assert _param_devices(model.init_params(torch.Generator().manual_seed(0))) == {"cpu"}


@CUDA
def test_flatten_succeeds_on_a_cuda_built_model():
    """The exact call that raised: cat over a dict split across two devices."""
    model = CausalTransformer(**SMALL, device="cuda")
    flat = model.flatten(model.init_params(torch.Generator().manual_seed(0)))
    assert flat.device.type == "cpu"
    assert flat.numel() == model.num_params


@CUDA
def test_theta0_is_bit_identical_across_devices():
    """Why init_params must not take a device: same seed, same vector, either way."""
    on_cpu = CausalTransformer(**SMALL, device="cpu")
    on_gpu = CausalTransformer(**SMALL, device="cuda")
    a = on_cpu.flatten(on_cpu.init_params(torch.Generator().manual_seed(7)))
    b = on_gpu.flatten(on_gpu.init_params(torch.Generator().manual_seed(7)))
    assert torch.equal(a, b)


# --- buffers, and the runner's real call path ---------------------------------- #


def test_buffers_follow_the_requested_device_on_cpu():
    assert _buffer_devices(CausalTransformer(**SMALL, device="cpu")) == {"cpu"}


@CUDA
def test_buffers_land_on_cuda_when_asked():
    assert _buffer_devices(CausalTransformer(**SMALL, device="cuda")) == {"cuda"}


def test_build_model_from_config_resolves_the_run_device():
    config = type("Cfg", (), {})()
    config.run = type("Run", (), {"dtype": "float64", "device": "auto"})()
    config.model = type("M", (), {"name": "causal_transformer", "context": 8, "d_model": 4,
                                  "n_heads": 2, "d_ff": 8})()
    model = build_model_from_config(config)
    assert _buffer_devices(model) == {"cuda" if torch.cuda.is_available() else "cpu"}


@CUDA
def test_forward_and_vjp_through_the_runners_path():
    """`flatten` -> `.to(device)` -> `unflatten`, which is what run_one does.

    Not by moving each parameter by hand: that is how the first version of this file
    masked the second fault.
    """
    model = CausalTransformer(**SMALL, device="cuda")
    theta0 = model.flatten(model.init_params(torch.Generator().manual_seed(0))).to("cuda")
    params = model.unflatten(theta0)
    x = torch.zeros(2, 8, dtype=torch.float64, device="cuda")

    out = model.forward(params, x)
    flat = model.vjp(params, x, torch.ones(2, 8, dtype=torch.float64, device="cuda"))

    assert out.shape == (2, 8) and out.device.type == "cuda" and torch.isfinite(out).all()
    assert flat.shape == (model.num_params,) and torch.isfinite(flat).all()
