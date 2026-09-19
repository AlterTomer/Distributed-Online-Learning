"""Name to builder, so the runner selects a model without importing one."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from dekf_bench.models.base import ModelError
from dekf_bench.models.linear_ar import LinearAR
from dekf_bench.models.mlp import MLP
from dekf_bench.models.transformer import CausalTransformer


class ModelBuilder(Protocol):
    def __call__(self, config: Any, dtype: torch.dtype, device: str) -> Any: ...


def _mlp(config: Any, dtype: torch.dtype, device: str) -> MLP:
    # `device` is unused here and that is correct: the MLP owns no tensors of its
    # own, so every tensor it touches arrives through `params`, already placed.
    # `build_model` places it generically anyway, which is what keeps this honest
    # if a buffer is ever added.
    return MLP(
        input_size=config.input_size,
        hidden=tuple(config.hidden),
        output_dim=config.output_dim,
        activation=config.activation,
        dtype=dtype,
    )


def _linear_probe(config: Any, dtype: torch.dtype, device: str) -> MLP:
    if config.hidden:
        raise ModelError(
            f"linear_probe must have no hidden layers, got {config.hidden}. The point of "
            "the probe is that theta -> logits is linear, which makes the EKF an exact KF "
            "with no linearisation error; a hidden layer removes exactly that property."
        )
    return _mlp(config, dtype, device)


def _causal_transformer(config: Any, dtype: torch.dtype, device: str) -> CausalTransformer:
    # The one model that owns tensors: a sinusoidal positional encoding and a causal
    # mask, both registered buffers. They are built on `device` rather than moved
    # afterwards, so they are never briefly in the wrong place.
    return CausalTransformer(
        context=config.context,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dtype=dtype,
        device=device,
    )


def _linear_ar(config: Any, dtype: torch.dtype, device: str) -> LinearAR:
    return LinearAR(context=config.context, dtype=dtype)


#: Every model a config may name. `cnn` arrives with its own builder; it has no
#: entry here rather than a stub, so an unknown name fails at load with a list of
#: what is available.
BUILDERS: dict[str, ModelBuilder] = {
    "mlp": _mlp,
    "mlp_small": _mlp,
    "linear_probe": _linear_probe,
    # The series task (docs/mackey_glass_plan.md, WP3).
    "causal_transformer": _causal_transformer,
    "linear_ar": _linear_ar,
}


def build_model(config: Any, dtype: torch.dtype = torch.float32, device: str = "cpu") -> Any:
    """The model a `model:` config block asks for, built for ``device``.

    Models here are *functional*: parameters arrive as an argument, already on the
    run's device, so a model that owns no tensors runs correctly wherever it was
    built. A model that owns **buffers** does not -- those travel with the module,
    and `functional_call` substitutes parameters without touching them. Left on the
    CPU while the parameters are on a GPU, they raise mid-run (D104).

    The placement below is therefore generic rather than per-model: every model
    wraps its network as ``_module``, and moving it is a no-op for the ones that
    hold nothing. ``device`` defaults to CPU so every existing caller is unchanged.
    """
    if config.name not in BUILDERS:
        raise ModelError(f"unknown model {config.name!r}; available: {sorted(BUILDERS)}")
    model = BUILDERS[config.name](config, dtype, device)
    module = getattr(model, "_module", None)
    if isinstance(module, torch.nn.Module):
        module.to(device)
    return model


def build_model_from_config(config: Any) -> Any:
    """The model a whole run config asks for, at the run's dtype and device.

    ``run.device`` may be ``auto``; it is resolved here through the same helper the
    environment uses, so the model and the data cannot disagree about where they are.
    """
    from dekf_bench.utils.determinism import resolve_device  # noqa: PLC0415

    dtype = torch.float64 if config.run.dtype == "float64" else torch.float32
    return build_model(config.model, dtype, resolve_device(config.run.device))
