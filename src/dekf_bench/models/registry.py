"""Name to builder, so the runner selects a model without importing one."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from dekf_bench.models.base import ModelError
from dekf_bench.models.mlp import MLP


class ModelBuilder(Protocol):
    def __call__(self, config: Any, dtype: torch.dtype) -> Any: ...


def _mlp(config: Any, dtype: torch.dtype) -> MLP:
    return MLP(
        input_size=config.input_size,
        hidden=tuple(config.hidden),
        output_dim=config.output_dim,
        activation=config.activation,
        dtype=dtype,
    )


def _linear_probe(config: Any, dtype: torch.dtype) -> MLP:
    if config.hidden:
        raise ModelError(
            f"linear_probe must have no hidden layers, got {config.hidden}. The point of "
            "the probe is that theta -> logits is linear, which makes the EKF an exact KF "
            "with no linearisation error; a hidden layer removes exactly that property."
        )
    return _mlp(config, dtype)


#: Every model a config may name. `cnn` arrives with its own builder; it has no
#: entry here rather than a stub, so an unknown name fails at load with a list of
#: what is available.
BUILDERS: dict[str, ModelBuilder] = {
    "mlp": _mlp,
    "mlp_small": _mlp,
    "linear_probe": _linear_probe,
}


def build_model(config: Any, dtype: torch.dtype = torch.float32) -> Any:
    """The model a `model:` config block asks for."""
    if config.name not in BUILDERS:
        raise ModelError(f"unknown model {config.name!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[config.name](config, dtype)


def build_model_from_config(config: Any) -> Any:
    """The model a whole run config asks for, at the run's dtype."""
    dtype = torch.float64 if config.run.dtype == "float64" else torch.float32
    return build_model(config.model, dtype)
