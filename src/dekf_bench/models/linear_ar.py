r"""A linear autoregressive predictor: the exact-Kalman gate and a baseline.

$\hat x_{i+1}=\sum_{j=0}^{L-2} w_j\,x_{i-j}+b$, with the window zero-padded where a
block's early positions have less past. Same inputs and outputs as the Transformer
-- a block of $L-1$ in, $L-1$ one-step predictions out -- with $p=L=32$ parameters.

**Why it earns a module** (`docs/mackey_glass_plan.md`, decision 11). The map
$\bm\theta\mapsto\bm h$ is *linear*: every output is $\bm X\bm w+b$, with no term
free of $\bm\theta$. Under a Gaussian likelihood the EKF is then an exact Kalman
filter -- recursive least squares -- and `prop:complete_graph` holds with no
linearisation error at all. That makes it the strongest correctness gate the second
task has, the role the linear probe plays on MNIST. It is also the classical
predictor for this system, so it is reported: a reviewer will ask whether the
Transformer is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from dekf_bench.models import functional as F
from dekf_bench.models.base import ModelError, ParamDict, ParamGroup


class _CausalLinear(nn.Module):
    def __init__(self, context: int, dtype: torch.dtype):
        super().__init__()
        self.context = context
        self.weight = nn.Parameter(torch.zeros(context, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, context) -> (batch, context)
        # Row i of the lag matrix is [x_i, x_{i-1}, ..., x_{i-L+2}], zero before the
        # block starts, so weight[j] always multiplies the sample j steps back.
        padded = nn.functional.pad(x, (self.context - 1, 0))
        lagged = padded.unfold(-1, self.context, 1).flip(-1)
        return lagged @ self.weight + self.bias


@dataclass(frozen=True)
class LinearAR:
    """Linear AR over the causal window, evaluated functionally.

    Attributes:
        context: $L-1$, inputs and outputs per block, and the AR order.
        dtype: parameter dtype.
    """

    context: int = 31
    dtype: torch.dtype = torch.float32
    _module: nn.Module = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.context < 1:
            raise ModelError(f"context must be >= 1, got {self.context}")
        object.__setattr__(self, "_module", _CausalLinear(self.context, self.dtype))

    @property
    def input_dim(self) -> int:
        return self.context

    @property
    def output_dim(self) -> int:
        return self.context

    @property
    def num_params(self) -> int:
        return self.context + 1

    @property
    def is_linear(self) -> bool:
        """Always: this is what makes the EKF exact on it."""
        return True

    @property
    def names(self) -> tuple[str, ...]:
        return F.parameter_names(self._module)

    @property
    def shapes(self) -> tuple[torch.Size, ...]:
        return tuple(parameter.shape for _, parameter in self._module.named_parameters())

    def init_params(self, generator: torch.Generator | None = None) -> ParamDict:
        """Uniform weights at the Xavier scale for one output, zero bias."""
        bound = math.sqrt(6.0 / (self.context + 1))
        return {
            "weight": torch.empty(self.context, dtype=self.dtype).uniform_(
                -bound, bound, generator=generator
            ),
            "bias": torch.zeros(1, dtype=self.dtype),
        }

    def flatten(self, params: ParamDict) -> torch.Tensor:
        return F.flatten(params, self.names)

    def unflatten(self, vector: torch.Tensor) -> ParamDict:
        return F.unflatten(vector, self.names, self.shapes)

    def param_groups(self) -> tuple[ParamGroup, ...]:
        return F.build_param_groups(self._module, self.names)

    def forward(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        return F.call(self._module, params, x)

    def vjp(self, params: ParamDict, x: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        return self.flatten(F.vector_jacobian_product(self._module, params, x, cotangent))

    def jvp(self, params: ParamDict, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return F.jacobian_vector_product(self._module, params, x, self.unflatten(tangent))

    def jacobian(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        return F.jacobian(self._module, params, x, self.names)

    def per_sample_jacobian(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        return F.per_sample_jacobian(self._module, params, x, self.names)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "linear_ar",
            "context": self.context,
            "num_params": self.num_params,
            "dtype": str(self.dtype),
        }
