"""The multilayer perceptron, and the linear probe as its no-hidden-layer case.

**The activation defaults to GELU, not ReLU, and that is a phase-5 requirement
rather than a preference.** The research note's smoothness assumption
(As. 3) bounds the second-order remainder of the linearisation,

$$\\lVert\\bm h(\\bm\\theta)-\\bm h(\\bm m)-\\bm H(\\bm\\theta-\\bm m)\\rVert
\\le \\tfrac{\\kappa}{2}\\lVert\\bm\\theta-\\bm m\\rVert^2,$$

and that bound **fails for ReLU at the kink set** -- the note says so explicitly
and recommends a smooth activation. Since D1 commits phases 1--4 to the same
model phase 5 will filter, the choice has to be made here. ReLU stays available
and costs nothing on MNIST, but a run using it is outside the regime the theory
covers, and that should be a deliberate choice rather than an inherited default.

The linear probe is the same class with no hidden layers. It matters because
$\\bm\\theta\\mapsto\\bm h$ is then **linear**, the EKF becomes an exact KF, and
complete-graph exactness holds with no linearisation error at all -- the one
setting where the theory is exact rather than approximate
(research note §6.3, IMPLEMENTATION.md §13.9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from dekf_bench.models import functional as F
from dekf_bench.models.base import ModelError, ParamDict, ParamGroup

#: Smooth activations satisfy the research note's bounded-remainder assumption;
#: ReLU does not, at its kink set.
SMOOTH_ACTIVATIONS = ("gelu", "tanh", "silu")
ACTIVATIONS = (*SMOOTH_ACTIVATIONS, "relu")

_MODULES = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}

#: Xavier gains. GELU and SiLU have no standard value; 1.0 is the conventional
#: choice and the depths here are too shallow for it to matter.
_GAINS = {"tanh": 5.0 / 3.0, "relu": math.sqrt(2.0), "gelu": 1.0, "silu": 1.0}


@dataclass(frozen=True)
class MLP:
    """A fully connected network, evaluated functionally.

    Attributes:
        input_size: image side length. The flattened input is its square.
        hidden: hidden widths. Empty gives a linear probe.
        output_dim: number of classes, $q$.
        activation: see the module docstring.
        dtype: parameter dtype. float64 for the exactness check.
    """

    input_size: int = 14
    hidden: tuple[int, ...] = (14,)
    output_dim: int = 10
    activation: str = "gelu"
    dtype: torch.dtype = torch.float32
    _module: nn.Module = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.input_size < 1:
            raise ModelError(f"input_size must be >= 1, got {self.input_size}")
        if self.output_dim < 2:
            raise ModelError(f"output_dim must be >= 2, got {self.output_dim}")
        if any(width < 1 for width in self.hidden):
            raise ModelError(f"hidden widths must be >= 1, got {self.hidden}")
        if self.activation not in ACTIVATIONS:
            raise ModelError(
                f"unknown activation {self.activation!r}; expected one of {list(ACTIVATIONS)}"
            )
        object.__setattr__(self, "_module", self._build_module())

    def _build_module(self) -> nn.Module:
        """A template. Its own parameter values are never read."""
        widths = [self.input_dim, *self.hidden, self.output_dim]
        layers: list[nn.Module] = [nn.Flatten()]
        for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            layers.append(nn.Linear(fan_in, fan_out, dtype=self.dtype))
            if index < len(widths) - 2:
                layers.append(_MODULES[self.activation]())
        return nn.Sequential(*layers)

    # -- shape ------------------------------------------------------------- #

    @property
    def input_dim(self) -> int:
        return self.input_size * self.input_size

    @property
    def num_params(self) -> int:
        widths = [self.input_dim, *self.hidden, self.output_dim]
        return sum(a * b + b for a, b in zip(widths[:-1], widths[1:], strict=True))

    @property
    def is_linear(self) -> bool:
        """Whether $\\bm\\theta\\mapsto\\bm h$ is linear, making the EKF exact."""
        return not self.hidden

    @property
    def names(self) -> tuple[str, ...]:
        return F.parameter_names(self._module)

    @property
    def shapes(self) -> tuple[torch.Size, ...]:
        return tuple(parameter.shape for _, parameter in self._module.named_parameters())

    # -- parameters -------------------------------------------------------- #

    def init_params(self, generator: torch.Generator | None = None) -> ParamDict:
        """Xavier-uniform weights, zero biases, drawn from an explicit generator.

        Implemented directly rather than through ``torch.nn.init`` so the
        generator is threaded explicitly: the global-RNG variants would make
        $\\bm\\theta_0$ depend on whatever else drew first, and every agent must
        receive the *same* initial parameters (design notes D9, D18).
        """
        params: ParamDict = {}
        for name, template in self._module.named_parameters():
            if name.endswith("bias"):
                params[name] = torch.zeros_like(template)
                continue
            fan_out, fan_in = template.shape
            gain = _GAINS[self.activation]
            bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
            params[name] = torch.empty(template.shape, dtype=self.dtype).uniform_(
                -bound, bound, generator=generator
            )
        return params

    def flatten(self, params: ParamDict) -> torch.Tensor:
        return F.flatten(params, self.names)

    def unflatten(self, vector: torch.Tensor) -> ParamDict:
        return F.unflatten(vector, self.names, self.shapes)

    def param_groups(self) -> tuple[ParamGroup, ...]:
        return F.build_param_groups(self._module, self.names)

    def last_layer(self) -> ParamGroup:
        """The read-out block -- the subspace Bayesian-last-layer filtering uses."""
        return self.param_groups()[-1]

    # -- evaluation -------------------------------------------------------- #

    def forward(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        return F.call(self._module, params, x)

    def vjp(self, params: ParamDict, x: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        """$\\bm u^{\\mathsf T}\\bm H$, returned flat."""
        return self.flatten(F.vector_jacobian_product(self._module, params, x, cotangent))

    def jvp(self, params: ParamDict, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """$\\bm H\\bm v$ for a flat direction ``v`` in parameter space."""
        return F.jacobian_vector_product(self._module, params, x, self.unflatten(tangent))

    def jacobian(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        """The full $\\bm H$. Tests and small models only -- see functional.py."""
        return F.jacobian(self._module, params, x, self.names)

    def per_sample_jacobian(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        """Every sample's $\\bm H$, shape ``(n, q, p)``. What the filter needs."""
        return F.per_sample_jacobian(self._module, params, x, self.names)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "linear_probe" if self.is_linear else "mlp",
            "input_size": self.input_size,
            "input_dim": self.input_dim,
            "hidden": list(self.hidden),
            "output_dim": self.output_dim,
            "activation": self.activation,
            "smooth": self.activation in SMOOTH_ACTIVATIONS,
            "num_params": self.num_params,
            "dtype": str(self.dtype),
            "dense_covariance_entries": self.num_params**2,
        }
