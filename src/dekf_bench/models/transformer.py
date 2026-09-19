r"""A one-block causal Transformer for blockwise one-step prediction.

The Mackey--Glass model (`docs/mackey_glass_plan.md`, decisions 9--10). A block of
$L-1=31$ inputs $[x_1..x_{L-1}]$ produces 31 predictions $[\hat x_2..\hat x_L]$ in
one pass, each seeing only its own past through a causal mask. That makes the
per-block Jacobian $31\times p$, of rank up to 31 -- the fix for the rank-1 problem
of scalar regression (P5.25).

**Architecture, and why each choice.** Input projection $1\to16$; fixed sinusoidal
positions (no parameters, so $p$ stays 2 273); one **pre-LayerNorm** block with
2-head causal self-attention and a GELU feed-forward of width 32; a scalar head at
every position. No final LayerNorm and no dropout.

* **GELU, not ReLU** -- the note's smoothness assumption bounds the linearisation
  remainder, and fails at ReLU's kinks. The same reason as `mlp.py`.
* **No dropout** -- the model is evaluated at a given $\bm\theta$, deterministically,
  by every learner; a stochastic forward pass has no place in a filter.
* **Our own block, not ``nn.TransformerEncoderLayer``** -- the library layer
  defaults to post-LN and ReLU, and its fused attention paths are not guaranteed
  under ``torch.func`` transforms. Attention here is written out: a masked
  softmax of $\bm q\bm k^{\mathsf T}/\sqrt{d_h}$, which ``vmap``/``jacrev`` handle
  exactly.

$p$ counts: input 32, two LayerNorms 32 each, attention 816 + 272, feed-forward
544 + 528, head 17 -- 2 273, the design doc's figure.

Functional, like every model here: parameters are an argument, the module is a
template whose own values are never read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from dekf_bench.models import functional as F
from dekf_bench.models.base import ModelError, ParamDict, ParamGroup


def sinusoidal_positions(length: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    """The fixed encoding of Vaswani et al.: sines and cosines at geometric wavelengths."""
    position = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float64) * (-math.log(10_000.0) / width)
    )
    table = torch.zeros(length, width, dtype=torch.float64)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency[: width // 2])
    return table.to(dtype)


class _CausalBlock(nn.Module):
    def __init__(self, context: int, d_model: int, n_heads: int, d_ff: int, dtype: torch.dtype,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.embed = nn.Linear(1, d_model, dtype=dtype)
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.qkv = nn.Linear(d_model, 3 * d_model, dtype=dtype)
        self.proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ff1 = nn.Linear(d_model, d_ff, dtype=dtype)
        self.ff2 = nn.Linear(d_ff, d_model, dtype=dtype)
        self.head = nn.Linear(d_model, 1, dtype=dtype)
        # Buffers, not parameters: positions are fixed (decision 9) and the mask is
        # structure. Neither enters p, and functional_call leaves them in place.
        self.register_buffer(
            "positions", sinusoidal_positions(context, d_model, dtype), persistent=False
        )
        self.register_buffer(
            "causal", torch.tril(torch.ones(context, context, dtype=torch.bool)), persistent=False
        )
        # ⚠ Parameters reach `forward` as an argument, already on the run's device.
        # BUFFERS do not: they travel with the module. Built on the CPU and left
        # there, `positions` meets a cuda tensor inside `functional_call` and raises
        # "Expected all tensors to be on the same device". Placing the module at
        # construction is what keeps the two together. This is the only model that
        # owns tensors, which is why it is the only one that ever hit this.
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, context) -> (batch, context)
        batch, context = x.shape
        head_width = self.d_model // self.n_heads
        h = self.embed(x.unsqueeze(-1)) + self.positions

        q, k, v = self.qkv(self.norm1(h)).split(self.d_model, dim=-1)
        q, k, v = (
            t.reshape(batch, context, self.n_heads, head_width).transpose(1, 2) for t in (q, k, v)
        )
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)
        scores = scores.masked_fill(~self.causal, float("-inf"))
        attended = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(batch, context, -1)
        h = h + self.proj(attended)

        h = h + self.ff2(nn.functional.gelu(self.ff1(self.norm2(h))))
        return self.head(h).squeeze(-1)


@dataclass(frozen=True)
class CausalTransformer:
    """The Mackey--Glass predictor, evaluated functionally.

    Attributes:
        context: $L-1$, inputs and outputs per block.
        d_model, n_heads, d_ff: the block's widths.
        dtype: parameter dtype. float64 for the filter.
    """

    context: int = 31
    d_model: int = 16
    n_heads: int = 2
    d_ff: int = 32
    dtype: torch.dtype = torch.float32
    #: Where the module's *buffers* live. Parameters are an argument and carry their
    #: own placement; the positional encoding and causal mask do not, so the module
    #: is built where the run will evaluate it. Defaults to CPU, so every existing
    #: caller and test is unchanged.
    device: str | torch.device = "cpu"
    _module: nn.Module = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.context < 1:
            raise ModelError(f"context must be >= 1, got {self.context}")
        if self.d_model < 2 or self.d_model % 2:
            raise ModelError(f"d_model must be even and >= 2, got {self.d_model}")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ModelError(f"n_heads={self.n_heads} must divide d_model={self.d_model}")
        if self.d_ff < 1:
            raise ModelError(f"d_ff must be >= 1, got {self.d_ff}")
        object.__setattr__(
            self,
            "_module",
            _CausalBlock(
                self.context, self.d_model, self.n_heads, self.d_ff, self.dtype, self.device
            ),
        )

    # -- shape ------------------------------------------------------------- #

    @property
    def input_dim(self) -> int:
        return self.context

    @property
    def output_dim(self) -> int:
        """$q$: one prediction per position."""
        return self.context

    @property
    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self._module.parameters())

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def names(self) -> tuple[str, ...]:
        return F.parameter_names(self._module)

    @property
    def shapes(self) -> tuple[torch.Size, ...]:
        return tuple(parameter.shape for _, parameter in self._module.named_parameters())

    # -- parameters -------------------------------------------------------- #

    def init_params(self, generator: torch.Generator | None = None) -> ParamDict:
        """Xavier-uniform weights, zero biases, unit LayerNorm scales.

        Drawn from an explicit generator, as `mlp.py` does, so every agent can be
        handed the same $\\bm\\theta_0$ (design notes D9, D18).
        """
        params: ParamDict = {}
        for name, template in self._module.named_parameters():
            # Built explicitly rather than with `*_like`, which would inherit the
            # module's device. The module may now live on a GPU (D104), while these
            # must not: a cuda tensor needs a cuda generator, which draws a different
            # random stream, and theta_0 would then depend on the device -- breaking
            # reproducibility and the D9/D18 requirement that every agent start from
            # the *same* vector. init_params is CPU-only by contract; `run_one` moves
            # the flat vector once it is assembled.
            if name.startswith(("norm1", "norm2")):
                fill = 1.0 if name.endswith("weight") else 0.0
                params[name] = torch.full(template.shape, fill, dtype=self.dtype)
            elif name.endswith("bias"):
                params[name] = torch.zeros(template.shape, dtype=self.dtype)
            else:
                fan_out, fan_in = template.shape
                bound = math.sqrt(6.0 / (fan_in + fan_out))
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

    # -- evaluation -------------------------------------------------------- #

    def forward(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        """``(n_blocks, context)`` inputs to ``(n_blocks, context)`` predictions."""
        return F.call(self._module, params, x)

    def vjp(self, params: ParamDict, x: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        return self.flatten(F.vector_jacobian_product(self._module, params, x, cotangent))

    def jvp(self, params: ParamDict, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return F.jacobian_vector_product(self._module, params, x, self.unflatten(tangent))

    def jacobian(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        """The full $\\bm H$. Tests only."""
        return F.jacobian(self._module, params, x, self.names)

    def per_sample_jacobian(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        """Every block's $\\bm H$, shape ``(n_blocks, context, p)``."""
        return F.per_sample_jacobian(self._module, params, x, self.names)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "causal_transformer",
            "context": self.context,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "positions": "sinusoidal",
            "norm": "pre-layernorm",
            "activation": "gelu",
            "num_params": self.num_params,
            "dtype": str(self.dtype),
            "dense_covariance_entries": self.num_params**2,
        }
