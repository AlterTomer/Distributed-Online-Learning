r"""The coupled-network GNN: a causal decoder per node, then graph layers (D3c, D139).

Per node $v$, at every position $t$ of its 31-sample window:

1. **encoder** -- the Mackey--Glass causal Transformer decoder block, without its head
   (`transformer._CausalBlock.features`): $\bm h_v^{(0)}$, ``(31, d)``;
2. **graph layers** $l=1..L$, aggregate then transform, per position:
   $\mathrm{agg}^{(l)}_v=\sum_{u\ne v}a_{vu}\bm h_u^{(l-1)}$ with the Metropolis weights the
   physics couples by, and $\bm h_v^{(l)}=\bm h_v^{(l-1)}+\mathrm{gelu}(\bm W^{(l)}_{\text{self}}
   \mathrm{LN}(\bm h_v^{(l-1)})+\bm W^{(l)}_{\text{nbr}}\mathrm{agg}^{(l)}_v)$;
3. **head** -- one prediction per position, causal: position $t$ only ever sees
   positions $\le t$, its own and (through the aggregates) its neighbours'.

**The per-node model takes the aggregates as inputs.** Its input is *packed*: the window
and one aggregate per layer per position, ``(31, 1 + L d)``. So
:meth:`CoupledGNN.per_sample_jacobian` differentiates through the node's own
computation and holds what it received fixed -- the **truncated** Jacobian every
distributed learner uses (decided 2026-10-07). :meth:`CoupledGNN.network_forward`
computes what each node actually receives, each from its own parameters; with one
shared $\bm\theta$, :meth:`CoupledGNN.network_jacobian` is the **exact** Jacobian through
every path in the graph, the centralised learners'.

Functional, like every model here: parameters are an argument, the module a template.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap

from dekf_bench.models import functional as F
from dekf_bench.models.base import ModelError, ParamDict, ParamGroup
from dekf_bench.models.transformer import _CausalBlock


class _GraphLayer(nn.Module):
    def __init__(self, d_model: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, dtype=dtype)
        self.self_ = nn.Linear(d_model, d_model, dtype=dtype)
        # No bias: the self transform carries one, and two would be one parameter twice.
        self.nbr = nn.Linear(d_model, d_model, bias=False, dtype=dtype)

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return h + nn.functional.gelu(self.self_(self.norm(h)) + self.nbr(agg))


class _GNNModule(nn.Module):
    def __init__(self, context: int, d_model: int, n_heads: int, d_ff: int, layers: int,
                 dtype: torch.dtype, device: str | torch.device) -> None:
        super().__init__()
        self.context, self.d_model, self.layers = context, d_model, layers
        self.encoder = _CausalBlock(context, d_model, n_heads, d_ff, dtype, device,
                                    with_head=False)
        self.graph = nn.ModuleList(_GraphLayer(d_model, dtype) for _ in range(layers))
        self.head = nn.Linear(d_model, 1, dtype=dtype)
        self.to(device)

    def _split(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = packed[..., 0]
        aggs = packed[..., 1:].reshape(*packed.shape[:-1], self.layers, self.d_model)
        return window, aggs

    def forward(self, packed: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        """Packed ``(b, context, 1 + L d)`` to predictions ``(b, context)``; with ``upto``
        = l, the embedding $\\bm h^{(l)}$ a node sends for layer $l+1$ instead."""
        window, aggs = self._split(packed)
        h = self.encoder.features(window)
        for index, layer in enumerate(self.graph):
            if upto is not None and index == upto:
                return h
            h = layer(h, aggs[..., index, :])
        if upto is not None:
            return h
        return self.head(h).squeeze(-1)

    def network(self, windows: torch.Tensor, off_weights: torch.Tensor) -> torch.Tensor:
        """One snapshot, one shared $\\bm\\theta$: ``(N, context)`` to ``(N, context)``,
        through every path in the graph -- the centralised model."""
        h = self.encoder.features(windows)
        for layer in self.graph:
            h = layer(h, torch.einsum("vu,ucd->vcd", off_weights, h))
        return self.head(h).squeeze(-1)


def off_diagonal(weights: torch.Tensor) -> torch.Tensor:
    """$a_{vu}$ for $u\\ne v$: the aggregation's weights (the self term is $\\bm W_{\\text{self}}$'s)."""
    out = weights.clone()
    out.fill_diagonal_(0)
    return out


@dataclass(frozen=True)
class CoupledGNN:
    """The coupled-network predictor, evaluated functionally.

    Attributes:
        context: $L-1$, inputs and outputs per block (31).
        d_model, n_heads, d_ff: the encoder's widths; the graph layers are $d\\times d$.
        layers: graph layers, 2 to start (D139), configurable.
    """

    context: int = 31
    d_model: int = 40
    n_heads: int = 2
    d_ff: int = 80
    layers: int = 2
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"
    _module: nn.Module = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.layers < 1:
            raise ModelError(f"a GNN needs at least one graph layer, got {self.layers}")
        if self.d_model < 2 or self.d_model % 2 or self.d_model % self.n_heads:
            raise ModelError(f"d_model={self.d_model} must be even and divisible by n_heads")
        object.__setattr__(self, "_module", _GNNModule(
            self.context, self.d_model, self.n_heads, self.d_ff, self.layers, self.dtype,
            self.device))

    # -- shape --------------------------------------------------------------- #

    @property
    def input_dim(self) -> int:
        return self.context

    @property
    def packed_width(self) -> int:
        """Per position: the window's sample, then one aggregate per layer."""
        return 1 + self.layers * self.d_model

    @property
    def output_dim(self) -> int:
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

    # -- parameters ------------------------------------------------------------ #

    def init_params(self, generator: torch.Generator | None = None) -> ParamDict:
        """Xavier-uniform weights, zero biases, unit LayerNorm scales; CPU, as the
        Transformer's (D104)."""
        params: ParamDict = {}
        for name, template in self._module.named_parameters():
            if ".norm" in name or name.startswith("norm"):
                fill = 1.0 if name.endswith("weight") else 0.0
                params[name] = torch.full(template.shape, fill, dtype=self.dtype)
            elif name.endswith("bias"):
                params[name] = torch.zeros(template.shape, dtype=self.dtype)
            else:
                fan_out, fan_in = template.shape
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                params[name] = torch.empty(template.shape, dtype=self.dtype).uniform_(
                    -bound, bound, generator=generator)
        return params

    def flatten(self, params: ParamDict) -> torch.Tensor:
        return F.flatten(params, self.names)

    def unflatten(self, vector: torch.Tensor) -> ParamDict:
        return F.unflatten(vector, self.names, self.shapes)

    def param_groups(self) -> tuple[ParamGroup, ...]:
        return F.build_param_groups(self._module, self.names)

    # -- the per-node model (truncated) ----------------------------------------- #

    def pack(self, window: torch.Tensor, aggregates: list[torch.Tensor]) -> torch.Tensor:
        """``(b, context)`` and L aggregates ``(b, context, d)`` to ``(b, context, 1 + L d)``;
        missing trailing layers are zeros (they are not read before they exist)."""
        missing = self.layers - len(aggregates)
        zeros = window.new_zeros(*window.shape, self.d_model)
        parts = [window.unsqueeze(-1), *aggregates, *([zeros] * missing)]
        return torch.cat(parts, dim=-1)

    def forward(self, params: ParamDict, packed: torch.Tensor) -> torch.Tensor:
        """``(b, context, 1 + L d)`` to ``(b, context)``: the node's prediction."""
        return F.call(self._module, params, packed)

    def embedding(self, params: ParamDict, packed: torch.Tensor, layer: int) -> torch.Tensor:
        """$\\bm h^{(l)}$, what a node sends for layer $l+1$'s aggregate, ``(b, context, d)``."""
        return functional_call(self._module, params, (packed,), {"upto": layer})

    def per_sample_jacobian(self, params: ParamDict, packed: torch.Tensor) -> torch.Tensor:
        """Every block's truncated $\\bm H$, ``(n, context, p)``: the aggregates are inputs."""
        return F.per_sample_jacobian(self._module, params, packed, self.names)

    def jacobian(self, params: ParamDict, packed: torch.Tensor) -> torch.Tensor:
        return F.jacobian(self._module, params, packed, self.names)

    # -- the network ------------------------------------------------------------- #

    def network_forward(self, params: ParamDict | list[ParamDict], windows: torch.Tensor,
                        weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The distributed forward pass: every node with its own parameters.

        ``windows`` ``(b, N, context)``; ``params`` one ParamDict per node, or one shared.
        Returns the predictions ``(b, N, context)`` and every node's packed input
        ``(b, N, context, 1 + L d)`` -- what it received, which is what a node's
        truncated update and a one-hop neighbour's message are built from.
        """
        n = windows.shape[1]
        per_node = params if isinstance(params, list) else [params] * n
        if len(per_node) != n:
            raise ModelError(f"{len(per_node)} parameter sets for {n} nodes")
        off = off_diagonal(weights).to(dtype=windows.dtype, device=windows.device)
        received: list[list[torch.Tensor]] = [[] for _ in range(n)]
        for layer in range(self.layers):
            sent = torch.stack([self.embedding(per_node[v], self.pack(windows[:, v], received[v]),
                                               layer) for v in range(n)], dim=1)
            aggregate = torch.einsum("vu,bucd->bvcd", off, sent)
            for v in range(n):
                received[v].append(aggregate[:, v])
        packed = torch.stack([self.pack(windows[:, v], received[v]) for v in range(n)], dim=1)
        predictions = torch.stack([self.forward(per_node[v], packed[:, v]) for v in range(n)],
                                  dim=1)
        return predictions, packed

    def network_jacobian(self, params: ParamDict, windows: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
        """The exact $\\bm H$ of every snapshot under one shared $\\bm\\theta$, through every
        path in the graph: ``(b, N context, p)`` -- the centralised learners'."""
        off = off_diagonal(weights).to(dtype=windows.dtype, device=windows.device)

        def network(flat: torch.Tensor, snapshot: torch.Tensor) -> torch.Tensor:
            return _call_network(self._module, self.unflatten(flat), snapshot, off).reshape(-1)

        flat = self.flatten(params)
        return vmap(jacrev(network), in_dims=(None, 0))(flat, windows)

    def network_predict(self, params: ParamDict, windows: torch.Tensor,
                        weights: torch.Tensor) -> torch.Tensor:
        """One shared $\\bm\\theta$: ``(b, N, context)`` to ``(b, N, context)``."""
        off = off_diagonal(weights).to(dtype=windows.dtype, device=windows.device)
        return vmap(lambda snapshot: _call_network(self._module, params, snapshot, off))(windows)

    def summary(self) -> dict[str, Any]:
        return {"kind": "coupled_gnn", "context": self.context, "d_model": self.d_model,
                "n_heads": self.n_heads, "d_ff": self.d_ff, "layers": self.layers,
                "aggregation": "sum_u a_vu h_u (Metropolis, off-diagonal)",
                "num_params": self.num_params, "dtype": str(self.dtype)}


class _NetworkCall(nn.Module):
    """Routes ``functional_call`` to ``_GNNModule.network`` (it only calls ``forward``)."""

    def __init__(self, inner: _GNNModule) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, windows: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        return self.inner.network(windows, off)


def _call_network(module: _GNNModule, params: ParamDict, snapshot: torch.Tensor,
                  off: torch.Tensor) -> torch.Tensor:
    wrapper = _NetworkCall(module)
    return functional_call(wrapper, {f"inner.{k}": v for k, v in params.items()},
                           (snapshot, off))
