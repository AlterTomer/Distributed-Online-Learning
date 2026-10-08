r"""The GNN's place in the run: what each learner's forward pass sends (D3c, D139).

A GNN node predicts from its own window **and** the aggregates its neighbours' embeddings
make, each computed with the sender's own parameters. So the data a learner trains and
is scored on depends on that learner: this module builds it, and nothing else in the
run needs to know a GNN exists.

* :func:`pack_observations` -- one step's observations for one learner: every node's
  window packed with the aggregates computed from the learner's own per-node
  parameters (the forward-communication round). Prequential scoring and ``adapt``
  then run unchanged: a distributed learner's Jacobian or gradient on a packed input
  is the **truncated** one, and the receiver-point one-hop ships a neighbour's packed
  input with its target -- the window, the target and what it received, which is the
  GNN one-hop as decided.
* :class:`NetworkModel` -- the centralised learners' model for ``adapt_pooled``: the
  pooled snapshot through every path of the graph, one shared $\bm\theta$, the
  **exact** Jacobian. Their per-node predictions stay on the per-node model, which
  under a shared $\bm\theta$ gives the same values (tested).
* :class:`NetworkEvalSets` -- held-out network snapshots packed for one learner, handed
  to the protocol one node at a time.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch.func import vjp as func_vjp

from dekf_bench.evaluation.series_evalsets import SeriesEvalSet


def is_network_model(model: Any) -> bool:
    return hasattr(model, "network_forward")


def forward_scalars(model: Any, observations: dict[int, Any], n_edges: int) -> int:
    """Scalars one step's forward pass sends: per layer, every node's embedding over
    every link both ways -- ``layers * 2|E| * blocks * context * d``."""
    blocks = int(next(iter(observations.values())).x.shape[0])
    return model.layers * 2 * n_edges * blocks * model.context * model.d_model


def _node_params(learner: Any, model: Any, nodes: list[int]) -> list[Any]:
    return [model.unflatten(learner.flat_params(node).detach()) for node in nodes]


def pack_observations(learner: Any, observations: dict[int, Any],
                      weights: torch.Tensor) -> dict[int, Any]:
    """The step as ``learner`` sees it: copies of the observations, each ``x`` packed.

    Messages come from each node's current parameters -- the posterior of the last
    step, the same point ``predict`` uses. Values only: nothing here is differentiated.
    """
    model = learner.model
    nodes = sorted(observations)
    windows = torch.stack([observations[v].x for v in nodes], dim=1)      # (b, N, context)
    with torch.no_grad():
        _pred, packed = model.network_forward(_node_params(learner, model, nodes), windows,
                                              weights)
    return {v: dataclasses.replace(observations[v], x=packed[:, i]) for i, v in enumerate(nodes)}


@dataclasses.dataclass(frozen=True)
class NetworkModel:
    """The coupled network under one $\\bm\\theta$, on the pooled batch (node-major rows).

    ``x`` is ``environment.pool``'s stack, ``(N b, context)``, node by node; every
    output row is one node's block, so the information pair and the gradient treat a
    node's outputs as one sample -- exact, since the nodes' noise is independent.
    """

    gnn: Any
    weights: torch.Tensor
    n_nodes: int

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gnn, name)

    def _snapshots(self, x: torch.Tensor) -> torch.Tensor:
        blocks = x.shape[0] // self.n_nodes
        return x.reshape(self.n_nodes, blocks, -1).transpose(0, 1)        # (b, N, context)

    def _rows(self, out: torch.Tensor) -> torch.Tensor:
        return out.transpose(0, 1).reshape(-1, *out.shape[2:])            # node-major

    def forward(self, params: Any, x: torch.Tensor) -> torch.Tensor:
        return self._rows(self.gnn.network_predict(params, self._snapshots(x), self.weights))

    def per_sample_jacobian(self, params: Any, x: torch.Tensor) -> torch.Tensor:
        snapshots = self._snapshots(x)
        jac = self.gnn.network_jacobian(params, snapshots, self.weights)  # (b, N ctx, p)
        b, _rows, p = jac.shape
        return self._rows(jac.reshape(b, self.n_nodes, self.gnn.context, p))

    def vjp(self, params: Any, x: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        flat = self.gnn.flatten(params)
        _out, pullback = func_vjp(lambda f: self.forward(self.gnn.unflatten(f), x), flat)
        return pullback(cotangent)[0]


class NetworkEvalSets:
    """Held-out network snapshots, packed for one learner and served per node.

    ``at(name, step)`` without a node is None, which sends the protocol down its
    per-node path; ``at(name, step, node)`` is that node's packed inputs and targets.
    Packed once per (set, step) and dropped when the step moves on.
    """

    per_node = True

    def __init__(self, base: Any, learner: Any, weights: torch.Tensor) -> None:
        self._base, self._learner, self._weights = base, learner, weights
        self._packed: dict[tuple[str, int], tuple[Any, torch.Tensor]] = {}

    def at(self, name: str, step: int, node: int | None = None) -> SeriesEvalSet | None:
        if node is None:
            return None
        key = (name, step)
        if key not in self._packed:
            self._packed = {k: v for k, v in self._packed.items() if k[1] == step}
            base = self._base.at(name, step)
            if base is None:
                return None
            model = self._learner.model
            nodes = list(range(base.inputs.shape[1]))
            with torch.no_grad():
                _pred, packed = model.network_forward(
                    _node_params(self._learner, model, nodes), base.inputs, self._weights)
            self._packed[key] = (base, packed)
        base, packed = self._packed[key]
        return SeriesEvalSet(name=base.name, inputs=packed[:, node], targets=base.targets[:, node],
                             rotation_degrees=base.rotation_degrees, step=base.step)
