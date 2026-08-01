"""Communication topologies, combination weights, and graph diagnostics.

Two matrices describe the network, and keeping them distinct is the single most
important thing in this module.

**Adjacency** $\\bm E$ is who talks to whom. Its diagonal is **zero**: an agent
does not send itself a message, and counting a self-edge would inflate the
communication ledger by $N$ vectors per step.

**Combination weights** $\\bm A = [a_{vu}]$ are how much an agent trusts each
estimate it holds. Its diagonal is **strictly positive**: agent $v$ always keeps
some of its own estimate, because it just computed that estimate from its own
data. Metropolis weights guarantee $a_{vv} \\ge 1/(1+d_v) > 0$, so an agent can
never discard itself entirely.

Conflating the two is easy and produces two distinct failure modes: a zero
diagonal in $\\bm A$ makes an agent throw away the update it just computed, and
a non-zero diagonal in $\\bm E$ bills the run for messages nobody sent.

**Two graphs, not one.** :class:`Graphs` holds ``comm`` and ``data`` separately.
The *communication* graph carries diffusion messages; the *data-coupling* graph
is the one a predictor's forward pass would exchange information over. For every
Class L architecture -- MLP, CNN, RNN, Transformer -- the data graph is
**empty**, and it stays empty for this whole project phase. It exists now because
retrofitting a second graph into an environment that assumed one touches every
file (IMPLEMENTATION.md section 13.11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import torch

#: How many times to redraw a random graph that comes out disconnected before
#: giving up. Small: if p is so low that 20 draws all fail, the parameters are
#: wrong and silently retrying forever would hide that.
MAX_CONNECTIVITY_ATTEMPTS = 20


class GraphError(ValueError):
    """Raised for an unbuildable topology or an invalid weight matrix."""


# --------------------------------------------------------------------------- #
# the graph object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Graph:
    """A fixed, undirected graph over ``n_nodes`` agents.

    Attributes:
        adjacency: ``(N, N)`` float, symmetric, **zero diagonal**, entries in
            ``{0, 1}``. Who exchanges messages with whom.
        weights: ``(N, N)`` row-stochastic combination weights with a strictly
            positive diagonal, or ``None`` for a graph nothing is combined over
            (the data graph). ``weights[v, u] > 0`` requires ``u == v`` or
            ``adjacency[v, u] == 1``.
        topology: the name it was built from, for logging and error messages.
    """

    adjacency: torch.Tensor
    weights: torch.Tensor | None
    topology: str

    def __post_init__(self) -> None:
        adjacency = self.adjacency
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise GraphError(f"adjacency must be square, got {tuple(adjacency.shape)}")
        if not torch.equal(adjacency, adjacency.T):
            raise GraphError(f"{self.topology}: adjacency must be symmetric")
        if bool(torch.any(torch.diagonal(adjacency) != 0)):
            raise GraphError(
                f"{self.topology}: adjacency has a non-zero diagonal. An agent does not "
                "send itself a message; self-weight lives in the combination matrix, not "
                "in the topology."
            )
        if not bool(torch.all((adjacency == 0) | (adjacency == 1))):
            raise GraphError(f"{self.topology}: adjacency entries must be 0 or 1")
        if self.weights is not None:
            self._validate_weights(self.weights, adjacency, self.topology)

    @staticmethod
    def _validate_weights(weights: torch.Tensor, adjacency: torch.Tensor, topology: str) -> None:
        if weights.shape != adjacency.shape:
            raise GraphError(
                f"{topology}: weights {tuple(weights.shape)} do not match adjacency "
                f"{tuple(adjacency.shape)}"
            )
        if bool(torch.any(weights < 0)):
            raise GraphError(f"{topology}: combination weights must be non-negative")

        row_sums = weights.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-10):
            worst = int(torch.argmax((row_sums - 1).abs()))
            raise GraphError(
                f"{topology}: weights must be row-stochastic; row {worst} sums to "
                f"{float(row_sums[worst])!r}"
            )
        if bool(torch.any(torch.diagonal(weights) <= 0)):
            raise GraphError(
                f"{topology}: every agent must keep a positive share of its own estimate, "
                "or it discards the update it just computed from its own data"
            )

        # A weight may be non-zero only on an edge or on the diagonal.
        allowed = adjacency.bool() | torch.eye(
            adjacency.shape[0], dtype=torch.bool, device=adjacency.device
        )
        if bool(torch.any((weights > 0) & ~allowed)):
            offenders = torch.nonzero((weights > 0) & ~allowed)[0].tolist()
            raise GraphError(
                f"{topology}: weight {offenders} is non-zero but those agents are not "
                "neighbours -- the combination step would use an estimate it never received"
            )

    # -- basic structure ---------------------------------------------------- #

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def degrees(self) -> torch.Tensor:
        """Number of neighbours per agent, excluding itself."""
        return self.adjacency.sum(dim=1).to(torch.int64)

    @property
    def n_edges(self) -> int:
        """Undirected edges. Self-loops do not exist, so this is not inflated."""
        return int(self.adjacency.sum() // 2)

    def neighbours(self, node: int) -> torch.Tensor:
        """The one-hop neighbourhood of ``node``, **excluding** ``node``."""
        return torch.nonzero(self.adjacency[node], as_tuple=True)[0]

    def closed_neighbourhood(self, node: int) -> torch.Tensor:
        """$\\mathcal N_v \\cup \\{v\\}$ -- whose estimates agent ``node`` combines."""
        return torch.cat([self.neighbours(node), torch.tensor([node])]).sort().values

    # -- diagnostics -------------------------------------------------------- #

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n_nodes))
        rows, cols = torch.nonzero(torch.triu(self.adjacency, diagonal=1), as_tuple=True)
        graph.add_edges_from(zip(rows.tolist(), cols.tolist(), strict=True))
        return graph

    @property
    def is_connected(self) -> bool:
        if self.n_nodes == 0:
            return False
        return nx.is_connected(self.to_networkx())

    @property
    def n_components(self) -> int:
        return nx.number_connected_components(self.to_networkx())

    @property
    def diameter(self) -> int | None:
        """Longest shortest path, or ``None`` when disconnected.

        ``None`` rather than ``inf``: a disconnected graph has no diameter, and
        returning a float that silently propagates into a plot axis is worse
        than a value that has to be handled.
        """
        if not self.is_connected:
            return None
        return int(nx.diameter(self.to_networkx()))

    @property
    def is_doubly_stochastic(self) -> bool:
        """Whether the columns sum to one as well as the rows.

        Metropolis weights are; relative-degree and uniform weights are only on
        a *regular* graph, where every agent has the same degree.
        """
        if self.weights is None:
            return False
        column_sums = self.weights.sum(dim=0)
        return bool(torch.allclose(column_sums, torch.ones_like(column_sums), atol=1e-10))

    @property
    def spectral_gap(self) -> float:
        r"""$1 - \rho$ with $\rho = \lVert \bm A - \tfrac1N \bm 1\bm 1^{\mathsf T}\rVert_2$.

        The scalar summary of how fast information mixes, and the natural x-axis
        for "what does connectivity cost". A complete graph gives 1; a path
        gives something near 0.

        **Defined only for doubly stochastic weights.** The $\tfrac1N\bm 1\bm
        1^{\mathsf T}$ term is the projector onto the consensus direction only
        when the all-ones vector is a left eigenvector as well as a right one.
        For merely row-stochastic weights the norm can exceed 1 and this returns
        a *negative* number: relative-degree weights on a 10-star give $-1.56$,
        while the star in fact mixes very fast because the hub aggregates
        everything in one hop. Rather than hand back a number that is not merely
        imprecise but wrong in sign, this raises and points at
        :attr:`mixing_gap`, which is valid for any row-stochastic matrix.
        """
        if self.weights is None:
            raise GraphError(f"{self.topology}: spectral gap needs combination weights")
        if not self.is_doubly_stochastic:
            raise GraphError(
                f"{self.topology}: the spectral gap 1 - ||A - 11^T/N||_2 is only a mixing "
                "measure for doubly stochastic weights, and these are not "
                "(relative-degree and uniform weights are doubly stochastic only on a "
                "regular graph). It would return a negative value here. Use mixing_gap, "
                "or switch to metropolis weights."
            )
        n = self.n_nodes
        averaging = torch.full((n, n), 1.0 / n, dtype=self.weights.dtype)
        rho = torch.linalg.matrix_norm(self.weights - averaging, ord=2)
        return float(1.0 - rho)

    @property
    def slem(self) -> float:
        """Second-largest eigenvalue modulus of the weight matrix.

        The asymptotic rate at which disagreement decays under repeated
        combining, and unlike :attr:`spectral_gap` it is well defined for any
        row-stochastic matrix. The largest eigenvalue is always 1, with the
        all-ones right eigenvector -- that is consensus, and it is the *second*
        one that says how fast everything else dies away.
        """
        if self.weights is None:
            raise GraphError(f"{self.topology}: slem needs combination weights")
        if self.n_nodes == 1:
            return 0.0
        moduli = torch.linalg.eigvals(self.weights).abs().sort(descending=True).values
        return float(moduli[1])

    @property
    def mixing_gap(self) -> float:
        """$1 - \\text{SLEM}$. Always in $[0, 1]$, and equal to
        :attr:`spectral_gap` whenever the weights are doubly stochastic."""
        return 1.0 - self.slem

    def summary(self) -> dict[str, Any]:
        """Everything worth denormalizing into a log row.

        ``spectral_gap`` is ``None`` when the weights are not doubly stochastic,
        because the quantity is undefined there; ``mixing_gap`` is always
        populated and is the safe axis for a plot.
        """
        degrees = self.degrees
        has_weights = self.weights is not None
        return {
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "min_degree": int(degrees.min()) if self.n_nodes else 0,
            "max_degree": int(degrees.max()) if self.n_nodes else 0,
            "is_connected": self.is_connected,
            "n_components": self.n_components,
            "diameter": self.diameter,
            "is_doubly_stochastic": self.is_doubly_stochastic,
            "spectral_gap": self.spectral_gap if self.is_doubly_stochastic else None,
            "mixing_gap": self.mixing_gap if has_weights else None,
        }

    @classmethod
    def empty(cls, n_nodes: int, topology: str = "empty") -> Graph:
        """A graph with no edges and no combination weights.

        This is what the data-coupling graph is for every Class L architecture.
        ``weights`` is ``None`` rather than the identity, because nothing is
        combined over the data graph and an identity matrix would imply
        otherwise.
        """
        return cls(
            adjacency=torch.zeros(n_nodes, n_nodes, dtype=torch.float64),
            weights=None,
            topology=topology,
        )


@dataclass(frozen=True)
class Graphs:
    """The communication graph and the data-coupling graph, kept separate."""

    comm: Graph
    data: Graph

    def __post_init__(self) -> None:
        if self.comm.n_nodes != self.data.n_nodes:
            raise GraphError(
                f"comm graph has {self.comm.n_nodes} agents but data graph has "
                f"{self.data.n_nodes}"
            )

    @property
    def n_nodes(self) -> int:
        return self.comm.n_nodes


# --------------------------------------------------------------------------- #
# topologies
# --------------------------------------------------------------------------- #


def _complete(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    return nx.complete_graph(n)


def _ring(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    # nx.cycle_graph(1) returns a node with a SELF-LOOP, which would put a 1 on
    # the adjacency diagonal and bill the ledger for a message an agent sent
    # itself. Below three nodes a cycle is a path anyway.
    if n < 3:
        return nx.path_graph(n)
    return nx.cycle_graph(n)


def _path(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    return nx.path_graph(n)


def _star(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    if n == 0:
        return nx.empty_graph(0)
    # nx.star_graph(k) has k+1 nodes: one hub, k leaves. Node 0 is the hub.
    return nx.star_graph(n - 1)


def _grid2d(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    rows, cols = params.get("rows"), params.get("cols")
    if rows is None or cols is None:
        raise GraphError("grid2d needs 'rows' and 'cols' in graph.params")
    if rows * cols != n:
        raise GraphError(f"grid2d {rows}x{cols} holds {rows * cols} nodes but n_nodes is {n}")
    grid = nx.grid_2d_graph(rows, cols)
    # Relabel (r, c) -> r * cols + c, so node ids are row-major integers.
    return nx.relabel_nodes(grid, {(r, c): r * cols + c for r in range(rows) for c in range(cols)})


def _erdos_renyi(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    p = params.get("p")
    if p is None:
        raise GraphError("erdos_renyi needs 'p' in graph.params")
    if not 0.0 <= p <= 1.0:
        raise GraphError(f"erdos_renyi p must lie in [0, 1], got {p}")
    ensure_connected = params.get("ensure_connected", True)
    seed = _seed_from(generator)

    for attempt in range(MAX_CONNECTIVITY_ATTEMPTS):
        graph = nx.gnp_random_graph(n, p, seed=seed + attempt)
        if not ensure_connected or n <= 1 or nx.is_connected(graph):
            return graph
    raise GraphError(
        f"erdos_renyi with n={n}, p={p} produced a disconnected graph in "
        f"{MAX_CONNECTIVITY_ATTEMPTS} attempts. The connectivity threshold is about "
        f"ln(n)/n = {torch.log(torch.tensor(float(max(n, 2)))).item() / max(n, 1):.3f}; "
        "raise p, or set params.ensure_connected=false if a disconnected draw is intended."
    )


def _watts_strogatz(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    k = params.get("k", 4)
    beta = params.get("beta", 0.1)
    if k >= n:
        raise GraphError(f"watts_strogatz needs k < n_nodes, got k={k}, n={n}")
    ensure_connected = params.get("ensure_connected", True)
    seed = _seed_from(generator)

    for attempt in range(MAX_CONNECTIVITY_ATTEMPTS):
        graph = nx.watts_strogatz_graph(n, k, beta, seed=seed + attempt)
        if not ensure_connected or nx.is_connected(graph):
            return graph
    raise GraphError(f"watts_strogatz with n={n}, k={k}, beta={beta} stayed disconnected")


def _disconnected(n: int, params: dict[str, Any], generator: torch.Generator | None) -> nx.Graph:
    """The negative control: several complete components, no edge between them.

    Each component is internally complete, so *within* a component diffusion
    works perfectly. Any shortfall against the connected case is therefore
    attributable to the missing cross-component links and nothing else.
    """
    n_components = params.get("n_components", 2)
    if n_components < 1:
        raise GraphError(f"disconnected needs n_components >= 1, got {n_components}")
    if n_components > n:
        raise GraphError(f"disconnected: {n_components} components but only {n} agents")

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    start = 0
    for index in range(n_components):
        # Sizes differ by at most one, so no component is degenerate.
        size = n // n_components + (1 if index < n % n_components else 0)
        members = range(start, start + size)
        graph.add_edges_from((u, v) for u in members for v in members if u < v)
        start += size
    return graph


TOPOLOGY_BUILDERS = {
    "complete": _complete,
    "ring": _ring,
    "path": _path,
    "star": _star,
    "grid2d": _grid2d,
    "erdos_renyi": _erdos_renyi,
    "watts_strogatz": _watts_strogatz,
    "disconnected": _disconnected,
}


def _seed_from(generator: torch.Generator | None) -> int:
    if generator is None:
        return 0
    return int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())


def grid_dimensions(n_nodes: int) -> tuple[int, int]:
    """The most square ``(rows, cols)`` with ``rows * cols == n_nodes``.

    A prime $N$ degenerates to a $1 \\times N$ grid, which *is* a path -- worth
    knowing before reading a topology sweep that shows the two behaving
    identically.
    """
    rows = max(d for d in range(1, int(n_nodes**0.5) + 1) if n_nodes % d == 0)
    return rows, n_nodes // rows


def default_topology_params(n_nodes: int) -> dict[str, dict[str, Any]]:
    """Sensible parameters for every topology at a given size.

    Used by the inspection script and by sweeps, so the grid shape and the
    Erdos-Renyi density are chosen in one place rather than in each caller.
    """
    rows, cols = grid_dimensions(n_nodes)
    return {
        "complete": {},
        "ring": {},
        "path": {},
        "star": {},
        "grid2d": {"rows": rows, "cols": cols},
        # Twice the ln(n)/n connectivity threshold: reliably connected, but far
        # enough from 1 that the graph is still sparse and worth distinguishing
        # from `complete`. At N=10 this is p = 0.46.
        "erdos_renyi": {"p": min(1.0, 2.0 * math.log(max(n_nodes, 2)) / max(n_nodes, 2))},
        "watts_strogatz": {"k": min(4, max(2, n_nodes - 1)), "beta": 0.2},
        "disconnected": {"n_components": 2},
    }


# --------------------------------------------------------------------------- #
# combination weights
# --------------------------------------------------------------------------- #


def metropolis_weights(adjacency: torch.Tensor) -> torch.Tensor:
    r"""Metropolis--Hastings weights.

    $a_{vu} = 1/(1 + \max(d_v, d_u))$ on an edge, and $a_{vv}$ takes up the
    slack. Symmetric and doubly stochastic, and $a_{vv} \ge 1/(1+d_v) > 0$
    always, so no agent can discard its own estimate.
    """
    degrees = adjacency.sum(dim=1)
    pairwise_max = torch.maximum(degrees.unsqueeze(1), degrees.unsqueeze(0))
    weights = adjacency / (1.0 + pairwise_max)
    weights = weights * adjacency  # zero anything off an edge
    diagonal = 1.0 - weights.sum(dim=1)
    return weights + torch.diag(diagonal)


def relative_degree_weights(adjacency: torch.Tensor) -> torch.Tensor:
    r"""Relative-degree weights, $a_{vu} = d_u / \sum_{j \in \mathcal N_v \cup \{v\}} d_j$.

    Row-stochastic but generally not symmetric: an agent leans toward
    better-connected neighbours. An isolated agent has an all-zero denominator,
    so it keeps all of its own weight -- the only sensible reading of "combine
    with nobody".
    """
    degrees = adjacency.sum(dim=1)
    closed = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
    numerator = closed * degrees.unsqueeze(0)  # weight of u seen from v
    denominator = numerator.sum(dim=1, keepdim=True)

    isolated = (denominator.squeeze(1) == 0).nonzero(as_tuple=True)[0]
    weights = torch.where(denominator > 0, numerator / denominator.clamp(min=1e-30), numerator)
    for node in isolated.tolist():
        weights[node] = 0.0
        weights[node, node] = 1.0
    return weights


def uniform_weights(adjacency: torch.Tensor) -> torch.Tensor:
    r"""Uniform over the closed neighbourhood, $a_{vu} = 1/(1 + d_v)$.

    On a complete graph $d_v = N-1$, so this is exactly the $a_{vu} = 1/N$ the
    exactness check requires. Defining it over the closed neighbourhood rather
    than as a flat $1/N$ means it also respects a sparser adjacency instead of
    quietly assigning weight to agents that never sent anything.
    """
    closed = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
    return closed / closed.sum(dim=1, keepdim=True)


WEIGHT_RULES = {
    "metropolis": metropolis_weights,
    "relative_degree": relative_degree_weights,
    "uniform": uniform_weights,
}


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def build_graph(
    topology: str,
    n_nodes: int,
    weights: str = "metropolis",
    params: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> Graph:
    """Build one graph and its combination weights.

    Args:
        topology: one of :data:`TOPOLOGY_BUILDERS`.
        n_nodes: $N$.
        weights: one of :data:`WEIGHT_RULES`.
        params: topology-specific parameters, e.g. ``{"rows": 2, "cols": 5}``.
        generator: seeded generator for the random topologies. Comes from the
            ``graph`` seed stream, so a topology can be redrawn without
            disturbing the partition or the sample order.
        dtype: float64 by default. The weights are exact small rationals and the
            exactness check compares at 1e-12; float32 would spend a third of
            that budget on the combination step alone.
    """
    if topology not in TOPOLOGY_BUILDERS:
        raise GraphError(f"unknown topology {topology!r}; have {sorted(TOPOLOGY_BUILDERS)}")
    if weights not in WEIGHT_RULES:
        raise GraphError(f"unknown weight rule {weights!r}; have {sorted(WEIGHT_RULES)}")
    if n_nodes < 1:
        raise GraphError(f"n_nodes must be >= 1, got {n_nodes}")

    graph = TOPOLOGY_BUILDERS[topology](n_nodes, params or {}, generator)
    adjacency = _adjacency_from(graph, n_nodes, dtype)
    return Graph(
        adjacency=adjacency,
        weights=WEIGHT_RULES[weights](adjacency),
        topology=topology,
    )


def _adjacency_from(graph: nx.Graph, n_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    adjacency = torch.zeros(n_nodes, n_nodes, dtype=dtype)
    for u, v in graph.edges():
        if u == v:
            # networkx will not produce these for the builders here, but a
            # self-loop reaching the adjacency would corrupt every degree.
            raise GraphError(f"builder produced a self-loop at node {u}")
        adjacency[u, v] = 1.0
        adjacency[v, u] = 1.0
    return adjacency


def build_graphs(config: Any, generator: torch.Generator | None = None) -> Graphs:
    """The communication and data graphs for a run.

    The data graph is empty and stays that way: MNIST with an MLP is Class L, so
    every agent's forward pass is purely local.
    """
    comm = build_graph(
        topology=config.graph.topology,
        n_nodes=config.graph.n_nodes,
        weights=config.graph.weights,
        params=config.graph.params,
        generator=generator,
    )
    return Graphs(comm=comm, data=Graph.empty(config.graph.n_nodes, topology="class_l_empty"))
