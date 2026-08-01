"""Topologies, combination weights and graph diagnostics.

Organised in four layers:

1. invariants every topology must satisfy, run parametrically over all of them;
2. the exact structure of each named topology -- edge sets, degrees, diameters,
   counted by hand rather than by re-running the builder;
3. the weight rules, checked against hand-computed matrices;
4. degenerate inputs and rejected malformed graphs.

The distinction the whole module turns on -- **zero diagonal in the adjacency,
positive diagonal in the weights** -- is asserted in layer 1 and again, from the
other side, in layer 3.
"""

from __future__ import annotations

import math

import pytest
import torch

from dekf_bench.env.graph import (
    TOPOLOGY_BUILDERS,
    Graph,
    GraphError,
    Graphs,
    build_graph,
    build_graphs,
    metropolis_weights,
    relative_degree_weights,
    uniform_weights,
)
from dekf_bench.utils.config import load_config

N = 10

#: Every topology, with parameters that make sense at N=10.
ALL_TOPOLOGIES = [
    ("complete", {}),
    ("ring", {}),
    ("path", {}),
    ("star", {}),
    ("grid2d", {"rows": 2, "cols": 5}),
    ("erdos_renyi", {"p": 0.4}),
    ("watts_strogatz", {"k": 4, "beta": 0.2}),
    ("disconnected", {"n_components": 2}),
]

CONNECTED_TOPOLOGIES = [entry for entry in ALL_TOPOLOGIES if entry[0] != "disconnected"]

WEIGHT_RULES = ["metropolis", "relative_degree", "uniform"]


def make(topology: str, params: dict, n: int = N, weights: str = "metropolis") -> Graph:
    generator = torch.Generator().manual_seed(0)
    return build_graph(topology, n, weights, params, generator)


def weights_of(graph: Graph) -> torch.Tensor:
    """The combination weights, narrowed from ``Tensor | None``.

    ``Graph.weights`` is optional because the data graph has none, but every
    graph ``build_graph`` returns has them. This asserts that once, so the tests
    below read as tensor arithmetic rather than as None-handling.
    """
    assert graph.weights is not None, f"{graph.topology}: build_graph must populate weights"
    return graph.weights


# =========================================================================== #
# 1. invariants that hold for every topology
# =========================================================================== #


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
@pytest.mark.parametrize("rule", WEIGHT_RULES)
def test_adjacency_has_no_self_loops(topology: str, params: dict, rule: str) -> None:
    """An agent does not send itself a message.

    A self-edge here would inflate the communication ledger by one p-vector per
    agent per step -- N extra vectors that were never transmitted.
    """
    graph = make(topology, params, weights=rule)
    assert bool(torch.all(torch.diagonal(graph.adjacency) == 0))


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
@pytest.mark.parametrize("rule", WEIGHT_RULES)
def test_weights_have_a_strictly_positive_diagonal(topology: str, params: dict, rule: str) -> None:
    """The other half of the same distinction.

    An agent always keeps some of its own estimate -- it just computed that
    estimate from its own data. A zero here would throw that update away.
    """
    graph = make(topology, params, weights=rule)
    assert graph.weights is not None
    assert bool(torch.all(torch.diagonal(graph.weights) > 0))


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_adjacency_is_symmetric(topology: str, params: dict) -> None:
    graph = make(topology, params)
    assert torch.equal(graph.adjacency, graph.adjacency.T)


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_adjacency_is_binary(topology: str, params: dict) -> None:
    graph = make(topology, params)
    assert bool(torch.all((graph.adjacency == 0) | (graph.adjacency == 1)))


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
@pytest.mark.parametrize("rule", WEIGHT_RULES)
def test_weights_are_row_stochastic(topology: str, params: dict, rule: str) -> None:
    graph = make(topology, params, weights=rule)
    assert graph.weights is not None
    row_sums = graph.weights.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-12)


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
@pytest.mark.parametrize("rule", WEIGHT_RULES)
def test_weights_are_non_negative(topology: str, params: dict, rule: str) -> None:
    graph = make(topology, params, weights=rule)
    assert graph.weights is not None
    assert bool(torch.all(graph.weights >= 0))


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
@pytest.mark.parametrize("rule", WEIGHT_RULES)
def test_weights_are_zero_off_the_closed_neighbourhood(
    topology: str, params: dict, rule: str
) -> None:
    """Weight on a non-neighbour means combining an estimate never received."""
    graph = make(topology, params, weights=rule)
    assert graph.weights is not None
    allowed = graph.adjacency.bool() | torch.eye(graph.n_nodes, dtype=torch.bool)
    assert not bool(torch.any((graph.weights > 0) & ~allowed))


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_degrees_agree_with_the_adjacency(topology: str, params: dict) -> None:
    graph = make(topology, params)
    assert torch.equal(graph.degrees, graph.adjacency.sum(dim=1).to(torch.int64))


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_edge_count_is_half_the_adjacency_sum(topology: str, params: dict) -> None:
    """Undirected edges. Each carries one p-vector per direction per step, so
    the ledger bills 2 * n_edges vectors per round."""
    graph = make(topology, params)
    assert graph.n_edges == int(graph.adjacency.sum()) // 2
    assert graph.n_edges == int(graph.degrees.sum()) // 2


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_neighbours_exclude_self_but_closed_neighbourhood_includes_it(
    topology: str, params: dict
) -> None:
    graph = make(topology, params)
    for node in range(graph.n_nodes):
        neighbours = graph.neighbours(node)
        closed = graph.closed_neighbourhood(node)
        assert node not in neighbours.tolist()
        assert node in closed.tolist()
        assert len(closed) == len(neighbours) + 1
        assert len(neighbours) == int(graph.degrees[node])


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_neighbour_relation_is_mutual(topology: str, params: dict) -> None:
    graph = make(topology, params)
    for node in range(graph.n_nodes):
        for neighbour in graph.neighbours(node).tolist():
            assert node in graph.neighbours(neighbour).tolist()


@pytest.mark.parametrize(("topology", "params"), CONNECTED_TOPOLOGIES)
def test_connected_topologies_are_connected(topology: str, params: dict) -> None:
    graph = make(topology, params)
    assert graph.is_connected
    assert graph.n_components == 1


@pytest.mark.parametrize(("topology", "params"), CONNECTED_TOPOLOGIES)
def test_diameter_is_within_the_theoretical_bounds(topology: str, params: dict) -> None:
    """At least 1 for any graph with an edge, at most N-1 for any connected one."""
    graph = make(topology, params)
    diameter = graph.diameter
    assert diameter is not None
    assert 1 <= diameter <= graph.n_nodes - 1


@pytest.mark.parametrize(("topology", "params"), CONNECTED_TOPOLOGIES)
def test_no_isolated_agent_in_a_connected_topology(topology: str, params: dict) -> None:
    graph = make(topology, params)
    assert int(graph.degrees.min()) >= 1


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
@pytest.mark.parametrize("rule", WEIGHT_RULES)
def test_mixing_gap_lies_in_the_unit_interval(topology: str, params: dict, rule: str) -> None:
    """Valid for every row-stochastic weight matrix, unlike the spectral gap."""
    graph = make(topology, params, weights=rule)
    assert -1e-12 <= graph.mixing_gap <= 1.0 + 1e-12


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_spectral_gap_lies_in_the_unit_interval_for_metropolis(topology: str, params: dict) -> None:
    graph = make(topology, params, weights="metropolis")
    assert 0.0 <= graph.spectral_gap <= 1.0 + 1e-12


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_summary_reports_every_field(topology: str, params: dict) -> None:
    summary = make(topology, params).summary()
    assert set(summary) == {
        "topology",
        "n_nodes",
        "n_edges",
        "min_degree",
        "max_degree",
        "is_connected",
        "n_components",
        "diameter",
        "is_doubly_stochastic",
        "spectral_gap",
        "mixing_gap",
    }


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_summary_omits_the_spectral_gap_when_it_is_undefined(topology: str, params: dict) -> None:
    """A plot axis must never receive a number that is wrong in sign."""
    summary = make(topology, params, weights="relative_degree").summary()
    if not summary["is_doubly_stochastic"]:
        assert summary["spectral_gap"] is None
    assert summary["mixing_gap"] is not None


# =========================================================================== #
# 2. exact structure, per topology
# =========================================================================== #


def test_complete_graph_structure() -> None:
    graph = make("complete", {})
    assert graph.n_edges == N * (N - 1) // 2 == 45
    assert torch.all(graph.degrees == N - 1)
    assert graph.diameter == 1
    off_diagonal = graph.adjacency + torch.eye(N, dtype=graph.adjacency.dtype)
    assert bool(torch.all(off_diagonal == 1)), "every pair must be adjacent"


def test_ring_connects_each_node_to_its_two_cyclic_neighbours() -> None:
    graph = make("ring", {})
    for node in range(N):
        expected = sorted([(node - 1) % N, (node + 1) % N])
        assert sorted(graph.neighbours(node).tolist()) == expected


def test_ring_structure() -> None:
    graph = make("ring", {})
    assert graph.n_edges == N
    assert torch.all(graph.degrees == 2)
    assert graph.diameter == N // 2 == 5


def test_path_connects_each_node_to_its_linear_neighbours() -> None:
    graph = make("path", {})
    for node in range(N):
        expected = [n for n in (node - 1, node + 1) if 0 <= n < N]
        assert sorted(graph.neighbours(node).tolist()) == expected


def test_path_structure() -> None:
    graph = make("path", {})
    assert graph.n_edges == N - 1
    assert int(graph.degrees[0]) == 1 and int(graph.degrees[-1]) == 1
    assert torch.all(graph.degrees[1:-1] == 2)
    assert graph.diameter == N - 1 == 9, "the two endpoints are N-1 hops apart"


def test_path_and_ring_differ_by_exactly_the_closing_edge() -> None:
    ring, path = make("ring", {}), make("path", {})
    assert ring.n_edges == path.n_edges + 1
    difference = ring.adjacency - path.adjacency
    assert float(difference[0, N - 1]) == 1.0
    assert float(difference[N - 1, 0]) == 1.0
    assert int(difference.sum()) == 2


def test_star_has_one_hub_and_n_minus_one_leaves() -> None:
    graph = make("star", {})
    assert int(graph.degrees[0]) == N - 1, "node 0 is the hub"
    assert torch.all(graph.degrees[1:] == 1)
    assert graph.n_edges == N - 1
    assert graph.diameter == 2, "leaf to hub to leaf"
    for leaf in range(1, N):
        assert graph.neighbours(leaf).tolist() == [0]


def test_grid2d_structure() -> None:
    rows, cols = 2, 5
    graph = make("grid2d", {"rows": rows, "cols": cols})
    assert graph.n_edges == rows * (cols - 1) + cols * (rows - 1) == 13
    assert graph.diameter == (rows - 1) + (cols - 1) == 5, "Manhattan distance across the grid"
    assert int(graph.degrees[0]) == 2, "a corner has two neighbours"


def test_grid2d_uses_row_major_node_ids() -> None:
    """Node (r, c) is r * cols + c, so a shard assignment can be read off."""
    rows, cols = 2, 5
    graph = make("grid2d", {"rows": rows, "cols": cols})
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            expected = []
            if c > 0:
                expected.append(r * cols + c - 1)
            if c < cols - 1:
                expected.append(r * cols + c + 1)
            if r > 0:
                expected.append((r - 1) * cols + c)
            if r < rows - 1:
                expected.append((r + 1) * cols + c)
            assert sorted(graph.neighbours(node).tolist()) == sorted(expected)


def test_watts_strogatz_preserves_the_edge_count_under_rewiring() -> None:
    """Rewiring moves edges; it does not create or destroy them."""
    k = 4
    graph = make("watts_strogatz", {"k": k, "beta": 0.2})
    assert graph.n_edges == N * k // 2 == 20


def test_erdos_renyi_is_connected_by_default() -> None:
    for seed in range(5):
        generator = torch.Generator().manual_seed(seed)
        assert build_graph("erdos_renyi", N, "metropolis", {"p": 0.4}, generator).is_connected


def test_erdos_renyi_may_be_left_disconnected_on_request() -> None:
    graph = make("erdos_renyi", {"p": 0.05, "ensure_connected": False})
    assert graph.n_components >= 1  # the point is that it does not raise


def test_erdos_renyi_raises_rather_than_looping_forever() -> None:
    with pytest.raises(GraphError, match="disconnected graph in"):
        make("erdos_renyi", {"p": 0.0})


def test_disconnected_control_has_no_cross_component_edges() -> None:
    graph = make("disconnected", {"n_components": 2})
    assert graph.n_components == 2
    assert not graph.is_connected
    assert graph.diameter is None, "a disconnected graph has no diameter"
    first, second = range(0, N // 2), range(N // 2, N)
    for u in first:
        for v in second:
            assert float(graph.adjacency[u, v]) == 0.0


def test_disconnected_components_are_internally_complete() -> None:
    """Each component is complete, so any shortfall is attributable to the
    missing cross-component links and to nothing else."""
    graph = make("disconnected", {"n_components": 2})
    half = N // 2
    assert graph.n_edges == 2 * (half * (half - 1) // 2) == 20
    assert torch.all(graph.degrees == half - 1)


def test_disconnected_splits_as_evenly_as_possible() -> None:
    graph = build_graph("disconnected", 7, "metropolis", {"n_components": 2})
    sizes = sorted(len(component) for component in _components(graph))
    assert sizes == [3, 4]


def _components(graph: Graph) -> list[set[int]]:
    import networkx as nx

    return [set(component) for component in nx.connected_components(graph.to_networkx())]


# =========================================================================== #
# 3. combination weights
# =========================================================================== #


def path3() -> torch.Tensor:
    """0 -- 1 -- 2, degrees [1, 2, 1]. Small enough to compute by hand."""
    return torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)


def test_metropolis_matches_the_hand_computed_matrix() -> None:
    """a_vu = 1/(1 + max(d_v, d_u)) on an edge, diagonal takes the slack."""
    expected = torch.tensor(
        [
            [2 / 3, 1 / 3, 0.0],
            [1 / 3, 1 / 3, 1 / 3],
            [0.0, 1 / 3, 2 / 3],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(metropolis_weights(path3()), expected)


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_metropolis_is_symmetric_and_doubly_stochastic(topology: str, params: dict) -> None:
    """Column sums matter too: diffusion preserves the network average only if
    the weight matrix is doubly stochastic."""
    weights = make(topology, params, weights="metropolis").weights
    assert weights is not None
    assert torch.allclose(weights, weights.T, atol=1e-12)
    column_sums = weights.sum(dim=0)
    assert torch.allclose(column_sums, torch.ones_like(column_sums), atol=1e-12)


@pytest.mark.parametrize(("topology", "params"), ALL_TOPOLOGIES)
def test_metropolis_self_weight_meets_its_lower_bound(topology: str, params: dict) -> None:
    """a_vv >= 1/(1 + d_v) > 0, which is why an agent can never discard itself."""
    graph = make(topology, params, weights="metropolis")
    assert graph.weights is not None
    bound = 1.0 / (1.0 + graph.degrees.to(torch.float64))
    assert bool(torch.all(torch.diagonal(graph.weights) >= bound - 1e-12))


def test_relative_degree_matches_the_hand_computed_matrix() -> None:
    """a_vu = d_u / sum over the closed neighbourhood."""
    expected = torch.tensor(
        [
            [1 / 3, 2 / 3, 0.0],
            [1 / 4, 2 / 4, 1 / 4],
            [0.0, 2 / 3, 1 / 3],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(relative_degree_weights(path3()), expected)


def test_relative_degree_is_not_symmetric_in_general() -> None:
    """It leans toward better-connected neighbours, unlike Metropolis."""
    weights = relative_degree_weights(path3())
    assert not torch.allclose(weights, weights.T)


def test_relative_degree_leaves_an_isolated_agent_with_all_its_own_weight() -> None:
    """The denominator is zero there; keeping everything is the only sensible
    reading of 'combine with nobody'."""
    adjacency = torch.zeros(3, 3, dtype=torch.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0  # node 2 is isolated
    weights = relative_degree_weights(adjacency)
    assert float(weights[2, 2]) == 1.0
    assert float(weights[2].sum()) == 1.0


def test_uniform_matches_the_hand_computed_matrix() -> None:
    expected = torch.tensor(
        [
            [1 / 2, 1 / 2, 0.0],
            [1 / 3, 1 / 3, 1 / 3],
            [0.0, 1 / 2, 1 / 2],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(uniform_weights(path3()), expected)


def test_uniform_on_a_complete_graph_is_exactly_one_over_n() -> None:
    """This is the a_vu = 1/N the X0 exactness identity requires."""
    graph = make("complete", {}, weights="uniform")
    assert graph.weights is not None
    assert torch.allclose(graph.weights, torch.full((N, N), 1.0 / N, dtype=torch.float64))


def test_metropolis_and_uniform_coincide_on_a_complete_graph() -> None:
    """Both give 1/N, so X0 is not sensitive to which rule the config names."""
    metropolis = make("complete", {}, weights="metropolis").weights
    uniform = make("complete", {}, weights="uniform").weights
    assert metropolis is not None and uniform is not None
    assert torch.allclose(metropolis, uniform)


def test_uniform_respects_a_sparse_adjacency() -> None:
    """A flat 1/N would assign weight to agents that never sent anything."""
    graph = make("ring", {}, weights="uniform")
    assert graph.weights is not None
    assert float(graph.weights[0, 5]) == 0.0
    assert float(graph.weights[0, 0]) == pytest.approx(1 / 3)


# =========================================================================== #
# 4. spectral gap
# =========================================================================== #


def test_complete_graph_has_spectral_gap_one() -> None:
    assert make("complete", {}).spectral_gap == pytest.approx(1.0, abs=1e-12)


def test_spectral_gap_refuses_non_doubly_stochastic_weights() -> None:
    """The spec formula rho = ||A - 11^T/N||_2 assumes the all-ones vector is a
    *left* eigenvector too. On a star with relative-degree weights it returns
    -1.56, which is not merely imprecise but wrong in sign -- so it raises
    instead of handing that to a plot."""
    graph = make("star", {}, weights="relative_degree")
    assert not graph.is_doubly_stochastic
    with pytest.raises(GraphError, match="only a mixing measure for doubly stochastic"):
        _ = graph.spectral_gap


def test_mixing_gap_is_defined_where_the_spectral_gap_is_not() -> None:
    graph = make("star", {}, weights="relative_degree")
    assert 0.0 < graph.mixing_gap < 1.0


def test_star_with_relative_degree_weights_actually_mixes_fast() -> None:
    """The hub aggregates the whole network in one hop, so this should mix far
    better than a ring -- the opposite of what the spec formula reported."""
    star = make("star", {}, weights="relative_degree")
    ring = make("ring", {}, weights="relative_degree")
    assert star.mixing_gap > ring.mixing_gap


def test_the_two_gap_measures_agree_when_both_are_defined() -> None:
    for topology, params in ALL_TOPOLOGIES:
        graph = make(topology, params, weights="metropolis")
        assert graph.is_doubly_stochastic
        assert graph.spectral_gap == pytest.approx(graph.mixing_gap, abs=1e-9), topology


def test_metropolis_is_doubly_stochastic_everywhere() -> None:
    for topology, params in ALL_TOPOLOGIES:
        assert make(topology, params, weights="metropolis").is_doubly_stochastic, topology


def test_all_rules_coincide_on_a_regular_graph() -> None:
    """A ring is 2-regular, so every rule gives 1/3 to each of the closed
    neighbourhood -- and all three are then doubly stochastic."""
    matrices = [weights_of(make("ring", {}, weights=rule)) for rule in WEIGHT_RULES]
    for other in matrices[1:]:
        assert torch.allclose(matrices[0], other)


def test_irregular_graphs_break_double_stochasticity_for_degree_based_rules() -> None:
    """A star is maximally irregular, which is exactly where the spec formula
    stops being usable."""
    assert not make("star", {}, weights="relative_degree").is_doubly_stochastic
    assert not make("star", {}, weights="uniform").is_doubly_stochastic
    assert not make("path", {}, weights="relative_degree").is_doubly_stochastic


def test_disconnected_graph_has_no_mixing() -> None:
    assert make("disconnected", {"n_components": 2}).spectral_gap == pytest.approx(0.0, abs=1e-12)


def test_spectral_gap_orders_the_topologies_as_expected() -> None:
    """The price of connectivity, in one line: a path mixes worst, a complete
    graph best. This ordering is the x-axis of the topology sweep."""
    gaps = {name: make(name, params).spectral_gap for name, params in ALL_TOPOLOGIES}
    assert gaps["disconnected"] < gaps["path"] < gaps["ring"] < gaps["complete"]
    assert gaps["path"] < gaps["grid2d"]
    assert gaps["complete"] > gaps["erdos_renyi"] > gaps["path"]


def test_denser_erdos_renyi_mixes_faster() -> None:
    sparse = make("erdos_renyi", {"p": 0.3}).spectral_gap
    dense = make("erdos_renyi", {"p": 0.9}).spectral_gap
    assert dense > sparse


def test_larger_ring_mixes_more_slowly() -> None:
    """Mixing degrades as the ring grows, which is why N interacts with the
    horizon rather than being a free parameter."""
    small = build_graph("ring", 6, "metropolis").spectral_gap
    large = build_graph("ring", 40, "metropolis").spectral_gap
    assert small > large


# =========================================================================== #
# 5. determinism
# =========================================================================== #


def test_random_topologies_are_reproducible_from_a_seed() -> None:
    first = build_graph(
        "erdos_renyi", N, "metropolis", {"p": 0.4}, torch.Generator().manual_seed(7)
    )
    second = build_graph(
        "erdos_renyi", N, "metropolis", {"p": 0.4}, torch.Generator().manual_seed(7)
    )
    assert torch.equal(first.adjacency, second.adjacency)


def test_different_seeds_give_different_random_topologies() -> None:
    graphs = [
        build_graph("erdos_renyi", 20, "metropolis", {"p": 0.3}, torch.Generator().manual_seed(s))
        for s in range(4)
    ]
    assert any(not torch.equal(graphs[0].adjacency, other.adjacency) for other in graphs[1:])


def test_deterministic_topologies_ignore_the_seed() -> None:
    a = build_graph("ring", N, "metropolis", {}, torch.Generator().manual_seed(1))
    b = build_graph("ring", N, "metropolis", {}, torch.Generator().manual_seed(2))
    assert torch.equal(a.adjacency, b.adjacency)


def test_weights_are_float64_by_default() -> None:
    """The exactness check compares at 1e-12; float32 combination weights would
    spend a third of that budget before the gradients are even involved."""
    graph = make("ring", {})
    assert graph.adjacency.dtype == torch.float64
    assert graph.weights is not None and graph.weights.dtype == torch.float64


# =========================================================================== #
# 6. degenerate sizes
# =========================================================================== #


@pytest.mark.parametrize(
    ("topology", "params"),
    [
        ("complete", {}),
        ("ring", {}),
        ("path", {}),
        ("star", {}),
        ("grid2d", {"rows": 1, "cols": 1}),
        ("erdos_renyi", {"p": 0.5}),
        ("disconnected", {"n_components": 1}),
    ],
)
def test_single_agent_network_is_well_formed(topology: str, params: dict) -> None:
    """N=1 must work: WORKPLAN section 7.2 requires every method to coincide there."""
    graph = build_graph(topology, 1, "metropolis", params)
    assert graph.n_nodes == 1
    assert graph.n_edges == 0
    assert graph.weights is not None
    assert float(graph.weights[0, 0]) == 1.0
    assert graph.is_connected
    assert graph.diameter == 0
    assert graph.spectral_gap == pytest.approx(1.0, abs=1e-12)


def test_two_agent_ring_degenerates_to_a_single_edge() -> None:
    graph = build_graph("ring", 2, "metropolis")
    assert graph.n_edges == 1
    assert graph.diameter == 1


def test_complete_ring_and_path_coincide_at_two_agents() -> None:
    matrices = [
        build_graph(name, 2, "metropolis").adjacency for name in ("complete", "ring", "path")
    ]
    assert all(torch.equal(matrices[0], other) for other in matrices[1:])


# =========================================================================== #
# 7. rejected inputs
# =========================================================================== #


def test_unknown_topology_lists_the_available_ones() -> None:
    with pytest.raises(GraphError, match="unknown topology"):
        build_graph("hypercube", N, "metropolis")


def test_unknown_weight_rule_is_rejected() -> None:
    with pytest.raises(GraphError, match="unknown weight rule"):
        build_graph("ring", N, "laplacian")


def test_zero_agents_is_rejected() -> None:
    with pytest.raises(GraphError, match="n_nodes must be >= 1"):
        build_graph("ring", 0, "metropolis")


def test_grid_dimensions_must_multiply_to_the_node_count() -> None:
    with pytest.raises(GraphError, match="holds 6 nodes but n_nodes is 10"):
        build_graph("grid2d", 10, "metropolis", {"rows": 2, "cols": 3})


def test_grid_without_dimensions_is_rejected() -> None:
    with pytest.raises(GraphError, match="needs 'rows' and 'cols'"):
        build_graph("grid2d", 10, "metropolis", {})


def test_erdos_renyi_without_p_is_rejected() -> None:
    with pytest.raises(GraphError, match="needs 'p'"):
        build_graph("erdos_renyi", N, "metropolis", {})


def test_erdos_renyi_probability_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(GraphError, match=r"p must lie in \[0, 1\]"):
        build_graph("erdos_renyi", N, "metropolis", {"p": 1.5})


def test_watts_strogatz_needs_k_below_n() -> None:
    with pytest.raises(GraphError, match="needs k < n_nodes"):
        build_graph("watts_strogatz", 4, "metropolis", {"k": 4})


def test_more_components_than_agents_is_rejected() -> None:
    with pytest.raises(GraphError, match="components but only"):
        build_graph("disconnected", 3, "metropolis", {"n_components": 5})


# --- malformed graphs assembled by hand ------------------------------------ #


def test_asymmetric_adjacency_is_rejected() -> None:
    adjacency = torch.zeros(3, 3, dtype=torch.float64)
    adjacency[0, 1] = 1.0
    with pytest.raises(GraphError, match="must be symmetric"):
        Graph(adjacency=adjacency, weights=None, topology="hand")


def test_self_loop_in_the_adjacency_is_rejected() -> None:
    adjacency = torch.eye(3, dtype=torch.float64)
    with pytest.raises(GraphError, match="non-zero diagonal"):
        Graph(adjacency=adjacency, weights=None, topology="hand")


def test_non_binary_adjacency_is_rejected() -> None:
    adjacency = torch.zeros(2, 2, dtype=torch.float64)
    adjacency[0, 1] = adjacency[1, 0] = 0.5
    with pytest.raises(GraphError, match="entries must be 0 or 1"):
        Graph(adjacency=adjacency, weights=None, topology="hand")


def test_non_square_adjacency_is_rejected() -> None:
    with pytest.raises(GraphError, match="must be square"):
        Graph(adjacency=torch.zeros(2, 3), weights=None, topology="hand")


def test_weights_that_do_not_sum_to_one_are_rejected() -> None:
    adjacency = path3()
    weights = torch.eye(3, dtype=torch.float64) * 0.5
    with pytest.raises(GraphError, match="row-stochastic"):
        Graph(adjacency=adjacency, weights=weights, topology="hand")


def test_zero_self_weight_is_rejected() -> None:
    adjacency = path3()
    weights = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=torch.float64)
    with pytest.raises(GraphError, match="positive share of its own estimate"):
        Graph(adjacency=adjacency, weights=weights, topology="hand")


def test_negative_weight_is_rejected() -> None:
    adjacency = path3()
    weights = torch.tensor(
        [[1.5, -0.5, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.0, 1 / 3, 2 / 3]], dtype=torch.float64
    )
    with pytest.raises(GraphError, match="non-negative"):
        Graph(adjacency=adjacency, weights=weights, topology="hand")


def test_weight_on_a_non_edge_is_rejected() -> None:
    """Node 0 and node 2 are not adjacent, so 0 cannot weight 2's estimate."""
    adjacency = path3()
    weights = torch.tensor(
        [[0.5, 0.25, 0.25], [1 / 3, 1 / 3, 1 / 3], [0.0, 1 / 3, 2 / 3]], dtype=torch.float64
    )
    with pytest.raises(GraphError, match="not neighbours"):
        Graph(adjacency=adjacency, weights=weights, topology="hand")


def test_mismatched_weight_shape_is_rejected() -> None:
    with pytest.raises(GraphError, match="do not match adjacency"):
        Graph(adjacency=path3(), weights=torch.eye(2, dtype=torch.float64), topology="hand")


def test_graph_is_frozen() -> None:
    import dataclasses

    graph = make("ring", {})
    # setattr, not `graph.topology = ...`: a direct assignment to a frozen field
    # is a static error the IDE flags in red, even though raising is the point.
    # setattr still routes through __setattr__, so the check is unchanged.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(graph, "topology", "path")  # noqa: B010


# =========================================================================== #
# 8. the two graphs
# =========================================================================== #


def test_empty_graph_has_no_edges_and_no_weights() -> None:
    """The data graph combines nothing, so an identity weight matrix would
    misleadingly imply that it does."""
    graph = Graph.empty(N)
    assert graph.n_edges == 0
    assert graph.weights is None
    assert torch.all(graph.degrees == 0)


def test_spectral_gap_without_weights_is_an_error_not_a_number() -> None:
    with pytest.raises(GraphError, match="needs combination weights"):
        _ = Graph.empty(N).spectral_gap


def test_data_graph_is_empty_for_class_l() -> None:
    """MNIST with an MLP is Class L: every forward pass is purely local."""
    graphs = build_graphs(load_config("x1_stationary"))
    assert graphs.data.n_edges == 0
    assert graphs.data.weights is None
    assert graphs.comm.n_edges > 0


def test_the_two_graphs_are_distinct_objects() -> None:
    graphs = build_graphs(load_config("x1_stationary"))
    assert graphs.comm is not graphs.data
    assert not torch.equal(graphs.comm.adjacency, graphs.data.adjacency)


def test_graph_pair_rejects_mismatched_sizes() -> None:
    with pytest.raises(GraphError, match="but data graph has"):
        Graphs(comm=build_graph("ring", 10, "metropolis"), data=Graph.empty(4))


# =========================================================================== #
# 9. integration with the configs
# =========================================================================== #


def test_x1_builds_a_ring_of_ten() -> None:
    graphs = build_graphs(load_config("x1_stationary"))
    assert graphs.comm.topology == "ring"
    assert graphs.comm.n_nodes == 10
    assert graphs.comm.is_connected


def test_x0_builds_the_complete_graph_with_uniform_weights() -> None:
    """The exactness preconditions, verified on the object rather than the YAML."""
    graphs = build_graphs(load_config("x0_exactness"))
    comm = graphs.comm
    assert comm.topology == "complete"
    assert comm.weights is not None
    assert torch.allclose(comm.weights, torch.full((10, 10), 0.1, dtype=torch.float64))
    assert comm.spectral_gap == pytest.approx(1.0, abs=1e-12)


def test_grid_config_matches_its_node_count() -> None:
    graphs = build_graphs(load_config("x1_stationary", overrides={"include": {"graph": "grid2d"}}))
    assert graphs.comm.n_edges == 13
    assert graphs.comm.diameter == 5


def test_every_shipped_graph_config_builds() -> None:
    """Every file in configs/graph/ must be usable, or the docs describe a
    topology nobody can select."""
    from dekf_bench.utils.config import default_configs_dir

    shipped = sorted(path.stem for path in (default_configs_dir() / "graph").glob("*.yaml"))
    assert set(shipped) == set(TOPOLOGY_BUILDERS), "a builder without a config, or vice versa"

    for name in shipped:
        config = load_config("x1_stationary", overrides={"include": {"graph": name}})
        graph = build_graphs(config, torch.Generator().manual_seed(0)).comm
        assert graph.n_nodes == 10
        assert graph.weights is not None


def test_documented_parameter_overrides_reach_the_graph() -> None:
    """The docs/configs.md section 3.2 snippets, executed."""
    config = load_config(
        "x1_stationary",
        overrides={"include": {"graph": "erdos_renyi"}, "graph": {"params": {"p": 0.5}}},
    )
    assert config.graph.params["p"] == 0.5
    graph = build_graphs(config, torch.Generator().manual_seed(0)).comm
    assert graph.n_edges == 29

    config = load_config(
        "x1_stationary",
        overrides={"include": {"graph": "grid2d"}, "graph": {"params": {"rows": 5, "cols": 2}}},
    )
    assert config.graph.params == {"rows": 5, "cols": 2}


def test_fully_disconnected_control_makes_combine_the_identity() -> None:
    """n_components == n_nodes must reproduce local_only exactly, which is a
    cheap consistency check on the learners in phase 3."""
    graph = build_graph("disconnected", 10, "metropolis", {"n_components": 10})
    assert graph.weights is not None
    assert torch.equal(graph.weights, torch.eye(10, dtype=torch.float64))
    assert graph.n_edges == 0


def test_ring_mixing_time_is_consistent_with_its_diameter() -> None:
    """Information crosses one hop per step, so the diameter is a lower bound on
    how long a sample at one agent takes to reach the far side of the ring."""
    graph = make("ring", {})
    assert graph.diameter == 5
    reach = graph.adjacency.clone()
    for _ in range(graph.diameter - 1):
        reach = ((reach @ graph.adjacency) + reach).clamp(max=1.0)
    assert bool(
        torch.all(reach + torch.eye(N, dtype=reach.dtype) > 0)
    ), "every pair must be reachable within `diameter` hops"


def test_hop_count_below_the_diameter_leaves_pairs_unreachable() -> None:
    graph = make("path", {})
    reach = graph.adjacency.clone()
    for _ in range(3):
        reach = ((reach @ graph.adjacency) + reach).clamp(max=1.0)
    assert float(reach[0, N - 1]) == 0.0, "endpoints of a 10-path are 9 hops apart"


def test_message_volume_per_round_is_twice_the_edge_count() -> None:
    """Each edge carries one p-vector in each direction; the ledger bills that."""
    for topology, params in ALL_TOPOLOGIES:
        graph = make(topology, params)
        directed = int(graph.adjacency.sum())
        assert directed == 2 * graph.n_edges
        assert directed == int(graph.degrees.sum())


def test_no_agent_combines_an_estimate_it_did_not_receive() -> None:
    """The end-to-end statement of the adjacency/weights distinction: the set of
    estimates an agent uses is exactly its neighbours plus itself."""
    for topology, params in ALL_TOPOLOGIES:
        for rule in WEIGHT_RULES:
            graph = make(topology, params, weights=rule)
            assert graph.weights is not None
            for node in range(graph.n_nodes):
                used = set(torch.nonzero(graph.weights[node], as_tuple=True)[0].tolist())
                available = set(graph.closed_neighbourhood(node).tolist())
                assert used <= available, f"{topology}/{rule}: node {node} used {used - available}"


def test_ring_diameter_formula_holds_across_sizes() -> None:
    for n in (3, 4, 5, 8, 9, 20, 21):
        assert build_graph("ring", n, "metropolis").diameter == n // 2


def test_path_diameter_formula_holds_across_sizes() -> None:
    for n in (2, 3, 7, 15):
        assert build_graph("path", n, "metropolis").diameter == n - 1


def test_complete_graph_diameter_is_one_at_every_size() -> None:
    for n in (2, 5, 30):
        assert build_graph("complete", n, "metropolis").diameter == 1


def test_star_diameter_is_two_at_every_size() -> None:
    for n in (3, 10, 50):
        assert build_graph("star", n, "metropolis").diameter == 2


def test_grid_diameter_is_the_manhattan_span() -> None:
    for rows, cols in ((2, 5), (3, 4), (4, 4), (1, 6)):
        graph = build_graph("grid2d", rows * cols, "metropolis", {"rows": rows, "cols": cols})
        assert graph.diameter == (rows - 1) + (cols - 1)


def test_spectral_gap_of_a_ring_decays_like_one_over_n_squared() -> None:
    """Sanity on the order of magnitude: doubling a ring should cut the gap by
    roughly four, which is what makes large rings hopeless for diffusion."""
    gap_10 = build_graph("ring", 10, "metropolis").spectral_gap
    gap_20 = build_graph("ring", 20, "metropolis").spectral_gap
    ratio = gap_10 / gap_20
    assert 3.0 < ratio < 5.0, f"expected roughly 4x, got {ratio:.2f}"


def test_grid_dimensions_are_as_square_as_the_factorisation_allows() -> None:
    from dekf_bench.env.graph import grid_dimensions

    assert grid_dimensions(10) == (2, 5)
    assert grid_dimensions(16) == (4, 4)
    assert grid_dimensions(12) == (3, 4)
    assert grid_dimensions(1) == (1, 1)


def test_prime_node_count_makes_the_grid_a_path() -> None:
    """Worth knowing before reading a sweep in which the two look identical."""
    from dekf_bench.env.graph import grid_dimensions

    assert grid_dimensions(7) == (1, 7)
    grid = build_graph("grid2d", 7, "metropolis", {"rows": 1, "cols": 7})
    path = build_graph("path", 7, "metropolis")
    assert torch.equal(grid.adjacency, path.adjacency)


def test_default_params_build_every_topology_at_several_sizes() -> None:
    from dekf_bench.env.graph import default_topology_params

    for n in (4, 9, 10, 16, 25):
        for topology, params in default_topology_params(n).items():
            graph = build_graph(topology, n, "metropolis", params)
            assert graph.n_nodes == n, f"{topology} at N={n}"


def test_default_erdos_renyi_density_stays_sparse_but_connected() -> None:
    """Near-p=1 would make the random graph indistinguishable from complete."""
    from dekf_bench.env.graph import default_topology_params

    for n in (10, 20, 50):
        p = default_topology_params(n)["erdos_renyi"]["p"]
        assert math.log(n) / n < p < 1.0
        graph = build_graph("erdos_renyi", n, "metropolis", {"p": p})
        assert graph.is_connected
        assert graph.n_edges < n * (n - 1) // 2, "must be sparser than complete"


def test_isolated_agent_keeps_a_valid_weight_row() -> None:
    """A singleton component under the disconnected control still needs a
    row-stochastic weight row, or the combine step produces NaN."""
    graph = build_graph("disconnected", 3, "metropolis", {"n_components": 3})
    assert graph.weights is not None
    assert torch.allclose(graph.weights, torch.eye(3, dtype=torch.float64))
    assert not bool(torch.any(torch.isnan(graph.weights)))


def test_no_weight_matrix_contains_nan_or_inf() -> None:
    for topology, params in ALL_TOPOLOGIES:
        for rule in WEIGHT_RULES:
            weights = make(topology, params, weights=rule).weights
            assert weights is not None
            assert bool(torch.all(torch.isfinite(weights))), f"{topology}/{rule}"


def test_averaging_is_a_fixed_point_of_the_combine_step() -> None:
    """If every agent already agrees, combining must not move anything -- this is
    row-stochasticity doing its job, and it is what makes X0 hold inductively."""
    for topology, params in ALL_TOPOLOGIES:
        for rule in WEIGHT_RULES:
            graph = make(topology, params, weights=rule)
            assert graph.weights is not None
            consensus = torch.full((graph.n_nodes, 5), 3.7, dtype=torch.float64)
            assert torch.allclose(graph.weights @ consensus, consensus, atol=1e-12)


def test_combine_stays_inside_the_convex_hull() -> None:
    """Lemma 1 of the research note: nothing an agent receives can push its
    estimate outside the range its neighbours collectively propose."""
    generator = torch.Generator().manual_seed(0)
    for topology, params in ALL_TOPOLOGIES:
        graph = make(topology, params)
        assert graph.weights is not None
        estimates = torch.randn(graph.n_nodes, 4, generator=generator, dtype=torch.float64)
        combined = graph.weights @ estimates
        for node in range(graph.n_nodes):
            neighbourhood = estimates[graph.closed_neighbourhood(node)]
            assert bool(torch.all(combined[node] >= neighbourhood.min(dim=0).values - 1e-12))
            assert bool(torch.all(combined[node] <= neighbourhood.max(dim=0).values + 1e-12))


def test_metropolis_preserves_the_network_average() -> None:
    """Doubly stochastic means the mean estimate is invariant under combining --
    information is redistributed, never created or destroyed."""
    generator = torch.Generator().manual_seed(1)
    for topology, params in ALL_TOPOLOGIES:
        graph = make(topology, params, weights="metropolis")
        assert graph.weights is not None
        estimates = torch.randn(graph.n_nodes, 3, generator=generator, dtype=torch.float64)
        before = estimates.mean(dim=0)
        after = (graph.weights @ estimates).mean(dim=0)
        assert torch.allclose(before, after, atol=1e-12), topology


def test_repeated_combining_reaches_consensus_on_a_connected_graph() -> None:
    """And fails to, on the disconnected control -- which is the whole point of
    keeping that topology around."""
    steps = 500
    connected = make("ring", {})
    assert connected.weights is not None
    estimates = torch.arange(N, dtype=torch.float64).unsqueeze(1)
    mixed = estimates.clone()
    for _ in range(steps):
        mixed = connected.weights @ mixed
    assert float(mixed.std()) < 1e-9, "a connected graph must reach consensus"

    split = make("disconnected", {"n_components": 2})
    assert split.weights is not None
    mixed = estimates.clone()
    for _ in range(steps):
        mixed = split.weights @ mixed
    assert float(mixed.std()) > 1.0, "components cannot agree with each other"
    assert math.isclose(float(mixed[0]), float(mixed[N // 2 - 1]), abs_tol=1e-9)
