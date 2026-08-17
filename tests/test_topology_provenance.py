r"""Which topology each shipped experiment runs on.

The repo deliberately holds **two** topologies at once. Erdos-Renyi(0.3) is the
standard from 2026-08-17, but x9 and X11 stay on ring because they are a matched
pair -- comparing repeated abrupt shifts against smooth drift at matched average
speed only means something if the graph is the same on both sides (design note
D52).

That is a fine distinction to hold in a comment and an easy one to lose in a
default. These tests pin it, so changing `base.yaml` cannot quietly move a
finished experiment onto a different graph and invalidate a comparison already
drawn from it.
"""

from __future__ import annotations

import pytest

from dekf_bench.utils.config import load_config

#: Ring, because these were run on ring and their results are on disk.
RING = (
    "x1_stationary",
    "x1b_atc_vs_cta",
    "x2_rotating",
    "x5_abrupt_shift",
    "x7_sinusoidal",
    "x9_rate_ramp",
    "x9_control",
    "x11_control",
)

#: Erdos-Renyi, the standard for everything from X8 onward.
ERDOS_RENYI = ("x8_per_node_drift", "x8_global", "x10_prior_drift")


@pytest.mark.parametrize("experiment", RING)
def test_the_ring_experiments_stay_on_ring(experiment: str) -> None:
    """Their results are already recorded. Moving them would silently
    invalidate every comparison drawn against those numbers."""
    assert load_config(experiment).graph.topology == "ring"


@pytest.mark.parametrize("experiment", ERDOS_RENYI)
def test_the_new_experiments_use_the_standard_topology(experiment: str) -> None:
    config = load_config(experiment)
    assert config.graph.topology == "erdos_renyi"
    assert config.graph.params["p"] == 0.3


def test_the_exactness_check_keeps_its_complete_graph() -> None:
    """X0's identity only holds on a complete graph -- one combine step has to
    reach full consensus. A default that moved it would turn the highest-value
    test in the suite into a test of nothing."""
    assert load_config("x0_exactness").graph.topology == "complete"


def test_the_two_defaults_agree() -> None:
    """The standard is written in `configs/base.yaml` *and* in the GraphConfig
    dataclass. Every shipped config loads through base.yaml, so the dataclass
    default rarely applies -- which is exactly why a stale value there would go
    unnoticed until something built a GraphConfig directly and silently got the
    old topology."""
    import yaml

    from dekf_bench.utils.config import GraphConfig, default_configs_dir

    base = yaml.safe_load((default_configs_dir() / "base.yaml").read_text(encoding="utf-8"))
    assert GraphConfig().topology == base["graph"]["topology"]


def test_the_density_is_not_in_the_shared_defaults() -> None:
    """`params` merges key by key, so a density in base.yaml would follow every
    other topology around: a grid would report {p, rows, cols} and misdescribe
    itself. It belongs in configs/graph/erdos_renyi.yaml, which the experiments
    include."""
    import yaml

    from dekf_bench.utils.config import GraphConfig, default_configs_dir

    base = yaml.safe_load((default_configs_dir() / "base.yaml").read_text(encoding="utf-8"))
    assert base["graph"]["params"] in ({}, None)
    assert GraphConfig().params == {}

    grid = load_config("x1_stationary", overrides={"include": {"graph": "grid2d"}})
    assert "p" not in grid.graph.params, "the density must not leak into other topologies"


def test_x8_and_its_twin_differ_only_in_scope() -> None:
    """The pairing X8 exists for. If they differed in topology as well, no later
    step could say whether a result came from the scope or from the graph."""
    per_node = load_config("x8_per_node_drift")
    global_scope = load_config("x8_global")

    assert per_node.env.drift_scope == "per_node"
    assert global_scope.env.drift_scope == "global"
    assert per_node.graph.topology == global_scope.graph.topology
    assert per_node.run.horizon == global_scope.run.horizon
    assert per_node.run.eval_every == global_scope.run.eval_every
    assert [e.name for e in per_node.learners] == [e.name for e in global_scope.learners]
