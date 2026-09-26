"""The AdamW family on the image task: three names, the same definitions as mg-task.

The AdamW pass (schedule.md) adds these arms to every MNIST experiment the paper
cites, so the names must load, dispatch and be costed exactly as the series branch
defines them -- otherwise the two tasks would compare against baselines that share a
name and nothing else.
"""

from __future__ import annotations

import pytest

from dekf_bench.learners.registry import BUILDERS, DIFFUSING, POOLING
from dekf_bench.metrics.communication import cost_for
from dekf_bench.utils.config import load_config

ADAMW = ("centralized_adamw", "diffusion_atc_adamw", "local_adamw")


def _learner(name: str):
    config = load_config("x1_stationary", overrides={"learners": [{"name": name, "lr": 1e-3}]})
    return config.learners[0]


@pytest.mark.parametrize("name", ADAMW)
def test_each_name_loads_as_adamw_with_both_moments_mixed(name: str) -> None:
    """The YAML carries the optimizer, so a runner entry needs only a name and a rate.

    `all` is inert for the pooled and the local arm, but an adaptive optimizer must
    name a mixing policy: unmixed adaptive state is refused outright.
    """
    learner = _learner(name)
    assert learner.optimizer == "adamw"
    assert learner.mix_optimizer_state == "all"
    assert learner.lr == 1e-3


def test_dispatch_matches_the_sgd_arm_each_name_mirrors() -> None:
    """Pooled, diffusing and local, the same split as centralized_sgd, ATC and local_only."""
    assert set(ADAMW) <= set(BUILDERS)
    assert "centralized_adamw" in POOLING and "centralized_adamw" not in DIFFUSING
    assert "diffusion_atc_adamw" in DIFFUSING and "diffusion_atc_adamw" not in POOLING
    assert "local_adamw" not in POOLING and "local_adamw" not in DIFFUSING


def test_the_ledger_bills_only_the_diffusing_arm() -> None:
    """ATC AdamW sends psi and both moments, 3p per link; the other two send nothing
    on the graph. Matching literal names billed the pooled and local arms 3p too."""
    p, edges = 2908, 15
    shape = {"n_nodes": 10, "samples_per_step": 4, "input_dim": 196}
    atc = cost_for(_learner("diffusion_atc_adamw"), p, edges, **shape)
    pooled = cost_for(_learner("centralized_adamw"), p, edges, **shape)
    alone = cost_for(_learner("local_adamw"), p, edges, **shape)
    assert atc.diffuses and atc.vectors_per_link == 3
    assert atc.scalars_per_step == 3 * p * 2 * edges
    assert not pooled.diffuses and pooled.vectors_per_link == 0
    assert pooled.scalars_per_step == 10 * 4 * 197
    assert not alone.diffuses and alone.scalars_per_step == 0
