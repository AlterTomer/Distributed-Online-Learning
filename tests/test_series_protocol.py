"""The protocol on the series task: regression rows, and calibration only when asked."""

from __future__ import annotations

import math

import pytest
import torch

from dekf_bench.env.series import build_series_environment
from dekf_bench.evaluation import protocol
from dekf_bench.evaluation.series_evalsets import build_series_evalsets
from dekf_bench.likelihoods.gaussian import Gaussian
from dekf_bench.models.transformer import CausalTransformer
from dekf_bench.utils.config import load_config

DTYPE = torch.float64


@pytest.fixture(scope="module")
def setting():
    config = load_config(
        "x1_stationary",
        overrides={
            "run": {"name": "probe", "horizon": 6, "seeds": [0], "dtype": "float64",
                    "device": "cpu"},
            "env": {"dataset": "mackey_glass", "series": {"burn_in": 50.0}},
            "model": {"name": "causal_transformer", "likelihood": "gaussian", "output_dim": 31},
        },
    )
    environment = build_series_environment(config, 0)
    model = CausalTransformer(dtype=DTYPE)
    params = model.init_params(torch.Generator().manual_seed(0))
    return config, environment, model, params


def _predict(model, params):
    return lambda node, x: model.forward(params, x)


def test_prequential_records_regression_rows(setting) -> None:
    config, environment, model, params = setting
    likelihood = Gaussian(output_dim=31, variance=0.02)
    rows = protocol.prequential(
        environment.step(2), _predict(model, params), likelihood, step=2
    ).as_rows()
    metrics = {row["metric"] for row in rows}
    assert metrics == {"mse", "rmse", "rmse_full_context", "nll"}
    assert "error_rate" not in metrics
    by_node = {(r["node_id"], r["metric"]): r["value"] for r in rows}
    assert by_node[(0, "rmse")] == pytest.approx(math.sqrt(by_node[(0, "mse")]))
    assert all(row["drift_state"] == pytest.approx(config.env.series.beta) for row in rows)


def test_full_evaluation_adds_calibration_only_with_a_variance(setting) -> None:
    config, environment, model, params = setting
    likelihood = Gaussian(output_dim=31, variance=0.02)
    evalsets = build_series_evalsets(config, environment)
    plain = protocol.full_evaluate(
        evalsets, _predict(model, params), likelihood, step=0, nodes=[0, 1],
        evalsets=["prequential", "current", "canonical"],
    ).as_rows()
    assert not any(row["metric"].startswith("coverage") for row in plain)

    constant = lambda node, x: torch.full((x.shape[0], 31), 0.01, dtype=DTYPE)  # noqa: E731
    calibrated = protocol.full_evaluate(
        evalsets, _predict(model, params), likelihood, step=0, nodes=[0, 1],
        evalsets=["current"], predict_variance=constant,
    ).as_rows()
    metrics = {row["metric"] for row in calibrated}
    assert {"coverage_90", "variance_ratio", "predictive_nll", "pit_0", "pit_9"} <= metrics
    assert {row["evalset"] for row in calibrated} == {"current"}


def test_the_variance_hook_is_ignored_for_classification() -> None:
    """MNIST never pays for H P H^T: the hook is read only on regression tasks."""
    from dekf_bench.likelihoods.categorical import Categorical

    calls = []

    def variance(node, x):
        calls.append(node)
        return torch.ones(x.shape[0], 4)

    class Set:
        name, step, rotation_degrees = "current", 0, 0.0

        def batches(self, size):
            yield torch.rand(3, 1, 4, 4), torch.tensor([0, 1, 2])

    protocol._score_evalset(
        0, Set(), lambda n, x: torch.randn(x.shape[0], 4), Categorical(4), 100, True, variance
    )
    assert calls == []
