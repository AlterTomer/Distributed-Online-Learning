"""The series task end to end: the real runner, environment, sets, model and learners.

A short run, but through every piece a sweep will use -- `build_task`, the learner
registry, the Gaussian likelihood from config, `simulate.run` and the protocol's
regression branch -- so that an M-series sweep cannot be the first thing to
discover that two of them disagree.
"""

from __future__ import annotations

import torch

from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.registry import build_likelihood
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.runner import simulate
from dekf_bench.runner.task import build_task
from dekf_bench.utils.config import load_config

FILTER = {
    "transition": "scalar", "gamma": 1.0, "process_noise_q": 1.0e-5,
    "lambda_forget": 1.0, "prior_scale": 0.01,
}


def test_a_short_series_run_records_regression_and_calibration_rows() -> None:
    config = load_config(
        "x1_stationary",
        overrides={
            "run": {"name": "series_smoke", "horizon": 4, "seeds": [0], "dtype": "float64",
                    "device": "cpu", "eval_every": 2},
            "graph": {"topology": "ring"},
            "env": {"dataset": "mackey_glass", "series": {"burn_in": 50.0}},
            "model": {"name": "causal_transformer", "likelihood": "gaussian", "output_dim": 31,
                      "observation_variance": 0.02},
            # Rates on the Gaussian-NLL scale, ~775x below a per-value MSE rate at
            # R = 0.02 (see test_series_exactness.py).
            "learners": [
                {"name": "centralized_sgd", "optimizer": "sgd", "lr": 1.0e-5, "momentum": 0.0,
                 "mix_optimizer_state": "none"},
                {"name": "diffusion_sgd_atc", "optimizer": "sgd_momentum", "lr": 1.0e-5},
                {"name": "diffusion_ekf", **FILTER},
                {"name": "diffusion_ekf_onehop_mean_receiver", **FILTER},
            ],
            "eval": {"evalsets": ["prequential", "current", "canonical"]},
        },
    )
    environment, evalsets = build_task(config, 0)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))

    records = simulate.run(
        config, environment, learners, evalsets, likelihood, theta0.to(environment.device)
    )
    rows = [row for record in records for row in record.rows]
    metrics = {row["metric"] for row in rows}
    assert {"rmse", "mse", "nll", "rmse_full_context"} <= metrics
    assert "error_rate" not in metrics

    # Calibration only for the learners that hold a covariance.
    calibrated = {row["learner"] for row in rows if row["metric"] == "coverage_90"}
    assert calibrated == {"diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"}

    # Every value is finite and every learner moved off theta_0.
    assert all(torch.isfinite(torch.tensor(row["value"])) for row in rows)
    for learner in learners.values():
        assert not torch.equal(learner.flat_params(0), theta0)

    # The one-hop learner prices its raw block at L = 32 scalars (D94).
    onehop = learners["diffusion_ekf_onehop_mean_receiver"]
    p = model.num_params
    assert onehop.comm_scalars_per_step(1) == (p + 32) * 2
