"""Many-to-one readout (M2O, D126): windows, the last-position model, and pairing.

The comparison M2O exists for -- "why many-to-many?" -- is only honest if the two
readouts see the same series and score the same targets. So the load-bearing tests
here are identities: a many-to-one window's target *is* the many-to-many target at
that position, the window *is* the 31 samples before it, and the last-position
model *is* the sequence model's last column -- Jacobian included.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dekf_bench.data.mackey_glass import to_blocks, to_windows, window_positions
from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.registry import build_likelihood
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.models.transformer import CausalTransformer
from dekf_bench.runner import simulate
from dekf_bench.runner.task import build_task
from dekf_bench.utils.config import ConfigError, load_config

L = 32

# =========================================================================== #
# windows
# =========================================================================== #


def test_window_positions_keep_the_last_and_thin_from_it() -> None:
    assert window_positions(L, 1) == list(range(1, L))
    assert window_positions(L, L - 1) == [L - 1]
    assert window_positions(L, 10) == [1, 11, 21, 31]


def test_every_window_target_is_the_many_to_many_target_at_that_position() -> None:
    rng = np.random.default_rng(0)
    prefix = L - 1
    series = rng.standard_normal((3, prefix + 4 * L))
    inputs, targets = to_windows(series, L, prefix, stride=1)
    _block_inputs, block_targets = to_blocks(series[:, prefix:], L)
    assert inputs.shape == (3, 4, L - 1, L - 1) and targets.shape == (3, 4, L - 1, 1)
    np.testing.assert_array_equal(targets[..., 0], block_targets)
    # Each window is exactly the L-1 samples before its target, prefix included.
    for b in range(4):
        for w, j in enumerate(window_positions(L, 1)):
            end = prefix + b * L + j
            np.testing.assert_array_equal(inputs[:, b, w], series[:, end - (L - 1):end])


def test_stride_l_minus_one_is_the_block_itself_scored_at_its_last_position() -> None:
    """M2O-c: one window per block -- the block's own inputs, its last target."""
    rng = np.random.default_rng(1)
    prefix = L - 1
    series = rng.standard_normal((2, prefix + 3 * L))
    inputs, targets = to_windows(series, L, prefix, stride=L - 1)
    block_inputs, block_targets = to_blocks(series[:, prefix:], L)
    np.testing.assert_array_equal(inputs[:, :, 0], block_inputs)
    np.testing.assert_array_equal(targets[:, :, 0, 0], block_targets[..., -1])


def test_a_short_prefix_is_refused() -> None:
    with pytest.raises(Exception, match="history"):
        to_windows(np.zeros((1, 10 + L)), L, prefix=10)


# =========================================================================== #
# the model
# =========================================================================== #


def test_the_last_readout_is_the_sequence_models_last_column_jacobian_included() -> None:
    sequence = CausalTransformer(dtype=torch.float64)
    last = CausalTransformer(dtype=torch.float64, readout="last")
    params = sequence.init_params(torch.Generator().manual_seed(0))
    x = torch.randn(5, 31, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    assert last.output_dim == 1 and last.num_params == sequence.num_params
    torch.testing.assert_close(last.forward(params, x), sequence.forward(params, x)[:, -1:])
    torch.testing.assert_close(last.per_sample_jacobian(params, x),
                               sequence.per_sample_jacobian(params, x)[:, -1:, :])


# =========================================================================== #
# config and the end-to-end pairing
# =========================================================================== #

FILTER = {"transition": "scalar", "gamma": 1.0, "process_noise_q": 1.0e-5,
          "lambda_forget": 1.0, "prior_scale": 0.01}


def _config(readout: str, prefix: int = L - 1, stride: int = 1, name: str = "m2o_test"):
    return load_config("x1_stationary", overrides={
        "run": {"name": name, "horizon": 4, "seeds": [0], "dtype": "float64",
                "device": "cpu", "eval_every": 2},
        "graph": {"topology": "ring"},
        "env": {"dataset": "mackey_glass",
                "series": {"burn_in": 50.0, "history_prefix": prefix, "window_stride": stride}},
        "model": {"name": "causal_transformer", "likelihood": "gaussian",
                  "output_dim": 1 if readout == "last" else 31, "readout": readout,
                  "observation_variance": 0.02},
        "learners": [
            {"name": "centralized_sgd", "optimizer": "sgd", "lr": 1.0e-5, "momentum": 0.0,
             "mix_optimizer_state": "none"},
            {"name": "diffusion_ekf_onehop_mean_receiver", **FILTER},
        ],
        "eval": {"evalsets": ["prequential", "current"]},
    })


def test_the_last_readout_needs_a_full_prefix_and_one_output() -> None:
    with pytest.raises(ConfigError, match="history_prefix"):
        _config("last", prefix=10)
    with pytest.raises(ConfigError, match="must be 1"):
        load_config("x1_stationary", overrides={
            "env": {"dataset": "mackey_glass", "series": {"history_prefix": 31}},
            "model": {"name": "causal_transformer", "likelihood": "gaussian",
                      "output_dim": 31, "readout": "last"}})


def test_both_readouts_see_the_same_series_and_the_same_targets() -> None:
    """The pairing M2O rests on: with the same prefix, the many-to-one targets are
    the many-to-many targets, sample for sample, every round of every agent."""
    many, _sets = build_task(_config("sequence"), 0)
    one, _sets = build_task(_config("last"), 0)
    assert one.inputs.shape[-2:] == (L - 1, L - 1) and one.targets.shape[-1] == 1
    torch.testing.assert_close(one.targets[..., 0], many.targets[:, :, 0, :])


def test_a_many_to_one_run_goes_end_to_end_and_its_last_rmse_is_its_rmse() -> None:
    config = _config("last")
    environment, evalsets = build_task(config, 0)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    records = simulate.run(config, environment, learners, evalsets, likelihood,
                           theta0.to(environment.device))
    rows = [row for record in records for row in record.rows
            if row.get("evalset") == "current"]
    by_metric: dict[str, list[float]] = {}
    for row in rows:
        by_metric.setdefault(row["metric"], []).append(row["value"])
    assert "rmse_full_context" not in by_metric  # one position: nothing to split
    assert by_metric["rmse"] == pytest.approx(by_metric["rmse_last"])
    assert "variance_ratio" in by_metric  # the filter's predictive is scored


def test_a_many_to_many_run_now_records_its_last_position_rmse() -> None:
    """M2O-a: rmse_last on a sequence run, never above what a full context allows
    to be checked here -- only that it is recorded and differs from the average."""
    config = _config("sequence")
    environment, evalsets = build_task(config, 0)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    records = simulate.run(config, environment, learners, evalsets, likelihood,
                           theta0.to(environment.device))
    metrics = {row["metric"] for record in records for row in record.rows}
    assert {"rmse", "rmse_last", "rmse_full_context"} <= metrics
