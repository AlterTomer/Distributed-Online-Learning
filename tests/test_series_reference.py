"""The series reference lines: persistence exactly, e* sanely, and the cache honestly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dekf_bench.evaluation.series_reference import (
    ReferenceSettings,
    persistence_rmse,
    train_reference,
)
from dekf_bench.utils.config import load_config

TINY = ReferenceSettings(
    train_trajectories=4, blocks_per_trajectory=25, val_blocks=20, test_blocks=40,
    epochs=2, patience=1, batch=16,
)


@pytest.fixture(scope="module")
def series():
    config = load_config(
        "x1_stationary",
        overrides={
            "run": {"name": "probe", "horizon": 10, "seeds": [0]},
            "env": {"dataset": "mackey_glass", "series": {"burn_in": 50.0}},
            "model": {"name": "causal_transformer", "likelihood": "gaussian", "output_dim": 31},
        },
    )
    return config.env.series


def test_persistence_on_a_known_case() -> None:
    inputs = np.array([[0.0, 1.0, 2.0]])
    targets = np.array([[1.0, 2.0, 3.0]])
    assert persistence_rmse(inputs, targets) == pytest.approx(1.0)


def test_the_reference_trains_and_reports_every_line(series) -> None:
    score = train_reference(series, 0.2, TINY, cache_dir=None)
    assert score.train_blocks == 100
    assert 1 <= score.best_epoch <= TINY.epochs
    assert np.isfinite([score.rmse, score.rmse_full_context, score.persistence_rmse]).all()
    assert score.noise_floor == pytest.approx(series.sigma)


def test_the_cache_returns_what_it_stored(series, tmp_path) -> None:
    first = train_reference(series, 0.2, TINY, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 1
    second = train_reference(series, 0.2, TINY, cache_dir=tmp_path)
    assert first == second


def test_a_different_law_is_a_different_cache_entry(series, tmp_path) -> None:
    train_reference(series, 0.2, TINY, cache_dir=tmp_path)
    train_reference(series, 0.22, TINY, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_training_is_deterministic(series) -> None:
    assert train_reference(series, 0.2, TINY, cache_dir=None) == train_reference(
        series, 0.2, TINY, cache_dir=None
    )
    assert torch.get_default_dtype() == torch.float32  # no global state was touched
