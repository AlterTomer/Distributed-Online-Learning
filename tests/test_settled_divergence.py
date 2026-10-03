"""The tuning scorer must treat a partly diverged learner as unusable (D126).

`run_m3_rates.settled` scores a rate by its settled RMSE. A diverged regression learner
records NaN, and pandas' mean skips NaN, so a rate that diverged on one seed of two used
to score as the other seed alone -- which is how M2O's extended grid selected 1e-3 over
the stable 3e-4. These pin the corrected behaviour: any NaN in the window is inf.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_m3_rates  # noqa: E402


def _run(root: Path, name: str, seeds: dict[int, list[float]]) -> None:
    """A completed run of one learner, one value per evaluation step, per seed."""
    directory = root / "results" / name
    directory.mkdir(parents=True)
    for seed, values in seeds.items():
        pd.DataFrame({
            "learner": "centralized_sgd", "metric": "rmse", "evalset": "current",
            "t": [25 * (i + 1) for i in range(len(values))], "value": values,
        }).to_parquet(directory / f"seed_{seed}.parquet")
    (directory / "_complete").write_text("", encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_m3_rates, "ROOT", tmp_path)
    return tmp_path


def test_stable_run_scores_its_settled_mean(root):
    _run(root, "stable", {0: [0.3, 0.2, 0.1, 0.1, 0.1], 1: [0.3, 0.2, 0.2, 0.2, 0.2]})
    # The last 20% of t: only the final evaluation of each seed (t >= 100).
    assert run_m3_rates.settled("stable", "centralized_sgd") == pytest.approx(0.15)


def test_one_diverged_seed_makes_the_rate_unusable(root):
    _run(root, "half", {0: [0.3] + [float("nan")] * 4, 1: [0.3, 0.2, 0.1, 0.1, 0.1]})
    assert run_m3_rates.settled("half", "centralized_sgd") == float("inf")


def test_a_single_nan_in_the_window_is_enough(root):
    _run(root, "late", {0: [0.3, 0.2, 0.1, 0.1, float("nan")], 1: [0.3, 0.2, 0.1, 0.1, 0.1]})
    assert run_m3_rates.settled("late", "centralized_sgd") == float("inf")


def test_nan_before_the_window_does_not_count(root):
    _run(root, "early", {0: [float("nan"), 0.2, 0.1, 0.1, 0.1]})
    assert run_m3_rates.settled("early", "centralized_sgd") == pytest.approx(0.1)


def test_missing_run_is_unusable(root):
    assert run_m3_rates.settled("absent", "centralized_sgd") == float("inf")
