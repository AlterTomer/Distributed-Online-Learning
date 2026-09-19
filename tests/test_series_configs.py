"""The shipped M-series configs load, and carry what M0 decided (D96)."""

from __future__ import annotations

import pytest

from dekf_bench.likelihoods.registry import build_likelihood
from dekf_bench.utils.config import load_config

EXPERIMENTS = ["m_stationary", "m_linear", "m_abrupt", "m_linear_ar"]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_m_config_loads_as_a_series_run(name: str) -> None:
    config = load_config(name)
    assert config.env.is_series
    assert config.env.series.sigma == pytest.approx(0.1)
    assert config.env.series.beta == pytest.approx(0.22)
    assert config.env.series.span == pytest.approx(0.02)
    assert config.run.dtype == "float64"
    assert "backward" not in config.eval.evalsets


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_likelihood_carries_the_measured_profile(name: str) -> None:
    likelihood = build_likelihood(load_config(name))
    assert likelihood.variances is not None and len(likelihood.variances) == 31
    worst = max(range(31), key=lambda i: likelihood.variances[i])
    # The Transformer's worst position is the first (one sample of context); the
    # linear AR's is the second (D96).
    assert worst == (1 if name == "m_linear_ar" else 0)


def test_the_adamw_baselines_build_and_price_their_moments() -> None:
    """Decision 19: AdamW arms beside the SGD ones, under their own names."""
    from dekf_bench.learners.registry import DIFFUSING, POOLING, build_learners
    from dekf_bench.models.registry import build_model_from_config

    config = load_config(
        "m_stationary",
        overrides={"learners": ["centralized_adamw", "diffusion_atc_adamw", "local_adamw",
                                "diffusion_sgd_atc"]},
    )
    model = build_model_from_config(config)
    learners = build_learners(config, model, build_likelihood(config))
    p = model.num_params
    assert "centralized_adamw" in POOLING and "diffusion_atc_adamw" in DIFFUSING
    # Both moments travel with the parameters: 3p per link, against momentum ATC's 2p.
    assert learners["diffusion_atc_adamw"].comm_scalars_per_step(1) == 3 * p * 2
    assert learners["diffusion_sgd_atc"].comm_scalars_per_step(1) == 2 * p * 2
    assert learners["local_adamw"].comm_scalars_per_step(1) == 0


def test_the_drift_stays_inside_the_chaotic_window() -> None:
    """Every beta the abrupt and linear schedules visit lies in [0.20, 0.24]."""
    from dekf_bench.env.drift import build_drift
    from dekf_bench.env.series import channel_value

    for name in ("m_linear", "m_abrupt"):
        config = load_config(name)
        drift = build_drift(config)
        values = [float(channel_value(config.env.series, drift.rotation_at(t)))
                  for t in range(config.run.horizon)]
        assert min(values) >= 0.20 - 1e-12 and max(values) <= 0.24 + 1e-12, name
