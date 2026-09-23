"""The shipped M-series configs load, and carry what M0 decided (D96)."""

from __future__ import annotations

import pytest

from dekf_bench.likelihoods.registry import build_likelihood
from dekf_bench.utils.config import load_config

EXPERIMENTS = ["m_stationary", "m_linear", "m_abrupt", "m_linear_ar",
               "m_tau_spread", "m_tau_control"]

#: Delays measured chaotic from ALL FIVE initial histories at beta = 0.22, with
#: the pilot's own estimator at span 3000. Not a general claim about
#: Mackey--Glass: the window is ragged. 18.4-18.6 and 19.3-19.6 are
#: start-dependent -- some histories settle on a periodic attractor and others do
#: not -- and 18.7 and 19.2 are marginal at min lambda near 0.001. Longer spans
#: do not help: once the gap saturates it re-enters the fit window and flattens
#: the slope, so span 3000 is the reference and 6000 reads lower.
VERIFIED_CHAOTIC_TAU = {16.4, 16.8, 17.2, 17.6, 18.0, 18.2, 18.3, 18.9, 19.1,
                        19.8, 20.0}


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


def test_the_delays_stay_chaotic() -> None:
    """The tau analogue of the window guard above (M8, decision 22).

    beta has a contiguous window and the guard above walks the whole schedule.
    tau does not: the chaotic region is ragged, and heterogeneity picks a SET of
    delays rather than traversing an interval, so each value is checked on its
    own. A tau edited into 18.5 would put some agents on a periodic attractor
    and leave others chaotic *inside one run* -- agents draw their initial
    histories from [0.5, 1.5] -- which is a quiet failure, not a loud one.

    The whole-step check is not belt and braces: the integrator raises on a tau
    that is not a multiple of dt, so 16.25 is rejected outright. The legal grid
    is 0.1, which is also the staircase any future tau drift would move on.
    """
    from dekf_bench.data.mackey_glass import DT

    spread = load_config("m_tau_spread").env.series
    control = load_config("m_tau_control").env.series
    assert len(spread.tau_values) == 10, "one delay per agent, cycled otherwise"
    for tau in [*spread.tau_values, spread.tau, control.tau]:
        assert tau in VERIFIED_CHAOTIC_TAU, f"{tau} has not been measured chaotic"
        steps = tau / DT
        assert abs(steps - round(steps)) < 1e-9, f"{tau} is not a whole number of dt"


def test_the_delay_cells_match_on_the_mean_and_the_evalset_law() -> None:
    """`tau` and `tau_values` feed different paths, and the spread cell needs both.

    `agent_laws` gives each agent a value from `tau_values`; `law_blocks` builds
    the held-out sets from the scalar `tau`. Left at the 17.0 default, the
    spread cell's evalsets would sit at a delay no agent follows -- offset by
    1.2 from their mean -- and `spread - control` would carry that offset rather
    than the heterogeneity. Matching the two is what makes the contrast clean.

    The mean must also match, because tau = 17 cannot sit at the centre of a set
    wide enough to be detectable: chaos ends just below 16.4, so the set runs
    upward and averages 18.21.
    """
    spread = load_config("m_tau_spread").env.series
    control = load_config("m_tau_control").env.series
    assert spread.tau == pytest.approx(control.tau)
    assert sum(spread.tau_values) / len(spread.tau_values) == pytest.approx(
        control.tau, abs=0.05
    )
    assert not control.tau_values, "the control cell must be homogeneous"
