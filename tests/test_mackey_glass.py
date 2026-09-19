"""The Mackey--Glass generator: accuracy, determinism, and the batch contract.

The integrator is the one piece of the second task that cannot be checked against
the first, so it is checked against the mathematics instead: an exact equilibrium,
convergence under step refinement, and independence of trajectories batched
together.
"""

from __future__ import annotations

import numpy as np
import pytest

from dekf_bench.data import mackey_glass as mg


def test_the_equilibrium_is_held_exactly() -> None:
    """x* = (beta/gamma - 1)^(1/n) = 1 at the default law, and f(1, 1) is exactly 0.

    Unstable, but in exact arithmetic nothing perturbs it: a single bug in the
    delayed-value bookkeeping moves it off at once.
    """
    clean = mg.integrate(200, 1.0, burn_in=0.0)
    assert np.max(np.abs(clean - 1.0)) < 1e-12


def test_the_first_delay_interval_matches_the_closed_form() -> None:
    r"""On [0, tau] the delayed term is the constant history, so the DDE is a linear ODE.

    x' = K - gamma x with K = beta c / (1 + c^n): x(t) = K/gamma + (c - K/gamma) e^{-gamma t}.
    This checks RK4 itself, before any interpolation is involved.
    """
    c = 0.9
    k = mg.BETA * c / (1.0 + c**mg.EXPONENT)
    t = np.arange(17.0)
    exact = k / mg.GAMMA + (c - k / mg.GAMMA) * np.exp(-mg.GAMMA * t)
    assert np.max(np.abs(mg.integrate(17, c, burn_in=0.0)[0] - exact)) < 1e-10


def _refinement_ratios(span: int) -> list[float]:
    fine = mg.integrate(span, 0.9, burn_in=0.0, dt=0.025)
    errors = [
        np.max(np.abs(mg.integrate(span, 0.9, burn_in=0.0, dt=dt) - fine)) for dt in (0.2, 0.1, 0.05)
    ]
    return [errors[0] / errors[1], errors[1] / errors[2]]


def test_rk4_is_fourth_order_across_the_first_kink() -> None:
    """Up to the second breaking point (2 tau = 34) halving dt divides the error by ~16.

    This window contains the history's kink arriving at t = tau -- the step a
    centred stencil gets wrong. Before the one-sided stencil the ratio was 4-5.
    """
    assert all(ratio > 12.0 for ratio in _refinement_ratios(33))


def test_rk4_converges_under_step_refinement() -> None:
    """dt and dt/2 agree to far below the noise levels the task uses, over 200 units.

    Past 2 tau the x'' jump costs an order at a few steps (measured ratio ~9, third
    order), and chaos amplifies what remains by about e^(0.006 * 200) ~ 3, so this
    is a bound, not an order test. Measured 2.8e-8; it was 2.4e-5 before the
    one-sided stencil.
    """
    coarse = mg.integrate(200, 0.9, burn_in=0.0, dt=0.1)
    fine = mg.integrate(200, 0.9, burn_in=0.0, dt=0.05)
    assert np.max(np.abs(coarse - fine)) < 1e-7


def test_the_series_is_deterministic() -> None:
    first = mg.integrate(300, [0.7, 1.2], burn_in=50.0)
    second = mg.integrate(300, [0.7, 1.2], burn_in=50.0)
    assert np.array_equal(first, second)


def test_batched_trajectories_do_not_interact() -> None:
    """Integrating agents together is the same as integrating each alone."""
    together = mg.integrate(300, [0.7, 1.2, 0.95], burn_in=50.0)
    for row, start in enumerate([0.7, 1.2, 0.95]):
        alone = mg.integrate(300, start, burn_in=50.0)
        assert np.array_equal(together[row], alone[0])


def test_per_trajectory_delays_match_solo_runs() -> None:
    """The gather path for mixed tau gives what the slicing path gives alone."""
    mixed = mg.integrate(300, [0.8, 0.8], tau=[17.0, 20.0], burn_in=50.0)
    assert np.array_equal(mixed[0], mg.integrate(300, 0.8, tau=17.0, burn_in=50.0)[0])
    assert np.array_equal(mixed[1], mg.integrate(300, 0.8, tau=20.0, burn_in=50.0)[0])


def test_a_constant_beta_array_is_the_scalar() -> None:
    scalar = mg.integrate(100, 0.9, beta=0.2, burn_in=10.0)
    per_sample = mg.integrate(100, 0.9, beta=np.full((1, 100), 0.2), burn_in=10.0)
    assert np.array_equal(scalar, per_sample)


def test_beta_drift_takes_effect_from_its_sample_on() -> None:
    """Changing beta at sample j leaves samples 0..j untouched and moves later ones."""
    base = np.full((1, 200), 0.2)
    switched = base.copy()
    switched[:, 100:] = 0.25
    before = mg.integrate(200, 0.9, beta=base, burn_in=10.0)
    after = mg.integrate(200, 0.9, beta=switched, burn_in=10.0)
    assert np.array_equal(before[:, :101], after[:, :101])
    assert not np.allclose(before[:, 110:], after[:, 110:])


def test_the_system_is_chaotic_at_the_default_law() -> None:
    """Two histories 1e-8 apart diverge at the known rate, about 0.006 per time unit.

    The benchmark relies on agents' trajectories decorrelating. Were the default law
    periodic, every agent would eventually hold the same data. The published largest
    Lyapunov exponent at tau = 17 is about 0.006; a first version of this test
    demanded a 1e-2 gap within 2000 units, which that rate does not reach.
    """
    pair = mg.integrate(3000, [0.9, 0.9 + 1e-8], burn_in=0.0)
    gap = np.abs(pair[0] - pair[1])
    window = np.nonzero((gap > 1e-7) & (gap < 1e-3))[0]
    rate = np.polyfit(window, np.log(gap[window]), 1)[0]
    assert 0.003 < rate < 0.012, rate
    assert gap[-500:].max() > 1e3 * gap[:50].max()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tau": 17.05}, "tau"),
        ({"delta": 0.25}, "delta"),
        ({"tau": 0.2}, "three integration steps"),
    ],
)
def test_invalid_steps_are_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(mg.MackeyGlassError, match=match):
        mg.integrate(10, 0.9, burn_in=0.0, **kwargs)


def test_reference_moments_reproduce() -> None:
    """The standardisation constants are a fixed function of their arguments."""
    mg.reference_moments.cache_clear()
    first = mg.reference_moments(n_samples=2000)
    mg.reference_moments.cache_clear()
    assert mg.reference_moments(n_samples=2000) == first


def test_standardised_series_has_roughly_unit_spread() -> None:
    moments = mg.reference_moments(n_samples=2000)
    rng = np.random.default_rng(3)
    clean = mg.integrate(2000, mg.initial_histories(4, rng))
    z = mg.standardise(clean, moments)
    assert abs(float(z.mean())) < 0.2
    assert 0.8 < float(z.std()) < 1.2


def test_the_sensor_adds_noise_at_the_requested_level() -> None:
    rng = np.random.default_rng(0)
    clean = np.zeros((3, 20_000))
    z = mg.observe(clean, np.array([0.01, 0.05, 0.1]), rng)
    assert np.allclose(z.std(axis=1), [0.01, 0.05, 0.1], rtol=0.05)


def test_gain_and_bias_act_on_the_measurement_only() -> None:
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    clean = np.linspace(-1, 1, 50)[None, :]
    plain = mg.observe(clean, 0.0, rng_a)
    shifted = mg.observe(clean, 0.0, rng_b, gain=2.0, bias=0.5)
    assert np.allclose(shifted, 2.0 * plain + 0.5)


def test_blocks_never_reuse_a_target() -> None:
    """Every sample is a target at most once; the boundary transition is dropped."""
    series = np.arange(2 * 100, dtype=np.float64).reshape(2, 100)
    inputs, targets = mg.to_blocks(series, length=32)
    assert inputs.shape == targets.shape == (2, 3, 31)
    flat = targets[0].reshape(-1)
    assert len(np.unique(flat)) == flat.size
    # Targets are one step ahead of the inputs within a block...
    assert np.array_equal(targets[..., :-1], inputs[..., 1:])
    # ...and block b+1 starts fresh: its first input is not predicted by block b.
    assert inputs[0, 1, 0] == 32.0 and targets[0, 0, -1] == 31.0
