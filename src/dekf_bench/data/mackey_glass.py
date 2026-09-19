r"""The Mackey--Glass delay system: the second task's data, generated rather than loaded.

Each agent observes its own realisation of

$$\frac{\mathrm dx}{\mathrm dt}=\beta\,\frac{x(t-\tau)}{1+x(t-\tau)^{n}}-\gamma x(t),
\qquad \beta=0.2,\ \gamma=0.1,\ n=10,\ \tau=17,$$

sampled every $\Delta=1$ time unit (`docs/mackey_glass_plan.md`, decision 1). At
$\tau=17$ the system is mildly chaotic, so agents started from different histories
decorrelate and each holds genuinely different data from one shared law.

**Why this module exists rather than a cached dataset.** On MNIST, drift is a
transform applied to images loaded once. Here drift is in the *law*: a time-varying
$\beta_t$ changes the trajectory itself, and $\beta$ at time $t$ shapes $x$ at every
later time. So the data are a function of the drift schedule and the seed, and are
integrated per run -- seconds of work, deterministic, never cached to disk.

**The integrator.** Classical RK4 at a fixed step $dt=0.1$, so that $\tau$ is a whole
number of steps and the delayed value at a grid point is read from the history
exactly. RK4 also needs the delayed value half a step off the grid; it comes from
4-point cubic interpolation of the history, which keeps the scheme fourth-order.
Linear interpolation there would silently drop it to second order.

**The kink a constant history leaves.** The solution is continuous at $t=0$ but its
derivative jumps there -- zero before, $f(c,c)$ after -- and the delay delivers that
kink into the right-hand side at $t=\tau$. A centred stencil straddling it is only
first-order accurate locally, and a first version that ignored it converged at
**second** order globally (error ratio 4--5 per halving of $dt$, measured), burn-in
included, since the error is made during the burn-in and carried out of it. So the
stencil is one-sided there: inside the constant history the delayed value *is* the
constant, and on the first step past the kink the four points come from the smooth
side. Measured after the fix: ratio 16 per halving up to $2\tau$, and the
$dt=0.1$ against $dt=0.05$ gap over 200 units fell from $2.4\times10^{-5}$ to
$2.8\times10^{-8}$. The next breaking point -- a jump in $x''$ at $t=\tau$, felt at
$2\tau$ -- costs one order at a handful of steps (ratio about 9 past it, third order)
and is left alone. All of it lies in
the burn-in, and at $dt=0.1$ the whole effect was $7\times10^{-6}$ before the fix --
far below the smallest noise level used -- so the fix is about the scheme being
what it says, not about the benchmark.

Everything is vectorised across trajectories: the first axis of every array is the
trajectory (an agent, or an agent in one seed), so ten agents cost one loop, not ten.
Each trajectory may carry its own $\beta$ (per sample, for drift), its own $\tau$
(for heterogeneity) and its own initial history.

**Units.** :func:`integrate` returns the clean state in its natural units.
:func:`standardise` maps it through *fixed* constants from one long stationary run
(decision 3), never per-run statistics -- a drift that changes the amplitude must
stay visible as drift. :func:`observe` then adds the sensor: gain, bias and noise, all
in standardised units, so $\sigma$ reads as a fraction of the signal's spread.
"""

from __future__ import annotations

from functools import cache

import numpy as np

#: The law, as settled in the design doc.
BETA = 0.2
GAMMA = 0.1
EXPONENT = 10.0
TAU = 17.0

#: Integration step and sampling step, in time units (decision 1).
DT = 0.1
DELTA = 1.0

#: Discarded before the first sample, so every trajectory starts on the attractor
#: rather than in the transient from its constant history (decision 4).
BURN_IN = 1000.0

#: Where a constant initial history is drawn from. Both ends lie in the attractor's
#: range at the default law, so no trajectory starts somewhere implausible.
HISTORY_LOW = 0.5
HISTORY_HIGH = 1.5


class MackeyGlassError(ValueError):
    """Raised for a law or a step size the integrator cannot honour."""


def _whole_steps(length: np.ndarray | float, dt: float, what: str) -> np.ndarray:
    steps = np.asarray(length, dtype=np.float64) / dt
    rounded = np.rint(steps)
    if np.any(np.abs(steps - rounded) > 1e-9):
        raise MackeyGlassError(
            f"{what} must be a whole number of integration steps of {dt}; got {length}"
        )
    return rounded.astype(np.int64)


def integrate(
    n_samples: int,
    initial: np.ndarray | float,
    *,
    beta: np.ndarray | float = BETA,
    gamma: float = GAMMA,
    exponent: float = EXPONENT,
    tau: np.ndarray | float = TAU,
    dt: float = DT,
    delta: float = DELTA,
    burn_in: float = BURN_IN,
) -> np.ndarray:
    r"""Integrate one or more trajectories; return ``(n_trajectories, n_samples)``.

    Args:
        n_samples: samples to return per trajectory, taken every ``delta`` after the
            burn-in.
        initial: the constant history $x(t)=c$ for $t\le0$, one per trajectory.
        beta: a scalar, one value per trajectory ``(n,)``, or one per trajectory
            per sample ``(n, n_samples)``. Sample $j$ is integrated forward with
            column $j$; the burn-in uses column 0.
        tau: a scalar or one delay per trajectory; each a whole number of ``dt``.
        dt, delta, burn_in: in time units; ``delta`` and ``burn_in`` must be whole
            multiples of ``dt``.

    Returns the **clean** state. Noise and scaling belong to :func:`observe`.
    """
    if n_samples < 1:
        raise MackeyGlassError(f"n_samples must be >= 1, got {n_samples}")
    initial = np.atleast_1d(np.asarray(initial, dtype=np.float64))
    n = initial.size
    per_sample = int(_whole_steps(delta, dt, "delta"))
    burn = int(_whole_steps(burn_in, dt, "burn_in"))
    lag = np.broadcast_to(_whole_steps(tau, dt, "tau"), (n,)).copy()
    if lag.min() < 3:
        # Three, because the one-sided stencil past the history's kink reads three
        # grid points beyond the delayed one, and all of them must already exist.
        raise MackeyGlassError(
            f"tau must span at least three integration steps; got {np.asarray(tau)} at dt={dt}"
        )

    rates = np.asarray(beta, dtype=np.float64)
    if rates.ndim == 0:
        rates = np.full((n, n_samples), float(rates))
    elif rates.ndim == 1:
        rates = np.broadcast_to(rates.reshape(n, 1), (n, n_samples))
    elif rates.shape != (n, n_samples):
        raise MackeyGlassError(
            f"beta must be a scalar, (n,) or (n, n_samples) = ({n}, {n_samples}); "
            f"got {rates.shape}"
        )

    steps = burn + n_samples * per_sample
    # One extra slot of history behind the longest delay: the cubic interpolation
    # reads one point before the delayed grid value.
    offset = int(lag.max()) + 2
    buffer = np.empty((n, offset + steps + 1), dtype=np.float64)
    buffer[:, : offset + 1] = initial[:, None]

    rows = np.arange(n)
    uniform = bool(np.all(lag == lag[0]))
    shift = int(lag[0])

    def rhs(x: np.ndarray, delayed: np.ndarray, rate: np.ndarray) -> np.ndarray:
        return rate * delayed / (1.0 + delayed**exponent) - gamma * x

    for k in range(steps):
        i = offset + k
        sample = (k - burn) // per_sample
        rate = rates[:, sample] if sample >= 0 else rates[:, 0]
        if uniform:
            # Plain slicing when every trajectory shares tau: the common case, and
            # several times faster than a gather per step.
            j = i - shift
            before, at, after, beyond, further = (
                buffer[:, j - 1],
                buffer[:, j],
                buffer[:, j + 1],
                buffer[:, j + 2],
                buffer[:, j + 3],
            )
        else:
            j = i - lag
            before, at, after, beyond, further = (
                buffer[rows, j - 1],
                buffer[rows, j],
                buffer[rows, j + 1],
                buffer[rows, j + 2],
                buffer[rows, j + 3],
            )
        # x(t + dt/2 - tau), between grid points j and j+1 of the history. Cubic
        # Lagrange at the midpoint, centred (-1, 9, 9, -1)/16 where the history is
        # smooth -- but index `offset` (t = 0) is a kink, and a stencil straddling
        # it is first-order. Inside the constant history the value is the constant;
        # on the first step past the kink the stencil is one-sided, (5, 15, -5, 1)/16
        # on points j..j+3. See the module docstring.
        centred = (-before + 9.0 * at + 9.0 * after - beyond) / 16.0
        forward = (5.0 * at + 15.0 * after - 5.0 * beyond + further) / 16.0
        if uniform:
            if j + 1 <= offset:
                half = at
            elif j == offset:
                half = forward
            else:
                half = centred
        else:
            half = np.where(j + 1 <= offset, at, np.where(j == offset, forward, centred))
        x = buffer[:, i]
        k1 = rhs(x, at, rate)
        k2 = rhs(x + 0.5 * dt * k1, half, rate)
        k3 = rhs(x + 0.5 * dt * k2, half, rate)
        k4 = rhs(x + dt * k3, after, rate)
        buffer[:, i + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    start = offset + burn
    return buffer[:, start : start + n_samples * per_sample : per_sample].copy()


def initial_histories(
    n: int, rng: np.random.Generator, low: float = HISTORY_LOW, high: float = HISTORY_HIGH
) -> np.ndarray:
    """One constant history per trajectory, uniform on ``[low, high)``."""
    return rng.uniform(low, high, size=n)


@cache
def reference_moments(
    beta: float = BETA, n_trajectories: int = 10, n_samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    r"""Mean and standard deviation of the clean series at a fixed law.

    The standardisation constants (decision 3). Computed from a long *stationary*
    run with a fixed seed, so they are the same for every experiment and a drift in
    amplitude is never normalised away. Cached: every caller in a process shares one
    integration.
    """
    rng = np.random.default_rng(seed)
    clean = integrate(n_samples, initial_histories(n_trajectories, rng), beta=beta)
    return float(clean.mean()), float(clean.std())


def standardise(clean: np.ndarray, moments: tuple[float, float] | None = None) -> np.ndarray:
    """Map the clean state through the fixed reference constants."""
    mean, std = reference_moments() if moments is None else moments
    return (clean - mean) / std


def observe(
    standardised: np.ndarray,
    sigma: np.ndarray | float,
    rng: np.random.Generator,
    *,
    gain: np.ndarray | float = 1.0,
    bias: np.ndarray | float = 0.0,
) -> np.ndarray:
    r"""The sensor: $z=g\,x+b+\sigma\varepsilon$, all in standardised units.

    ``gain`` and ``bias`` default to the identity sensor; varying them over time is
    the sensor-drift channel (decision 8), which changes what is measured without
    touching the dynamics. ``sigma`` may differ per trajectory -- the heterogeneous
    sensor-quality condition (decision 22).
    """
    noise = rng.standard_normal(standardised.shape)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    return np.asarray(gain) * standardised + np.asarray(bias) + sigma * noise


def to_blocks(series: np.ndarray, length: int = 32) -> tuple[np.ndarray, np.ndarray]:
    r"""Cut each trajectory into non-overlapping blocks; return ``(inputs, targets)``.

    A block of ``length`` samples $[x_1..x_L]$ yields inputs $[x_1..x_{L-1}]$ and
    targets $[x_2..x_L]$, so every target is one step ahead of the last input it can
    see. Blocks never overlap, and the transition from one block's last sample into
    the next block's first is dropped: **every sample is a target at most once**,
    the temporal counterpart of the reachability guard (P5.25).

    Shapes ``(n, n_blocks, length - 1)`` for both; a trailing partial block is
    discarded.
    """
    n, samples = series.shape
    n_blocks = samples // length
    if n_blocks == 0:
        raise MackeyGlassError(f"{samples} samples cannot fill one block of {length}")
    blocks = series[:, : n_blocks * length].reshape(n, n_blocks, length)
    return blocks[..., :-1].copy(), blocks[..., 1:].copy()
