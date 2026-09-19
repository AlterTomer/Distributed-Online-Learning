r"""The series environment: what each agent observes each round, on Mackey--Glass.

The counterpart of `environment.py` for a task whose data are *generated from a
law* rather than loaded (`docs/mackey_glass_plan.md`, WP2). It answers the same
question -- what does agent $v$ see at step $t$? -- through the same interface
(``step``, ``observe``, ``pool``, ``assert_unmodified``, ``graph``, ``n_nodes``,
``horizon``), so the runner, the learners and the filter need no second code path.

**Round $t$ is a fixed stretch of physical time.** Each agent's whole series for
the run is integrated once, at build time: $T$ rounds of $n_b$ blocks of $L$
samples. Round $t$ of agent $v$ is always the same $n_bL$ samples, so an
unavailable block is a *sensor dropout* -- the series moves on, the agent just
does not see that stretch -- rather than a sample held back for later.

**Blocks never overlap and every target is used once** (`to_blocks`): the
temporal counterpart of the reachability guard.

**Drift acts on the law, through the shared schedules.** A schedule's displacement
$d$ (in its degree units, under the 45-degree cap) moves the configured channel by
``span * d / 45``: $\beta$ for the law, gain or bias for the sensor. $\beta$ is
applied per sample inside the integration, so it shapes every later sample, as
drift in a law must. Gain and bias act on the observation only and leave the
dynamics untouched (decision 8).

**Twins by construction.** Initial histories and observation noise are drawn from
seed streams that do not depend on the drift, so a stationary twin shares every
history and noise draw with its drifting run, and the two differ in the drift
alone.

**Observations are shared and read-only**, exactly as on MNIST: one instance goes
to every learner in the run, and :meth:`SeriesEnvironment.assert_unmodified` lets
the runner check it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dekf_bench.data import mackey_glass as mg
from dekf_bench.env.drift import MAX_WELL_POSED_DEGREES, Drift, DriftState, build_drift
from dekf_bench.env.graph import Graph, Graphs, build_graphs
from dekf_bench.env.stream import _label_mask
from dekf_bench.runner.seeding import Seeds
from dekf_bench.utils.determinism import resolve_device


class SeriesError(RuntimeError):
    """Raised when the series environment is misconfigured or corrupted."""


@dataclass(frozen=True)
class SeriesObservation:
    """What one agent receives in one round.

    The fields every learner reads -- ``x``, ``y``, ``has_label``, ``n_samples``,
    ``node``, ``step`` -- with the shapes of this task: ``x`` and ``y`` are
    ``(n_blocks, L - 1)`` inputs and one-step-ahead targets, and a "sample" is a
    block. ``drift_value`` is the drifting channel's value this round ($\\beta$,
    gain or bias).
    """

    x: torch.Tensor
    y: torch.Tensor | None
    has_label: bool
    n_samples: int
    node: int
    step: int
    drift_value: float

    def __post_init__(self) -> None:
        if self.has_label != (self.n_samples > 0):
            raise SeriesError(
                f"agent {self.node} at step {self.step}: has_label={self.has_label} "
                f"disagrees with n_samples={self.n_samples}"
            )
        if self.x.shape[0] != self.n_samples:
            raise SeriesError(
                f"agent {self.node} at step {self.step}: {self.x.shape[0]} blocks but "
                f"n_samples={self.n_samples}"
            )
        if self.has_label and (self.y is None or self.y.shape != self.x.shape):
            raise SeriesError(
                f"agent {self.node} at step {self.step}: targets must match the inputs' "
                f"shape {tuple(self.x.shape)}"
            )
        if not self.has_label and self.y is not None:
            raise SeriesError(f"agent {self.node} at step {self.step}: idle but y is not None")

    @property
    def rotation_degrees(self) -> float:
        """The drift state under the name the protocol and the schema read.

        On this task it is the channel's value, not an angle; the schema records it
        as ``drift_state`` either way.
        """
        return self.drift_value

    def __len__(self) -> int:
        return self.n_samples

    def checksum(self) -> tuple[float, float]:
        """A cheap fingerprint, for detecting in-place mutation by a learner."""
        if not self.n_samples:
            return 0.0, 0.0
        return float(self.x.to(torch.float64).sum()), float(self.y.to(torch.float64).sum())


def pool(observations: dict[int, SeriesObservation]) -> tuple[torch.Tensor, torch.Tensor]:
    """Every available agent's blocks, stacked: what a pooled learner trains on."""
    available = [obs for obs in observations.values() if obs.has_label]
    if not available:
        example = next(iter(observations.values()))
        empty = torch.empty(
            (0, example.x.shape[1]), dtype=example.x.dtype, device=example.x.device
        )
        return empty, empty.clone()
    return (
        torch.cat([obs.x for obs in available]),
        torch.cat([obs.y for obs in available]),  # type: ignore[misc]
    )


# --------------------------------------------------------------------------- #
# the law each agent follows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Laws:
    """Per-agent law and sensor, before any drift: heterogeneity (decision 22)."""

    beta: np.ndarray
    tau: np.ndarray
    sigma: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "beta": self.beta.tolist(),
            "tau": self.tau.tolist(),
            "sigma": self.sigma.tolist(),
        }


def _evenly(center: float, half_width: float, n: int) -> np.ndarray:
    if n == 1 or half_width == 0.0:
        return np.full(n, center)
    return center + np.linspace(-half_width, half_width, n)


def agent_laws(series: Any, n_nodes: int) -> Laws:
    """Evenly spread offsets rather than random draws, as `Drift` spreads rates:
    the heterogeneity is then the same in every seed, and only the data vary."""
    beta = _evenly(series.beta, series.beta_spread, n_nodes)
    sigma = _evenly(series.sigma, series.sigma * series.sigma_spread, n_nodes)
    if series.tau_values:
        tau = np.array([series.tau_values[v % len(series.tau_values)] for v in range(n_nodes)])
    else:
        tau = np.full(n_nodes, series.tau)
    return Laws(beta=beta, tau=tau, sigma=sigma)


def channel_value(series: Any, displacement: float | np.ndarray, base_beta: Any = None):
    """The drifting channel's value at a schedule displacement (degree units)."""
    fraction = np.asarray(displacement, dtype=np.float64) / MAX_WELL_POSED_DEGREES
    if series.channel == "beta":
        base = series.beta if base_beta is None else base_beta
        return base + series.span * fraction
    if series.channel == "gain":
        return 1.0 + series.span * fraction
    return series.span * fraction  # bias


def law_blocks(
    series: Any,
    value: float,
    n_trajectories: int,
    blocks_each: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Blocks from fresh trajectories at one *fixed* channel value.

    Samples the law's stationary distribution at that value, through the central
    sensor -- what the held-out sets and the offline reference both need. Histories
    are drawn from ``rng`` before the noise, in that order. Shapes
    ``(n_trajectories, blocks_each, L - 1)``.
    """
    histories = mg.initial_histories(n_trajectories, rng)
    clean = mg.integrate(
        blocks_each * series.length,
        histories,
        beta=value if series.channel == "beta" else series.beta,
        gamma=series.gamma,
        exponent=series.exponent,
        tau=series.tau,
        dt=series.dt,
        delta=series.delta,
        burn_in=series.burn_in,
    )
    z = mg.observe(
        mg.standardise(clean),
        series.sigma,
        rng,
        gain=value if series.channel == "gain" else 1.0,
        bias=value if series.channel == "bias" else 0.0,
    )
    return mg.to_blocks(z, series.length)


def generate(
    series: Any,
    laws: Laws,
    values: np.ndarray,
    histories: np.ndarray,
    noise: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate, standardise and observe; return blocks ``(N, T, n_blocks, L - 1)``.

    ``values`` is the channel per agent per round, ``(N, T)``.
    """
    n_nodes, n_rounds = values.shape
    per_round = series.n_blocks * series.length
    per_sample = np.repeat(values, per_round, axis=1)
    beta = per_sample if series.channel == "beta" else laws.beta
    clean = mg.integrate(
        n_rounds * per_round,
        histories,
        beta=beta,
        gamma=series.gamma,
        exponent=series.exponent,
        tau=laws.tau,
        dt=series.dt,
        delta=series.delta,
        burn_in=series.burn_in,
    )
    gain = per_sample if series.channel == "gain" else 1.0
    bias = per_sample if series.channel == "bias" else 0.0
    z = mg.observe(mg.standardise(clean), laws.sigma, noise, gain=gain, bias=bias)
    inputs, targets = mg.to_blocks(z, series.length)
    shape = (n_nodes, n_rounds, series.n_blocks, series.length - 1)
    return inputs.reshape(shape), targets.reshape(shape)


# --------------------------------------------------------------------------- #
# the environment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeriesEnvironment:
    """Graph, law, drift and data, composed. Frozen and positional."""

    config: Any
    seeds: Seeds
    graphs: Graphs
    drift: Drift
    laws: Laws
    #: The channel's value per agent per round, ``(N, T)``.
    values: np.ndarray
    inputs: torch.Tensor
    targets: torch.Tensor
    #: Which rounds each agent observes, ``(N, T)`` -- block availability.
    available: torch.Tensor

    @property
    def graph(self) -> Graph:
        return self.graphs.comm

    @property
    def horizon(self) -> int:
        return int(self.config.run.horizon)

    @property
    def n_nodes(self) -> int:
        return self.graphs.n_nodes

    @property
    def device(self) -> torch.device:
        return self.inputs.device

    def drift_state(self, step: int, node: int | None = None) -> DriftState:
        self._check_step(step)
        return DriftState(
            step=step, rotation_degrees=float(self.values[node or 0, step]), node=node
        )

    def step(self, step: int) -> dict[int, SeriesObservation]:
        self._check_step(step)
        return {node: self.observe(node, step) for node in range(self.n_nodes)}

    def observe(self, node: int, step: int) -> SeriesObservation:
        self._check_step(step)
        value = float(self.values[node, step])
        if not bool(self.available[node, step]):
            empty = self.inputs.new_empty((0, self.inputs.shape[-1]))
            return SeriesObservation(
                x=empty, y=None, has_label=False, n_samples=0, node=node, step=step,
                drift_value=value,
            )
        # Clones, not views. A view would let a learner that mutates its input in
        # place corrupt the environment's own copy -- and assert_unmodified, which
        # re-reads that copy, could then never see it. MNIST avoids this by
        # re-transforming pristine images; here the copy is the only defence, and at
        # (n_blocks, 31) it costs nothing.
        return SeriesObservation(
            x=self.inputs[node, step].clone(),
            y=self.targets[node, step].clone(),
            has_label=True,
            n_samples=int(self.inputs.shape[2]),
            node=node,
            step=step,
            drift_value=value,
        )

    def pool(self, observations: dict[int, SeriesObservation]) -> tuple[torch.Tensor, torch.Tensor]:
        return pool(observations)

    def assert_unmodified(self, observations: dict[int, SeriesObservation], step: int) -> None:
        for node, observed in observations.items():
            if observed.checksum() != self.observe(node, step).checksum():
                raise SeriesError(
                    f"agent {node}'s observation at step {step} was modified in place. "
                    "Observations are shared by every learner in the run."
                )

    def summary(self) -> dict[str, Any]:
        series = self.config.env.series
        return {
            "n_nodes": self.n_nodes,
            "horizon": self.horizon,
            "task": "mackey_glass",
            "channel": series.channel,
            "span": series.span,
            "length": series.length,
            "n_blocks": series.n_blocks,
            "availability": float(self.available.float().mean()),
            "laws": self.laws.summary(),
            **{f"graph_{k}": v for k, v in self.graph.summary().items()},
            **{f"drift_{k}": v for k, v in self.drift.summary(self.horizon).items()},
        }

    def reset(self, master_seed: int) -> SeriesEnvironment:
        return build_series_environment(self.config, master_seed)

    def _check_step(self, step: int) -> None:
        if not 0 <= step < self.horizon:
            raise SeriesError(f"step {step} outside 0..{self.horizon - 1}")


def build_series_environment(config: Any, master_seed: int) -> SeriesEnvironment:
    """Assemble the series environment for one seed.

    Histories, noise and availability each draw from their own seed stream, none
    of which depends on the drift -- which is what makes a twin a twin.
    """
    seeds = Seeds.from_master(master_seed)
    series = config.env.series
    n_nodes, horizon = config.graph.n_nodes, config.run.horizon

    graphs = build_graphs(config, seeds.torch_generator("graph"))
    # The jump draw varies with the run seed (D98). On MNIST a recurring schedule
    # pins its jump_seed so the shift *pattern* is held while the data vary; here
    # the channel's centre is the interesting law and either side is a different
    # difficulty, so one fixed draw makes a run's realised mean law a fixed offset
    # from its twin's -- measured at 0.2111 against 0.22, with a 12-degree spread
    # across draws. Deriving it per seed averages that out over a cell's seeds, so
    # the damage metric measures the drift rather than the draw.
    drift = build_drift(config, jump_seed=seeds.sub("stream", "jumps"))
    laws = agent_laws(series, n_nodes)

    displacement = np.array(
        [[drift.rotation_at(step, node) for step in range(horizon)] for node in range(n_nodes)]
    )
    values = channel_value(series, displacement, laws.beta[:, None])

    histories = mg.initial_histories(n_nodes, seeds.numpy_rng("stream", "histories"))
    inputs, targets = generate(
        series, laws, values, histories, seeds.numpy_rng("stream", "noise")
    )
    available = _label_mask(
        n_nodes, horizon, config.env.label_availability, seeds.torch_generator("stream", "blocks")
    )

    dtype = torch.float64 if config.run.dtype == "float64" else torch.float32
    device = resolve_device(config.run.device)
    return SeriesEnvironment(
        config=config,
        seeds=seeds,
        graphs=graphs,
        drift=drift,
        laws=laws,
        values=values,
        inputs=torch.tensor(inputs, dtype=dtype, device=device),
        targets=torch.tensor(targets, dtype=dtype, device=device),
        available=available,
    )
