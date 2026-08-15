"""Configuration: load, compose, validate.

Composable YAML with inheritance from ``configs/base.yaml``. An experiment config
selects one entry from each of ``env/``, ``graph/``, ``model/`` and lists the
learners it runs, then overrides whatever else it needs.

Two principles govern this module:

*Validation happens once, at load, against a dataclass schema.* A typo in a key
is an error, not a silently ignored field. The alternative -- a dict passed
around and read with ``.get()`` -- turns a misspelled ``label_availabilty`` into
a run that quietly uses the default and produces a plausible wrong curve.

*Derived quantities are derived, not configured.* The drift rate ``alpha`` is
``total_degrees / horizon``, so changing the horizon cannot silently change how
far the distribution travels. Supplying ``alpha`` directly is an error
(WORKPLAN.md section 4.3).
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

# Size of the MNIST training set, which is what the shard budget divides up.
MNIST_TRAIN_SIZE = 60_000

# Sections of an experiment config that may be composed from a named file in
# ``configs/<section>/<name>.yaml``. ``learners`` is handled separately because
# it is a list rather than a single selection.
INCLUDABLE_SECTIONS = ("env", "graph", "model")

DTYPES = ("float32", "float64")
DEVICES = ("cpu", "cuda", "auto")
TOPOLOGIES = (
    "complete",
    "ring",
    "path",
    "grid2d",
    "star",
    "erdos_renyi",
    "watts_strogatz",
    "disconnected",
)
WEIGHT_RULES = ("metropolis", "relative_degree", "uniform")
PARTITIONS = ("iid", "dirichlet")
SCHEDULES = ("stationary", "linear", "ramp", "piecewise", "sinusoidal")
DRIFT_SCOPES = ("global", "per_node")
OPTIMIZERS = ("sgd", "sgd_momentum", "adamw")
MIX_POLICIES = ("none", "momentum", "all")
ADAPT_SCOPES = ("local", "one_hop")
EVALSETS = ("prequential", "current", "backward", "canonical")
# Smooth activations satisfy the research note's bounded-remainder assumption
# (As. 3); ReLU does not, at its kink set. See models/mlp.py.
ACTIVATIONS = ("gelu", "tanh", "silu", "relu")
# Phase-5 state model. TRANSITIONS picks F_t; FORGETTING_RULES picks how the
# covariance is loosened. They are independent axes -- see design note D26.
#: How the per-rotation reference classifiers relate to one another.
INIT_STRATEGIES = ("shared_seed", "independent_seeds", "warm_start")
#: How the reference picks the epoch to report.
SELECTION_RULES = ("validation", "fixed_budget")
TRANSITIONS = ("identity", "scalar")
FORGETTING_RULES = ("lambda", "process_noise")


class ConfigError(ValueError):
    """Raised for any malformed, unknown or inconsistent configuration."""


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def _one_of(value: Any, allowed: tuple[str, ...], path: str) -> None:
    if value not in allowed:
        raise ConfigError(f"{path}: {value!r} is not one of {list(allowed)}")


@dataclass
class RunConfig:
    name: str = "unnamed"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    horizon: int = 1500
    eval_every: int = 25
    dtype: str = "float32"
    device: str = "cpu"
    out_dir: str = "results"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ConfigError(f"run.horizon must be >= 1, got {self.horizon}")
        if self.eval_every < 1:
            raise ConfigError(f"run.eval_every must be >= 1, got {self.eval_every}")
        if not self.seeds:
            raise ConfigError("run.seeds must list at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigError(f"run.seeds contains duplicates: {self.seeds}")
        _one_of(self.dtype, DTYPES, "run.dtype")
        _one_of(self.device, DEVICES, "run.device")
        if self.device != "cpu":
            raise ConfigError(
                f"run.device is {self.device!r}, but nothing in phases 1-4 moves tensors "
                "to it -- the field would be accepted and silently ignored.\n"
                "  It is not an oversight: CUDA is measurably SLOWER here (design note "
                "D43). At p=2908 with batches of 4-40, kernel launch overhead exceeds "
                "the arithmetic -- 0.69x at batch 4, and even the 20-minute reference "
                "trainer comes out slower (16 min vs 15).\n"
                "  Phase 5 is where it pays: a dense p x p covariance matmul is 120 ms "
                "on CPU against 8.6 ms on CUDA, a 14x win, and that is what makes the "
                "dense filter feasible at all. This guard comes off when phase 5 wires "
                "the device through."
            )


@dataclass
class GraphConfig:
    topology: str = "ring"
    n_nodes: int = 10
    weights: str = "metropolis"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _one_of(self.topology, TOPOLOGIES, "graph.topology")
        _one_of(self.weights, WEIGHT_RULES, "graph.weights")
        if self.n_nodes < 1:
            raise ConfigError(f"graph.n_nodes must be >= 1, got {self.n_nodes}")
        if self.topology == "grid2d":
            rows, cols = self.params.get("rows"), self.params.get("cols")
            if rows is None or cols is None:
                raise ConfigError("graph.params needs 'rows' and 'cols' for grid2d")
            if rows * cols != self.n_nodes:
                raise ConfigError(
                    f"graph: grid2d {rows}x{cols} holds {rows * cols} nodes "
                    f"but n_nodes is {self.n_nodes}"
                )


@dataclass
class PartitionConfig:
    kind: str = "iid"
    beta: float = 1.0

    def __post_init__(self) -> None:
        _one_of(self.kind, PARTITIONS, "env.partition.kind")
        if self.kind == "dirichlet" and self.beta <= 0:
            raise ConfigError(f"env.partition.beta must be > 0, got {self.beta}")


@dataclass
class DriftConfig:
    """Drift schedule. The per-step rate is derived, never configured."""

    schedule: str = "stationary"
    total_degrees: float = 45.0
    change_points: list[int] = field(default_factory=list)
    jump_degrees: float = 15.0
    amplitude_degrees: float = 30.0
    period: int = 500
    #: `ramp` only. The run ends at this multiple of the constant rate covering
    #: the same ground, so it is the width of the rate sweep. Must exceed 1:
    #: at 1 the ramp is the linear schedule under another name.
    ramp_exponent: float = 2.0
    #: Under `drift_scope: per_node`, agents drift at rates spread over
    #: [1 - spread, 1] times the configured rate. See env/drift.py for why the
    #: multipliers top out at 1 rather than straddling it.
    per_node_spread: float = 0.5

    #: Beyond roughly this much rotation MNIST labels stop being well defined
    #: (a 6 becomes a 9). See WORKPLAN.md section 4.3.
    MAX_WELL_POSED_DEGREES = 45.0

    def __post_init__(self) -> None:
        _one_of(self.schedule, SCHEDULES, "env.drift.schedule")
        if self.total_degrees < 0:
            raise ConfigError(f"env.drift.total_degrees must be >= 0, got {self.total_degrees}")
        if self.total_degrees > self.MAX_WELL_POSED_DEGREES:
            raise ConfigError(
                f"env.drift.total_degrees is {self.total_degrees}, above the "
                f"{self.MAX_WELL_POSED_DEGREES} degree cap. Past that, rotated MNIST labels "
                "stop being well defined and the gap to the reference measures label "
                "ambiguity rather than decentralization cost (WORKPLAN.md section 4.3)."
            )
        if self.period < 1:
            raise ConfigError(f"env.drift.period must be >= 1, got {self.period}")
        if any(t < 0 for t in self.change_points):
            raise ConfigError(f"env.drift.change_points must be non-negative: {self.change_points}")
        if list(self.change_points) != sorted(self.change_points):
            raise ConfigError(f"env.drift.change_points must be sorted: {self.change_points}")
        if abs(self.amplitude_degrees) > self.MAX_WELL_POSED_DEGREES:
            raise ConfigError(
                f"env.drift.amplitude_degrees is {self.amplitude_degrees}, above the "
                f"{self.MAX_WELL_POSED_DEGREES} degree cap"
            )
        if not 0.0 <= self.per_node_spread < 1.0:
            raise ConfigError(
                f"env.drift.per_node_spread must lie in [0, 1), got {self.per_node_spread}"
            )
        if self.schedule == "ramp" and self.ramp_exponent <= 1.0:
            raise ConfigError(
                f"env.drift.ramp_exponent must be > 1, got {self.ramp_exponent}. At 1.0 "
                "the ramp is exactly the linear schedule, and below it the drift "
                "decelerates -- which measures nothing the ramp exists to measure."
            )

    # Evaluating the schedule lives in env/drift.py, not here. This class
    # validates fields; turning a step into a rotation is behaviour, and having
    # two implementations of "where is the piecewise jump" is how they diverge.


@dataclass
class PriorDriftConfig:
    """Class-prior drift: the label-shift channel.

    Off by default. When on, it rides the *same* schedule as the rotation, so a
    run drifts in both channels together unless ``env.drift.total_degrees`` is
    set to 0 to isolate the prior.
    """

    enabled: bool = False
    #: Dirichlet concentration of the endpoint the priors travel toward. Small
    #: values concentrate an agent on a few classes, which makes the path long.
    beta: float = 0.5
    #: How far along the start-to-end path the run travels. The magnitude knob,
    #: and unlike degrees of rotation it has no well-posedness ceiling.
    total_shift: float = 1.0
    #: Start from 1/K rather than a Dirichlet draw, so at progress 0 the run is
    #: the same experiment as one without prior drift.
    uniform_start: bool = True

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ConfigError(f"env.prior_drift.beta must be > 0, got {self.beta}")
        if not 0.0 <= self.total_shift <= 1.0:
            raise ConfigError(
                f"env.prior_drift.total_shift must lie in [0, 1], got {self.total_shift}. "
                "Past 1 the interpolation leaves the simplex and asks for negative mass."
            )


@dataclass
class EnvConfig:
    samples_per_node_per_step: int = 2
    label_availability: float = 1.0
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    prior_drift: PriorDriftConfig = field(default_factory=PriorDriftConfig)
    drift_scope: str = "global"
    allow_epochs: bool = False

    def __post_init__(self) -> None:
        if self.samples_per_node_per_step < 1:
            raise ConfigError(
                f"env.samples_per_node_per_step must be >= 1, "
                f"got {self.samples_per_node_per_step}"
            )
        if not 0.0 <= self.label_availability <= 1.0:
            raise ConfigError(
                f"env.label_availability must lie in [0, 1], got {self.label_availability}"
            )
        _one_of(self.drift_scope, DRIFT_SCOPES, "env.drift_scope")


@dataclass
class ModelConfig:
    name: str = "mlp_small"
    input_size: int = 14
    hidden: list[int] = field(default_factory=lambda: [14])
    output_dim: int = 10
    #: GELU rather than ReLU by default: the research note's smoothness
    #: assumption bounds the linearisation remainder, and that bound fails for
    #: ReLU at its kink set. See models/mlp.py.
    activation: str = "gelu"

    def __post_init__(self) -> None:
        _one_of(self.activation, ACTIVATIONS, "model.activation")
        if self.input_size < 1:
            raise ConfigError(f"model.input_size must be >= 1, got {self.input_size}")
        if any(h < 1 for h in self.hidden):
            raise ConfigError(f"model.hidden widths must be >= 1, got {self.hidden}")
        if self.output_dim < 2:
            raise ConfigError(f"model.output_dim must be >= 2, got {self.output_dim}")

    @property
    def num_params(self) -> int:
        """Parameter count p, for the phase-5 covariance budget."""
        widths = [self.input_size**2, *self.hidden, self.output_dim]
        return sum(a * b + b for a, b in zip(widths[:-1], widths[1:], strict=True))


@dataclass
class LearnerConfig:
    name: str = "diffusion_sgd_atc"
    optimizer: str = "sgd_momentum"
    lr: float = 0.05
    momentum: float = 0.9
    mix_optimizer_state: str = "momentum"
    adapt_scope: str = "local"
    #: Stop adapting at this step, and stop transmitting with it. The
    #: non-adapting baseline the *comparative* break is measured against: it saw
    #: the same data with the same optimizer up to the same point and then
    #: stopped, so the comparison isolates continued adaptation from initial
    #: learning. `None` means never freeze.
    freeze_after: int | None = None

    # --- phase 5 state model; inert for the SGD learners ------------------- #
    #: F_t. "identity" is a driftless random walk; "scalar" makes F_t = gamma*I,
    #: an AR(1) that also pulls the mean toward the origin -- which is L2 weight
    #: decay in state-space form, not forgetting. See design note D26.
    transition: str = "identity"
    gamma: float = 1.0
    #: How the covariance is loosened. Multiplicative inflation is exactly
    #: structure-preserving in the information domain; additive process noise is
    #: not, but is anisotropic and cheap while the covariance stays dense.
    forgetting: str = "lambda"
    #: Effective memory is about 1/(1 - lambda) steps. The default gives ~333
    #: steps, over which the data rotates ~10 degrees at the capped drift rate.
    #: Pilot-calibrate this in phase 5 (WORKPLAN section 9).
    lambda_forget: float = 0.997
    process_noise_q: float = 1.0e-6
    prior_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.freeze_after is not None:
            if self.freeze_after < 1:
                raise ConfigError(
                    f"learner[{self.name}].freeze_after must be >= 1, got {self.freeze_after}. "
                    "A learner frozen at step 0 never leaves its random initialisation, which "
                    "measures nothing about whether adaptation pays."
                )
            if self.name == "centralized_sgd":
                raise ConfigError(
                    "learner[centralized_sgd].freeze_after is not supported: it adapts through "
                    "adapt_pooled(), which the runner calls without a step, so the freeze point "
                    "could not be honoured. Freeze a diffusion or local learner instead -- "
                    "'frozen_atc' is the intended baseline."
                )
        _one_of(self.optimizer, OPTIMIZERS, f"learner[{self.name}].optimizer")
        _one_of(self.mix_optimizer_state, MIX_POLICIES, f"learner[{self.name}].mix_optimizer_state")
        _one_of(self.adapt_scope, ADAPT_SCOPES, f"learner[{self.name}].adapt_scope")
        if self.lr <= 0:
            raise ConfigError(f"learner[{self.name}].lr must be > 0, got {self.lr}")
        if not 0.0 <= self.momentum < 1.0:
            raise ConfigError(
                f"learner[{self.name}].momentum must lie in [0, 1), got {self.momentum}"
            )
        if self.optimizer != "sgd" and self.mix_optimizer_state == "none":
            raise ConfigError(
                f"learner[{self.name}]: optimizer {self.optimizer!r} carries per-node state but "
                "mix_optimizer_state is 'none'. Unmixed adaptive state diverges across agents -- "
                "a known failure mode, not an open question (WORKPLAN.md section 3.4). Use plain "
                "'sgd' if you want no mixing."
            )
        _one_of(self.transition, TRANSITIONS, f"learner[{self.name}].transition")
        _one_of(self.forgetting, FORGETTING_RULES, f"learner[{self.name}].forgetting")
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigError(f"learner[{self.name}].gamma must lie in (0, 1], got {self.gamma}")
        if self.transition == "identity" and self.gamma != 1.0:
            raise ConfigError(
                f"learner[{self.name}]: gamma={self.gamma} has no effect under "
                "transition='identity', where F_t = I. Set transition='scalar' to use it, "
                "rather than leaving a value that silently does nothing."
            )
        if not 0.0 < self.lambda_forget <= 1.0:
            raise ConfigError(
                f"learner[{self.name}].lambda_forget must lie in (0, 1], got {self.lambda_forget}"
            )
        if self.process_noise_q <= 0.0:
            raise ConfigError(
                f"learner[{self.name}].process_noise_q must be > 0, got {self.process_noise_q}"
            )


@dataclass
class ReferenceConfig:
    r"""The offline reference classifier, $e^\star$.

    A fixed asset computed once and cached, not part of any experiment run. It
    is recomputed per rotation level, because a single reference would conflate
    decentralization cost with drift cost.
    """

    #: 100, not 20. At 20 epochs 9 of 16 rotation levels selected the final
    #: epoch, meaning the budget rather than convergence decided where training
    #: stopped -- and an under-trained e* is too high, which flatters every gap.
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.003
    #: shared_seed   - independent runs, all from the same theta_0 (default).
    #: independent_seeds - independent runs, each from its own theta_0.
    #: warm_start    - each level initialised from the previous one; cheaper,
    #:                 but e*(45) then depends on having passed through 40.
    init_strategy: str = "shared_seed"
    #: validation    - hold out `validation_size`, early-stop on it (default).
    #: fixed_budget  - train on everything for `epochs`; the test curve is
    #:                 recorded for inspection only, never for selection.
    selection: str = "validation"
    validation_size: int = 5000
    #: The grid covers every rotation the configured schedules actually visit:
    #: linear [0, 45], piecewise [0, 15], sinusoidal [-30, +30].
    rotation_min_degrees: float = -30.0
    rotation_max_degrees: float = 45.0
    rotation_step_degrees: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        _one_of(self.init_strategy, INIT_STRATEGIES, "reference.init_strategy")
        _one_of(self.selection, SELECTION_RULES, "reference.selection")
        if self.epochs < 1:
            raise ConfigError(f"reference.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ConfigError(f"reference.batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ConfigError(f"reference.lr must be > 0, got {self.lr}")
        if self.rotation_step_degrees <= 0:
            raise ConfigError(
                f"reference.rotation_step_degrees must be > 0, got {self.rotation_step_degrees}"
            )
        if self.rotation_max_degrees < self.rotation_min_degrees:
            raise ConfigError(
                f"reference rotation range is empty: [{self.rotation_min_degrees}, "
                f"{self.rotation_max_degrees}]"
            )
        if self.selection == "validation" and not 0 < self.validation_size < MNIST_TRAIN_SIZE:
            raise ConfigError(
                f"reference.validation_size must lie in (0, {MNIST_TRAIN_SIZE}), got "
                f"{self.validation_size}"
            )
        if self.selection == "fixed_budget" and self.validation_size:
            raise ConfigError(
                "reference.selection='fixed_budget' trains on the whole split, so "
                f"validation_size={self.validation_size} would be silently ignored. "
                "Set it to 0 to be explicit."
            )

    @property
    def rotations(self) -> list[float]:
        """Every grid point, inclusive of both ends."""
        span = self.rotation_max_degrees - self.rotation_min_degrees
        count = int(round(span / self.rotation_step_degrees)) + 1
        return [
            round(self.rotation_min_degrees + index * self.rotation_step_degrees, 6)
            for index in range(count)
        ]


@dataclass
class EvalConfig:
    evalsets: list[str] = field(default_factory=lambda: ["prequential", "current", "canonical"])
    #: The backward set is anchored by *rotation*, not by a step count. A fixed
    #: step offset degenerates: at offset == period the sinusoidal probe has
    #: identically zero separation, and the piecewise one collapses once t
    #: passes the last change point plus the offset. Specifying the separation
    #: and deriving the step is the same move D3 makes for alpha.
    backward_separation_degrees: float = 15.0
    batch_size: int = 1000

    def __post_init__(self) -> None:
        for name in self.evalsets:
            _one_of(name, EVALSETS, "eval.evalsets")
        if len(set(self.evalsets)) != len(self.evalsets):
            raise ConfigError(f"eval.evalsets contains duplicates: {self.evalsets}")
        if self.backward_separation_degrees <= 0:
            raise ConfigError(
                "eval.backward_separation_degrees must be > 0, got "
                f"{self.backward_separation_degrees}"
            )
        if self.batch_size < 1:
            raise ConfigError(f"eval.batch_size must be >= 1, got {self.batch_size}")


@dataclass
class Config:
    """A fully resolved configuration. Everything needed to reproduce one run."""

    run: RunConfig = field(default_factory=RunConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    learners: list[LearnerConfig] = field(default_factory=lambda: [LearnerConfig()])
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        if not self.learners:
            raise ConfigError("learners must list at least one learner")
        names = [learner.name for learner in self.learners]
        if len(set(names)) != len(names):
            raise ConfigError(f"learners must have unique names, got {names}")
        self._check_shard_budget()
        self._check_backward_evalset()

    def _check_shard_budget(self) -> None:
        """Reject a horizon the disjoint shards cannot supply.

        Each agent holds ``MNIST_TRAIN_SIZE / N`` samples and consumes ``n`` per
        step, so the run needs ``N * n * T`` samples in total. Discovering this
        at step 1400 rather than at load is how a long sweep gets wasted.
        """
        if self.env.allow_epochs:
            return
        required = self.graph.n_nodes * self.env.samples_per_node_per_step * self.run.horizon
        if required > MNIST_TRAIN_SIZE:
            per_agent = MNIST_TRAIN_SIZE // self.graph.n_nodes
            max_horizon = per_agent // self.env.samples_per_node_per_step
            raise ConfigError(
                f"shard budget exceeded: N={self.graph.n_nodes} agents x "
                f"n={self.env.samples_per_node_per_step} samples x T={self.run.horizon} steps "
                f"needs {required} samples but MNIST provides {MNIST_TRAIN_SIZE}. "
                f"Each shard holds {per_agent}, so the horizon can be at most {max_horizon}. "
                "Reduce the horizon, reduce n, reduce N, or set env.allow_epochs=true "
                "(which lets samples repeat and forfeits the exactly-once guarantee)."
            )

    def _check_backward_evalset(self) -> None:
        """The backward probe needs a separation the schedule can actually reach.

        Checked against what the schedule *does*, not against its fields: a
        separation larger than the run's total travel means the probe is
        undefined at every step, and reporting that as "no forgetting" would be
        a silent lie.
        """
        if "backward" not in self.eval.evalsets:
            return
        separation = self.eval.backward_separation_degrees
        if separation <= 0:
            raise ConfigError(f"eval.backward_separation_degrees must be > 0, got {separation}")
        reachable = self._schedule_travel()
        if separation > reachable + 1e-9:
            raise ConfigError(
                f"eval.backward_separation_degrees is {separation} but the "
                f"{self.env.drift.schedule} schedule only travels {reachable:.1f} degrees "
                "over the run, so the backward probe would be undefined at every step. "
                "Lower the separation, or drop 'backward' from eval.evalsets."
            )

    def _schedule_travel(self) -> float:
        """How far the configured schedule moves. Kept crude deliberately: the
        authoritative version lives in env/drift.py, and importing it here would
        invert the dependency (design note D19)."""
        drift = self.env.drift
        if drift.schedule == "stationary":
            return 0.0
        if drift.schedule in ("linear", "ramp"):
            # A ramp reaches the same place as the linear schedule; only the
            # rate along the way differs.
            return drift.total_degrees
        if drift.schedule == "piecewise":
            return drift.jump_degrees * max(len(drift.change_points), 1)
        return 2.0 * abs(drift.amplitude_degrees)

    def learner(self, name: str) -> LearnerConfig:
        for entry in self.learners:
            if entry.name == name:
                return entry
        raise KeyError(f"no learner named {name!r}; have {[e.name for e in self.learners]}")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, for writing the resolved config beside the results."""
        return _to_plain(self)


# --------------------------------------------------------------------------- #
# strict construction
# --------------------------------------------------------------------------- #


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """Unwrap ``X | None`` into ``(True, X)``.

    Both spellings have to be recognised: ``Optional[X]`` has origin
    ``typing.Union``, while the ``X | None`` syntax used in this module has
    origin ``types.UnionType``. Checking only the former silently treats every
    optional field as required.
    """
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, annotation


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    optional, annotation = _is_optional(annotation)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{path}: null is not allowed here")

    if is_dataclass(annotation):
        return _build(annotation, value, path)

    origin = get_origin(annotation)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_type,) = get_args(annotation)
        return [_coerce(item_type, item, f"{path}[{i}]") for i, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return dict(value)

    # bool is a subclass of int, so it has to be checked first or True passes as 1.
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _build(cls: Any, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        suggestions = {key: _closest(key, known) for key in unknown}
        detail = ", ".join(
            f"{key!r}" + (f" (did you mean {match!r}?)" if match else "")
            for key, match in suggestions.items()
        )
        raise ConfigError(
            f"{path or 'config'}: unknown key(s): {detail}. Known keys: {sorted(known)}"
        )

    kwargs = {
        f.name: _coerce(hints[f.name], data[f.name], f"{path}.{f.name}" if path else f.name)
        for f in fields(cls)
        if f.name in data
    }
    return cls(**kwargs)


def _closest(key: str, candidates: set[str]) -> str | None:
    """Cheap typo suggestion; good enough to catch a transposed letter."""
    import difflib

    matches = difflib.get_close_matches(key, sorted(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_plain(value) for key, value in obj.items()}
    return obj


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Mappings merge key by key; every other type, lists included, is replaced
    wholesale. Element-wise list merging would make it impossible to *shorten* a
    list in an override, which is the more common need.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _reject_configured_alpha(raw: dict[str, Any]) -> None:
    """``alpha`` is derived from ``total_degrees`` and the horizon, never set."""
    drift = raw.get("env", {}).get("drift", {})
    if isinstance(drift, dict) and ("alpha" in drift or "alpha_per_step" in drift):
        raise ConfigError(
            "env.drift.alpha is derived, not configured. Set env.drift.total_degrees "
            "instead; the per-step rate is total_degrees / run.horizon, so that changing "
            "the horizon cannot silently change how far the distribution travels "
            "(WORKPLAN.md section 4.3)."
        )


def _resolve_learners(entries: list[Any], configs_dir: Path) -> list[dict[str, Any]]:
    """Turn learner names or partial dicts into full learner mappings.

    Accepts ``"local_only"`` or ``{"name": "local_only", "lr": 0.1}``; both load
    ``configs/learner/local_only.yaml`` and apply any overrides on top.
    """
    resolved: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            name, overrides = entry, {}
        elif isinstance(entry, dict):
            if "name" not in entry:
                raise ConfigError(f"learners[{index}]: a mapping entry needs a 'name' key")
            name = entry["name"]
            overrides = {k: v for k, v in entry.items() if k != "name"}
        else:
            raise ConfigError(
                f"learners[{index}]: expected a name or a mapping, got {type(entry).__name__}"
            )

        path = configs_dir / "learner" / f"{name}.yaml"
        if not path.is_file():
            available = sorted(p.stem for p in (configs_dir / "learner").glob("*.yaml"))
            raise ConfigError(f"unknown learner {name!r}; available: {available}")
        resolved.append(deep_merge(_read_yaml(path), overrides))
    return resolved


def resolve(raw: dict[str, Any], configs_dir: Path) -> dict[str, Any]:
    """Compose a raw experiment mapping into a complete configuration mapping.

    Precedence, lowest first: ``base.yaml``, then each ``include:`` selection,
    then the experiment file's own keys.
    """
    merged = _read_yaml(configs_dir / "base.yaml")
    raw = copy.deepcopy(raw)

    includes = raw.pop("include", {})
    if not isinstance(includes, dict):
        raise ConfigError(f"include: expected a mapping, got {type(includes).__name__}")
    for section, name in includes.items():
        if section not in INCLUDABLE_SECTIONS:
            raise ConfigError(
                f"include.{section}: not an includable section; "
                f"expected one of {list(INCLUDABLE_SECTIONS)}. "
                "Learners are selected through the top-level 'learners' list."
            )
        path = configs_dir / section / f"{name}.yaml"
        if not path.is_file():
            available = sorted(p.stem for p in (configs_dir / section).glob("*.yaml"))
            raise ConfigError(f"include.{section}: unknown {name!r}; available: {available}")
        merged[section] = deep_merge(merged.get(section, {}), _read_yaml(path))

    learner_entries = raw.pop("learners", None)
    if learner_entries is not None:
        if not isinstance(learner_entries, list):
            raise ConfigError(
                f"learners: expected a list, got {type(learner_entries).__name__}. "
                "Several learners share one environment per run (WORKPLAN.md section 6.1)."
            )
        merged["learners"] = _resolve_learners(learner_entries, configs_dir)
    elif isinstance(merged.get("learners"), list):
        merged["learners"] = _resolve_learners(merged["learners"], configs_dir)

    merged = deep_merge(merged, raw)
    _reject_configured_alpha(merged)
    return merged


def load_config(
    path: str | Path,
    configs_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load, compose and validate a config file.

    Args:
        path: an experiment config. A bare name such as ``"x1_stationary"``
            resolves inside ``configs/experiment/``.
        configs_dir: the ``configs/`` tree. Defaults to the one in the repo.
        overrides: a mapping merged into the experiment file before composition,
            so it sits above the includes and below nothing. Useful for sweeps
            and for one-off edits from a script or a notebook.

    Overrides are applied to the *raw* mapping rather than to the resolved one,
    so that ``{"learners": ["local_only"]}`` and ``{"include": {...}}`` go
    through the same resolution as if they had been written in the file. Merging
    them afterwards would leave a learner name as a bare string that never gets
    looked up.
    """
    configs_dir = Path(configs_dir) if configs_dir is not None else default_configs_dir()
    path = Path(path)
    if not path.suffix:
        path = configs_dir / "experiment" / f"{path.name}.yaml"

    raw = _read_yaml(path)
    if overrides:
        raw = deep_merge(raw, overrides)
    merged = resolve(raw, configs_dir)
    return _build(Config, merged, "")


def default_configs_dir() -> Path:
    """The repository's ``configs/`` directory."""
    return Path(__file__).resolve().parents[3] / "configs"


def dump_config(config: Config, path: str | Path) -> Path:
    """Write the fully resolved config, so a result traces back to its settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, default_flow_style=False)
    return path
