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
SCHEDULES = ("stationary", "linear", "piecewise", "sinusoidal")
DRIFT_SCOPES = ("global", "per_node")
OPTIMIZERS = ("sgd", "sgd_momentum", "adamw")
MIX_POLICIES = ("none", "momentum", "all")
ADAPT_SCOPES = ("local", "one_hop")
EVALSETS = ("prequential", "current", "backward", "canonical")


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

    def alpha_per_step(self, horizon: int) -> float:
        """Degrees per step, derived from the horizon.

        Zero for every schedule other than ``linear``: piecewise drift moves in
        jumps and sinusoidal drift is governed by amplitude and period.
        """
        if self.schedule != "linear":
            return 0.0
        return self.total_degrees / horizon

    def rotation_at(self, t: int, horizon: int) -> float:
        """Total rotation in degrees applied to the data at step ``t``."""
        import math

        if self.schedule == "stationary":
            return 0.0
        if self.schedule == "linear":
            return self.alpha_per_step(horizon) * t
        if self.schedule == "piecewise":
            return self.jump_degrees * sum(1 for cp in self.change_points if t >= cp)
        if self.schedule == "sinusoidal":
            return self.amplitude_degrees * math.sin(2.0 * math.pi * t / self.period)
        raise ConfigError(f"unhandled drift schedule {self.schedule!r}")  # pragma: no cover


@dataclass
class EnvConfig:
    samples_per_node_per_step: int = 2
    label_availability: float = 1.0
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
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

    def __post_init__(self) -> None:
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
    # Phase 5 only; inert for the SGD learners.
    lambda_forget: float | None = None
    process_noise_q: float | None = None
    prior_scale: float = 1.0

    def __post_init__(self) -> None:
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
        if self.lambda_forget is not None and self.process_noise_q is not None:
            raise ConfigError(
                f"learner[{self.name}]: lambda_forget and process_noise_q are two "
                "parameterisations of the same forgetting effect and are jointly "
                "unidentifiable. Set exactly one."
            )
        if self.lambda_forget is not None and not 0.0 < self.lambda_forget <= 1.0:
            raise ConfigError(
                f"learner[{self.name}].lambda_forget must lie in (0, 1], got {self.lambda_forget}"
            )


@dataclass
class EvalConfig:
    evalsets: list[str] = field(default_factory=lambda: ["prequential", "current", "canonical"])
    # 500 steps, not 200. At the capped drift rate of 45 deg / 1500 steps, a
    # 200-step lookback separates the backward set from the current one by only
    # 6 degrees -- too little to distinguish forgetting from noise. 500 steps
    # gives 15 degrees, a third of the total range, while still leaving 1000
    # steps of the run over which the probe is defined.
    backward_offset: int = 500
    batch_size: int = 1000

    def __post_init__(self) -> None:
        for name in self.evalsets:
            _one_of(name, EVALSETS, "eval.evalsets")
        if len(set(self.evalsets)) != len(self.evalsets):
            raise ConfigError(f"eval.evalsets contains duplicates: {self.evalsets}")
        if self.backward_offset < 0:
            raise ConfigError(f"eval.backward_offset must be >= 0, got {self.backward_offset}")
        if self.batch_size < 1:
            raise ConfigError(f"eval.batch_size must be >= 1, got {self.batch_size}")


@dataclass
class Config:
    """A fully resolved configuration. Everything needed to reproduce one run."""

    run: RunConfig = field(default_factory=RunConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
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
        if "backward" in self.eval.evalsets and self.eval.backward_offset >= self.run.horizon:
            raise ConfigError(
                f"eval.backward_offset ({self.eval.backward_offset}) must be smaller than "
                f"run.horizon ({self.run.horizon}) or the backward evaluation set never exists"
            )

    @property
    def alpha_per_step(self) -> float:
        """Drift rate in degrees per step, derived from the horizon."""
        return self.env.drift.alpha_per_step(self.run.horizon)

    def rotation_at(self, t: int) -> float:
        """Rotation in degrees applied to the data at step ``t``."""
        return self.env.drift.rotation_at(t, self.run.horizon)

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
