r"""The column contract.

**One long-format table: one row per (run, seed, step, learner, node, metric).**
Long format costs disk but makes every plot a groupby, and adding a metric never
changes the schema -- which is what keeps a new aggregate from requiring a
re-run.

**Counts travel alongside the value where they exist.** ``value`` is present on
every row, as `IMPLEMENTATION.md` §7 specifies, so a plot that only wants the
number keeps working. Accuracy-type rows additionally carry ``n_correct`` and
``n_samples``, because *averaging rates across unequal batch sizes silently
mis-weights*: at $\pi_{\text{lab}} < 1$ the number of samples behind a
prequential score varies step to step, and
$\text{mean}(\text{rates}) \ne \sum\text{correct} / \sum\text{samples}$.
Counts sum correctly by construction, so any aggregation -- across agents,
across a time window, across seeds -- is exact.

**Where the two disagree, the counts win.** ``value`` is the convenience column;
``n_correct / n_samples`` is the measurement.

``n_correct`` always means *correct*, never *wrong*, whatever the metric is
called. So an aggregated ``error_rate`` is

.. code-block:: python

    1 - group["n_correct"].sum() / group["n_samples"].sum()

and not the ratio itself. Storing the complement under a field named
``n_correct`` would make the natural expression silently return accuracy while
the ``metric`` column said error -- a discrepancy nobody would look for.

**Run-constant fields are denormalized into every row.** ``topology``,
``spectral_gap``, ``n_nodes`` and so on repeat identically 700 000 times, which
sounds wasteful and is not: parquet dictionary- and run-length-encodes a
constant column to near nothing, and the alternative is a join before every
plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Bumping this invalidates cached results, because a reader written against an
#: older layout would silently mis-parse them.
SCHEMA_VERSION = 1

#: Columns every row carries. Order is the on-disk column order.
COLUMNS: dict[str, str] = {
    # provenance -- what produced this row
    "run_id": "str",
    "git_sha": "str",
    "schema_version": "int",
    "experiment": "str",
    "seed": "int",
    # the run's fixed settings, denormalized so a plot needs no join
    "learner": "str",
    "topology": "str",
    "n_nodes": "int",
    "spectral_gap": "float",
    "mixing_gap": "float",
    "samples_per_node": "int",
    "label_availability": "float",
    "drift_schedule": "str",
    "drift_param": "float",
    # where in the run
    "t": "int",
    "drift_state": "float",
    "node_id": "str",
    "evalset": "str",
    # the measurement
    "metric": "str",
    "value": "float",
    "n_correct": "int",
    "n_samples": "int",
    # communication, cumulative to this step
    "cum_scalars_tx": "int",
    "cum_rounds": "int",
    "wallclock_s": "float",
}

#: Columns that must be present and non-null on every row.
REQUIRED = (
    "run_id",
    "experiment",
    "seed",
    "learner",
    "t",
    "node_id",
    "metric",
    "value",
)

#: Metrics that carry counts. Anything else leaves n_correct/n_samples null.
COUNTED_METRICS = frozenset({"error_rate", "accuracy"})

#: node_id is a string because aggregate rows use names rather than integers --
#: 'mean' for a network-level quantity, 'reference' for e*. Mixing int and str
#: in one column is what forces that.
AGGREGATE_NODES = frozenset({"mean", "reference", "network"})


class SchemaError(ValueError):
    """Raised when a row does not satisfy the contract."""


@dataclass(frozen=True)
class RunContext:
    """The fields that are constant for a whole run.

    Held once and stamped onto every row, rather than passed at each log call --
    a caller that forgets one would produce rows that look fine and cannot be
    grouped.
    """

    run_id: str
    git_sha: str
    experiment: str
    seed: int
    topology: str
    n_nodes: int
    spectral_gap: float | None
    mixing_gap: float
    samples_per_node: int
    label_availability: float
    drift_schedule: str
    drift_param: float
    schema_version: int = SCHEMA_VERSION

    def stamp(self, row: dict[str, Any]) -> dict[str, Any]:
        """Fill the run-constant fields, without overwriting anything set."""
        base = {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "seed": self.seed,
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "spectral_gap": self.spectral_gap,
            "mixing_gap": self.mixing_gap,
            "samples_per_node": self.samples_per_node,
            "label_availability": self.label_availability,
            "drift_schedule": self.drift_schedule,
            "drift_param": self.drift_param,
        }
        base.update(row)
        return base

    @classmethod
    def from_config(
        cls, config: Any, seed: int, graph: Any, run_id: str, git_sha: str
    ) -> RunContext:
        summary = graph.summary()
        return cls(
            run_id=run_id,
            git_sha=git_sha,
            experiment=config.run.name,
            seed=seed,
            topology=config.graph.topology,
            n_nodes=config.graph.n_nodes,
            spectral_gap=summary["spectral_gap"],
            mixing_gap=summary["mixing_gap"],
            samples_per_node=config.env.samples_per_node_per_step,
            label_availability=config.env.label_availability,
            drift_schedule=config.env.drift.schedule,
            drift_param=config.env.drift.total_degrees,
        )


@dataclass
class Row:
    """One measurement, validated on construction."""

    t: int
    node_id: str | int
    metric: str
    value: float
    evalset: str = "prequential"
    drift_state: float = 0.0
    n_correct: int | None = None
    n_samples: int | None = None
    cum_scalars_tx: int = 0
    cum_rounds: int = 0
    wallclock_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "node_id": str(self.node_id),
            "metric": self.metric,
            "value": float(self.value),
            "evalset": self.evalset,
            "drift_state": self.drift_state,
            "n_correct": self.n_correct,
            "n_samples": self.n_samples,
            "cum_scalars_tx": self.cum_scalars_tx,
            "cum_rounds": self.cum_rounds,
            "wallclock_s": self.wallclock_s,
            **self.extra,
        }


def validate(row: dict[str, Any], strict: bool = True) -> None:
    """Check one row against the contract.

    ``strict`` rejects unknown columns, matching how the config loader treats
    unknown keys: a misspelled metric name that silently becomes a new column
    produces a file whose groupby quietly drops rows.
    """
    missing = [name for name in REQUIRED if row.get(name) is None]
    if missing:
        raise SchemaError(f"row is missing required field(s): {missing}")

    if strict:
        unknown = sorted(set(row) - set(COLUMNS))
        if unknown:
            raise SchemaError(
                f"row has unknown column(s) {unknown}; known columns are "
                f"{sorted(COLUMNS)}. Add the column to schema.COLUMNS if it is "
                "intended -- a typo that becomes a new column makes every groupby "
                "silently incomplete."
            )

    if row["metric"] in COUNTED_METRICS:
        correct, samples = row.get("n_correct"), row.get("n_samples")
        if correct is None or samples is None:
            raise SchemaError(
                f"metric {row['metric']!r} must carry n_correct and n_samples: "
                "averaging rates across unequal batch sizes mis-weights, and batch "
                "sizes vary as soon as label_availability < 1"
            )
        if samples <= 0:
            raise SchemaError(f"n_samples must be positive, got {samples}")
        if not 0 <= correct <= samples:
            raise SchemaError(f"n_correct {correct} outside [0, {samples}]")

    node = str(row["node_id"])
    if node not in AGGREGATE_NODES and not node.isdigit():
        raise SchemaError(
            f"node_id {node!r} is neither an integer nor one of {sorted(AGGREGATE_NODES)}"
        )


def empty_frame():
    """An empty DataFrame with the right columns and dtypes.

    Returned when a run produced nothing, so a caller can concatenate without
    special-casing -- an empty frame with no columns silently drops every column
    it is concatenated with.
    """
    import pandas as pd

    dtypes = {
        "str": "object",
        "int": "Int64",  # nullable, because n_correct is null on most rows
        "float": "float64",
    }
    return pd.DataFrame({name: pd.Series(dtype=dtypes[kind]) for name, kind in COLUMNS.items()})
