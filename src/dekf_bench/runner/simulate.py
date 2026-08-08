r"""The simulation loop: one environment, several learners, stepped together.

``env.step(t)`` is called **once** per step and the result is handed to every
learner. That is what makes X0 exact by construction rather than contingent on
two runs' RNG draws lining up, and it is what makes $E_{\text{cent}}$ available
without a second pass (design note D4).

The per-learner body is **adapt, then combine**, and must not change when
Diff-EKF arrives -- the filter differs from diffusion SGD in ``adapt`` alone.

**Prequential scoring happens before the update, structurally.** The protocol is
handed a *predict function*, not a learner, so a call that scores cannot train.
The runner's ordering is the other half of that guarantee and is asserted in the
tests.

**The exactness preconditions are checked at start, not assumed.** An X0 run
whose config has drifted -- momentum left on, float32 inherited from a sweep --
fails by about $10^{-3}$, which reads as a numerical issue rather than as a
broken precondition, and invites loosening the tolerance until it "passes". The
check names the offending field instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from dekf_bench.env.environment import Environment, pool
from dekf_bench.evaluation import protocol
from dekf_bench.evaluation.evalsets import EvalSetBuilder
from dekf_bench.metrics import disagreement

#: The learner an experiment compares everything else against, when present.
REFERENCE_LEARNER = "centralized_sgd"


class SimulationError(RuntimeError):
    """Raised when a run cannot proceed as configured."""


@dataclass
class StepRecord:
    """What one step produced, before it reaches the recorder."""

    step: int
    learner: str
    rows: list[dict[str, Any]] = field(default_factory=list)


def check_exactness_preconditions(config: Any) -> None:
    r"""Refuse to start an exactness run whose preconditions have drifted.

    The identity

    .. math::
        \sum_v \tfrac1N\bigl(\bm\theta - \eta\nabla L_v\bigr)
        = \bm\theta - \eta\,\tfrac1N\sum_v\nabla L_v

    needs all four of: a complete graph with uniform weights ($a_{vu} = 1/N$
    exactly), plain SGD with no optimizer state, equal batch sizes across agents
    ($\pi_{\text{lab}} = 1$), and float64. Each failure produces a *small,
    plausible* residual rather than an obvious break, which is the case this
    check exists for.
    """
    problems = []
    if config.run.dtype != "float64":
        problems.append(
            f"run.dtype is {config.run.dtype!r}; the identity is checked at 1e-12 and "
            "float32 carries ~1e-7 of accumulation"
        )
    if config.graph.topology != "complete":
        problems.append(
            f"graph.topology is {config.graph.topology!r}; the identity holds only on a "
            "complete graph, where one combine step reaches full consensus"
        )
    if config.graph.weights != "uniform":
        problems.append(f"graph.weights is {config.graph.weights!r}; the identity needs a_vu = 1/N")
    if config.env.label_availability != 1.0:
        problems.append(
            f"env.label_availability is {config.env.label_availability}; the average of "
            "per-agent means equals the pooled mean only for equal batch sizes"
        )
    for learner in config.learners:
        # Plain SGD is required as the *canonical* configuration, not because
        # every other optimizer breaks the identity. Heavy-ball momentum is
        # linear in the gradients, so averaging commutes and the identity
        # survives it; AdamW carries g^2 and does not (design note D35). Pinning
        # plain SGD keeps X0 testing the diffusion algebra alone, without
        # leaning on that additional fact.
        if learner.optimizer != "sgd":
            problems.append(
                f"learner[{learner.name}].optimizer is {learner.optimizer!r}; X0 is "
                "specified at plain SGD so the check depends on nothing but the "
                "diffusion algebra"
            )
        if learner.momentum != 0.0:
            problems.append(f"learner[{learner.name}].momentum is {learner.momentum}, not 0")

    if problems:
        raise SimulationError(
            "exactness preconditions not met:\n  - "
            + "\n  - ".join(problems)
            + "\n\nThese are not preferences. Each one produces a small, plausible, non-zero "
            "residual rather than an obvious failure -- which is exactly the failure mode "
            "the check exists to catch (WORKPLAN.md section 7.1)."
        )


def run(
    config: Any,
    environment: Environment,
    learners: dict[str, Any],
    evalsets: EvalSetBuilder,
    likelihood: Any,
    theta0: torch.Tensor,
    recorder: Any = None,
    verify_observations: bool = True,
    progress_every: int = 0,
    stop_after: int | None = None,
) -> list[StepRecord]:
    """Run one seed to the horizon.

    Args:
        verify_observations: check on evaluation steps that no learner mutated a
            shared observation in place. Cheap because the environment is
            positional, and it guards a corruption that would otherwise surface
            as an unexplained exactness residual.
        stop_after: stop once this step has completed, leaving the run resumable.
            Not a config field on purpose: the horizon *is* part of the config
            fingerprint, because alpha = total_degrees / T means changing T
            changes the data at every step. Interrupting is not reconfiguring.
    """
    if config.run.name.startswith("x0") or config.run.name.endswith("exactness"):
        check_exactness_preconditions(config)

    for learner in learners.values():
        learner.init(theta0)

    # Resume where a previous run stopped, if it did. Exact rather than
    # approximate: the loop consumes no randomness, so there is no RNG state to
    # restore (design note D38).
    start_step = recorder.resume(learners) if recorder is not None else 0
    if start_step:
        print(f"  resuming from step {start_step}")

    nodes = list(range(environment.n_nodes))
    n_edges = environment.graph.n_edges
    weights = environment.graph.weights
    if weights is None:  # pragma: no cover - build_graph always populates them
        raise SimulationError("the communication graph has no combination weights")

    per_node_drift = config.env.drift_scope == "per_node"
    records: list[StepRecord] = []

    last_step = (
        environment.horizon - 1 if stop_after is None else min(stop_after, environment.horizon - 1)
    )

    for step in range(start_step, last_step + 1):
        observations = environment.step(step)
        pooled_x, pooled_y = pool(observations)
        full_eval = protocol.should_evaluate(step, config.run.eval_every, environment.horizon)

        for name, learner in learners.items():
            # Test-then-train: score first, on the batch about to be learned
            # from. `predict` cannot update anything.
            preq = protocol.prequential(observations, learner.predict, likelihood, step=step)
            rows = preq.as_rows()

            _advance(learner, name, observations, nodes, weights, pooled_x, pooled_y)

            if full_eval:
                scores = protocol.full_evaluate(
                    evalsets,
                    learner.predict,
                    likelihood,
                    step=step,
                    nodes=nodes,
                    evalsets=config.eval.evalsets,
                    batch_size=config.eval.batch_size,
                    per_node_drift=per_node_drift,
                )
                rows.extend(scores.as_rows())
                rows.extend(_disagreement_rows(learner, learners, nodes, step))

            # Cumulative communication, so F2 plots error against it directly
            # rather than joining against the ledger.
            per_step = learner.comm_scalars_per_step(n_edges)
            stamped = [
                {
                    **row,
                    "learner": name,
                    "cum_scalars_tx": per_step * (step + 1),
                    "cum_rounds": (step + 1) if per_step else 0,
                }
                for row in rows
            ]
            records.append(StepRecord(step=step, learner=name, rows=stamped))
            if recorder is not None:
                recorder.log_many(stamped)

        if full_eval and verify_observations:
            environment.assert_unmodified(observations, step)

        # Flush where a step's rows are complete for every learner. A fixed row
        # budget would land mid-step and leave a partial file incoherent.
        if full_eval and recorder is not None:
            recorder.flush(step, learners)

        if progress_every and step % progress_every == 0:
            _report(step, environment.horizon, records)

    return records


def _advance(
    learner: Any,
    name: str,
    observations: dict[int, Any],
    nodes: list[int],
    weights: torch.Tensor,
    pooled_x: torch.Tensor,
    pooled_y: torch.Tensor,
) -> None:
    """One adapt/combine cycle.

    The centralized learner is the one special case: it consumes the *pooled*
    batch rather than adapting per agent. That is not a wart in the interface --
    it is the definition of the method, and `pool()` builds the union the X0
    identity is stated over.
    """
    if name == REFERENCE_LEARNER:
        learner.adapt_pooled(pooled_x, pooled_y)
        return

    intermediates = {node: learner.adapt(node, observations[node]) for node in nodes}
    learner.combine(intermediates, weights)


def _disagreement_rows(
    learner: Any, learners: dict[str, Any], nodes: list[int], step: int
) -> list[dict[str, Any]]:
    r"""$E_{\text{agree}}$, and $E_{\text{cent}}$ where a reference exists.

    $E_{\text{cent}}$ is omitted rather than zeroed for the centralized learner
    itself and for any run without one: a zero there would be read as "these
    coincide" when the truth is "there is nothing to compare against".
    """
    parameters = {node: learner.flat_params(node) for node in nodes}
    reference = learners.get(REFERENCE_LEARNER)
    centralized = (
        reference.flat_params(0) if reference is not None and learner is not reference else None
    )
    measured = disagreement.measure(parameters, centralized)
    return [{**row, "t": step, "node_id": "mean"} for row in measured.as_rows()]


def _report(step: int, horizon: int, records: list[StepRecord]) -> None:
    recent = [
        row
        for record in records[-8:]
        for row in record.rows
        if row.get("evalset") == "prequential" and row.get("metric") == "error_rate"
    ]
    if not recent:
        return
    mean = sum(float(row["value"]) for row in recent) / len(recent)
    print(f"  t={step:>5}/{horizon}   recent prequential error {mean:.3f}")
