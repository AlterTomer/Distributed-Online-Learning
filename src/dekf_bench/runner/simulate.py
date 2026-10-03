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

from dekf_bench.compression import Channel, CodecChannel, data_bits, working_bits
from dekf_bench.env.environment import Environment, pool
from dekf_bench.evaluation import belief, protocol
from dekf_bench.evaluation.evalsets import EvalSetBuilder
from dekf_bench.learners.registry import POOLING
from dekf_bench.metrics import disagreement

#: The learner an experiment compares everything else against, when present.
REFERENCE_LEARNER = "centralized_sgd"
#: The pooled filters, in the order one becomes the filters' own E_cent reference
#: (P5.12, D134). The random walk first: where a run carries both, as M6 does, the
#: gamma = 0.9995 twin is a reference line, not the tuned filter.
FILTER_REFERENCES = ("centralized_ekf_walk", "centralized_ekf_gamma")


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
    _attach_channels(config, learners, environment.seeds, theta0)

    # Resume where a previous run stopped, if it did. Exact rather than
    # approximate: the loop consumes no randomness, so there is no RNG state to
    # restore (design note D38). Stochastic rounding is the one exception, so a
    # compressed run refuses to resume rather than redraw.
    start_step = recorder.resume(learners) if recorder is not None else 0
    if start_step:
        if config.comm.compressor == "stochastic":
            raise SimulationError(
                "a run with stochastic rounding cannot resume: its channel draws from a "
                "generator whose state the checkpoint does not hold. Re-run it from step 0.")
        if config.comm.compressor == "codec":
            raise SimulationError(
                "a codec run cannot resume: its public copies and counts are not in the "
                "checkpoint. Re-run it from step 0.")
        print(f"  resuming from step {start_step}")
    pricing = _bit_prices(config)
    cum_bits = {name: 0 for name in learners}

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
        score_beliefs = belief.due(config, step, environment.horizon, full_eval)

        for name, learner in learners.items():
            # Test-then-train: score first, on the batch about to be learned
            # from. `predict` cannot update anything.
            preq = protocol.prequential(observations, learner.predict, likelihood, step=step)
            rows = preq.as_rows()

            # P5.14 scores the belief before combine as well as after, so the
            # filter is asked to keep it -- on this step only, since under full
            # sharing it holds N more p x p matrices.
            if score_beliefs and hasattr(learner, "retain_pre_combine"):
                learner.retain_pre_combine = True
            channel = getattr(learner, "channel", None)
            sent_before = (channel.bits, channel.scalars) if channel is not None else (0, 0)
            _advance(learner, name, observations, nodes, weights, pooled_x, pooled_y)
            bits_this_step = _step_bits(learner, name, channel, sent_before, n_edges, pricing)
            if step == start_step and start_step:
                # A resumed run starts its running sum where the scalar column would:
                # exact whenever the per-step cost is constant, which it is for every
                # compressor here (only a frozen learner changes it, and to zero).
                cum_bits[name] = bits_this_step * start_step
            cum_bits[name] += bits_this_step

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
            if score_beliefs and belief.scoreable(learner):
                rows.extend(belief.evaluate(evalsets, learner, likelihood, config, step,
                                            nodes, environment.seeds))
            if hasattr(learner, "release_pre_combine"):
                learner.release_pre_combine()

            # Cumulative communication, so F2 plots error against it directly
            # rather than joining against the ledger.
            per_step = learner.comm_scalars_per_step(n_edges)
            stamped = [
                {
                    **row,
                    "learner": name,
                    "cum_scalars_tx": per_step * (step + 1),
                    "cum_bits_tx": cum_bits[name],
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

    # A channel with something to keep -- the codec's counts, scales and totals --
    # writes it beside the seed's parquet, once the seed has run to the end.
    if recorder is not None and last_step == environment.horizon - 1:
        for name, learner in learners.items():
            channel = getattr(learner, "channel", None)
            if channel is not None:
                channel.finish(recorder.out_dir, recorder.context.seed, name)
    return records


def _attach_channels(config: Any, learners: dict[str, Any], seeds: Any,
                     theta0: torch.Tensor | None = None) -> None:
    """Give every diffusing learner the run's compressor (compression.py, Track C).

    One channel per learner, each with its own seed stream when it rounds
    stochastically, so adding or removing a learner cannot change another's draws.
    The codec's channel is the learner's own too: its public copies start at
    theta_0, and its tables and the moments' scales are looked up by learner name.
    """
    for name, learner in learners.items():
        if not hasattr(learner, "channel"):
            continue
        if config.comm.compressor == "codec":
            learner.channel = _codec_channel(config, name, learner, theta0)
            continue
        stochastic = config.comm.compressor == "stochastic"
        learner.channel = Channel(
            compressor=config.comm.compressor,
            precision=config.comm.precision,
            generator=seeds.torch_generator("channel", name) if stochastic else None,
        )


def _codec_channel(config: Any, name: str, learner: Any, theta0: torch.Tensor | None) -> Any:
    """A learner's codec channel: its layers, theta_0, its tables and moment scales."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from dekf_bench.codec import LayerCode, module_layers  # noqa: PLC0415

    if theta0 is None:
        raise SimulationError("the codec's public copies start at theta_0, which was not given")
    comm = config.comm
    layers = module_layers(learner.model)
    names = [layer for layer, _part in layers]

    def per_layer(entry: dict, what: str, build: Any) -> list:
        missing = [layer for layer in names if layer not in entry]
        if missing:
            raise SimulationError(f"{name}: {what} has no entry for layers {missing}")
        return [build(entry[layer]) for layer in names]

    scales, codes = {}, {}
    if comm.codec_scales:
        table = json.loads(Path(comm.codec_scales).read_text(encoding="utf-8"))
        for kind, entry in table.get("learners", {}).get(name, {}).items():
            scales[kind] = per_layer(entry, f"the scales for {kind!r}", float)
    if comm.codec_mode == "code":
        table = json.loads(Path(comm.codec_tables).read_text(encoding="utf-8"))
        if abs(float(table["c"]) - comm.codec_c) > 1e-12 * comm.codec_c:
            raise SimulationError(
                f"the tables were trained at c={table['c']}, the run asks for c={comm.codec_c}")
        for kind, entry in table.get("learners", {}).get(name, {}).items():
            codes[kind] = per_layer(entry, f"the tables for {kind!r}", LayerCode.from_json)
    return CodecChannel(c=comm.codec_c, mode=comm.codec_mode, layers=layers,
                        theta0=theta0.detach().clone(), moment_scales=scales, codes=codes)


def _bit_prices(config: Any) -> dict[str, int]:
    """Bits per scalar for what does not go through the channel (C-1, C-8)."""
    dtype = getattr(torch, config.run.dtype)
    return {"covariance": working_bits(dtype), "prior": working_bits(dtype),
            "data": data_bits(config.env.dataset, dtype)}


def _step_bits(learner: Any, name: str, channel: Any, before: tuple[int, int],
               n_edges: int, pricing: dict[str, int]) -> int:
    """This step's bits: the channel's, plus the rest of the payload at its own price.

    Checks the channel mixed exactly the vector scalars the learner's ledger declares,
    which is what keeps the bits column and the scalar column describing one run.
    """
    if not hasattr(learner, "comm_payload"):
        return 0
    payload = learner.comm_payload(n_edges)
    mixed_bits = channel.bits - before[0] if channel is not None else 0
    mixed_scalars = channel.scalars - before[1] if channel is not None else 0
    if mixed_scalars != payload.get("vectors", 0):
        raise SimulationError(
            f"{name}: the channel mixed {mixed_scalars} vector scalars this step but the "
            f"ledger declares {payload.get('vectors', 0)}. A learner is mixing something "
            "it does not pay for, or paying for something it does not mix.")
    return mixed_bits + sum(count * pricing[kind] for kind, count in payload.items()
                            if kind != "vectors")


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

    Pooled learners are the one special case: they consume the *union* of every
    agent's batch rather than adapting per agent. That is not a wart in the
    interface -- it is the definition of those methods, and `pool()` builds the
    union the X0 identity is stated over.

    Dispatched on membership in `POOLING` rather than on the reference learner's
    name, because the centralized EKF is pooled for exactly the same reason
    `centralized_sgd` is while being a different method entirely.
    """
    if name in POOLING:
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

    A learner that holds a covariance also gets ``e_cent_filter``, the same
    distance to the pooled *filter* (P5.12): centralised SGD is the wrong
    reference for asking whether a diffusion filter's agents approach the
    centralised belief. Omitted by the same rule.
    """
    parameters = {node: learner.flat_params(node) for node in nodes}
    reference = learners.get(REFERENCE_LEARNER)
    centralized = (
        reference.flat_params(0) if reference is not None and learner is not reference else None
    )
    pooled_filter = next((learners[n] for n in FILTER_REFERENCES if n in learners), None)
    holds_covariance = "P" in learner.state(nodes[0]).extras
    filter_centre = (
        pooled_filter.flat_params(0)
        if pooled_filter is not None and learner is not pooled_filter and holds_covariance
        else None
    )
    measured = disagreement.measure(parameters, centralized, filter_centre)
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
