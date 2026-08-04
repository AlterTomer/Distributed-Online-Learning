"""When things are scored: prequential every step, full evaluation every K.

**Prequential is test-then-train.** At each step every agent is scored on the
batch it is about to learn from, *before* it learns from it. Cheap, unbiased,
and it gives a per-step signal where the full evaluation is affordable only
every $K$ steps.

The ordering is a correctness property, not a convention: score after training
and the model has already seen the answer, so the curve reports memorisation
rather than generalisation and does so smoothly enough to look plausible. This
module cannot enforce the order by itself -- the runner calls it -- so
:func:`prequential` takes a **predict function** rather than a learner, which
means it structurally *cannot* update anything. A protocol that could train
would be a protocol that might.

**Scoring is by node, and aggregates are derived later.** Only per-agent rows
are returned; the mean, the spread and the gap are computed at plot time from
those (IMPLEMENTATION.md §8.1).

**Idle agents are skipped, not scored as zero.** With $\\pi_{\\text{lab}}<1$ an
agent may receive no samples at a step. It has nothing to be right or wrong
about, and recording a 0% or 100% error rate there would bias the mean by an
amount that grows as labels get sparser -- precisely along the axis X4 sweeps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from dekf_bench.evaluation.evalsets import EvalSet, EvalSetBuilder
from dekf_bench.metrics import calibration
from dekf_bench.metrics.classification import correct_count

#: A learner's prediction interface: (node, inputs) -> logits. Deliberately
#: narrower than the learner itself, so a protocol cannot mutate one.
PredictFn = Callable[[int, torch.Tensor], torch.Tensor]


class ProtocolError(ValueError):
    """Raised when an evaluation cannot be carried out as asked."""


@dataclass(frozen=True)
class NodeScore:
    """One agent's score on one evaluation set at one step."""

    node: int
    evalset: str
    step: int
    n_samples: int
    n_correct: int
    rotation_degrees: float
    nll: float | None = None
    calibration: calibration.CalibrationScores | None = None

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_samples

    @property
    def error_rate(self) -> float:
        return 1.0 - self.accuracy

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            {
                "node_id": self.node,
                "evalset": self.evalset,
                "t": self.step,
                "drift_state": self.rotation_degrees,
                "metric": "error_rate",
                "value": self.error_rate,
            }
        ]
        if self.nll is not None:
            rows.append({**rows[0], "metric": "nll", "value": self.nll})
        if self.calibration is not None:
            for entry in self.calibration.as_rows():
                if entry["metric"] != "nll":  # already recorded above
                    rows.append({**rows[0], **entry})
        return rows


@dataclass(frozen=True)
class StepScores:
    """Every agent's scores at one step, across the sets that were evaluated."""

    step: int
    scores: tuple[NodeScore, ...] = field(default_factory=tuple)
    skipped: tuple[int, ...] = field(default_factory=tuple)

    def for_evalset(self, name: str) -> tuple[NodeScore, ...]:
        return tuple(score for score in self.scores if score.evalset == name)

    def error_rates(self, name: str) -> dict[int, float]:
        return {score.node: score.error_rate for score in self.for_evalset(name)}

    def as_rows(self) -> list[dict[str, Any]]:
        return [row for score in self.scores for row in score.as_rows()]


# --------------------------------------------------------------------------- #
# prequential
# --------------------------------------------------------------------------- #


def prequential(
    observations: dict[int, Any],
    predict: PredictFn,
    likelihood: Any,
    step: int,
    with_calibration: bool = False,
) -> StepScores:
    """Score every agent on the batch it is about to train on.

    Must be called *before* the adapt step. ``predict`` cannot update anything,
    which is the structural half of that guarantee; the runner's ordering is the
    other half.
    """
    scores: list[NodeScore] = []
    skipped: list[int] = []

    for node in sorted(observations):
        observation = observations[node]
        if not observation.has_label:
            skipped.append(node)
            continue

        logits = predict(node, observation.x)
        _check_logits(logits, observation, node)
        scores.append(
            _score(
                node=node,
                evalset="prequential",
                step=step,
                logits=logits,
                targets=observation.y,
                rotation=observation.rotation_degrees,
                likelihood=likelihood,
                with_calibration=with_calibration,
            )
        )

    return StepScores(step=step, scores=tuple(scores), skipped=tuple(skipped))


# --------------------------------------------------------------------------- #
# periodic full evaluation
# --------------------------------------------------------------------------- #


def full_evaluate(
    builder: EvalSetBuilder,
    predict: PredictFn,
    likelihood: Any,
    step: int,
    nodes: list[int],
    evalsets: list[str],
    batch_size: int = 1000,
    per_node_drift: bool = False,
    with_calibration: bool = True,
) -> StepScores:
    """Score every agent on the held-out sets.

    Args:
        evalsets: which sets to build. ``prequential`` is ignored here -- it
            scores the incoming training batch and is handled above.
        per_node_drift: when true, ``current`` is built per agent at that
            agent's own rotation, and ``current_mean`` is additionally scored so
            the per-agent spread can be separated from the rotation spread.
    """
    scores: list[NodeScore] = []
    wanted = [name for name in evalsets if name != "prequential"]
    if per_node_drift and "current" in wanted and "current_mean" not in wanted:
        wanted.append("current_mean")

    for name in wanted:
        shared = None if (name == "current" and per_node_drift) else builder.at(name, step)
        for node in nodes:
            evalset = shared if shared is not None else builder.at(name, step, node)
            if evalset is None:
                # The backward probe is undefined here. Recording nothing is the
                # point: "cannot measure" and "measured no forgetting" must not
                # log the same value.
                continue
            scores.append(
                _score_evalset(node, evalset, predict, likelihood, batch_size, with_calibration)
            )

    return StepScores(step=step, scores=tuple(scores))


def should_evaluate(step: int, eval_every: int, horizon: int) -> bool:
    """Whether ``step`` is a full-evaluation step.

    The final step always is, so every run ends with a complete measurement
    whatever the horizon and cadence happen to be -- otherwise the last point of
    every curve would move with $T \\bmod K$.
    """
    if eval_every < 1:
        raise ProtocolError(f"eval_every must be >= 1, got {eval_every}")
    return step % eval_every == 0 or step == horizon - 1


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _score_evalset(
    node: int,
    evalset: EvalSet,
    predict: PredictFn,
    likelihood: Any,
    batch_size: int,
    with_calibration: bool,
) -> NodeScore:
    """Score one agent on one set, in batches so a 10 000-image pass is not one
    allocation."""
    all_logits, all_targets = [], []
    for images, labels in evalset.batches(batch_size):
        all_logits.append(predict(node, images))
        all_targets.append(labels)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    return _score(
        node=node,
        evalset=evalset.name,
        step=evalset.step,
        logits=logits,
        targets=targets,
        rotation=evalset.rotation_degrees,
        likelihood=likelihood,
        with_calibration=with_calibration,
    )


def _score(
    node: int,
    evalset: str,
    step: int,
    logits: torch.Tensor,
    targets: torch.Tensor,
    rotation: float,
    likelihood: Any,
    with_calibration: bool,
) -> NodeScore:
    predictions = likelihood.predictions(logits)
    return NodeScore(
        node=node,
        evalset=evalset,
        step=step,
        n_samples=int(targets.numel()),
        n_correct=correct_count(predictions, targets),
        rotation_degrees=rotation,
        nll=float(likelihood.nll(logits, targets, reduction="mean")),
        calibration=(calibration.score(logits, targets, likelihood) if with_calibration else None),
    )


def _check_logits(logits: torch.Tensor, observation: Any, node: int) -> None:
    if logits.shape[0] != observation.n_samples:
        raise ProtocolError(
            f"agent {node}: predict returned {logits.shape[0]} rows for "
            f"{observation.n_samples} samples"
        )
