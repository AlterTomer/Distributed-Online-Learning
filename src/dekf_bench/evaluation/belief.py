r"""Scoring the filters' beliefs during a run (P5.11, P5.14).

Called by the runner at full-evaluation steps inside the settled window, for every
learner that holds a covariance, once per belief it can report: after combine
(``post``), and for the diffusion filter also before it (``pre``). The metrics are
`metrics/belief.py`'s; this module only decides what they are computed on.

**One fixed subset of the ``current`` set**, the first ``eval.belief_subset``
images, so every agent, stage and step is scored on the same inputs and the
comparisons between them are paired. The Jacobians are what cost: $n\times q\times p$
per agent, which at $n=1000$, $q=10$, $p=2908$ is 29 M entries, formed in chunks.

Rows carry the stage in the metric name -- ``kappa_star`` after combine,
``pre_kappa_star`` before it -- so the recording schema needs no new column and a
reader cannot mistake one for the other.
"""

from __future__ import annotations

from typing import Any

import torch

from dekf_bench.evaluation.evalsets import EvalSetBuilder
from dekf_bench.metrics.belief import classification_scores
from dekf_bench.runner.seeding import derive_seed

#: Jacobians are formed this many samples at a time: 250 x 10 x 2908 floats is
#: about 29 MB, small beside the covariance itself.
JACOBIAN_CHUNK = 250


def due(config: Any, step: int, horizon: int, full_eval: bool) -> bool:
    """Whether the beliefs are scored at this step."""
    return (bool(config.eval.belief_calibration) and full_eval
            and step >= int(config.eval.belief_from * horizon))


def scoreable(learner: Any) -> bool:
    """Learners that hold a belief to score: the ones with a `belief` method."""
    return callable(getattr(learner, "belief", None))


def predictive(model: Any, theta: torch.Tensor, covariance: torch.Tensor,
               x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""$\bm h(\bm\theta)$ and $\bm H\bm P\bm H^{\mathsf T}$ on ``x``, Jacobians in chunks."""
    params = model.unflatten(theta)
    logits = model.forward(params, x)
    blocks = []
    for start in range(0, x.shape[0], JACOBIAN_CHUNK):
        jacobians = model.per_sample_jacobian(params, x[start:start + JACOBIAN_CHUNK])
        # (n, q, p) @ (p, p) first, then against H again: one p x p product per chunk.
        projected = torch.einsum("nqp,pr->nqr", jacobians, covariance)
        blocks.append(torch.einsum("nqr,nsr->nqs", projected, jacobians))
    return logits, torch.cat(blocks)


def evaluate(
    builder: EvalSetBuilder,
    learner: Any,
    config: Any,
    step: int,
    nodes: list[int],
    seeds: Any,
) -> list[dict[str, Any]]:
    """Every belief this learner holds, scored on the fixed subset at ``step``."""
    evalset = builder.at("current", step)
    if evalset is None:  # pragma: no cover - `current` always exists
        return []
    x = evalset.images[: config.eval.belief_subset]
    y = evalset.labels[: config.eval.belief_subset]
    rows: list[dict[str, Any]] = []
    for stage in learner.belief_stages():
        prefix = "" if stage == "post" else f"{stage}_"
        for node in nodes:
            theta, covariance = learner.belief(node, stage)
            logits, logit_cov = predictive(learner.model, theta, covariance, x)
            # Its own keyed seed per (step, node, stage), derived from the master
            # seed by the same hash as every stream: reproducible, and drawing
            # nothing from the data, graph or initialisation streams.
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(derive_seed(seeds.master, "belief_mc", step, node, stage))
            scores = classification_scores(logits, logit_cov, y,
                                           config.eval.belief_mc_samples, generator)
            rows.extend(
                {"node_id": node, "evalset": "current", "t": step,
                 "drift_state": evalset.rotation_degrees,
                 "metric": f"{prefix}{name}", "value": value}
                for name, value in scores.items())
    return rows
