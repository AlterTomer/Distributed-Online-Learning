"""Name to learner, and the phase-5 stub."""

from __future__ import annotations

from typing import Any

import torch

from dekf_bench.learners.base import Intermediate, LearnerError, LearnerState
from dekf_bench.learners.optim_state import build_optimizer
from dekf_bench.learners.sgd import (
    CentralizedSGD,
    DiffusionSGDATC,
    DiffusionSGDCTA,
    LocalOnly,
)
from dekf_bench.models.base import Model


class DiffusionEKF:
    """Phase 5. The interface only, so the type checker exercises it now.

    Present rather than absent because IMPLEMENTATION.md §13.5 asks for the
    adapt/combine split to be *exercised* before the filter arrives: if the
    interface were shaped only around SGD, discovering that the filter does not
    fit would happen at the point where changing it is most expensive.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._name = "diffusion_ekf"

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_nodes(self) -> int:
        raise NotImplementedError(_MESSAGE)

    def init(self, theta0: torch.Tensor) -> None:
        raise NotImplementedError(_MESSAGE)

    def adapt(self, node: int, observation: Any) -> Intermediate:
        raise NotImplementedError(_MESSAGE)

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        raise NotImplementedError(_MESSAGE)

    def predict(self, node: int, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_MESSAGE)

    def state(self, node: int) -> LearnerState:
        raise NotImplementedError(_MESSAGE)

    def flat_params(self, node: int) -> torch.Tensor:
        raise NotImplementedError(_MESSAGE)

    def comm_scalars_per_step(self, n_edges: int) -> int:
        raise NotImplementedError(_MESSAGE)


_MESSAGE = (
    "diffusion_ekf arrives in phase 5. The interface exists now so that the "
    "adapt/combine split is exercised before the filter is written -- see "
    "IMPLEMENTATION.md section 13.5."
)

#: Every learner a config may name. `diffusion_sgd_atc_plain` shares the ATC
#: implementation; it differs only in carrying no optimizer state, which is what
#: makes its payload p rather than 2p (design note D29).
BUILDERS = {
    "centralized_sgd": CentralizedSGD,
    "local_only": LocalOnly,
    "diffusion_sgd_atc": DiffusionSGDATC,
    "diffusion_sgd_atc_plain": DiffusionSGDATC,
    "diffusion_sgd_cta": DiffusionSGDCTA,
    "diffusion_ekf": DiffusionEKF,
    # The non-adapting baseline. Shares the ATC implementation and differs only
    # in carrying a `freeze_after`, so "what does continuing to adapt buy?" is
    # answered against the same algorithm rather than against a different one.
    "frozen_atc": DiffusionSGDATC,
}

#: Learners whose combine step actually transmits. `centralized_sgd` and
#: `local_only` are both False, for opposite reasons.
DIFFUSING = {"diffusion_sgd_atc", "diffusion_sgd_atc_plain", "diffusion_sgd_cta", "diffusion_ekf"}


def build_learner(learner_config: Any, model: Model, likelihood: Any, n_nodes: int) -> Any:
    """The learner a `learners:` entry asks for."""
    name = learner_config.name
    if name not in BUILDERS:
        raise LearnerError(f"unknown learner {name!r}; available: {sorted(BUILDERS)}")
    if name == "diffusion_ekf":
        return DiffusionEKF()

    return BUILDERS[name](
        name=name,
        model=model,
        likelihood=likelihood,
        optimizer=build_optimizer(learner_config),
        n_nodes=n_nodes,
        mix_policy=getattr(learner_config, "mix_optimizer_state", "none"),
        freeze_after=getattr(learner_config, "freeze_after", None),
    )


def build_learners(config: Any, model: Model, likelihood: Any) -> dict[str, Any]:
    """Every learner in a run, in config order.

    They share one environment and one `theta_0` (design note D4), so the
    comparison between them is paired by construction rather than by seed.
    """
    return {
        entry.name: build_learner(entry, model, likelihood, config.graph.n_nodes)
        for entry in config.learners
    }
