"""Name to learner, and the phase-5 stub."""

from __future__ import annotations

from typing import Any

import torch

from dekf_bench.learners.base import Intermediate, LearnerError, LearnerState
from dekf_bench.learners.ekf import CentralizedEKF
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
    # Two names, one class. The gamma and lambda families are the same recursion
    # under different transition models, and the config picks which by setting
    # `transition` -- so a run that names both gets a genuine comparison rather
    # than two implementations that might disagree for uninteresting reasons
    # (design note D56).
    "centralized_ekf_gamma": CentralizedEKF,
    "centralized_ekf_lambda": CentralizedEKF,
    # The gamma family pinned at gamma = 1: the driftless random walk, which is
    # the canonical state model rather than a tuned variant of it. A separate
    # name because X13 could not separate it from a shrinking gamma -- the whole
    # gamma span was 0.0012 against a 0.0013 threshold -- so both are carried
    # forward, and two entries in one run need two names (design note D71).
    "centralized_ekf_walk": CentralizedEKF,
    # The non-adapting baseline. Shares the ATC implementation and differs only
    # in carrying a `freeze_after`, so "what does continuing to adapt buy?" is
    # answered against the same algorithm rather than against a different one.
    "frozen_atc": DiffusionSGDATC,
}

#: Learners whose combine step actually transmits. `centralized_sgd` and
#: `local_only` are both False, for opposite reasons.
DIFFUSING = {"diffusion_sgd_atc", "diffusion_sgd_atc_plain", "diffusion_sgd_cta", "diffusion_ekf"}

#: Learners that consume the *pooled* batch instead of adapting per agent. A set
#: rather than a name check in the runner: the centralized filter joined
#: `centralized_sgd` here the moment it existed, and the next pooled method
#: should not require editing `simulate.py` again to be dispatched correctly.
POOLING = {
    "centralized_sgd",
    "centralized_ekf_gamma",
    "centralized_ekf_lambda",
    "centralized_ekf_walk",
}

#: Learners holding a covariance, so metrics may ask them for predictive spread.
#: Nothing outside this set can answer an uncertainty question at all.
BAYESIAN = {
    "centralized_ekf_gamma",
    "centralized_ekf_lambda",
    "centralized_ekf_walk",
    "diffusion_ekf",
}

#: Every name the centralized filter answers to. One class, three names, each
#: asserting a different constraint on the state model.
CENTRALIZED_EKF = {"centralized_ekf_gamma", "centralized_ekf_lambda", "centralized_ekf_walk"}


def build_learner(learner_config: Any, model: Model, likelihood: Any, n_nodes: int) -> Any:
    """The learner a `learners:` entry asks for."""
    name = learner_config.name
    if name not in BUILDERS:
        raise LearnerError(f"unknown learner {name!r}; available: {sorted(BUILDERS)}")
    if name == "diffusion_ekf":
        return DiffusionEKF()
    if name in CENTRALIZED_EKF:
        return _build_centralized_ekf(learner_config, model, likelihood, n_nodes)

    return BUILDERS[name](
        name=name,
        model=model,
        likelihood=likelihood,
        optimizer=build_optimizer(learner_config),
        n_nodes=n_nodes,
        mix_policy=getattr(learner_config, "mix_optimizer_state", "none"),
        freeze_after=getattr(learner_config, "freeze_after", None),
    )


def _build_centralized_ekf(
    learner_config: Any, model: Model, likelihood: Any, n_nodes: int
) -> CentralizedEKF:
    r"""The centralized filter, with the transition fixed by the learner's name.

    **The name chooses the state model, and the config may not contradict it.**
    A `centralized_ekf_lambda` entry carrying $\gamma=0.99$ is not a variant of
    the $\lambda$ family; it is a mistake that would run happily and produce a
    third model nobody chose. Since $\gamma=1$ *is* the random walk, the two
    families overlap at exactly one point, and letting a config express that
    point twice is how a sweep ends up with duplicate cells that disagree
    (design note D56).
    """
    name = learner_config.name
    gamma = getattr(learner_config, "gamma", 1.0)
    lambda_forget = getattr(learner_config, "lambda_forget", 1.0)
    process_noise_q = getattr(learner_config, "process_noise_q", 0.0)
    transition = getattr(learner_config, "transition", "identity")

    # `transition` is checked rather than overwritten. Deriving it from the name
    # would let a config say `transition: identity` under the gamma learner and
    # be quietly ignored, which is the failure mode this whole function exists
    # to prevent.
    expected = "identity" if name == "centralized_ekf_lambda" else "scalar"
    if transition != expected:
        raise LearnerError(
            f"learner[{name}] requires transition={expected!r}, got {transition!r}. The "
            "name and the state model must agree; they are the same choice written twice."
        )

    if name == "centralized_ekf_walk" and gamma != 1.0:
        raise LearnerError(
            f"{name} is the driftless random walk, which fixes gamma = 1, but "
            f"gamma={gamma} was set. Use centralized_ekf_gamma for a shrinking "
            "transition -- the two are carried separately precisely so that the "
            "difference between them stays visible (design note D71)."
        )

    if name in ("centralized_ekf_gamma", "centralized_ekf_walk"):
        if lambda_forget != 1.0:
            raise LearnerError(
                f"{name} is the gamma family (F = gamma I, P <- gamma^2 P + Q) and has no "
                f"forgetting factor, but lambda_forget={lambda_forget} was set. Use "
                "centralized_ekf_lambda for the forgetting model."
            )
    else:
        if gamma != 1.0:
            raise LearnerError(
                f"{name} is the lambda family (F = I, P <- P / lambda), which fixes "
                f"gamma = 1, but gamma={gamma} was set. Use centralized_ekf_gamma for a "
                "shrinking transition."
            )
        if process_noise_q != 0.0:
            raise LearnerError(
                f"{name} inflates the covariance by 1/lambda rather than by adding Q, but "
                f"process_noise_q={process_noise_q} was set. Two inflation mechanisms at "
                "once makes neither hyperparameter interpretable."
            )

    if not 0.0 < lambda_forget <= 1.0:
        raise LearnerError(f"lambda_forget must lie in (0, 1], got {lambda_forget}")
    if process_noise_q < 0.0:
        raise LearnerError(f"process_noise_q must be >= 0, got {process_noise_q}")

    return CentralizedEKF(
        name=name,
        model=model,
        likelihood=likelihood,
        n_nodes=n_nodes,
        transition=transition,
        gamma=gamma,
        lambda_forget=lambda_forget,
        process_noise_q=process_noise_q,
        prior_scale=getattr(learner_config, "prior_scale", 1.0),
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
