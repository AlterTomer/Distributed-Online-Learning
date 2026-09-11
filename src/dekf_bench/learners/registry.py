"""Name to learner, and the phase-5 stub."""

from __future__ import annotations

from typing import Any

from dekf_bench.learners.base import LearnerError
from dekf_bench.learners.diffusion_ekf import DiffusionEKF
from dekf_bench.learners.ekf import TRUST_REGION_RATIO, CentralizedEKF
from dekf_bench.learners.optim_state import build_optimizer
from dekf_bench.learners.sgd import (
    CentralizedSGD,
    DiffusionSGDATC,
    DiffusionSGDCTA,
    LocalOnly,
)
from dekf_bench.models.base import Model

#: The diffusion filter's three names, and the (adapt_scope, covariance_sharing)
#: each pins. Separate names rather than one name with config fields, for the
#: reason D71 records: two variants have to appear in a *single* run to be
#: compared as a paired difference, and one name cannot appear twice.
DIFFUSION_EKF_VARIANTS = {
    # The expensive ceiling: shares everything the derivation makes available, so
    # whatever diffusion can achieve on a graph it achieves here. Implemented and
    # measured first precisely because it bounds the cheap one.
    "diffusion_ekf_full": ("local", "full"),
    # The deployable reduction, and what the communication claim will rest on.
    # Its gap to the above is the price of not shipping covariances.
    "diffusion_ekf": ("local", "local"),
    # Not a competitor: the correctness fixture. On a complete graph one-hop
    # makes the measurement set the whole vertex set, which is the hypothesis of
    # prop:complete_graph, and the filter must then reproduce the centralized one
    # exactly. Without it the diffusion filter has no analogue of X0.
    "diffusion_ekf_onehop": ("one_hop", "full"),
    # The deployable form of the fix D79 identifies, and after X19 the only
    # one-hop variant worth measuring: covariance sharing was shown to buy
    # +0.0002 to +0.0007 for 2909x the bandwidth, so the sensible pairing with a
    # one-hop adapt is mean-only.
    #
    # The exactness gate is unaffected by dropping full sharing: on a complete
    # graph one-hop makes every agent's (psi, P^psi) identical, so averaging the
    # covariances and keeping one's own give the same answer. The fixture above
    # keeps full sharing only because that is the variant the proposition is
    # written for.
    "diffusion_ekf_onehop_mean": ("one_hop", "local"),
}

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
    "diffusion_ekf_full": DiffusionEKF,
    "diffusion_ekf_onehop": DiffusionEKF,
    "diffusion_ekf_onehop_mean": DiffusionEKF,
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
DIFFUSING = {
    "diffusion_sgd_atc",
    "diffusion_sgd_atc_plain",
    "diffusion_sgd_cta",
    *DIFFUSION_EKF_VARIANTS,
}

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
    *DIFFUSION_EKF_VARIANTS,
}

#: Every name the centralized filter answers to. One class, three names, each
#: asserting a different constraint on the state model.
CENTRALIZED_EKF = {"centralized_ekf_gamma", "centralized_ekf_lambda", "centralized_ekf_walk"}


def build_learner(learner_config: Any, model: Model, likelihood: Any, n_nodes: int) -> Any:
    """The learner a `learners:` entry asks for."""
    name = learner_config.name
    if name not in BUILDERS:
        raise LearnerError(f"unknown learner {name!r}; available: {sorted(BUILDERS)}")
    if name in DIFFUSION_EKF_VARIANTS:
        return _build_diffusion_ekf(learner_config, model, likelihood, n_nodes)
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


def _build_diffusion_ekf(
    learner_config: Any, model: Model, likelihood: Any, n_nodes: int
) -> DiffusionEKF:
    r"""The diffusion filter, with both axes fixed by the learner's name.

    **The name chooses the variant, and the config may not contradict it**, for
    the reason `_build_centralized_ekf` documents: a config that says
    `covariance_sharing: local` under `diffusion_ekf_full` would run happily and
    produce a variant nobody chose, and in a sweep that is how two cells end up
    disagreeing for a reason no one can find afterwards.

    The state model is the gamma family throughout. The lambda family was
    rejected for the centralized filter in D76 -- multiplicative forgetting has
    no sustainable level when the information arriving is bounded below only by
    a softmax Fisher that decays to zero -- and nothing about diffusing the
    belief repairs that argument. Sharing covariances would in fact make it
    worse: an inflated P propagates to neighbours.
    """
    name = learner_config.name
    scope, sharing = DIFFUSION_EKF_VARIANTS[name]

    for field, chosen, expected in (
        ("adapt_scope", getattr(learner_config, "adapt_scope", scope), scope),
        (
            "covariance_sharing",
            getattr(learner_config, "covariance_sharing", sharing),
            sharing,
        ),
    ):
        if chosen != expected:
            raise LearnerError(
                f"learner[{name}] requires {field}={expected!r}, got {chosen!r}. The name "
                "and the variant must agree; they are the same choice written twice."
            )

    if getattr(learner_config, "lambda_forget", 1.0) != 1.0:
        raise LearnerError(
            f"{name} is the gamma family (F = gamma I, P <- gamma^2 P + Q) and has no "
            "forgetting factor. Multiplicative forgetting was rejected for the "
            "centralized filter in design note D76, and sharing covariances would "
            "propagate an inflated P to every neighbour rather than contain it."
        )

    return DiffusionEKF(
        name=name,
        model=model,
        likelihood=likelihood,
        n_nodes=n_nodes,
        transition="scalar",
        gamma=getattr(learner_config, "gamma", 1.0),
        lambda_forget=1.0,
        process_noise_q=getattr(learner_config, "process_noise_q", 0.0),
        prior_scale=getattr(learner_config, "prior_scale", 1.0),
        trust_region_ratio=getattr(learner_config, "trust_region_ratio", TRUST_REGION_RATIO),
        adapt_scope=scope,
        covariance_sharing=sharing,
        # Dials, not variant identity, so unlike scope and sharing these are read
        # from the config rather than pinned by the name.
        adapt_rounds=getattr(learner_config, "adapt_rounds", 1),
        combine_exponent=getattr(learner_config, "combine_exponent", 1.0),
        information_exponent=getattr(learner_config, "information_exponent", 0.0),
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
        trust_region_ratio=getattr(learner_config, "trust_region_ratio", 1.0e6),
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
