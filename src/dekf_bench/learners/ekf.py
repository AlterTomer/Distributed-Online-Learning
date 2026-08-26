r"""The centralised extended Kalman filter.

One belief over the pooled data from every agent -- the reference the diffusion
filter is measured against, and what it reduces to on a complete graph. See
``docs/filter.md`` for the derivation and the decisions; this module is the
implementation of it.

**Two state models, selected by config** (design note D56):

``transition: scalar`` -- the gamma family, $\bm F=\gamma\bm I$::

    m <- gamma * m                       P <- gamma^2 * P + Q

``transition: identity`` -- the lambda family, $\bm F=\bm I$::

    m <- m                               P <- P / lambda

At $\gamma=1$ the first is the random walk, which is why ``identity`` forces
$\gamma=1$: they are the same model and the config refuses to express it twice.

**The update is stacked, not sequential** (D57). A step's $n$ samples enter as
one update with $\bar{\bm H}$ of shape $(nq)\times p$, which gives the filter
exactly one linearisation per step -- the same budget every SGD baseline gets.
"""

from __future__ import annotations

from typing import Any

import torch

from dekf_bench.learners.base import Intermediate, LearnerError, LearnerState
from dekf_bench.models.base import Model


class FilterError(LearnerError):
    """Raised when the filter's belief stops being a belief."""


class CentralizedEKF:
    r"""One Gaussian belief $(\bm m,\bm P)$ over the pooled batch.

    Every agent reports the same parameters, exactly as ``CentralizedSGD``
    does, so $E_{\text{agree}}$ is identically zero and $E_{\text{cent}}$ is
    measured against this.

    **The covariance is stored once, not per agent.** A centralised filter has
    one belief by definition, and at $p=2908$ a per-agent copy would be ten
    covariances of 68 MB for no information. ``state(node)`` therefore hands
    back the shared tensor rather than a clone.
    """

    def __init__(
        self,
        name: str,
        model: Model,
        likelihood: Any,
        n_nodes: int,
        transition: str = "identity",
        gamma: float = 1.0,
        lambda_forget: float = 1.0,
        process_noise_q: float = 0.0,
        prior_scale: float = 1.0,
        **_ignored: Any,
    ) -> None:
        self._name = name
        self.model = model
        self.likelihood = likelihood
        self._n_nodes = n_nodes
        self.transition = transition
        self.gamma = gamma
        self.lambda_forget = lambda_forget
        self.process_noise_q = process_noise_q
        self.prior_scale = prior_scale

        self._mean: torch.Tensor | None = None
        self._covariance: torch.Tensor | None = None
        self._steps = 0

    # -- identity ----------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def comm_scalars_per_step(self, n_edges: int) -> int:
        """Zero. A centralised filter has one processor and sends nothing."""
        return 0

    # -- state -------------------------------------------------------------- #

    def init(self, theta0: torch.Tensor) -> None:
        r"""$\bm m_0=\bm\theta_0$ and $\bm P_0=\sigma_0^2\bm I$.

        $\sigma_0^2$ acts as an initial learning rate and is the method's most
        sensitive hyperparameter, which is why it is swept rather than chosen.
        """
        if theta0.ndim != 1:
            raise FilterError(f"theta0 must be flat, got shape {tuple(theta0.shape)}")
        if theta0.numel() != self.model.num_params:
            raise FilterError(
                f"theta0 has {theta0.numel()} entries but the model has "
                f"{self.model.num_params} parameters"
            )
        if self.prior_scale <= 0:
            raise FilterError(f"prior_scale must be > 0, got {self.prior_scale}")

        self._mean = theta0.clone()
        self._covariance = torch.eye(
            theta0.numel(), dtype=theta0.dtype, device=theta0.device
        ) * self.prior_scale

    def state(self, node: int) -> LearnerState:
        self._check_initialised()
        self._check_node(node)
        assert self._mean is not None and self._covariance is not None
        return LearnerState(theta=self._mean, extras={"P": self._covariance})

    def flat_params(self, node: int) -> torch.Tensor:
        self._check_initialised()
        self._check_node(node)
        assert self._mean is not None
        return self._mean

    @property
    def covariance(self) -> torch.Tensor:
        self._check_initialised()
        assert self._covariance is not None
        return self._covariance

    def predict(self, node: int, x: torch.Tensor) -> torch.Tensor:
        params = self.model.unflatten(self.flat_params(node))
        return self.model.forward(params, x)

    # -- the step ----------------------------------------------------------- #

    def adapt(self, node: int, observation: Any) -> Intermediate:
        raise FilterError(
            f"{self._name} filters the pooled batch, not per agent. The runner calls "
            "adapt_pooled(); a per-agent adapt is the diffusion filter, which is a "
            "different method rather than this one restricted."
        )

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        """A no-op. One belief, nothing to combine."""

    def adapt_pooled(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Predict, then update on every agent's samples at once."""
        self._check_initialised()
        self._steps += 1
        self._predict()
        if x.shape[0] != 0:
            # An empty batch means no labels anywhere this step. The prediction
            # step still ran, which is the whole content of an unlabelled step:
            # the belief keeps its mean and loses confidence.
            self._update(x, y)
        self._check_belief()

    def _check_belief(self) -> None:
        r"""Stop on divergence instead of returning NaN to the metrics.

        The EKF mean update is a Gauss--Newton step whose trust region is
        $\bm P$ itself, so **too large a $\sigma_0^2$ diverges** -- at
        $\sigma_0^2=1$ the first step moves $\lVert\bm m\rVert$ by more than
        $\lVert\bm\theta_0\rVert$, which leaves the region the linearisation
        describes, and every later linearisation is taken somewhere worse.
        That is the method behaving as derived, not a bug, and it is why the
        prior scale is swept rather than picked (design note D61).

        The check is deliberately $O(p)$: a Cholesky every step would cost more
        than the update it guards. A non-positive diagonal entry is enough to
        rule out positive definiteness, and is free.
        """
        assert self._mean is not None and self._covariance is not None
        if not bool(torch.isfinite(self._mean).all()):
            raise FilterError(
                f"{self._name} diverged at step {self._steps}: the mean is not finite. "
                f"prior_scale={self.prior_scale} is most likely too large -- the update "
                "is a Gauss-Newton step and the covariance is its trust region."
            )
        smallest = float(self._covariance.diagonal().min())
        if not smallest > 0.0:
            raise FilterError(
                f"{self._name} lost positive definiteness at step {self._steps}: the "
                f"smallest variance is {smallest:.3e}. With no process noise and "
                "lambda at 1 the covariance collapses; give the filter a way to stay "
                "uncertain."
            )

    def _predict(self) -> None:
        assert self._mean is not None and self._covariance is not None
        if self.transition == "scalar":
            self._mean = self.gamma * self._mean
            self._covariance = self.gamma**2 * self._covariance
        if self.lambda_forget < 1.0:
            self._covariance = self._covariance / self.lambda_forget
        if self.process_noise_q > 0.0:
            self._covariance.diagonal().add_(self.process_noise_q)

    def _update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        r"""One stacked measurement update, via Woodbury.

        The information increment $\bar{\bm B}\bar{\bm B}^{\mathsf T}$ has rank
        at most $n(q-1)$ -- 360 here against $p=2908$ -- so the identity turns a
        $p\times p$ inverse into an $(nq)\times(nq)$ one. Direct inversion is
        $O(p^3)\approx2.5\times10^{10}$ flops a step, about an hour a seed
        before any sweep (D59).
        """
        assert self._mean is not None and self._covariance is not None
        mean, covariance = self._mean, self._covariance

        params = self.model.unflatten(mean)
        logits = self.model.forward(params, x)
        jacobians = self.model.per_sample_jacobian(params, x)  # (n, q, p)
        factor = self.likelihood.fisher_factor(logits)  # (n, q, q)

        batch, outputs, _ = jacobians.shape
        # B_i = H_i^T G_i, stacked along the column axis into (p, n*q).
        stacked = torch.einsum("nqp,nqr->pnr", jacobians, factor).reshape(-1, batch * outputs)
        # The score, not the innovation: they differ off the canonical link, and
        # the factor between them is not one the covariance can absorb (D60).
        score = torch.einsum("nqp,nq->p", jacobians, self.likelihood.score(logits, y))

        product = covariance @ stacked  # (p, n*q)
        gram = stacked.transpose(0, 1) @ product  # (n*q, n*q)
        gram.diagonal().add_(1.0)

        # $\bm I + \bar{\bm B}^{\mathsf T}\bm P\bar{\bm B}$ is symmetric positive
        # definite by construction -- identity plus a Gram matrix -- so it gets a
        # Cholesky solve rather than a general LU one. Cheaper, better
        # conditioned, and its failure carries information a general solver's
        # does not: the only way to lose definiteness here is for
        # $\bar{\bm B}^{\mathsf T}\bm P\bar{\bm B}$ to be so large that adding
        # the identity is lost to rounding, which is D61's divergence caught one
        # step before the mean goes non-finite.
        try:
            factor = torch.linalg.cholesky(gram)
        except torch.linalg.LinAlgError as failure:
            raise FilterError(
                f"{self._name} diverged at step {self._steps}: the innovation covariance "
                f"is singular. prior_scale={self.prior_scale} is far too large -- P has "
                "grown until the identity is negligible beside it."
            ) from failure

        solved = torch.cholesky_solve(product.transpose(0, 1), factor)  # (n*q, p)
        updated = covariance - product @ solved
        updated = 0.5 * (updated + updated.transpose(0, 1))

        self._covariance = updated
        self._mean = mean + updated @ score

    # -- what no SGD baseline can report ------------------------------------ #

    def logit_covariance(self, node: int, x: torch.Tensor) -> torch.Tensor:
        r"""$\bm H\bm P\bm H^{\mathsf T}$ per sample, shape ``(n, q, q)``.

        The belief's uncertainty pushed through the network, before any choice
        about how to turn it into class probabilities. Equation 47 adds $\bm R$
        for regression; under a categorical likelihood there is no $\bm R$, so
        this is the logit covariance and the mapping to probabilities is a
        separate decision.

        A delta-method approximation, and **systematically over-confident**: it
        ignores the linearisation remainder and model misspecification alike.
        """
        self._check_initialised()
        assert self._covariance is not None
        params = self.model.unflatten(self.flat_params(node))
        jacobians = self.model.per_sample_jacobian(params, x)
        return torch.einsum("nqp,pr,nsr->nqs", jacobians, self._covariance, jacobians)

    # -- diagnostics -------------------------------------------------------- #

    def summary(self) -> dict[str, Any]:
        self._check_initialised()
        assert self._covariance is not None
        diagonal = self._covariance.diagonal()
        return {
            "transition": self.transition,
            "gamma": self.gamma,
            "lambda_forget": self.lambda_forget,
            "process_noise_q": self.process_noise_q,
            "prior_scale": self.prior_scale,
            "trace_over_p": float(diagonal.mean()),
            "min_diagonal": float(diagonal.min()),
        }

    def _check_initialised(self) -> None:
        if self._mean is None or self._covariance is None:
            raise FilterError(f"{self._name} has no belief; call init(theta0) before stepping")

    def _check_node(self, node: int) -> None:
        if not 0 <= node < self._n_nodes:
            raise FilterError(f"no agent {node}; have 0..{self._n_nodes - 1}")
