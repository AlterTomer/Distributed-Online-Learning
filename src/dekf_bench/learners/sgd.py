r"""The four first-order methods, sharing one gradient and one optimizer.

They differ in exactly two places:

=====================  ===============================  ========================
learner                adapt sees                       combine
=====================  ===============================  ========================
``centralized_sgd``    the pooled batch $\bigcup_v$      nothing
``diffusion_sgd_atc``  agent $v$'s own batch            average the $\bm\psi$
``diffusion_sgd_cta``  agent $v$'s own batch            *before* the gradient
``local_only``         agent $v$'s own batch            nothing
=====================  ===============================  ========================

Everything else -- the gradient, the update rule, the state container -- is
shared, so a difference in the results is a difference in the method.

**ATC and CTA differ in ordering, not in cost.** ATC computes the gradient at
$\bm\theta_{v}$ and averages the result; CTA averages first and computes the
gradient at the *pre-combine* parameters. Both send one $\bm\psi$ per link per
step. ATC is primary because Diff-EKF is ATC: with both ATC, phase 5 differs in
the adapt step alone, and any advantage is attributable to the second-order
update rather than to the ordering (WORKPLAN §3.2).

**Centralized SGD is the online upper reference, not a competitor.** It sees the
pooled batch every step -- which in a real deployment means shipping every
sample to a centre -- and is plotted as a horizontal line on the communication
axis rather than as a point on it (design note D30).
"""

from __future__ import annotations

from typing import Any

import torch

from dekf_bench.learners.base import Intermediate, LearnerError, LearnerState, loss_gradient
from dekf_bench.learners.optim_state import Optimizer, mixed_entries
from dekf_bench.models.base import Model


class _SGDBase:
    """Shared machinery: state, prediction, the gradient, the ledger."""

    def __init__(
        self,
        name: str,
        model: Model,
        likelihood: Any,
        optimizer: Optimizer,
        n_nodes: int,
        mix_policy: str = "none",
        freeze_after: int | None = None,
    ) -> None:
        self._name = name
        self.model = model
        self.likelihood = likelihood
        self.optimizer = optimizer
        self._n_nodes = n_nodes
        self.mix_policy = mix_policy
        self.mixed = mixed_entries(optimizer, mix_policy)
        self._states: dict[int, LearnerState] = {}
        #: Stop adapting at this step. The non-adapting baseline the comparative
        #: break is measured against: it learned from the same data, with the
        #: same optimizer, up to the same point, and then simply stopped -- so
        #: the comparison isolates *continued adaptation* from initial learning.
        self.freeze_after = freeze_after
        self._last_step = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def init(self, theta0: torch.Tensor) -> None:
        r"""Every agent starts at the *same* $\bm\theta_0$.

        Cloned per agent rather than shared, or the first in-place update would
        change every agent at once -- and the run would look like perfect
        consensus for a reason that has nothing to do with the combine step.
        """
        if theta0.ndim != 1:
            raise LearnerError(f"theta0 must be flat, got shape {tuple(theta0.shape)}")
        if theta0.numel() != self.model.num_params:
            raise LearnerError(
                f"theta0 has {theta0.numel()} entries but the model has "
                f"{self.model.num_params} parameters"
            )
        self._states = {
            node: LearnerState(
                theta=theta0.clone(),
                extras=self.optimizer.init_state(theta0.numel(), theta0.dtype, theta0.device),
            )
            for node in range(self._n_nodes)
        }

    def state(self, node: int) -> LearnerState:
        self._check_initialised()
        if node not in self._states:
            raise LearnerError(f"no agent {node}; have 0..{self._n_nodes - 1}")
        return self._states[node]

    def flat_params(self, node: int) -> torch.Tensor:
        return self.state(node).theta

    def predict(self, node: int, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(self.model.unflatten(self.flat_params(node)), x)

    def comm_scalars_per_step(self, n_edges: int) -> int:
        """Zero for the non-diffusing methods; overridden where it is not."""
        return 0

    # -- shared internals --------------------------------------------------- #

    def _gradient(self, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return loss_gradient(self.model, theta, x, y, self.likelihood)

    def is_frozen(self, step: int | None = None) -> bool:
        """Whether adaptation has stopped by ``step`` (default: the last seen)."""
        if self.freeze_after is None:
            return False
        return (self._last_step if step is None else step) >= self.freeze_after

    def _local_step(self, node: int, observation: Any) -> torch.Tensor:
        r"""One optimizer step on agent ``node``'s own data, returning $\bm\psi$.

        An idle agent takes no step: with no label there is no gradient, and its
        estimate passes through unchanged. That is what lets it still benefit
        from the combine step, which is one of diffusion's more attractive
        properties.
        """
        self._last_step = int(observation.step)
        state = self.state(node)
        # A frozen learner takes no step for the same structural reason an idle
        # one does not: it contributes no update, and its estimate passes
        # through unchanged.
        if self.is_frozen(observation.step) or not observation.has_label:
            return state.theta.clone()
        gradient = self._gradient(state.theta, observation.x, observation.y)
        return self.optimizer.step(state.theta, gradient, state.extras)

    def _payload(self, node: int) -> dict[str, torch.Tensor]:
        """The optimizer entries that travel with psi, if any."""
        state = self.state(node)
        return {name: state.extras[name].clone() for name in self.mixed}

    def _check_initialised(self) -> None:
        if not self._states:
            raise LearnerError(f"{self._name} has no state; call init(theta0) before stepping")


class CentralizedSGD(_SGDBase):
    r"""The upper reference: one logical learner on the pooled batch.

    Every agent holds the same $\bm\theta$ by construction, so ``combine`` is a
    no-op and $E_{\text{agree}}$ is identically zero. Keeping $N$ copies rather
    than one keeps the runner uniform and lets $E_{\text{cent}}$ read this
    learner's parameters the same way it reads any other's.
    """

    def adapt(self, node: int, observation: Any) -> Intermediate:
        raise LearnerError(
            "centralized_sgd adapts on the pooled batch, not per agent. The runner "
            "calls adapt_pooled(); a per-agent adapt would be a different method."
        )

    def adapt_pooled(self, x: torch.Tensor, y: torch.Tensor) -> None:
        r"""One step on $\mathcal D_t = \bigcup_v \mathcal D_t^v$.

        The pooled batch must be *exactly* the union of the per-agent ones --
        ``environment.pool()`` builds it, and the X0 identity depends on that
        equality.
        """
        self._check_initialised()
        if x.shape[0] == 0:
            return
        shared = self._states[0]
        theta = self.optimizer.step(shared.theta, self._gradient(shared.theta, x, y), shared.extras)
        for node in range(self._n_nodes):
            self._states[node].theta = theta.clone()
            self._states[node].extras = {
                name: tensor.clone() for name, tensor in shared.extras.items()
            }

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        """A no-op. Every agent already holds the same parameters."""


class LocalOnly(_SGDBase):
    """The lower reference: no communication at all.

    The gap between this and the diffusion methods *is* the value of
    cooperation, and it is the cleanest answer to Q2.
    """

    def adapt(self, node: int, observation: Any) -> Intermediate:
        self.state(node).theta = self._local_step(node, observation)
        return Intermediate(node=node, psi=self.state(node).theta)

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        """A no-op. That is the point of this learner."""


class DiffusionSGDATC(_SGDBase):
    r"""Adapt-then-combine: step on local data, then average with neighbours.

    .. math::
        \bm\psi_{v} = \bm\theta_{v} - \eta\nabla L(\bm\theta_{v}; \mathcal D_v),
        \qquad
        \bm\theta_{v} \leftarrow \sum_{u} a_{vu}\bm\psi_{u}

    **Parameters are mixed, not gradients**, which is what drives the agents
    toward agreement. Olshevskyi et al. compare against a variant that runs
    consensus on gradients instead and find it needs roughly twice the
    message-passing rounds for the same error.
    """

    def adapt(self, node: int, observation: Any) -> Intermediate:
        return Intermediate(
            node=node, psi=self._local_step(node, observation), extras=self._payload(node)
        )

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        # A frozen learner stops communicating as well as stepping. Averaging
        # already-identical estimates would be a no-op numerically, but it would
        # still be *counted* by the ledger -- and a baseline that keeps paying
        # bandwidth to change nothing would distort every plot against cost.
        if self.is_frozen():
            return
        _combine_states(self._states, intermediates, weights, self.mixed)

    def comm_scalars_per_step(self, n_edges: int) -> int:
        # Asked once per step, so this reports the warmup cost honestly and
        # drops to zero only once the learner has actually stopped -- rather
        # than crediting the whole run with silence it did not keep.
        if self.is_frozen():
            return 0
        vectors = 1 + len(self.mixed)
        return vectors * self.model.num_params * 2 * n_edges


class DiffusionSGDCTA(_SGDBase):
    r"""Combine-then-adapt: average first, then step at the *pre-combine* point.

    .. math::
        \bm\theta_{v}(t+1) = \sum_{u} \bm W_{vu}\bm\theta_{u}(t)
        \;-\; \alpha_t\,\nabla L(\bm\theta_{v}(t); \mathcal D_v)

    Eq. (17) of Olshevskyi et al., the Nedic--Ozdaglar consensus-plus-local-
    gradient form. The gradient is evaluated **before** the averaging, which is
    the whole difference from ATC.

    Implemented in the same adapt/combine shape as ATC by having ``adapt``
    compute and stash the gradient while emitting the *un-stepped* parameters as
    the message, and ``combine`` apply the stashed gradient after averaging.
    Deferring the update is what makes the two orderings comparable through one
    runner (X1b) rather than through two loops that might differ elsewhere.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending: dict[int, torch.Tensor] = {}

    def adapt(self, node: int, observation: Any) -> Intermediate:
        state = self.state(node)
        self._pending[node] = (
            self._gradient(state.theta, observation.x, observation.y)
            if observation.has_label
            else torch.zeros_like(state.theta)
        )
        # The message is theta itself: under CTA the averaging happens on the
        # pre-gradient parameters.
        return Intermediate(node=node, psi=state.theta.clone(), extras=self._payload(node))

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        _combine_states(self._states, intermediates, weights, self.mixed)
        for node, gradient in self._pending.items():
            state = self._states[node]
            state.theta = self.optimizer.step(state.theta, gradient, state.extras)
        self._pending.clear()

    def comm_scalars_per_step(self, n_edges: int) -> int:
        vectors = 1 + len(self.mixed)
        return vectors * self.model.num_params * 2 * n_edges


# --------------------------------------------------------------------------- #
# the combine step
# --------------------------------------------------------------------------- #


def _combine_states(
    states: dict[int, LearnerState],
    intermediates: dict[int, Intermediate],
    weights: torch.Tensor,
    mixed: tuple[str, ...],
) -> None:
    r"""$\bm\theta_v \leftarrow \sum_u a_{vu}\bm\psi_u$, and the same for mixed state.

    Every new value is computed from the *old* messages before any is written
    back, so an agent combined earlier in the loop cannot feed its updated
    parameters to one combined later. Sequential in-place mixing would make the
    result depend on node ordering and quietly break the X0 identity.
    """
    if set(intermediates) != set(states):
        raise LearnerError(
            f"combine got intermediates for {sorted(intermediates)} but holds state for "
            f"{sorted(states)}"
        )

    order = sorted(states)
    psi = torch.stack([intermediates[node].psi for node in order])
    # Matched to psi's device as well as its dtype: the mixing matrix is built
    # from the graph, which has no reason to know where the parameters live, and
    # at N x N the move costs nothing next to the p-vectors it multiplies.
    mixing = weights.to(device=psi.device, dtype=psi.dtype)
    combined = mixing @ psi

    mixed_stacks = {
        name: mixing @ torch.stack([intermediates[node].extras[name] for node in order])
        for name in mixed
    }

    for index, node in enumerate(order):
        states[node].theta = combined[index]
        for name in mixed:
            states[node].extras[name] = mixed_stacks[name][index]
