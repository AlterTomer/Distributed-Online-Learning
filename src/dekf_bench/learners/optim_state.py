r"""How optimizer moments are updated, and whether the combine step mixes them.

Defined in exactly one place because every diffusion learner uses the same
policy, and a mixing rule that differs between ATC and CTA would confound the
X1b comparison with an implementation difference.

**Mixing is not optional bookkeeping.** Olshevskyi et al. (Fig. 2a) compare two
distributed adaptive methods: **D-Adam**, which mixes parameters but keeps the
moments local, *converges quickly and then diverges* -- the local velocities
drift apart until the agents are pulling in different directions. **D-AMSGrad**,
which runs consensus on the moments too, is their best distributed method. So
``mix_optimizer_state: none`` with a stateful optimizer is a known failure mode
rather than an open question, and the config rejects it.

**What "mix" means here.** The combine step treats the whole learner state as
one object the weights act on:

.. math::
    \begin{pmatrix}\bm\theta_v \\ \bm m_v\end{pmatrix}
    \leftarrow \sum_u a_{vu}\begin{pmatrix}\bm\psi_u \\ \bm m_u\end{pmatrix}

Every property established for $\bm A$ then covers the whole state rather than
$\bm\theta$ alone: row-stochasticity keeps the result inside the convex hull of
what the neighbours proposed, and double stochasticity preserves the network
average. Under the unmixed alternative $\bm\theta$ gets those guarantees and
$\bm m$ -- the part that actually diverges -- gets none.

Mixing the moment means **transmitting** it: a neighbour cannot average a buffer
it was never sent. That is why ``momentum`` costs $2p$ per link where plain SGD
costs $p$, and why the ledger charges it (design note D29).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

#: Which state entries each optimizer carries beyond theta.
OPTIMIZER_STATE = {
    "sgd": (),
    "sgd_momentum": ("momentum",),
    "adamw": ("momentum", "second_moment"),
}

#: Which of those entries the combine step mixes, per policy.
MIX_POLICIES = {
    "none": (),
    "momentum": ("momentum",),
    "all": ("momentum", "second_moment"),
}


class OptimStateError(ValueError):
    """Raised for an unknown optimizer or mixing policy."""


@dataclass(frozen=True)
class Optimizer:
    r"""A first-order update rule, applied to a flat $\bm\theta$.

    Implemented directly rather than via ``torch.optim`` for two reasons. Torch
    optimizers own their state internally and expose it only as
    ``optimizer.state[param]['momentum_buffer']``, which makes mixing across
    agents awkward. And the phase-5 filter has no torch optimizer at all, so
    routing phase 3 through one would leave the learner interface differing
    between phases at exactly the boundary that must not move.
    """

    kind: str = "sgd"
    lr: float = 0.05
    momentum: float = 0.0
    beta2: float = 0.999
    weight_decay: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.kind not in OPTIMIZER_STATE:
            raise OptimStateError(
                f"unknown optimizer {self.kind!r}; have {sorted(OPTIMIZER_STATE)}"
            )
        if self.lr <= 0:
            raise OptimStateError(f"lr must be > 0, got {self.lr}")
        if not 0.0 <= self.momentum < 1.0:
            raise OptimStateError(f"momentum must lie in [0, 1), got {self.momentum}")
        if self.kind == "sgd" and self.momentum != 0.0:
            raise OptimStateError(
                f"optimizer 'sgd' carries no momentum buffer, but momentum={self.momentum}. "
                "Use 'sgd_momentum', or set momentum to 0 -- a value that silently does "
                "nothing is worse than an error, and X0 depends on plain SGD being plain."
            )

    @property
    def state_names(self) -> tuple[str, ...]:
        return OPTIMIZER_STATE[self.kind]

    @property
    def is_stateful(self) -> bool:
        return bool(self.state_names)

    def init_state(
        self,
        num_params: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Zeroed buffers, one per state entry this optimizer carries.

        ``device`` defaults to the CPU rather than being required, so the SGD
        learners that predate phase 5 keep working unchanged; callers that hold
        parameters elsewhere pass ``theta.device``.
        """
        return {
            name: torch.zeros(num_params, dtype=dtype, device=device)
            for name in self.state_names
        }

    def step(
        self, theta: torch.Tensor, gradient: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        r"""One update, returning the new $\bm\theta$ and mutating ``state``.

        Returns rather than updating in place, because the ATC adapt step needs
        $\bm\psi$ as a *separate* object from $\bm\theta$ -- the agent's own
        parameters must survive until the combine step replaces them.
        """
        if self.weight_decay:
            gradient = gradient + self.weight_decay * theta

        if self.kind == "sgd":
            return theta - self.lr * gradient

        if self.kind == "sgd_momentum":
            # Heavy-ball: m <- beta*m + g, then theta <- theta - lr*m. Torch's
            # convention, so a run here matches one written with torch.optim.
            state["momentum"].mul_(self.momentum).add_(gradient)
            return theta - self.lr * state["momentum"]

        if self.kind == "adamw":
            state["momentum"].mul_(self.momentum).add_(gradient, alpha=1 - self.momentum)
            state["second_moment"].mul_(self.beta2).addcmul_(
                gradient, gradient, value=1 - self.beta2
            )
            # No bias correction: the step count would be per-agent state that
            # the combine step has no sensible way to average, and mixing
            # corrected moments with uncorrected ones is worse than neither.
            denominator = state["second_moment"].sqrt().add_(self.eps)
            return theta - self.lr * state["momentum"] / denominator

        raise OptimStateError(f"unhandled optimizer {self.kind!r}")  # pragma: no cover


def mixed_entries(optimizer: Optimizer, policy: str) -> tuple[str, ...]:
    """Which state entries travel with $\\bm\\psi$ and get averaged.

    The intersection of what the optimizer carries and what the policy mixes,
    so ``mix_optimizer_state: all`` on plain SGD is a no-op rather than an
    error -- there is simply nothing to mix.
    """
    if policy not in MIX_POLICIES:
        raise OptimStateError(f"unknown mix policy {policy!r}; have {sorted(MIX_POLICIES)}")
    wanted = set(MIX_POLICIES[policy])
    return tuple(name for name in optimizer.state_names if name in wanted)


def build_optimizer(config: Any) -> Optimizer:
    """The optimizer a learner's config asks for."""
    return Optimizer(
        kind=config.optimizer,
        lr=config.lr,
        momentum=config.momentum if config.optimizer != "sgd" else 0.0,
    )
