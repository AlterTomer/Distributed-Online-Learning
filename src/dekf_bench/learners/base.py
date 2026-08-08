r"""The learner interface: adapt, then combine.

Every method is expressed as an **adapt** step (use local data, no
communication) followed by a **combine** step (use neighbours, the only
communication). That split is not cosmetic: it is what lets phase 5 differ from
diffusion SGD *only in adapt*. A single ``step()`` would not accommodate the
filter, and the interface would have to be rewritten at exactly the point where
a rewrite is most expensive (IMPLEMENTATION.md §13.5).

**Per-agent state is a dict, not a bare tensor.** ``LearnerState`` carries
``theta`` plus whatever else the method needs -- a momentum buffer now, a
covariance in phase 5 -- so the filter can add one without touching
``simulate.py`` (§13.6).

**One gradient function, shared by every learner.** Centralized SGD and ATC
diffusion differ in *which data* they see and *how they combine*, and in nothing
else. If each computed its own gradient, an X0 failure could mean a diffusion
bug or two gradient implementations disagreeing about loss reduction -- and
distinguishing those is exactly the confusion the check exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from dekf_bench.models.base import Model


class LearnerError(ValueError):
    """Raised for a malformed learner configuration or state."""


@dataclass
class LearnerState:
    r"""One agent's state: $\bm\theta$ plus whatever the method carries.

    Mutable, unlike almost everything else here, because the runner updates it
    in place every step and copying a $p$-vector per agent per step would be
    the dominant cost of a run. The tensors it holds are owned by the learner
    and are never shared with the environment or with another agent.
    """

    theta: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key == "theta":
            return self.theta
        if key not in self.extras:
            raise KeyError(f"no state entry {key!r}; have 'theta' and {sorted(self.extras)}")
        return self.extras[key]

    def __contains__(self, key: str) -> bool:
        return key == "theta" or key in self.extras

    def keys(self) -> list[str]:
        return ["theta", *sorted(self.extras)]

    def clone(self) -> LearnerState:
        return LearnerState(
            theta=self.theta.clone(),
            extras={name: tensor.clone() for name, tensor in self.extras.items()},
        )


@dataclass(frozen=True)
class Intermediate:
    r"""What ``adapt`` produces and ``combine`` consumes.

    The $\bm\psi_{v,t}$ of the ATC recursion, plus any optimizer state that
    travels with it. Frozen: an intermediate is a message, and a message that
    can be edited after it is sent is not a message.
    """

    node: int
    psi: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def payload_vectors(self) -> int:
        """How many $p$-vectors this crosses a link as. Feeds the ledger."""
        return 1 + len(self.extras)


@runtime_checkable
class Learner(Protocol):
    """What ``simulate.py`` may assume about any method."""

    @property
    def name(self) -> str: ...

    @property
    def n_nodes(self) -> int: ...

    def init(self, theta0: torch.Tensor) -> None:
        r"""Seed every agent with the *same* $\bm\theta_0$.

        Shared initialisation is required for the filter -- independently
        initialised agents do not represent one Bayesian model -- and removes a
        confound from the SGD comparison (WORKPLAN §4.5).
        """

    def adapt(self, node: int, observation: Any) -> Intermediate:
        """Local computation. No communication."""

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        """The only communication."""

    def predict(self, node: int, x: torch.Tensor) -> torch.Tensor:
        """Logits at agent ``node``'s current parameters."""

    def state(self, node: int) -> LearnerState: ...

    def flat_params(self, node: int) -> torch.Tensor:
        r"""$\bm\theta^v$ as a flat vector, for $E_{\text{agree}}$ and $E_{\text{cent}}$."""

    def comm_scalars_per_step(self, n_edges: int) -> int:
        """What this learner transmits per step, for the ledger."""


# --------------------------------------------------------------------------- #
# the shared gradient
# --------------------------------------------------------------------------- #


def loss_gradient(
    model: Model,
    theta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    likelihood: Any,
) -> torch.Tensor:
    r"""$\nabla L(\bm\theta; \mathcal D)$ with **mean** reduction, as a flat vector.

    Computed as $-\bm H^{\mathsf T}\bm\nu / n$ rather than through autograd on
    the loss. The two are equal -- ``test_likelihoods.py`` asserts it -- and this
    route reuses the ``vjp`` the filter will need, so phase 5 shares the code
    path rather than acquiring a second one.

    **Mean, not sum, and this is a precondition of X0.** The identity

    .. math::
        \sum_v \tfrac1N\bigl(\bm\theta - \eta\nabla L_v\bigr)
        = \bm\theta - \eta\,\tfrac1N\sum_v \nabla L_v

    equates the average of per-agent *means* with the pooled mean, which holds
    only when every agent has the same batch size and every gradient is reduced
    the same way. A sum here and a mean there produces a residual of order
    $N$ that looks like a scaling bug rather than a reduction bug.
    """
    if x.shape[0] == 0:
        return torch.zeros_like(theta)

    params = model.unflatten(theta)
    logits = model.forward(params, x)
    innovation = likelihood.innovation(logits, y)
    # vjp with the innovation gives -grad; divide by n for the mean reduction.
    return -model.vjp(params, x, innovation) / x.shape[0]
