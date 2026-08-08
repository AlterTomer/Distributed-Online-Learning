"""The model interface.

Every model here is **functional**: the parameters are an argument, not state
owned by an object. ``model.forward(params, x)`` evaluates the network at
whatever $\\bm\\theta$ it is handed.

That is not a stylistic preference. Phase 1 needs it because ten agents hold ten
different parameter vectors and share one architecture, so a module carrying its
own weights would have to be cloned per agent. Phase 5 needs it more sharply: the
filter evaluates the network at the *predictive mean* $\\bm m_{t|t-1}$, and
computes Jacobians there, without ever mutating a module
(IMPLEMENTATION.md §13.2).

Phase-1 SGD uses only ``init_params``, ``forward`` and ``num_params``. The rest
exists now because retrofitting it later means touching every model
(IMPLEMENTATION.md §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

#: A model's parameters, keyed by name. The keys and their order are fixed for a
#: given model, which is what makes flatten/unflatten well defined.
ParamDict = dict[str, torch.Tensor]


class ModelError(ValueError):
    """Raised for a malformed model or a parameter vector that does not fit."""


@dataclass(frozen=True)
class ParamGroup:
    """One block of the flat parameter vector.

    Blocks are layers, not individual tensors: a layer's weight and bias belong
    to the same curvature block, which is what a block-diagonal or
    Kronecker-factored covariance is factored over (research note §6.2). Keeping
    ``fan_in``/``fan_out`` here means phase 5 does not have to re-derive the
    Kronecker shapes from the flat indices.

    Attributes:
        name: the layer's name, e.g. ``"layers.0"``.
        start, stop: the half-open span in the flat vector.
        fan_in, fan_out: the layer's input and output widths.
    """

    name: str
    start: int
    stop: int
    fan_in: int
    fan_out: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@runtime_checkable
class Model(Protocol):
    """What a learner may assume about a model."""

    @property
    def num_params(self) -> int:
        """$p$ -- the length of the flat parameter vector."""

    @property
    def output_dim(self) -> int:
        """$q$ -- the number of logits. Sizes the measurement covariance."""

    @property
    def input_dim(self) -> int:
        """Flattened input width."""

    def init_params(self, generator: torch.Generator | None = None) -> ParamDict:
        """A fresh parameter dict.

        Deterministic given the generator, so every agent can be handed the
        *same* $\\bm\\theta_0$ -- required because Diff-EKF agents must share a
        prior, and because it removes a confound from the SGD comparison
        (WORKPLAN.md §4.5).
        """

    def forward(self, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
        """Logits at the given parameters. No state is read or written."""

    def flatten(self, params: ParamDict) -> torch.Tensor:
        """The parameters as one $p$-vector."""

    def unflatten(self, vector: torch.Tensor) -> ParamDict:
        """The inverse of :meth:`flatten`."""

    def vjp(self, params: ParamDict, x: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        r"""$\\bm u^{\\mathsf T}\\bm H$, returned flat.

        Declared on the protocol rather than only on the concrete model because
        the *shared gradient* uses it -- every learner's update runs through
        $-\\bm H^{\\mathsf T}\\bm\\nu$ -- and phase 5 needs it for the information
        increment. It is not an implementation detail of one model.
        """

    def jvp(self, params: ParamDict, x: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        r"""$\\bm H\\bm v$ for a flat direction in parameter space."""

    def param_groups(self) -> tuple[ParamGroup, ...]:
        """Layer blocks of the flat vector, in order."""
