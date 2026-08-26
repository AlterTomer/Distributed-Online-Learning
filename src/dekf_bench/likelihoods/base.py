"""The likelihood interface.

A likelihood turns the network's logits $\\bm h$ into the four quantities the
filter consumes, and **nothing else**. It never sees a parameter, a Jacobian or a
covariance:

===============  ==========================================================
``mu``           the mean parameter $\\bm\\mu = \\nabla A(\\bm h)$
``fisher``       $\\bm\\Lambda = \\nabla^2 A(\\bm h) \\succeq \\bm 0$
``innovation``   $\\bm\\nu = \\bm y - \\bm\\mu$
``score``        $\\partial\\log p/\\partial\\bm h$ -- what the update multiplies
``nll``          the negative log-likelihood, for training and calibration
===============  ==========================================================

**``innovation`` and ``score`` are not the same quantity**, and conflating them
is a bug the categorical likelihood cannot expose. Under softmax the logits *are*
the natural parameter, so $\\partial\\log p/\\partial\\bm h = \\bm y-\\bm\\pi =
\\bm\\nu$ and the two coincide. Under a Gaussian the logits are the *mean*
parameter, and the score picks up the noise covariance:
$\\partial\\log p/\\partial\\bm h = \\bm R^{-1}(\\bm y-\\bm h) = \\bm\\Lambda\\bm\\nu$.
Gradient training never notices -- $\\sigma^{-2}$ is absorbed by the learning rate
-- but the filter does, because $\\bm P$ carries absolute units and has nothing
to absorb it into. The mean update takes the **score** (design note D60).

The two are tied together by an identity that holds in both families and is
worth testing: $\\bm\\Lambda = \\operatorname{Cov}(\\text{score})$.

That split is what lets classification, regression and next-token prediction run
through one update (research note §3.3).

**The factor is part of the interface, not an implementation detail.** The filter
never wants $\\bm\\Lambda$ itself; it wants $\\bm H^{\\mathsf T}\\bm\\Lambda\\bm H$.
Every likelihood therefore also supplies $\\bm G$ with
$\\bm\\Lambda = \\bm G\\bm G^{\\mathsf T}$ and $\\bm G \\in \\mathbb R^{q\\times q'}$,
$q' = \\operatorname{rank}\\bm\\Lambda$, so the information increment is

$$\\Delta\\bm\\Omega = \\bm B\\bm B^{\\mathsf T}, \\qquad
  \\bm B = \\bm H^{\\mathsf T}\\bm G \\in \\mathbb R^{p\\times q'},$$

always **low rank**. That is what lets Woodbury replace a $p\\times p$ inverse
with a $q'\\times q'$ one, and it is why $q'$ is exposed rather than left implicit.

**Reduction is explicit everywhere.** ``nll`` takes a ``reduction`` argument with
no default that silently averages. Mean-versus-sum is one of the four
preconditions of the X0 exactness identity, and a loss that quietly reduces one
way while the pooled learner reduces the other produces a residual that looks
like a numerical problem rather than a bug.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import torch

Reduction = Literal["none", "mean", "sum"]


class LikelihoodError(ValueError):
    """Raised for a malformed target, logit or reduction."""


def reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """Apply a reduction, rejecting anything unrecognised."""
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise LikelihoodError(f"unknown reduction {reduction!r}; expected 'none', 'mean' or 'sum'")


@runtime_checkable
class Likelihood(Protocol):
    """What the filter and the metrics may assume about an observation model."""

    @property
    def output_dim(self) -> int:
        """$q$ -- the width of the logit vector."""

    @property
    def fisher_rank(self) -> int:
        """$q'$ -- the rank of $\\bm\\Lambda$, and the rank of each update."""

    def mu(self, logits: torch.Tensor) -> torch.Tensor:
        """The mean parameter."""

    def fisher(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\bm\\Lambda$, shape ``(..., q, q)``."""

    def fisher_factor(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\bm G$ with $\\bm\\Lambda = \\bm G\\bm G^{\\mathsf T}$, shape ``(..., q, q')``."""

    def innovation(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """$\\bm\\nu = \\bm y - \\bm\\mu$ -- the only channel data enters the filter by."""

    def score(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """$\\partial\\log p/\\partial\\bm h$. Equals $\\bm\\nu$ only in canonical form."""

    def nll(
        self, logits: torch.Tensor, targets: torch.Tensor, reduction: Reduction
    ) -> torch.Tensor:
        """Negative log-likelihood."""
