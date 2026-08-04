"""The softmax likelihood, and its singular Fisher.

For $K$-class classification the logits are the natural parameter,
$\\bm\\mu = \\operatorname{softmax}(\\bm h) = \\bm\\pi$, the label is one-hot, and

$$\\bm\\Lambda = \\operatorname{diag}(\\bm\\pi) - \\bm\\pi\\bm\\pi^{\\mathsf T}
  \\succeq \\bm 0, \\qquad \\bm\\Lambda\\bm 1 = \\bm 0 .$$

**The singularity is correct and is kept.** Softmax is shift-invariant,
$\\operatorname{softmax}(\\bm h + c\\bm 1) = \\operatorname{softmax}(\\bm h)$, so the
$\\bm 1$ direction in logit space is genuinely unidentifiable and $\\bm\\Lambda$
reports exactly zero information there. The innovation lives in the same
subspace: $\\bm 1^{\\mathsf T}\\bm\\nu = 1 - 1 = 0$.

Three ways to deal with that were considered (design note D28). This module
implements the first: **use the information form and never invert anything.**
The gain form needs $\\bm R = \\bm\\Lambda^{-1}$, which does not exist; the
alternatives were a reduced $(q{-}1)$ output head, which breaks class symmetry
and changes the SGD baselines, and a ridge $\\bm\\Lambda + \\varepsilon\\bm I$,
which injects curvature along $\\bm 1$ and makes the filter confident about a
quantity it can never learn.

**The factorisation is exact, not numerical.** With $\\bm s = \\sqrt{\\bm\\pi}$
elementwise, $\\lVert\\bm s\\rVert^2 = \\sum_i \\pi_i = 1$, so
$\\bm I - \\bm s\\bm s^{\\mathsf T}$ is an orthogonal projector and therefore its
own square root. Since $\\operatorname{diag}(\\bm s)\\bm s = \\bm\\pi$,

$$\\operatorname{diag}(\\bm s)\\,(\\bm I - \\bm s\\bm s^{\\mathsf T})\\,
  \\operatorname{diag}(\\bm s)
  = \\operatorname{diag}(\\bm\\pi) - \\bm\\pi\\bm\\pi^{\\mathsf T} = \\bm\\Lambda,$$

so $\\bm G = \\operatorname{diag}(\\sqrt{\\bm\\pi})(\\bm I -
\\sqrt{\\bm\\pi}\\sqrt{\\bm\\pi}^{\\mathsf T})$ gives $\\bm\\Lambda = \\bm G\\bm
G^{\\mathsf T}$ with no eigendecomposition, no tolerance, and no failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as TF

from dekf_bench.likelihoods.base import LikelihoodError, Reduction, reduce


@dataclass(frozen=True)
class Categorical:
    """Softmax over ``output_dim`` classes.

    Targets are **class indices**, not one-hot vectors: that is what the dataset
    supplies and what ``cross_entropy`` wants, and it avoids materialising a
    one-hot matrix per batch. :meth:`innovation` builds the one-hot internally,
    where it is needed.
    """

    output_dim: int = 10

    def __post_init__(self) -> None:
        if self.output_dim < 2:
            raise LikelihoodError(f"output_dim must be >= 2, got {self.output_dim}")

    @property
    def fisher_rank(self) -> int:
        """$q-1$: softmax cannot identify a shift common to every logit."""
        return self.output_dim - 1

    # -- moments ----------------------------------------------------------- #

    def mu(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\bm\\pi = \\operatorname{softmax}(\\bm h)$."""
        self._check_logits(logits)
        return torch.softmax(logits, dim=-1)

    def fisher(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\operatorname{diag}(\\bm\\pi) - \\bm\\pi\\bm\\pi^{\\mathsf T}$, shape ``(..., q, q)``."""
        pi = self.mu(logits)
        return torch.diag_embed(pi) - pi.unsqueeze(-1) * pi.unsqueeze(-2)

    def fisher_factor(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\bm G$ with $\\bm\\Lambda = \\bm G\\bm G^{\\mathsf T}$, shape ``(..., q, q)``.

        Returned at full width $q$ rather than trimmed to $q-1$ columns: the
        rank deficiency lives in the column space, and any $q-1$ columns that
        span it would be an arbitrary choice of basis. Keeping $\\bm G$ square
        and singular means $\\bm B = \\bm H^{\\mathsf T}\\bm G$ has one redundant
        column, which Woodbury handles without complaint, and no basis is
        invented.
        """
        pi = self.mu(logits)
        root = pi.clamp_min(0.0).sqrt()
        outer = root.unsqueeze(-1) * root.unsqueeze(-2)
        projector = torch.diag_embed(torch.ones_like(root)) - outer
        return root.unsqueeze(-1) * projector

    # -- data -------------------------------------------------------------- #

    def innovation(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """$\\bm\\nu = \\bm e_c - \\bm\\pi$.

        Zero innovation means the sample was completely predicted and nothing is
        learned from it -- the only channel through which data enter the filter.
        """
        self._check_targets(logits, targets)
        one_hot = TF.one_hot(targets, num_classes=self.output_dim).to(logits.dtype)
        return one_hot - self.mu(logits)

    def nll(
        self, logits: torch.Tensor, targets: torch.Tensor, reduction: Reduction
    ) -> torch.Tensor:
        """Cross-entropy. ``reduction`` is required, never defaulted."""
        self._check_targets(logits, targets)
        per_sample = TF.cross_entropy(logits, targets, reduction="none")
        return reduce(per_sample, reduction)

    def predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Arg-max class, for accuracy."""
        self._check_logits(logits)
        return logits.argmax(dim=-1)

    # -- validation --------------------------------------------------------- #

    def _check_logits(self, logits: torch.Tensor) -> None:
        if logits.shape[-1] != self.output_dim:
            raise LikelihoodError(f"expected {self.output_dim} logits, got {logits.shape[-1]}")

    def _check_targets(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self._check_logits(logits)
        if targets.dtype != torch.int64:
            raise LikelihoodError(
                f"targets must be int64 class indices, got {targets.dtype}. "
                "One-hot targets are built internally where they are needed."
            )
        if targets.shape != logits.shape[:-1]:
            raise LikelihoodError(
                f"targets {tuple(targets.shape)} do not match logits " f"{tuple(logits.shape[:-1])}"
            )
        if targets.numel() and (int(targets.min()) < 0 or int(targets.max()) >= self.output_dim):
            raise LikelihoodError(
                f"targets must lie in [0, {self.output_dim - 1}], got "
                f"[{int(targets.min())}, {int(targets.max())}]"
            )
