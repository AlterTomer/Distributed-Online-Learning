"""The Gaussian likelihood, for regression.

$\\bm y = \\bm h(\\bm\\theta) + \\bm\\varepsilon$, $\\bm\\varepsilon \\sim
\\mathcal N(\\bm 0, \\bm R)$, so $\\bm\\mu = \\bm h$ and $\\bm\\Lambda = \\bm
R^{-1}$.

**No experiment in X0--X6 uses this**, and it is here anyway for one reason: it
is the only likelihood in the project with a **non-singular** $\\bm\\Lambda$, so
it is the only one that can exercise the filter's gain form. Without it that code
path would sit untested until a regression experiment appeared -- and untested
code the filter will one day depend on is exactly the kind that turns out to be
wrong. It also keeps the abstraction honest: an interface with one implementation
is shaped around that implementation whether or not anyone intended it.

The research note's E2 (distributed link-quality prediction) is the experiment
this would serve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dekf_bench.likelihoods.base import LikelihoodError, Reduction, reduce


@dataclass(frozen=True)
class Gaussian:
    """Gaussian observations with fixed, diagonal noise covariance.

    Attributes:
        output_dim: $q$.
        variance: the diagonal of $\\bm R$ when isotropic. Plays the role a
            learning rate plays in gradient training: it says how much to trust
            one sample.
        variances: a per-output diagonal, overriding ``variance`` when given. On
            the series task output $i$ predicts from $i$ samples of context, so the
            residual variance genuinely differs by position (Mackey--Glass plan,
            decision 14). ``None`` keeps the isotropic path exactly as it was.
    """

    output_dim: int = 1
    variance: float = 1.0
    variances: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.output_dim < 1:
            raise LikelihoodError(f"output_dim must be >= 1, got {self.output_dim}")
        if self.variance <= 0.0:
            raise LikelihoodError(f"variance must be > 0, got {self.variance}")
        if self.variances is not None:
            if len(self.variances) != self.output_dim:
                raise LikelihoodError(
                    f"variances has {len(self.variances)} entries for {self.output_dim} outputs"
                )
            if any(v <= 0.0 for v in self.variances):
                raise LikelihoodError(f"variances must all be > 0, got {self.variances}")

    def _diagonal(self, logits: torch.Tensor) -> torch.Tensor:
        """The per-output diagonal of R, on the logits' device and dtype."""
        return torch.tensor(self.variances, dtype=logits.dtype, device=logits.device)

    def _diagonal_matrix(self, logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(values).expand(
            *logits.shape[:-1], self.output_dim, self.output_dim
        )

    @property
    def fisher_rank(self) -> int:
        """Full rank -- which is what makes the gain form usable here."""
        return self.output_dim

    @property
    def is_regression(self) -> bool:
        """Scored by squared error and interval coverage rather than by argmax.

        Read by `evaluation/protocol.py` to pick its scoring branch. Explicit
        rather than inferred from a missing ``predictions`` method, so a future
        likelihood cannot land in the wrong branch by omission.
        """
        return True

    def mu(self, logits: torch.Tensor) -> torch.Tensor:
        """The identity link: the network output *is* the mean."""
        self._check(logits)
        return logits

    def fisher(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\bm R^{-1} = \\sigma^{-2}\\bm I$, broadcast over the batch."""
        self._check(logits)
        if self.variances is not None:
            return self._diagonal_matrix(logits, 1.0 / self._diagonal(logits))
        eye = torch.eye(self.output_dim, dtype=logits.dtype, device=logits.device)
        return eye.expand(*logits.shape[:-1], self.output_dim, self.output_dim) / self.variance

    def fisher_factor(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\bm G = \\sigma^{-1}\\bm I$. Trivially exact, and full rank."""
        self._check(logits)
        if self.variances is not None:
            return self._diagonal_matrix(logits, self._diagonal(logits).rsqrt())
        eye = torch.eye(self.output_dim, dtype=logits.dtype, device=logits.device)
        return eye.expand(*logits.shape[:-1], self.output_dim, self.output_dim) / math.sqrt(
            self.variance
        )

    def noise_covariance(self, logits: torch.Tensor) -> torch.Tensor:
        """$\\bm R$ itself. Defined only because $\\bm\\Lambda$ is invertible here;
        the categorical likelihood deliberately has no counterpart."""
        self._check(logits)
        if self.variances is not None:
            return self._diagonal_matrix(logits, self._diagonal(logits))
        eye = torch.eye(self.output_dim, dtype=logits.dtype, device=logits.device)
        return eye.expand(*logits.shape[:-1], self.output_dim, self.output_dim) * self.variance

    def innovation(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._check_targets(logits, targets)
        return targets - self.mu(logits)

    def score(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """$\\partial\\log p/\\partial\\bm h = \\bm R^{-1}(\\bm y-\\bm h) = \\bm\\nu/\\sigma^2$.

        **Not** the innovation. The identity link makes $\\bm h$ the mean rather
        than the natural parameter, so differentiating the log-density leaves the
        $\\sigma^{-2}$ behind. Dropping it is invisible to gradient training,
        which absorbs it into the step size, and wrong by exactly that factor in
        the filter's mean update (design note D60).
        """
        if self.variances is not None:
            return self.innovation(logits, targets) / self._diagonal(logits)
        return self.innovation(logits, targets) / self.variance

    def nll(
        self, logits: torch.Tensor, targets: torch.Tensor, reduction: Reduction
    ) -> torch.Tensor:
        """$\\tfrac12\\lVert\\bm y-\\bm h\\rVert^2/\\sigma^2$ plus the normaliser.

        The constant is kept rather than dropped: a negative log-likelihood that
        omits it is not comparable across noise levels, and calibration
        reporting compares exactly that.
        """
        self._check_targets(logits, targets)
        residual = targets - logits
        if self.variances is not None:
            diagonal = self._diagonal(logits)
            constant = 0.5 * torch.log(2.0 * math.pi * diagonal).sum()
            per_sample = 0.5 * (residual**2 / diagonal).sum(dim=-1) + constant
            return reduce(per_sample, reduction)
        constant = 0.5 * self.output_dim * math.log(2.0 * math.pi * self.variance)
        per_sample = 0.5 * (residual**2).sum(dim=-1) / self.variance + constant
        return reduce(per_sample, reduction)

    def _check(self, logits: torch.Tensor) -> None:
        if logits.shape[-1] != self.output_dim:
            raise LikelihoodError(f"expected {self.output_dim} outputs, got {logits.shape[-1]}")

    def _check_targets(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self._check(logits)
        if targets.shape != logits.shape:
            raise LikelihoodError(
                f"targets {tuple(targets.shape)} must match logits {tuple(logits.shape)}"
            )
