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


#: Default for `trust_region_ratio`: how far the mean may travel from
#: $\bm\theta_0$, as a multiple of $\lVert\bm\theta_0\rVert$, before the belief is
#: declared diverged. Overridable per learner in the config, because the value is
#: **chosen rather than derived** and a different model may need a different one.
#:
#: Measured on 2026-09-05 at the tuned setting: a healthy filter reaches a ratio
#: of about 1.3, and reaches it whether the data are stationary, drifting at
#: 0.025 deg/step, or shifting every 2 steps. The diverged X15 cell reached
#: 5.6e130.
#:
#: ⚠ **Tightened from 1e6 to 50 on 2026-09-13, because 1e6 did not do its job.**
#: X22's alpha=1 cell finished at chance error (0.8941 against a 0.9 floor) with
#: the mean 323x its initial norm, and was recorded as a clean run -- exactly the
#: failure D61 introduced this guard for, waved through because the threshold only
#: ever caught overflow-scale blowup, which finiteness already catches. The
#: separation is not close: every healthy run measured sits between 1.3x and 5.5x,
#: the centralised filter at 1.5x, so 50 is an order of magnitude above the worst
#: legitimate belief and six times below the worst destroyed one.
TRUST_REGION_RATIO = 50.0


def information_pair(
    model: Model, likelihood: Any, mean: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The pair $(\bar{\bm B},\ \bm H^{\mathsf T}\bm s)$ at one linearisation point.

    $\bar{\bm B}$ has shape $(p, nq)$ and satisfies $\bar{\bm B}\bar{\bm B}^{\mathsf T}
    =\sum_i\bm\Delta_i$, the information the batch contributes; the second return
    is the summed score.

    **Module level, and shared by both filters.** The centralised filter forms
    this once on the pooled batch; the diffusion filter forms it per agent, and
    under a one-hop adapt step concatenates several of them. That the three cases
    are one function is what makes `prop:complete_graph` testable as an
    *identity* rather than as an approximate agreement --- the same argument
    `base.py` makes for having exactly one gradient function.
    """
    params = model.unflatten(mean)
    logits = model.forward(params, x)
    jacobians = model.per_sample_jacobian(params, x)  # (n, q, p)
    factor = likelihood.fisher_factor(logits)  # (n, q, q)

    batch, outputs, _ = jacobians.shape
    # B_i = H_i^T G_i, stacked along the column axis into (p, n*q).
    stacked = torch.einsum("nqp,nqr->pnr", jacobians, factor).reshape(-1, batch * outputs)
    # The score, not the innovation: they differ off the canonical link, and the
    # factor between them is not one the covariance can absorb (D60).
    score = torch.einsum("nqp,nq->p", jacobians, likelihood.score(logits, y))
    return stacked, score


# --------------------------------------------------------------------------- #
# the block-diagonal covariance (D3, D139)
# --------------------------------------------------------------------------- #

#: `covariance_blocks`: "full" is the dense p x p of every run before D3; the rest are
#: block-diagonal, grouped by neuron (the decoupled EKF of Puskorius & Feldkamp 1991),
#: by layer (a module: weight and bias together) or by tensor.
COVARIANCE_BLOCKS = ("full", "neuron", "layer", "tensor")


def parameter_blocks(model: Model, structure: str) -> list[torch.Tensor]:
    r"""The flat indices of every block, in parameter order.

    ``neuron``: a 2-D ``<m>.weight`` of shape (out, in) gives ``out`` blocks, row $i$
    with ``<m>.bias[i]`` when the bias exists -- one output unit's incoming weights and
    its bias. Every other tensor (a LayerNorm's gain or bias, a lone vector) is one block
    (decided 2026-10-07). ``layer``: one block per module, weight and bias together.
    ``tensor``: one block per tensor. Every index appears in exactly one block.
    """
    if structure not in COVARIANCE_BLOCKS[1:]:
        raise FilterError(f"block structure must be one of {COVARIANCE_BLOCKS[1:]}, got {structure!r}")
    offsets, start = {}, 0
    for name, shape in zip(model.names, model.shapes, strict=True):
        size = int(torch.Size(shape).numel())
        offsets[name] = (start, tuple(shape))
        start += size
    blocks: list[torch.Tensor] = []
    if structure == "tensor":
        return [torch.arange(o, o + int(torch.Size(s).numel())) for o, s in offsets.values()]
    if structure == "layer":
        modules: dict[str, list[torch.Tensor]] = {}
        for name, (o, s) in offsets.items():
            modules.setdefault(name.rsplit(".", 1)[0], []).append(
                torch.arange(o, o + int(torch.Size(s).numel())))
        return [torch.cat(parts) for parts in modules.values()]
    used: set[str] = set()
    for name, (o, s) in offsets.items():
        if name in used:
            continue
        module, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        bias = f"{module}.bias" if module else "bias"
        if leaf == "weight" and len(s) == 2:
            out, fan_in = s
            rows = o + torch.arange(out)[:, None] * fan_in + torch.arange(fan_in)[None, :]
            if bias in offsets and offsets[bias][1] == (out,):
                rows = torch.cat([rows, (offsets[bias][0] + torch.arange(out))[:, None]], dim=1)
                used.add(bias)
            blocks.extend(rows)
        else:
            blocks.append(torch.arange(o, o + int(torch.Size(s).numel())))
        used.add(name)
    return blocks


class BlockCovariance:
    r"""$\bm P=\operatorname{blockdiag}(\bm P_1,\dots,\bm P_B)$, the blocks stacked by size.

    Stands in for the dense $\bm P$ wherever a filter holds one: it knows the products
    the filters take ($\bm P\bm M$, $\bm J\bm P\bm J^{\mathsf T}$, the diagonal), the
    prediction's inflation, and its own Woodbury update. Memory is $\sum_b s_b^2$ rather
    than $p^2$, which is what lets a larger model carry a covariance at all.

    **Closed under both combines.** A convex combination of block-diagonal matrices,
    weighted by $a$ or by $a^\beta$, is block-diagonal with the same blocks, so the
    diffusion step is exact; only the measurement update is approximate (see
    :meth:`woodbury`).
    """

    def __init__(self, groups: list[tuple[torch.Tensor, torch.Tensor]], p: int) -> None:
        #: (index (count, size) long, blocks (count, size, size)), ascending size.
        self.groups = groups
        self.p = p

    @classmethod
    def identity(cls, blocks: list[torch.Tensor], scale: float, dtype: torch.dtype,
                 device: torch.device | str) -> BlockCovariance:
        p = int(sum(b.numel() for b in blocks))
        covered = torch.cat(blocks).sort().values
        if not torch.equal(covered, torch.arange(p)):
            raise FilterError("the blocks must cover every parameter exactly once")
        by_size: dict[int, list[torch.Tensor]] = {}
        for index in blocks:
            by_size.setdefault(int(index.numel()), []).append(index)
        groups = []
        for size in sorted(by_size):
            index = torch.stack(by_size[size]).to(device)
            eye = torch.eye(size, dtype=dtype, device=device)
            groups.append((index, (eye * scale).expand(index.shape[0], size, size).clone()))
        return cls(groups, p)

    # -- shape and copies ------------------------------------------------------ #

    @property
    def dtype(self) -> torch.dtype:
        return self.groups[0][1].dtype

    @property
    def device(self) -> torch.device:
        return self.groups[0][1].device

    @property
    def sizes(self) -> list[tuple[int, int]]:
        """(count, size) per group."""
        return [(int(i.shape[0]), int(i.shape[1])) for i, _b in self.groups]

    @property
    def upper_entries(self) -> int:
        """Scalars in the upper triangles: what full sharing sends per message."""
        return sum(c * s * (s + 1) // 2 for c, s in self.sizes)

    def _like(self, blocks: list[torch.Tensor]) -> BlockCovariance:
        return BlockCovariance([(i, b) for (i, _o), b in zip(self.groups, blocks, strict=True)],
                               self.p)

    def clone(self) -> BlockCovariance:
        return self._like([b.clone() for _i, b in self.groups])

    def to(self, *args: Any, **kwargs: Any) -> BlockCovariance:
        """Blocks to a dtype or device; the integer indices follow the device only."""
        groups = []
        for index, blocks in self.groups:
            moved = blocks.to(*args, **kwargs)
            groups.append((index.to(moved.device), moved))
        return BlockCovariance(groups, self.p)

    def __mul__(self, scalar: float) -> BlockCovariance:
        return self._like([b * scalar for _i, b in self.groups])

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> BlockCovariance:
        return self._like([b / scalar for _i, b in self.groups])

    def add_diagonal_(self, value: float) -> BlockCovariance:
        for _i, blocks in self.groups:
            blocks.diagonal(dim1=-2, dim2=-1).add_(value)
        return self

    def symmetrised(self) -> BlockCovariance:
        return self._like([0.5 * (b + b.transpose(-1, -2)) for _i, b in self.groups])

    @staticmethod
    def weighted_sum(terms: list[tuple[float, BlockCovariance]]) -> BlockCovariance:
        """$\\sum_u w_u\\bm P_u$ for matching structures: the full-sharing combine."""
        first = terms[0][1]
        blocks = [torch.zeros_like(b) for _i, b in first.groups]
        for weight, cov in terms:
            for total, (_i, b) in zip(blocks, cov.groups, strict=True):
                total.add_(b, alpha=weight)
        return first._like(blocks)

    # -- products ------------------------------------------------------------- #

    def diagonal(self) -> torch.Tensor:
        out = torch.empty(self.p, dtype=self.dtype, device=self.device)
        for index, blocks in self.groups:
            out[index] = blocks.diagonal(dim1=-2, dim2=-1)
        return out

    def matmul(self, matrix: torch.Tensor) -> torch.Tensor:
        """$\\bm P\\bm M$ for $\\bm M$ of shape (p,) or (p, r)."""
        out = torch.empty_like(matrix)
        for index, blocks in self.groups:
            if matrix.ndim == 1:
                out[index] = torch.einsum("cst,ct->cs", blocks, matrix[index])
            else:
                out[index] = torch.einsum("cst,ctr->csr", blocks, matrix[index])
        return out

    def quadratic(self, jacobians: torch.Tensor) -> torch.Tensor:
        """$\\bm J\\bm P\\bm J^{\\mathsf T}$ per sample, ``jacobians`` (n, q, p) -> (n, q, q)."""
        n, q, _p = jacobians.shape
        out = torch.zeros(n, q, q, dtype=jacobians.dtype, device=jacobians.device)
        for index, blocks in self.groups:
            part = jacobians[:, :, index]                    # (n, q, count, size)
            out += torch.einsum("nqcs,cst,nrct->nqr", part, blocks, part)
        return out

    def dense(self) -> torch.Tensor:
        """The full matrix, for tests."""
        out = torch.zeros(self.p, self.p, dtype=self.dtype, device=self.device)
        for index, blocks in self.groups:
            for row, block in zip(index, blocks, strict=True):
                out[row[:, None], row[None, :]] = block
        return out

    # -- the update ------------------------------------------------------------- #

    def woodbury(self, mean: torch.Tensor, stacked: torch.Tensor, score: torch.Tensor,
                 context: str = "") -> tuple[torch.Tensor, BlockCovariance]:
        r"""The decoupled EKF's update: one shared gain, block-diagonal covariance.

        $\bm S=\bm I+\bar{\bm B}^{\mathsf T}\bm P\bar{\bm B}$ couples every block, so
        the gain is the full filter's. The **mean is exact for a block-diagonal prior**,
        $\bm m+\bm P\bm s-\bm P\bar{\bm B}\bm S^{-1}(\bm P\bar{\bm B})^{\mathsf T}\bm s$,
        low rank and never $p\times p$; the covariance keeps only each block of
        $\bm P-\bm P\bar{\bm B}\bm S^{-1}\bar{\bm B}^{\mathsf T}\bm P$, dropping the
        cross-block terms. That truncation is the whole approximation.
        """
        product = self.matmul(stacked)                       # P B, (p, r)
        gram = stacked.transpose(0, 1) @ product
        gram.diagonal().add_(1.0)
        try:
            factor = torch.linalg.cholesky(gram)
        except torch.linalg.LinAlgError as failure:
            raise FilterError(
                "the innovation covariance is singular, so P has grown until the identity "
                f"is negligible beside it. {context}"
            ) from failure
        solved = torch.cholesky_solve(product.transpose(0, 1), factor)    # S^-1 (PB)^T, (r, p)
        new_mean = mean + self.matmul(score) - product @ (solved @ score)
        blocks = []
        for index, block in self.groups:
            left = product[index]                            # (count, size, r)
            right = solved[:, index].permute(1, 0, 2)        # (count, r, size)
            updated = block - left @ right
            blocks.append(0.5 * (updated + updated.transpose(-1, -2)))
        return new_mean, self._like(blocks)


def add_diagonal(covariance: Any, value: float) -> Any:
    """$\\bm P+q\\bm I$ in place, dense or block."""
    if isinstance(covariance, BlockCovariance):
        return covariance.add_diagonal_(value)
    covariance.diagonal().add_(value)
    return covariance


def quadratic(jacobians: torch.Tensor, covariance: Any) -> torch.Tensor:
    """$\\bm J\\bm P\\bm J^{\\mathsf T}$ per sample, dense or block."""
    if isinstance(covariance, BlockCovariance):
        return covariance.quadratic(jacobians)
    return torch.einsum("nqp,pr,nsr->nqs", jacobians, covariance, jacobians)


def initial_covariance(model: Model, structure: str, scale: float, like: torch.Tensor) -> Any:
    """$\\sigma_0^2\\bm I$ in the requested structure."""
    if structure == "full":
        return torch.eye(like.numel(), dtype=like.dtype, device=like.device) * scale
    return BlockCovariance.identity(parameter_blocks(model, structure), scale, like.dtype,
                                    like.device)


def woodbury_update(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    stacked: torch.Tensor,
    score: torch.Tensor,
    context: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""One stacked measurement update, via Woodbury.

    The information increment $\bar{\bm B}\bar{\bm B}^{\mathsf T}$ has rank at
    most $n(q-1)$ -- 360 here against $p=2908$ -- so the identity turns a
    $p\times p$ inverse into an $(nq)\times(nq)$ one. Direct inversion is
    $O(p^3)\approx2.5\times10^{10}$ flops a step, about an hour a seed before any
    sweep (D59).

    ``context`` is appended to the divergence message: the caller knows which
    hyperparameter is the likely culprit and this function does not.

    A :class:`BlockCovariance` takes its own, block-wise update (D3).
    """
    if isinstance(covariance, BlockCovariance):
        return covariance.woodbury(mean, stacked, score, context)
    product = covariance @ stacked  # (p, n*q)
    gram = stacked.transpose(0, 1) @ product  # (n*q, n*q)
    gram.diagonal().add_(1.0)

    # $\bm I + \bar{\bm B}^{\mathsf T}\bm P\bar{\bm B}$ is symmetric positive
    # definite by construction -- identity plus a Gram matrix -- so it gets a
    # Cholesky solve rather than a general LU one. Cheaper, better conditioned,
    # and its failure carries information a general solver's does not: the only
    # way to lose definiteness here is for $\bar{\bm B}^{\mathsf T}\bm P\bar{\bm
    # B}$ to be so large that adding the identity is lost to rounding, which is
    # D61's divergence caught one step before the mean goes non-finite.
    try:
        factor = torch.linalg.cholesky(gram)
    except torch.linalg.LinAlgError as failure:
        raise FilterError(
            "the innovation covariance is singular, so P has grown until the identity "
            f"is negligible beside it. {context}"
        ) from failure

    solved = torch.cholesky_solve(product.transpose(0, 1), factor)  # (n*q, p)
    updated = covariance - product @ solved
    updated = 0.5 * (updated + updated.transpose(0, 1))
    return mean + updated @ score, updated


def iterated_update(
    linearise: Any,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    iterations: int,
    context: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The iterated EKF: ``iterations`` Gauss--Newton steps on the posterior (D139).

    ``linearise(theta)`` returns the information pair $(\bar{\bm B},\bm H^{\mathsf T}\bm s)$
    at ``theta`` -- whatever rescaling the caller applies included -- so the
    centralised and the diffusion filters share this loop as they share
    `woodbury_update`. From the prior $(\bm m^-,\bm P^-)$, with $\bm m_0=\bm m^-$:

    .. math::
        \bm P_i=\bigl((\bm P^-)^{-1}+\bar{\bm B}_i\bar{\bm B}_i^{\mathsf T}\bigr)^{-1},\qquad
        \bm m_{i+1}=\bm m^-+\bm P_i\bigl[\bm H_i^{\mathsf T}\bm s(\bm m_i)
        +\bar{\bm B}_i\bar{\bm B}_i^{\mathsf T}(\bm m_i-\bm m^-)\bigr]

    which is the Gauss--Newton step on $\tfrac12\lVert\bm m-\bm m^-\rVert^2_{(\bm P^-)^{-1}}
    -\log p(\bm y\mid\bm m)$, written so each iteration is one Woodbury solve
    against the *prior*. At $i=0$ the correction term vanishes and this is the
    plain update, operation for operation, so one iteration reproduces every run
    before D139 exactly.

    **The covariance is the last linearisation's** (decided 2026-10-06): $K$
    iterations cost $K$ Jacobians, the same $K$ passes the gradient learners get,
    rather than the textbook's $K+1$. **Plain Gauss--Newton, no damping**: the
    prior covariance is the trust region (D61), and the callers' divergence guards
    catch an iterate that leaves it.
    """
    if iterations < 1:
        raise FilterError(f"iterations must be >= 1, got {iterations}")
    prior_mean, iterate = mean, mean
    updated = covariance
    for index in range(iterations):
        stacked, score = linearise(iterate)
        if index:
            score = score + stacked @ (stacked.transpose(0, 1) @ (iterate - prior_mean))
        iterate, updated = woodbury_update(prior_mean, covariance, stacked, score, context)
    return iterate, updated


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
        trust_region_ratio: float = TRUST_REGION_RATIO,
        inner_steps: int = 1,
        covariance_blocks: str = "full",
        **_ignored: Any,
    ) -> None:
        if inner_steps < 1:
            raise FilterError(f"inner_steps must be >= 1, got {inner_steps}")
        if covariance_blocks not in COVARIANCE_BLOCKS:
            raise FilterError(
                f"covariance_blocks must be one of {COVARIANCE_BLOCKS}, got {covariance_blocks!r}")
        self._name = name
        self.model = model
        self.likelihood = likelihood
        self._n_nodes = n_nodes
        #: Gauss--Newton relinearisations per measurement update: the iterated EKF
        #: (`iterated_update`, D139). 1 is the plain filter.
        self.inner_steps = inner_steps
        #: The covariance's structure (D3): "full" is the dense P of every run before;
        #: "neuron", "layer" and "tensor" hold a `BlockCovariance`.
        self.covariance_blocks = covariance_blocks
        self.transition = transition
        self.gamma = gamma
        self.lambda_forget = lambda_forget
        self.process_noise_q = process_noise_q
        self.prior_scale = prior_scale
        self.trust_region_ratio = trust_region_ratio

        #: The belief, held as **one** `LearnerState` that `state(node)` returns
        #: for every agent. Not rebuilt per call: `recorder.resume()` restores a
        #: checkpoint by mutating the object `state(node)` hands back, so a fresh
        #: instance each time would swallow the restore silently -- the filter
        #: would resume from theta_0 while the recorder skipped to the checkpoint
        #: step, and the run would look clean and be nonsense (design note D68).
        self._state: LearnerState | None = None
        self._steps = 0
        #: Set by `init`. Zero means "not initialised from a theta_0", which
        #: disables the trust-region test rather than comparing against a scale
        #: that was never established -- a guard that cannot be calibrated should
        #: abstain, not guess.
        self._initial_norm = 0.0

    # `_mean` and `_covariance` read and write through the single state object,
    # so the update code below stays in the notation of the derivation.

    @property
    def _mean(self) -> torch.Tensor | None:
        return None if self._state is None else self._state.theta

    @_mean.setter
    def _mean(self, value: torch.Tensor) -> None:
        assert self._state is not None
        self._state.theta = value

    @property
    def _covariance(self) -> torch.Tensor | None:
        return None if self._state is None else self._state.extras["P"]

    @_covariance.setter
    def _covariance(self, value: torch.Tensor) -> None:
        assert self._state is not None
        self._state.extras["P"] = value

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

        self._state = LearnerState(
            theta=theta0.clone(),
            extras={"P": initial_covariance(self.model, self.covariance_blocks,
                                            self.prior_scale, theta0)},
        )
        # The scale the trust-region test in `_check_belief` is measured against.
        # Kept from theta0 rather than from a constant because "how far the mean
        # has travelled" is only meaningful relative to where it started, and
        # that depends on the model's initialisation.
        self._initial_norm = float(theta0.detach().norm())

    def state(self, node: int) -> LearnerState:
        """The one belief, for every agent. Not a copy, and not a fresh object."""
        self._check_initialised()
        self._check_node(node)
        assert self._state is not None
        return self._state

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
        # Finiteness alone is far too weak a test. float64 reaches 1.8e308, and a
        # network whose weights are 1e131 is meaningless long before it stops
        # being representable -- it evaluates without complaint, scores at chance,
        # and only fails once two matmuls push the logits past the float ceiling
        # and softmax returns NaN. At that point the failure surfaces in the
        # metrics rather than here, as a MetricError the sweep does not catch.
        #
        # The threshold is deliberately enormous: no run that is still learning
        # travels a millionfold from its initialisation, so this cannot reclassify
        # a healthy cell, and a diverging one crosses it within a few steps of
        # setting off.
        norm = float(self._mean.detach().norm())
        if self._initial_norm > 0.0 and norm > self.trust_region_ratio * self._initial_norm:
            raise FilterError(
                f"{self._name} left the trust region at step {self._steps}: the mean "
                f"norm is {norm:.3e}, {norm / self._initial_norm:.2e} times its "
                f"initial {self._initial_norm:.3e}. The update is a Gauss-Newton step "
                "valid only near the current linearisation point, so a mean this far "
                "from theta_0 is a diverged filter that has not yet overflowed "
                "(design note D61)."
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
            add_diagonal(self._covariance, self.process_noise_q)

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

        try:
            self._mean, self._covariance = iterated_update(
                lambda theta: information_pair(self.model, self.likelihood, theta, x, y),
                mean,
                covariance,
                self.inner_steps,
                context=self._divergence_context(covariance),
            )
        except FilterError as failure:
            raise FilterError(f"{self._name} diverged at step {self._steps}: {failure}") from failure

    def _divergence_context(self, covariance: torch.Tensor) -> str:
        """Which hyperparameter to blame, from the state that actually caused it.

        P can reach a singular innovation covariance by two routes -- too large a
        prior, or forgetting that outruns the information arriving -- and naming
        the wrong one sends the reader to the wrong hyperparameter. X13 hit it at
        prior_scale=0.01, which is mid-range and blameless; the culprit was
        lambda=0.993 inflating P by 0.7% a step for 860 steps.
        """
        variance = float(covariance.diagonal().mean())
        inflation = (
            f"lambda={self.lambda_forget} inflates P by "
            f"{1 / self.lambda_forget - 1:.2%} per step"
            if self.lambda_forget < 1.0
            else f"Q={self.process_noise_q} adds to P each step"
        )
        return (
            f"Mean variance is {variance:.3e}, from prior_scale={self.prior_scale}; "
            f"{inflation}. Either the prior is too large or the forgetting outruns "
            "the information arriving (design note D61)."
        )

    # -- what no SGD baseline can report ------------------------------------ #

    def belief_stages(self) -> tuple[str, ...]:
        """One belief and no combine, so there is no pre-combine stage to score."""
        return ("post",)

    def belief(self, node: int, stage: str = "post") -> tuple[torch.Tensor, torch.Tensor]:
        r"""$(\bm m,\bm P)$ -- the same pooled belief for every agent."""
        if stage != "post":
            raise FilterError(f"{self._name} has no combine, so no {stage!r} belief")
        self._check_initialised()
        self._check_node(node)
        assert self._mean is not None and self._covariance is not None
        return self._mean, self._covariance

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
        return quadratic(jacobians, self._covariance)

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
            "inner_steps": self.inner_steps,
            "covariance_blocks": self.covariance_blocks,
            "trace_over_p": float(diagonal.mean()),
            "min_diagonal": float(diagonal.min()),
        }

    def _check_initialised(self) -> None:
        if self._state is None:
            raise FilterError(f"{self._name} has no belief; call init(theta0) before stepping")

    def _check_node(self, node: int) -> None:
        if not 0 <= node < self._n_nodes:
            raise FilterError(f"no agent {node}; have 0..{self._n_nodes - 1}")
