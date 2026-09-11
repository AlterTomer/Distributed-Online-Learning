r"""The diffusion extended Kalman filter: one belief per agent, mixed each step.

The centralised filter of `ekf.py` is the reference this reduces to; both call
the same `information_pair` and `woodbury_update`, so the complete-graph identity
is an identity of one implementation rather than an agreement between two.

**Two independent choices, and this module implements both** (see the research
note, algorithm `alg:diffekf`):

``adapt_scope`` -- whose likelihood information agent $v$ uses, the set
$\mathcal M_{v,t}$ of `eq:adapt`::

    local     M = {v}                    no communication in adapt; the default
    one_hop   M = N^c_v union {v}        neighbours exchange (B_u, H_u^T s_u)

``covariance_sharing`` -- what an agent does with its *confidence* in combine::

    full      P_v <- sum_u a_vu^beta P^psi_u  eq:cov_combine, O(p^2) per link
    local     P_v <- P^psi_v                  eq:cov_local, O(p) per link

The mean update $\bm m_{v,t|t}=\sum_u a_{vu,t}\bm\psi_{u,t}$ is common to every
combination: the choice is about confidence, never about the estimate.

Two further dials, added after X19 measured what a local adapt costs. Unlike the
axes above these are **not pinned by the learner name**, being continuous
settings rather than variant identities:

``adapt_rounds`` -- how many hops of measurement information reach an agent. One
is the canonical incremental step; $L$ hops make $\mathcal M_{v,t}$ the $L$-hop
neighbourhood, and at $L\ge\operatorname{diam}(\G)$ it is the whole vertex set,
so the filter equals the centralised one on *any* connected graph rather than
only on a complete one. Reachability is a boolean set, never a sum along paths,
which is what keeps each agent's information counted exactly once.

``combine_exponent`` -- the $\beta$ above, in $[1,2]$. One is the conservative
bound of `lem:conservative`, which holds for any cross-correlation and is
attained when the neighbours' errors coincide; two is what *independent* errors
give and is smaller by about $|\mathcal M_v|$ -- the same factor a local adapt
leaves the belief inflated by. It costs no communication at all: it tells the
covariance what the mean update already did, which is the gap Cattivelli & Sayed
flag when they note their propagated matrices "do not represent the covariances
of the state estimation errors any longer, since the diffusion update is not
taken into account in the recursions for these matrices".

**One-hop is the canonical algorithm, not an optional extra.** The incremental
step of the diffusion Kalman filter of Cattivelli & Sayed (IEEE TAC 55(9), 2010)
loops over the neighbours of agent $v$ -- "for every neighboring node $l$ in
$N_k$, repeat" -- in both its covariance form (their Algorithm 1) and its
information form (Algorithm 2). A *local* adapt is a reduction of it, and X19
measured what that reduction costs: +0.0228 to +0.0411, growing with drift.

`prop:complete_graph` holds only for $\mathcal M_{v,t}=\V$, which one-hop
delivers on a complete graph. Under a local adapt the filter does **not** reduce
to the centralised one even there -- each agent updates on its own data and the
combine averages covariances, which is not summing information -- so without
one-hop there is no exact reference to validate against, and this filter would
lack the analogue of the X0 gate that every SGD learner has.

**Memory is the binding constraint, not compute.** Each agent holds a $p\times p$
covariance: at $p=2908$ in float64 that is 64.5 MiB, so $N=10$ agents cost
645 MiB. Under ``full`` sharing the combine needs every $\bm P^{\psi}_u$ intact
before any $\bm P_{v,t|t}$ is written, so a second set of the same size is live
during the mix --- about 1.3 GiB. Under ``local`` sharing no second set is needed
and the cost stays at 645 MiB. Compute, by contrast, is roughly centralised-equal:
$N$ updates on $1/N$ of the data each cost about what one update on the pooled
batch costs, and the combine adds $O(N d_v p^2)$.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from dekf_bench.learners.base import Intermediate, LearnerState
from dekf_bench.learners.ekf import (
    TRUST_REGION_RATIO,
    FilterError,
    information_pair,
    woodbury_update,
)
from dekf_bench.models.base import Model

#: The two axes, spelled out so a typo in a config is a refusal rather than a
#: silently different method.
ADAPT_SCOPES = ("local", "one_hop")
COVARIANCE_SHARING = ("full", "local")


class DiffusionEKF:
    r"""One Gaussian belief $(\bm m_v,\bm P_v)$ per agent, diffused each step.

    Unlike `CentralizedEKF`, the covariance is **per agent** and cannot be
    shared: the agents' beliefs are the object under study, and collapsing them
    to one tensor would erase the disagreement $E_{\text{agree}}$ exists to
    measure.
    """

    def __init__(
        self,
        name: str,
        model: Model,
        likelihood: Any,
        n_nodes: int,
        transition: str = "scalar",
        gamma: float = 1.0,
        lambda_forget: float = 1.0,
        process_noise_q: float = 0.0,
        prior_scale: float = 1.0,
        trust_region_ratio: float = TRUST_REGION_RATIO,
        adapt_scope: str = "local",
        covariance_sharing: str = "local",
        adapt_rounds: int = 1,
        combine_exponent: float = 1.0,
        information_exponent: float = 0.0,
        **_ignored: Any,
    ) -> None:
        if adapt_scope not in ADAPT_SCOPES:
            raise FilterError(f"adapt_scope must be one of {ADAPT_SCOPES}, got {adapt_scope!r}")
        if covariance_sharing not in COVARIANCE_SHARING:
            raise FilterError(
                f"covariance_sharing must be one of {COVARIANCE_SHARING}, "
                f"got {covariance_sharing!r}"
            )
        if adapt_rounds < 1:
            raise FilterError(
                f"adapt_rounds must be >= 1, got {adapt_rounds}. A local adapt is "
                "adapt_scope='local', not zero rounds of a one-hop one."
            )
        if not 1.0 <= combine_exponent <= 2.0:
            raise FilterError(
                f"combine_exponent must lie in [1, 2], got {combine_exponent}. 1 is the "
                "conservative bound of lem:conservative, 2 is what independent errors "
                "give; outside that range the covariance is neither."
            )
        if not 0.0 <= information_exponent <= 1.0:
            raise FilterError(
                f"information_exponent must lie in [0, 1], got {information_exponent}. "
                "0 uses the information actually gathered; 1 extrapolates one agent's "
                "batch to the whole network. Outside that range the filter is asserting "
                "evidence no estimator of the network total would give it."
            )
        self._name = name
        self.model = model
        self.likelihood = likelihood
        self._n_nodes = n_nodes
        self.adapt_rounds = adapt_rounds
        self.combine_exponent = combine_exponent
        self.information_exponent = information_exponent
        self._reach: torch.Tensor | None = None
        self.transition = transition
        self.gamma = gamma
        self.lambda_forget = lambda_forget
        self.process_noise_q = process_noise_q
        self.prior_scale = prior_scale
        self.trust_region_ratio = trust_region_ratio
        self.adapt_scope = adapt_scope
        self.covariance_sharing = covariance_sharing

        self._states: dict[int, LearnerState] = {}
        self._steps = 0
        self._initial_norm = 0.0
        #: One-hop stashes its information pair in adapt and applies it in
        #: combine, because the exchange it needs has not happened yet when
        #: `adapt` runs. `DiffusionSGDCTA` defers a gradient the same way and for
        #: the same reason -- the interface reserves communication for combine.
        self._pending: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    # -- identity ----------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def comm_scalars_per_step(self, n_edges: int) -> int:
        r"""Scalars crossing every link, both directions, in one step.

        $\bm\psi$ is always sent. ``full`` sharing adds the $p\times p$
        covariance, which is $p$ further $p$-vectors -- the reason it is
        undeployable and, at this scale, exactly why it is worth measuring once.
        ``one_hop`` adds the pair $(\bm B,\bm H^{\mathsf T}\bm s)$.
        """
        p = self.model.num_params
        vectors = 1
        if self.covariance_sharing == "full":
            vectors += p
        if self.adapt_scope == "one_hop":
            # Each extra round forwards what was received, so the information
            # payload is paid once per round.
            vectors += self.adapt_rounds * (self._fisher_rank() + 1)
        return vectors * p * 2 * n_edges

    def _fisher_rank(self) -> int:
        """$q'$: the columns $\\bm B$ carries per sample."""
        return int(getattr(self.likelihood, "fisher_rank", 0) or 0)

    # -- state -------------------------------------------------------------- #

    def init(self, theta0: torch.Tensor) -> None:
        r"""Every agent starts at $\bm m_0=\bm\theta_0$, $\bm P_0=\sigma_0^2\bm I$.

        Shared initialisation is not a convenience here: agents initialised
        independently do not represent one Bayesian model, and
        `prop:complete_graph` assumes a common predictive prior.

        The covariances are **separate tensors** despite being equal at $t=0$.
        They diverge from the first update, and aliasing them would make agent
        $v$'s update visible to agent $u$ before any message was sent.
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

        self._states = {
            node: LearnerState(
                theta=theta0.clone(),
                extras={
                    "P": torch.eye(theta0.numel(), dtype=theta0.dtype, device=theta0.device)
                    * self.prior_scale
                },
            )
            for node in range(self._n_nodes)
        }
        self._initial_norm = float(theta0.detach().norm())

    def state(self, node: int) -> LearnerState:
        self._check_initialised()
        self._check_node(node)
        return self._states[node]

    def flat_params(self, node: int) -> torch.Tensor:
        return self.state(node).theta

    def covariance(self, node: int) -> torch.Tensor:
        return self.state(node).extras["P"]

    def predict(self, node: int, x: torch.Tensor) -> torch.Tensor:
        params = self.model.unflatten(self.flat_params(node))
        return self.model.forward(params, x)

    # -- the step ----------------------------------------------------------- #

    def adapt_pooled(self, x: torch.Tensor, y: torch.Tensor) -> None:
        raise FilterError(
            f"{self._name} is a per-agent method; the runner calls adapt() per node. "
            "Pooling it would be the centralized filter, which is a different method."
        )

    def adapt(self, node: int, observation: Any) -> Intermediate:
        r"""Predict locally, then form $(\bm\psi_v,\bm P^{\psi}_v)$.

        Under ``one_hop`` the update cannot happen yet -- it needs the
        neighbours' information, which arrives in combine -- so this stashes the
        local pair and emits it as the message.
        """
        self._check_initialised()
        self._check_node(node)
        # Taken from the observation rather than incremented on node 0: the
        # runner's node order is not part of the interface, and a counter that
        # assumes it would misreport every divergence message.
        self._steps = int(observation.step)

        state = self._states[node]
        self._predict(state)
        mean, covariance = state.theta, state.extras["P"]

        stacked, score = None, None
        if observation.has_label and observation.x.shape[0] != 0:
            stacked, score = information_pair(
                self.model, self.likelihood, mean, observation.x, observation.y
            )

        if self.adapt_scope == "one_hop":
            # psi is the *predictive* mean here: the update is deferred, so what
            # travels is the prior plus the information to act on it.
            self._pending[node] = (stacked, score)
            extras = {} if stacked is None else {"B": stacked, "score": score}
            return Intermediate(node=node, psi=mean, extras=extras)

        if stacked is not None:
            stacked, score = self._rescale(stacked, score, seen=1)
            mean, covariance = self._apply(node, mean, covariance, stacked, score)
            state.theta, state.extras["P"] = mean, covariance
        # An unlabelled agent passes its prediction through unchanged and still
        # takes part in combine -- one of diffusion's more attractive properties,
        # and line `line:missing` of the algorithm.
        extras = {"P": covariance} if self.covariance_sharing == "full" else {}
        return Intermediate(node=node, psi=mean, extras=extras)

    def combine(self, intermediates: dict[int, Intermediate], weights: torch.Tensor) -> None:
        r"""The only communication: mix $\bm\psi$, and the covariance if shared."""
        self._check_initialised()
        if set(intermediates) != set(self._states):
            raise FilterError(
                f"combine got intermediates for {sorted(intermediates)} but holds state "
                f"for {sorted(self._states)}"
            )
        order = sorted(self._states)
        mixing = weights.to(
            device=self._states[order[0]].theta.device, dtype=self._states[order[0]].theta.dtype
        )

        psi = {node: intermediates[node].psi for node in order}
        cov_psi = {node: self._states[node].extras["P"] for node in order}
        if self.adapt_scope == "one_hop":
            psi, cov_psi = self._one_hop_update(order, mixing)

        # Every new value is read from the old messages before any is written
        # back: an agent combined earlier must not feed its updated belief to one
        # combined later, which would make the result depend on node ordering and
        # break the complete-graph identity.
        stacked_psi = torch.stack([psi[node] for node in order])
        combined_mean = mixing @ stacked_psi

        combined_cov: dict[int, torch.Tensor] = {}
        if self.covariance_sharing == "full":
            for row, node in enumerate(order):
                total = torch.zeros_like(cov_psi[node])
                for column, other in enumerate(order):
                    weight = float(mixing[row, column])
                    if weight != 0.0:
                        # a^beta, not a. beta = 1 is lem:conservative -- the bound
                        # that holds for any cross-correlation and is attained when
                        # the neighbours' errors coincide. beta = 2 is what those
                        # errors being *independent* gives, since
                        # Cov(sum a_u e_u) = sum a_u^2 Cov(e_u), and is smaller by
                        # about |M_v| -- the same factor D79 says the belief is
                        # inflated by. The truth lies between, because the errors
                        # are correlated through shared history but not perfectly.
                        #
                        # This is the one correction that costs no communication at
                        # all: the covariance is simply told that the estimate was
                        # averaged, which the mean update already did and the
                        # covariance recursion never learned. Cattivelli & Sayed
                        # note the same gap for the linear filter -- their
                        # propagated matrices "do not represent the covariances of
                        # the state estimation errors any longer, since the
                        # diffusion update is not taken into account".
                        total.add_(cov_psi[other], alpha=weight**self.combine_exponent)
                # Symmetrise: without it the recursion loses positive
                # definiteness within a few hundred steps, and under full sharing
                # this also repairs the asymmetry accumulated across blocks that
                # were symmetrised independently (line `line:symm`).
                combined_cov[node] = 0.5 * (total + total.transpose(0, 1))

        for row, node in enumerate(order):
            state = self._states[node]
            state.theta = combined_mean[row]
            if self.covariance_sharing == "full":
                state.extras["P"] = combined_cov[node]
            else:
                state.extras["P"] = cov_psi[node]
            self._check_belief(node)
        self._pending.clear()

    def _one_hop_update(
        self, order: list[int], mixing: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        r"""Apply $\sum_{u\in\mathcal M_v}\bm\Delta_u$ at each agent.

        The information sum of `eq:adapt` is **unweighted** -- $a_{vu,t}$ weights
        the combine, not the measurement set. The neighbourhood is read off the
        mixing matrix's sparsity, which for Metropolis weights is exactly
        $\N^{\mathrm c}_v\cup\{v\}$ with every entry strictly positive.
        """
        reach = self._reachability(mixing)
        psi: dict[int, torch.Tensor] = {}
        cov_psi: dict[int, torch.Tensor] = {}
        for row, node in enumerate(order):
            state = self._states[node]
            mean, covariance = state.theta, state.extras["P"]
            columns = [
                self._pending[other][0]
                for column, other in enumerate(order)
                if bool(reach[row, column]) and self._pending[other][0] is not None
            ]
            scores = [
                self._pending[other][1]
                for column, other in enumerate(order)
                if bool(reach[row, column]) and self._pending[other][1] is not None
            ]
            if columns:
                # Concatenating the B blocks column-wise gives sum_u B_u B_u^T in
                # one Woodbury solve, which is the same identity the centralized
                # filter uses on the pooled batch.
                stacked = torch.cat(columns, dim=1)
                score = torch.stack(scores).sum(dim=0)
                # `seen` is the number of agents that actually contributed, not
                # the neighbourhood size: an unlabelled neighbour supplies no
                # information, so counting it would scale by a factor the
                # evidence does not support.
                stacked, score = self._rescale(stacked, score, seen=len(columns))
                mean, covariance = self._apply(node, mean, covariance, stacked, score)
            psi[node], cov_psi[node] = mean, covariance
        return psi, cov_psi

    def _reachability(self, mixing: torch.Tensor) -> torch.Tensor:
        r"""Which agents' measurements reach $v$ in ``adapt_rounds`` hops.

        One round is $\N^{\mathrm c}_v\cup\{v\}$, the canonical diffusion Kalman
        filter's incremental step (Cattivelli & Sayed 2010, Algorithm 1: "for
        every neighboring node $l\in\N_k$, repeat"). $L$ rounds is the $L$-hop
        neighbourhood, and at $L\ge\operatorname{diam}(\G)$ it is the whole vertex
        set --- which is `prop:complete_graph`'s hypothesis, so the filter then
        *is* the centralised one on any connected graph rather than only on a
        complete one.

        **Reachability is boolean, and that is what keeps it incest-free.** Each
        agent's $\bm\Delta_u$ enters the sum once regardless of how many paths
        carry it, which is the source-tagging of a flooding scheme expressed as a
        set. Summing along paths instead would count the same evidence twice --
        the failure that conservative fusion and covariance intersection exist to
        avoid.

        Cached: the mixing matrix is rebuilt each step but the graph is fixed
        within a run, and a boolean matrix power per step would be waste.
        """
        if self._reach is not None:
            return self._reach
        adjacency = mixing != 0.0
        reach = adjacency.clone()
        for _round in range(self.adapt_rounds - 1):
            reach = (reach.to(torch.float32) @ adjacency.to(torch.float32)) != 0.0
        self._reach = reach
        return reach

    def _rescale(
        self, stacked: torch.Tensor, score: torch.Tensor, seen: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Extrapolate ``seen`` agents' evidence to the whole network.

        The centralised filter accumulates $\sum_{u=1}^{N}\bm\Delta_u\approx
        N\bar{\bm\Delta}$ per step; an agent that has gathered $\lvert\mathcal
        M_v\rvert$ of them holds an unbiased estimator of that total after
        multiplying by $c=N/\lvert\mathcal M_v\rvert$. Right mean, $c$ times the
        variance --- it is an extrapolation, not evidence.

        **Both the information and the score are scaled, and scaling one alone is
        a bug rather than a half-measure.** The step is $\bm\psi=\bm m+\bm
        P^{\psi}\sum\bm H^{\trans}\bm s$: shrinking $\bm P^{\psi}$ by $c$ while
        the score stays as gathered makes every update $c$ times too small, so the
        filter would report a confident belief it never moved toward. Since
        $\bm\Delta=\bm B\bm B^{\trans}$, scaling $\bm B$ by $\sqrt c$ scales the
        information by $c$.

        ``information_exponent`` is the interpolation: $c=(N/\lvert\mathcal
        M_v\rvert)^{\alpha}$, $\alpha=0$ using what was gathered and $\alpha=1$
        extrapolating fully. An interior value is expected to win, because the
        two ends fail in opposite directions --- D80 measures the filter as
        *under*-confident at $\alpha=0$, and full extrapolation asserts $N$
        agents' certainty from one agent's batch.

        ⚠ Unbiased **only under exchangeable agents**. Under Dirichlet skew it is
        not: an agent holding three classes would claim the network's confidence
        about all ten. P5.2 is where that breaks, and it should be measured there
        rather than assumed away.
        """
        if self.information_exponent == 0.0 or seen <= 0:
            return stacked, score
        scale = (self._n_nodes / seen) ** self.information_exponent
        return stacked * math.sqrt(scale), score * scale

    def _apply(
        self,
        node: int,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        stacked: torch.Tensor,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return woodbury_update(
                mean, covariance, stacked, score, context=self._divergence_context(covariance)
            )
        except FilterError as failure:
            raise FilterError(
                f"{self._name} diverged at agent {node}, step {self._steps}: {failure}"
            ) from failure

    def _predict(self, state: LearnerState) -> None:
        if self.transition == "scalar":
            state.theta = self.gamma * state.theta
            state.extras["P"] = self.gamma**2 * state.extras["P"]
        if self.lambda_forget < 1.0:
            state.extras["P"] = state.extras["P"] / self.lambda_forget
        if self.process_noise_q > 0.0:
            state.extras["P"].diagonal().add_(self.process_noise_q)

    # -- guards ------------------------------------------------------------- #

    def _check_belief(self, node: int) -> None:
        """The same two tests the centralised filter applies, per agent.

        One agent's divergence is not survivable by the others: the combine step
        mixes its covariance (under full sharing) or at least its mean into every
        neighbour, so a poisoned belief propagates. Failing the run is therefore
        the honest outcome rather than the harsh one.
        """
        state = self._states[node]
        if not bool(torch.isfinite(state.theta).all()):
            raise FilterError(
                f"{self._name} diverged at agent {node}, step {self._steps}: the mean is "
                f"not finite. prior_scale={self.prior_scale} is most likely too large -- "
                "the update is a Gauss-Newton step and the covariance is its trust region."
            )
        norm = float(state.theta.detach().norm())
        if self._initial_norm > 0.0 and norm > self.trust_region_ratio * self._initial_norm:
            raise FilterError(
                f"{self._name} left its trust region at agent {node}, step {self._steps}: "
                f"the mean has norm {norm:.3e}, more than {self.trust_region_ratio:.0e} "
                f"times its initial {self._initial_norm:.3e}. Finiteness is far too weak a "
                "test in float64 (design note D61)."
            )
        smallest = float(state.extras["P"].diagonal().min())
        if smallest <= 0.0:
            raise FilterError(
                f"{self._name} lost positive definiteness at agent {node}, step "
                f"{self._steps}: the smallest variance is {smallest:.3e}. With no process "
                "noise and lambda at 1 the covariance collapses; give the filter a way to "
                "stay uncertain."
            )

    def _divergence_context(self, covariance: torch.Tensor) -> str:
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

    def logit_covariance(self, node: int, x: torch.Tensor) -> torch.Tensor:
        r"""$\bm H_v\bm P_v\bm H_v^{\mathsf T}$ per sample, at agent ``node``."""
        params = self.model.unflatten(self.flat_params(node))
        jacobians = self.model.per_sample_jacobian(params, x)
        return torch.einsum("nqp,pr,nsr->nqs", jacobians, self.covariance(node), jacobians)

    def summary(self) -> dict[str, Any]:
        self._check_initialised()
        diagonals = torch.stack([s.extras["P"].diagonal() for s in self._states.values()])
        return {
            "adapt_scope": self.adapt_scope,
            "covariance_sharing": self.covariance_sharing,
            "adapt_rounds": self.adapt_rounds,
            "combine_exponent": self.combine_exponent,
            "information_exponent": self.information_exponent,
            "transition": self.transition,
            "gamma": self.gamma,
            "lambda_forget": self.lambda_forget,
            "process_noise_q": self.process_noise_q,
            "prior_scale": self.prior_scale,
            "trace_over_p": float(diagonals.mean()),
            "min_diagonal": float(diagonals.min()),
        }

    def _check_initialised(self) -> None:
        if not self._states:
            raise FilterError(f"{self._name} has no belief; call init(theta0) before stepping")

    def _check_node(self, node: int) -> None:
        if not 0 <= node < self._n_nodes:
            raise FilterError(f"no agent {node}; have 0..{self._n_nodes - 1}")
