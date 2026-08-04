"""The communication ledger: how many scalars each method actually sends.

Every accuracy curve is reported against this as well as against $t$, because a
method that sends more per step wins any per-step plot for uninteresting
reasons. That makes the ledger a load-bearing measurement rather than
bookkeeping, and it has to count the **real payload**.

**The payload is not always one $p$-vector.** The plan says diffusion SGD
exchanges "one $p$-vector per link per step", which is true of plain ATC. But the
primary configuration for X1--X6 is SGD *with momentum, momentum also mixed*, and
mixing the momentum means broadcasting it: $2p$ per link. The Diff-EKF broadcasts
$\\bm\\psi$ alone, so at that configuration it sends **half** what the baseline
does:

==============================  ===========  ===================================
learner                         per link     note
==============================  ===========  ===================================
``diffusion_sgd_atc`` none      $p$          matches Diff-EKF exactly
``diffusion_sgd_atc`` momentum  $2p$         the X1--X6 primary
``diffusion_sgd_atc`` all       $3p$         AdamW, two moments
``diffusion_ekf`` local         $p$          covariance stays local
``diffusion_ekf`` one_hop       $p(q'{+}1)$  exchanges $(\\bm B, \\bm H^{\\mathsf T}\\bm\\nu)$
==============================  ===========  ===================================

So "at identical communication" is a claim about a *particular* pairing, and X1
runs both SGD variants so phase 5 can state it against the matched one while
still showing the stronger baseline (design note D29).

**Centralized SGD has no diffusion cost and is not plotted on the communication
axis.** It is an upper reference for the online setting, not a deployable
competitor, so it appears on F2 as a horizontal line like $e^\\star$. The notional
cost of shipping its samples to a centre is recorded in the ledger anyway, so the
number exists when a reviewer asks -- and it is uncomfortable: at the defaults
that is far *less* traffic than ring diffusion, which is the fact the paradigm
has to justify itself against.

**Idle agents still communicate.** With $\\pi_{\\text{lab}}<1$ an agent with no
label passes its prediction through and still takes part in the combine step, so
the exchange happens regardless. Communication is a function of the graph and the
method, never of the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Scalars per parameter vector exchanged, by optimizer-state mixing policy.
#: A learner mixing momentum must send it, or the neighbours cannot mix it.
VECTORS_PER_MIX_POLICY = {"none": 1, "momentum": 2, "all": 3}


class LedgerError(ValueError):
    """Raised when a communication cost cannot be determined."""


@dataclass(frozen=True)
class CommunicationCost:
    """What one learner sends, per step and over a run.

    Attributes:
        learner: the learner's name.
        vectors_per_link: how many $p$-vectors cross each link, each direction.
        scalars_per_step: total scalars transmitted network-wide per step.
        rounds_per_step: exchange rounds. One for every method here; the
            consensus alternatives the research note rejects need many.
        diffuses: whether this learner communicates at all. False for
            ``centralized_sgd`` and ``local_only``, for opposite reasons.
        note: why the cost is what it is, carried into the ledger table.
    """

    learner: str
    vectors_per_link: int
    scalars_per_step: int
    rounds_per_step: int
    diffuses: bool
    note: str = ""

    def total_scalars(self, horizon: int) -> int:
        return self.scalars_per_step * horizon


def directed_links(n_edges: int) -> int:
    """Messages per round: each undirected edge carries one each way.

    Self-loops do not exist in the adjacency (design note D15), so this is not
    inflated by the $N$ messages an agent would otherwise be billed for sending
    to itself.
    """
    return 2 * n_edges


def diffusion_cost(
    learner: str,
    num_params: int,
    n_edges: int,
    mix_optimizer_state: str = "none",
    adapt_scope: str = "local",
    fisher_rank: int = 0,
) -> CommunicationCost:
    """The per-step cost of a diffusion learner.

    Args:
        num_params: $p$.
        n_edges: undirected edges in the communication graph.
        mix_optimizer_state: ``none``, ``momentum`` or ``all``. Mixing a moment
            means transmitting it.
        adapt_scope: ``local`` sends $\\bm\\psi$ only; ``one_hop`` additionally
            sends $(\\bm B, \\bm H^{\\mathsf T}\\bm\\nu)$, which is what makes the
            complete-graph exactness proposition hold.
        fisher_rank: $q'$, needed only for ``one_hop``.
    """
    if mix_optimizer_state not in VECTORS_PER_MIX_POLICY:
        raise LedgerError(
            f"unknown mix policy {mix_optimizer_state!r}; " f"have {sorted(VECTORS_PER_MIX_POLICY)}"
        )

    vectors = VECTORS_PER_MIX_POLICY[mix_optimizer_state]
    note = f"psi plus {vectors - 1} optimizer moment(s)" if vectors > 1 else "psi only"

    if adapt_scope == "one_hop":
        if fisher_rank < 1:
            raise LedgerError("one_hop adapt needs a positive fisher_rank")
        # (B, H^T nu): a p x q' factor plus one p-vector.
        vectors = fisher_rank + 1
        note = f"psi plus the information pair (B, H^T nu) at rank {fisher_rank}"

    return CommunicationCost(
        learner=learner,
        vectors_per_link=vectors,
        scalars_per_step=vectors * num_params * directed_links(n_edges),
        rounds_per_step=1,
        diffuses=True,
        note=note,
    )


def centralized_cost(
    learner: str, n_nodes: int, samples_per_step: int, input_dim: int
) -> CommunicationCost:
    """The notional cost of shipping every sample to a fusion centre.

    Recorded but **not** plotted on the communication axis: centralized SGD is an
    upper reference for the online setting, not a competitor, and it appears on
    F2 as a horizontal line. The number is kept because it is the honest answer
    to "what does pooling actually cost", and because it is smaller than ring
    diffusion at these sizes -- which is the uncomfortable fact the decentralized
    paradigm has to argue against on grounds other than raw bandwidth.
    """
    per_sample = input_dim + 1  # the image, plus its label
    return CommunicationCost(
        learner=learner,
        vectors_per_link=0,
        scalars_per_step=n_nodes * samples_per_step * per_sample,
        rounds_per_step=1,
        diffuses=False,
        note="notional: raw samples to a fusion centre; not plotted on the comms axis",
    )


def local_only_cost(learner: str) -> CommunicationCost:
    """Zero, and that is the point: the gap to it *is* the value of cooperation."""
    return CommunicationCost(
        learner=learner,
        vectors_per_link=0,
        scalars_per_step=0,
        rounds_per_step=0,
        diffuses=False,
        note="no communication",
    )


def cost_for(
    learner_config: Any, num_params: int, n_edges: int, **kwargs: Any
) -> CommunicationCost:
    """The cost of whichever learner a config names."""
    name = learner_config.name
    if name == "local_only":
        return local_only_cost(name)
    if name == "centralized_sgd":
        return centralized_cost(
            name,
            n_nodes=kwargs["n_nodes"],
            samples_per_step=kwargs["samples_per_step"],
            input_dim=kwargs["input_dim"],
        )
    return diffusion_cost(
        learner=name,
        num_params=num_params,
        n_edges=n_edges,
        mix_optimizer_state=getattr(learner_config, "mix_optimizer_state", "none"),
        adapt_scope=getattr(learner_config, "adapt_scope", "local"),
        fisher_rank=kwargs.get("fisher_rank", 0),
    )


def ledger(costs: list[CommunicationCost], horizon: int) -> list[dict[str, Any]]:
    """The table emitted once per experiment, in the style of Table I of [1].

    One row per method. The ``relative`` column is against the cheapest method
    that actually diffuses, which is the comparison the phase-5 claim rests on.
    """
    diffusing = [cost for cost in costs if cost.diffuses]
    baseline = min((cost.scalars_per_step for cost in diffusing), default=0)

    rows = []
    for cost in costs:
        relative = cost.scalars_per_step / baseline if baseline and cost.diffuses else None
        rows.append(
            {
                "learner": cost.learner,
                "diffuses": cost.diffuses,
                "vectors_per_link": cost.vectors_per_link,
                "rounds_per_step": cost.rounds_per_step,
                "scalars_per_step": cost.scalars_per_step,
                "total_scalars": cost.total_scalars(horizon),
                "relative_to_cheapest_diffusing": relative,
                "note": cost.note,
            }
        )
    return rows
