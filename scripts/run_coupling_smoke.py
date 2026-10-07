r"""D3b -- choose the coupled network's kappa: neighbours informative, nodes not in sync.

    python scripts/run_coupling_smoke.py              # the grid, on CPU, a few minutes
    python scripts/run_coupling_smoke.py --seeds 0 1 2

The network (decided 2026-10-07, D139): the communication graph's closed
neighbourhoods and Metropolis weights couple the nodes, each node keeps M8's delay
(`m_tau_spread`), and node $v$'s production reads $y_v=(1-\kappa)x_v(t-\tau_v)
+\kappa\sum_u a_{vu}x_u(t-\tau_v)$ (`mackey_glass.integrate`). For every kappa, per
run seed:

* **gain** -- an offline ridge AR predicting each node's next observation from its own
  31-sample window, against one that also sees its neighbours' windows; the relative
  RMSE reduction, averaged over nodes. What a GNN could at best add, linearly.
* **sync** -- the mean pairwise Pearson correlation of the clean series, over all
  pairs and over neighbour pairs. Near 1 the network has synchronised and neighbours
  repeat what a node already knows.

**The rule** (decided 2026-10-07): the largest gain with mean pairwise correlation
<= 0.5. This script prints the table and the kappa the rule picks; nothing is fixed
until the user has seen it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from _args import sweep_parser  # noqa: E402
from dekf_bench.data import mackey_glass as mg  # noqa: E402
from dekf_bench.env.graph import build_graphs  # noqa: E402
from dekf_bench.env.series import agent_laws  # noqa: E402
from dekf_bench.runner.seeding import Seeds  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

KAPPAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SYNC_LIMIT = 0.5
WINDOW = 31
SAMPLES = 6000
TRAIN = 0.7
RIDGE = 1e-3


def network(config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The run's communication graph as closed neighbourhoods and Metropolis weights."""
    graphs = build_graphs(config, Seeds.from_master(seed).torch_generator("graph"))
    weights = graphs.comm.weights.double().numpy()
    n = weights.shape[0]
    width = int((weights > 0).sum(axis=1).max())
    neighbours = np.zeros((n, width), dtype=np.int64)
    padded = np.zeros((n, width))
    for v in range(n):
        around = np.nonzero(weights[v])[0]
        neighbours[v, : around.size] = around
        neighbours[v, around.size:] = v               # padding: any index, weight 0
        padded[v, : around.size] = weights[v, around]
    return neighbours, padded


def windows(series: np.ndarray, length: int) -> np.ndarray:
    """(T - length, length) sliding windows of a 1-D series, last column most recent."""
    return np.lib.stride_tricks.sliding_window_view(series[:-1], length)


def ridge_rmse(features: np.ndarray, target: np.ndarray) -> float:
    split = int(TRAIN * len(target))
    x = np.hstack([features, np.ones((len(target), 1))])
    a, b = x[:split], target[:split]
    coef = np.linalg.solve(a.T @ a + RIDGE * np.eye(a.shape[1]), a.T @ b)
    residual = x[split:] @ coef - target[split:]
    return float(np.sqrt(np.mean(residual**2)))


def measure(config, seed: int, kappa: float) -> dict:
    series = config.env.series
    n = config.graph.n_nodes
    neighbours, weights = network(config, seed)
    taus = agent_laws(series, n).tau
    rng = np.random.default_rng(Seeds.from_master(seed).sub("stream", "coupling_smoke"))
    clean = mg.integrate(SAMPLES, mg.initial_histories(n, rng), beta=series.beta,
                         gamma=series.gamma, exponent=series.exponent, tau=taus,
                         dt=series.dt, delta=series.delta, burn_in=series.burn_in,
                         neighbours=neighbours, weights=weights, kappa=kappa)
    observed = mg.observe(mg.standardise(clean), series.sigma, rng)
    gains = []
    for v in range(n):
        target = observed[v, WINDOW:]
        own = windows(observed[v], WINDOW)
        around = [u for u, w in zip(neighbours[v], weights[v], strict=True) if w > 0 and u != v]
        joint = np.hstack([own, *[windows(observed[u], WINDOW) for u in around]])
        gains.append(1.0 - ridge_rmse(joint, target) / ridge_rmse(own, target))
    corr = np.corrcoef(clean)
    pairs = np.triu_indices(n, 1)
    adjacency = np.zeros((n, n), dtype=bool)
    for v in range(n):
        for u, w in zip(neighbours[v], weights[v], strict=True):
            if w > 0 and u != v:
                adjacency[v, u] = True
    return {"gain": float(np.mean(gains)), "sync": float(corr[pairs].mean()),
            "sync_nbr": float(corr[adjacency].mean())}


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2])
    args = parser.parse_args(argv)
    config = load_config("m_tau_spread")
    print(f"D3b coupling smoke: N = {config.graph.n_nodes}, tau = "
          f"{list(agent_laws(config.env.series, config.graph.n_nodes).tau)}, seeds {args.seeds}\n")
    print(f"  {'kappa':>6}{'gain':>9}{'(min)':>8}{'sync':>8}{'sync nbr':>10}")
    rows = []
    for kappa in KAPPAS:
        found = [measure(config, seed, kappa) for seed in args.seeds]
        row = {k: float(np.mean([f[k] for f in found])) for k in found[0]}
        row["gain_min"] = float(min(f["gain"] for f in found))
        rows.append((kappa, row))
        print(f"  {kappa:>6g}{row['gain']:>+9.3f}{row['gain_min']:>+8.3f}"
              f"{row['sync']:>8.3f}{row['sync_nbr']:>10.3f}", flush=True)
    allowed = [(row["gain"], kappa) for kappa, row in rows
               if kappa > 0 and row["sync"] <= SYNC_LIMIT]
    if allowed:
        gain, kappa = max(allowed)
        print(f"\n  rule (largest gain with sync <= {SYNC_LIMIT}): kappa = {kappa:g}, "
              f"gain {gain:+.3f}")
    else:
        print(f"\n  rule: no kappa > 0 keeps sync <= {SYNC_LIMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
