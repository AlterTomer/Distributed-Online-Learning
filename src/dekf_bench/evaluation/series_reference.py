r"""Reference lines for the series task: the noise floor, persistence, and $e^\star$.

`docs/mackey_glass_plan.md`, WP6 and decisions 17 and 20. Every online method is
read against three lines, each answering a different question:

* **The noise floor** $\sigma$ -- what no predictor of a noisy target can beat. With
  noisy *inputs* (decision 2) the achievable floor lies above it, so this line
  shows headroom rather than a target.
* **Persistence**, $\hat x_{i+1}=x_i$ -- the trivial predictor. A learner above it
  has learned nothing about the dynamics. Computed here rather than run as a
  learner: it has no parameters, and the runner's disagreement metrics assume
  every learner has some.
* **$e^\star$**, the offline reference: the same Transformer trained to
  convergence on the **network's whole data budget at a fixed law** -- $N\times T$
  blocks, the pooled data an online run sees -- with the epoch chosen on a held-out
  validation set, as `reference.py` does for MNIST. Recomputed per channel value,
  for the reason MNIST recomputes it per rotation: a gap measured against the
  undrifted law would charge the drifted law's own difficulty to the method.

Cached under ``data/reference/mackey_glass/`` by a hash of everything that
determines the answer, and independent of any run's seed, so re-running an
experiment never silently retrains it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dekf_bench.env.series import law_blocks
from dekf_bench.metrics import regression
from dekf_bench.models.transformer import CausalTransformer

#: Where the cached references live. `data/` is gitignored.
REFERENCE_DIR = Path(__file__).resolve().parents[3] / "data" / "reference" / "mackey_glass"


@dataclass(frozen=True)
class ReferenceSettings:
    """How the offline reference is trained. The defaults match one run's budget."""

    #: 40 x 375 = 15 000 blocks: N x T at N = 10, T = 1500, n_b = 1.
    train_trajectories: int = 40
    blocks_per_trajectory: int = 375
    val_blocks: int = 2000
    test_blocks: int = 4000
    #: A ceiling, not a budget: `patience` ends training. Raised from 40 when the
    #: first M2 level selected epoch 37 -- still improving at the cap, so not the
    #: converged reference $e^\star$ is defined as.
    epochs: int = 100
    patience: int = 5
    batch: int = 64
    lr: float = 3e-3
    seed: int = 0


@dataclass(frozen=True)
class ReferenceScore:
    value: float
    rmse: float
    rmse_full_context: float
    persistence_rmse: float
    noise_floor: float
    best_epoch: int
    train_blocks: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def persistence_rmse(inputs: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray) -> float:
    """RMSE of predicting each target by the input just before it."""
    inputs, targets = torch.as_tensor(inputs), torch.as_tensor(targets)
    return regression.rmse(inputs, targets)


def _key(series: Any, value: float, model: CausalTransformer, settings: ReferenceSettings) -> str:
    law = {
        name: getattr(series, name)
        for name in ("length", "sigma", "beta", "gamma", "exponent", "tau", "dt", "delta",
                     "burn_in", "channel")
    }
    payload = json.dumps(
        {"law": law, "value": round(value, 12), "settings": asdict(settings),
         "model": {"context": model.context, "d_model": model.d_model,
                   "n_heads": model.n_heads, "d_ff": model.d_ff}},
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def train_reference(
    series: Any,
    value: float,
    settings: ReferenceSettings = ReferenceSettings(),  # noqa: B008 -- frozen, so safe
    model: CausalTransformer | None = None,
    cache_dir: Path | None = REFERENCE_DIR,
) -> ReferenceScore:
    """$e^\\star$ at one channel value: train offline, select on validation, score on test.

    ``cache_dir=None`` disables the cache (tests).
    """
    model = model or CausalTransformer(dtype=torch.float64)
    if cache_dir is not None:
        path = cache_dir / f"{_key(series, value, model, settings)}.json"
        if path.exists():
            return ReferenceScore(**json.loads(path.read_text(encoding="utf-8")))

    width = series.length - 1

    def blocks(n: int, stream: int) -> tuple[torch.Tensor, torch.Tensor]:
        per = settings.blocks_per_trajectory
        rng = np.random.default_rng([settings.seed, stream])
        x, y = law_blocks(series, value, math.ceil(n / per), per, rng)
        return (
            torch.tensor(x.reshape(-1, width)[:n], dtype=torch.float64),
            torch.tensor(y.reshape(-1, width)[:n], dtype=torch.float64),
        )

    n_train = settings.train_trajectories * settings.blocks_per_trajectory
    train_x, train_y = blocks(n_train, 0)
    val_x, val_y = blocks(settings.val_blocks, 1)
    test_x, test_y = blocks(settings.test_blocks, 2)

    generator = torch.Generator().manual_seed(settings.seed)
    theta = torch.nn.Parameter(model.flatten(model.init_params(generator)))
    optimiser = torch.optim.AdamW([theta], lr=settings.lr)

    def score(x: torch.Tensor, y: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.forward(model.unflatten(at), x)

    best, best_epoch, waited = math.inf, 0, 0
    best_theta = theta.detach().clone()
    for epoch in range(1, settings.epochs + 1):
        order = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train - settings.batch + 1, settings.batch):
            idx = order[start : start + settings.batch]
            optimiser.zero_grad()
            prediction = model.forward(model.unflatten(theta), train_x[idx])
            ((prediction - train_y[idx]) ** 2).mean().backward()
            optimiser.step()
        val = regression.rmse(score(val_x, val_y, theta.detach()), val_y)
        if val < best:
            best, best_epoch, waited = val, epoch, 0
            best_theta = theta.detach().clone()
        else:
            waited += 1
            if waited >= settings.patience:
                break

    predictions = score(test_x, test_y, best_theta)
    result = ReferenceScore(
        value=value,
        rmse=regression.rmse(predictions, test_y),
        rmse_full_context=regression.rmse_from(predictions, test_y, regression.FULL_CONTEXT_FROM),
        persistence_rmse=persistence_rmse(test_x, test_y),
        noise_floor=float(series.sigma),
        best_epoch=best_epoch,
        train_blocks=n_train,
    )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return result


def train_gnn_reference(
    config: Any,
    graphs: list[Any],
    settings: ReferenceSettings = ReferenceSettings(),  # noqa: B008 -- frozen, so safe
    snapshot_batch: int = 8,
) -> dict[str, Any]:
    r"""The coupled task's offline reference, and the R profile it measures (D3c).

    M2's protocol on network snapshots (decided 2026-10-08): one run's budget --
    ``train_trajectories x blocks_per_trajectory`` node-blocks, i.e. that over $N$
    snapshots -- from fresh coupled networks at the configured law, spread evenly over
    ``graphs`` (the report seeds' communication graphs, so the profile belongs to the
    task rather than to one draw); one shared $\bm\theta$ through the whole network;
    AdamW, early stopping on a validation set. ``snapshot_batch`` snapshots per step,
    about M2's 64 blocks at $N=10$. Returns the test RMSE, $e^\star$'s line, and **R:
    the per-position residual variance on held-out snapshots, pooled over nodes**.
    """
    from dekf_bench.env.series import agent_laws, closed_neighbourhoods, network_blocks
    from dekf_bench.models.registry import build_model_from_config
    from dekf_bench.utils.determinism import resolve_device

    series = config.env.series
    n_nodes = config.graph.n_nodes
    laws = agent_laws(series, n_nodes)
    model = build_model_from_config(config)
    dtype = model.dtype
    # The model's buffers live on the run's device (D104); the data and theta follow.
    device = torch.device(resolve_device(config.run.device))
    value = float(series.beta) if series.channel == "beta" else 0.0

    def snapshots(total_blocks: int, stream: int) -> tuple[torch.Tensor, torch.Tensor, list]:
        per_graph = math.ceil(total_blocks / n_nodes / len(graphs))
        xs, ys, owners = [], [], []
        for index, graph in enumerate(graphs):
            neighbours, weights = closed_neighbourhoods(graph.weights)
            networks = max(1, math.ceil(per_graph / settings.blocks_per_trajectory))
            each = math.ceil(per_graph / networks)
            rng = np.random.default_rng([settings.seed, stream, index])
            x, y = network_blocks(series, laws, value, neighbours, weights, networks, each, rng)
            xs.append(torch.tensor(x[:per_graph], dtype=dtype, device=device))
            ys.append(torch.tensor(y[:per_graph], dtype=dtype, device=device))
            owners += [index] * min(per_graph, len(x))
        return torch.cat(xs), torch.cat(ys), owners

    train_x, train_y, train_g = snapshots(
        settings.train_trajectories * settings.blocks_per_trajectory, 0)
    val_x, val_y, val_g = snapshots(settings.val_blocks, 1)
    test_x, test_y, test_g = snapshots(settings.test_blocks, 2)
    weights = [graph.weights.to(dtype=dtype, device=device) for graph in graphs]

    def predict(at: torch.Tensor, x: torch.Tensor, owner: list) -> torch.Tensor:
        params = model.unflatten(at)
        out = torch.empty_like(x)
        for index in set(owner):
            rows = torch.tensor([i for i, g in enumerate(owner) if g == index], device=device)
            out[rows] = model.network_predict(params, x[rows], weights[index])
        return out

    generator = torch.Generator().manual_seed(settings.seed)
    theta = torch.nn.Parameter(model.flatten(model.init_params(generator)).to(device))
    optimiser = torch.optim.AdamW([theta], lr=settings.lr)
    best, best_epoch, waited = math.inf, 0, 0
    best_theta = theta.detach().clone()
    n_train = train_x.shape[0]
    for epoch in range(1, settings.epochs + 1):
        order = torch.randperm(n_train, generator=generator).tolist()
        for start in range(0, n_train - snapshot_batch + 1, snapshot_batch):
            idx = order[start : start + snapshot_batch]
            optimiser.zero_grad()
            prediction = predict(theta, train_x[idx], [train_g[i] for i in idx])
            ((prediction - train_y[idx]) ** 2).mean().backward()
            optimiser.step()
        with torch.no_grad():
            val = float(((predict(theta.detach(), val_x, val_g) - val_y) ** 2).mean().sqrt())
        if val < best:
            best, best_epoch, waited = val, epoch, 0
            best_theta = theta.detach().clone()
        else:
            waited += 1
            if waited >= settings.patience:
                break

    with torch.no_grad():
        residual = predict(best_theta, test_x, test_g) - test_y           # (S, N, positions)
    profile = residual.pow(2).mean(dim=(0, 1))
    return {
        "rmse": float(residual.pow(2).mean().sqrt()),
        "observation_variances": [float(v) for v in profile],
        "observation_variance": float(profile.mean()),
        "best_epoch": best_epoch,
        "train_snapshots": int(n_train),
        "graphs": len(graphs),
        "noise_floor": float(series.sigma),
        "persistence_rmse": persistence_rmse(test_x.reshape(-1, test_x.shape[-1]).cpu(),
                                             test_y.reshape(-1, test_y.shape[-1]).cpu()),
    }
