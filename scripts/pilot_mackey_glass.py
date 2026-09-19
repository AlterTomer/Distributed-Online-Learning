r"""M0 -- the Mackey--Glass pilot: does the task separate anything, and at what settings?

    python scripts/pilot_mackey_glass.py              # the full pilot, ~20-25 min CPU
    python scripts/pilot_mackey_glass.py --quick      # a smoke run, ~1 min

WP0 of `docs/mackey_glass_plan.md`, and a **hard gate**: nothing else is built if it
fails. No distributed machinery and no filter -- online gradient learners only, on
the CPU, so it never touches the GPU the MNIST sweeps hold.

## What it measures

1. **The standardisation constants** (decision 3), from one long stationary run.
2. **Headroom and the gate** (decisions 3, 23). At each candidate noise level
   $\sigma$: ten solo learners (one per agent, its own blocks only) against one
   learner pooling all ten agents' blocks. Each optimiser -- SGD, momentum SGD,
   AdamW -- gets its own learning rate, chosen on two seeds for solo and pooled
   separately, then runs five seeds. **Pass** if solo minus pooled exceeds three
   times the pooled learner's seed noise: cooperation must have something to buy.
3. **Residual structure** (decisions 12--14), on a model trained offline to
   convergence: the adjacent-position correlation $\rho_1$ (diagonal $\bm R$ needs
   $|\rho_1|<0.2$), the per-position variance ratio (per-position $\bm R$ if it
   exceeds 2), and the mean residual variance -- the centre of the $\bm R$ grid.
4. **The $\beta$ map** (decision 6): across $\beta\in[0.10,0.40]$, whether the
   system stays chaotic (divergence rate of two trajectories $10^{-8}$ apart), how
   its amplitude moves, and how much a model converged at $\beta=0.2$ loses when
   the law moves to $\beta$ -- the analogue of calibrating the 45-degree cap.

Everything is written to ``results/m0_pilot/report.json``; the console gets the
tables. Prequential RMSE is test-then-train throughout, in standardised units.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dekf_bench.data import mackey_glass as mg  # noqa: E402
from dekf_bench.models.transformer import CausalTransformer  # noqa: E402

DTYPE = torch.float64
N_AGENTS, HORIZON, LENGTH = 10, 1500, 32
SIGMAS = [0.01, 0.05, 0.1]
SEEDS = [0, 1, 2, 3, 4]
TUNE_SEEDS = [0, 1]
LEARNING_RATES = {
    "sgd": [0.01, 0.03, 0.1, 0.3],
    "momentum": [0.003, 0.01, 0.03, 0.1],
    "adamw": [3e-4, 1e-3, 3e-3, 1e-2],
}
BETAS = [round(0.10 + 0.02 * i, 2) for i in range(16)]
DAMAGE_SIGMA = 0.05
SETTLED_FRACTION = 0.8
OUT = ROOT / "results" / "m0_pilot"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


_CLEAN: dict[tuple[int, int, float], np.ndarray] = {}


def clean_series(seed: int, horizon: int, beta: float = mg.BETA) -> np.ndarray:
    """One seed's ten clean trajectories, integrated once and reused across noise levels."""
    key = (seed, horizon, beta)
    if key not in _CLEAN:
        histories = mg.initial_histories(N_AGENTS, np.random.default_rng([seed, 0]))
        _CLEAN[key] = mg.integrate(horizon * LENGTH, histories, beta=beta)
    return _CLEAN[key]


def agent_blocks(seed: int, sigma: float, horizon: int, beta: float = mg.BETA):
    """``(inputs, targets)`` of shape ``(N, horizon, 31)``: one seed's ten agents.

    Histories and noise come from separate streams, so the same seed at two noise
    levels is the same trajectory observed through two sensors.
    """
    clean = clean_series(seed, horizon, beta)
    z = mg.observe(mg.standardise(clean), sigma, np.random.default_rng([seed, 1, int(sigma * 1e4)]))
    inputs, targets = mg.to_blocks(z, LENGTH)
    return torch.tensor(inputs, dtype=DTYPE), torch.tensor(targets, dtype=DTYPE)


# --------------------------------------------------------------------------- #
# online learners, vectorised
# --------------------------------------------------------------------------- #


class Learner:
    """One or several independent parameter vectors trained online.

    ``n`` rows of ``theta``: ten for the solo arm (one per agent, updated together
    through ``vmap``), one for the pooled arm. The optimisers are written out so
    both arms share one code path.
    """

    def __init__(self, model: CausalTransformer, theta0: torch.Tensor, n: int, kind: str, lr: float):
        self.model, self.kind, self.lr = model, kind, lr
        self.theta = theta0.expand(n, -1).clone()
        self.m = torch.zeros_like(self.theta)
        self.v = torch.zeros_like(self.theta)
        self.t = 0
        module = model._module

        def loss(theta, x, y):
            params = model.unflatten(theta)
            return ((functional_call(module, params, (x,)) - y) ** 2).mean()

        def predict(theta, x):
            return functional_call(module, model.unflatten(theta), (x,))

        self._grad = vmap(grad(loss))
        self._predict = vmap(predict)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict(self.theta, x)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        g = self._grad(self.theta, x, y)
        self.t += 1
        if self.kind == "sgd":
            self.theta -= self.lr * g
        elif self.kind == "momentum":
            self.m = 0.9 * self.m + g
            self.theta -= self.lr * self.m
        else:  # adamw, torch's defaults
            b1, b2, eps, decay = 0.9, 0.999, 1e-8, 0.01
            self.m = b1 * self.m + (1 - b1) * g
            self.v = b2 * self.v + (1 - b2) * g * g
            m_hat = self.m / (1 - b1**self.t)
            v_hat = self.v / (1 - b2**self.t)
            self.theta -= self.lr * (m_hat / (v_hat.sqrt() + eps) + decay * self.theta)


def run_online(inputs, targets, kind: str, lr: float, pooled: bool, model, theta0) -> np.ndarray:
    """Per-round prequential RMSE over the network; NaN from the first divergence."""
    horizon = inputs.shape[1]
    if pooled:
        learner = Learner(model, theta0, 1, kind, lr)
    else:
        learner = Learner(model, theta0, N_AGENTS, kind, lr)
    curve = np.full(horizon, np.nan)
    for t in range(horizon):
        x, y = inputs[:, t], targets[:, t]  # (N, 31)
        if pooled:
            xb, yb = x.unsqueeze(0), y.unsqueeze(0)  # one learner, a batch of N blocks
        else:
            xb, yb = x.unsqueeze(1), y.unsqueeze(1)  # N learners, one block each
        with torch.no_grad():
            error = float(((learner.predict(xb) - yb) ** 2).mean())
        if not math.isfinite(error) or error > 1e6:
            break
        curve[t] = math.sqrt(error)
        learner.step(xb, yb)
    return curve


def settled(curve: np.ndarray) -> float:
    tail = curve[int(SETTLED_FRACTION * len(curve)) :]
    return float(np.mean(tail)) if np.all(np.isfinite(tail)) else math.inf


# --------------------------------------------------------------------------- #
# the pilot
# --------------------------------------------------------------------------- #


def gate(model, theta0, horizon, seeds, tune_seeds, report) -> None:
    print("\n== 2. headroom and the gate ==")
    data = {(s, sigma): agent_blocks(s, sigma, horizon) for s in seeds for sigma in SIGMAS}
    persistence = {
        sigma: float(np.mean([((d[1] - d[0]) ** 2).mean().sqrt() for (s, sg), d in data.items() if sg == sigma]))
        for sigma in SIGMAS
    }
    report["gate"] = {}
    for sigma in SIGMAS:
        entry = {"persistence_rmse": persistence[sigma], "noise_floor": sigma, "optimisers": {}}
        for kind, rates in LEARNING_RATES.items():
            choice = {}
            for arm in ("solo", "pooled"):
                scored = []
                for lr in rates:
                    values = [
                        settled(run_online(*data[(s, sigma)], kind, lr, arm == "pooled", model, theta0))
                        for s in tune_seeds
                    ]
                    scored.append((float(np.mean(values)), lr))
                choice[arm] = min(scored)[1]
            curves = {
                arm: [run_online(*data[(s, sigma)], kind, choice[arm], arm == "pooled", model, theta0) for s in seeds]
                for arm in ("solo", "pooled")
            }
            solo = np.array([settled(c) for c in curves["solo"]])
            pooled = np.array([settled(c) for c in curves["pooled"]])
            gap = solo - pooled
            noise = float(np.std(pooled, ddof=1)) if len(seeds) > 1 else math.nan
            t_stat = float(gap.mean() / (gap.std(ddof=1) / math.sqrt(len(gap)))) if len(gap) > 1 else math.nan
            passed = bool(gap.mean() > 3.0 * noise)
            entry["optimisers"][kind] = {
                "lr": choice,
                "solo_settled": solo.tolist(),
                "pooled_settled": pooled.tolist(),
                "gap_mean": float(gap.mean()),
                "pooled_seed_noise": noise,
                "gap_over_noise": float(gap.mean() / noise) if noise else math.nan,
                "paired_t": t_stat,
                "passes": passed,
                "pooled_over_floor": float(pooled.mean() / sigma),
                "curve_solo": np.nanmean(curves["solo"], axis=0)[::25].tolist(),
                "curve_pooled": np.nanmean(curves["pooled"], axis=0)[::25].tolist(),
            }
            print(
                f"  sigma={sigma:<5} {kind:<9} lr solo {choice['solo']:<7g} pooled {choice['pooled']:<7g}"
                f" | solo {solo.mean():.4f}  pooled {pooled.mean():.4f}  gap {gap.mean():+.4f}"
                f"  = {entry['optimisers'][kind]['gap_over_noise']:.1f}x noise  t={t_stat:.1f}"
                f"  {'PASS' if passed else 'fail'} | pooled/floor {pooled.mean() / sigma:.2f}",
                flush=True,
            )
        print(f"  sigma={sigma:<5} persistence {persistence[sigma]:.4f}")
        report["gate"][str(sigma)] = entry


def converged_model(
    model, theta0, sigma, seeds, horizon, epochs=3, batch=32, lr=3e-3, beta=mg.BETA
):
    """Offline AdamW over every seed but the last; the last is held out."""
    train = [agent_blocks(s, sigma, horizon, beta) for s in seeds[:-1]]
    x = torch.cat([d[0].reshape(-1, LENGTH - 1) for d in train])
    y = torch.cat([d[1].reshape(-1, LENGTH - 1) for d in train])
    learner = Learner(model, theta0, 1, "adamw", lr)
    order = torch.Generator().manual_seed(0)
    for _epoch in range(epochs):
        perm = torch.randperm(x.shape[0], generator=order)
        for start in range(0, x.shape[0] - batch + 1, batch):
            idx = perm[start : start + batch]
            learner.step(x[idx].unsqueeze(0), y[idx].unsqueeze(0))
    held_x, held_y = agent_blocks(seeds[-1], sigma, horizon, beta)
    return learner, held_x.reshape(-1, LENGTH - 1), held_y.reshape(-1, LENGTH - 1)


def residuals(model, theta0, seeds, horizon, report) -> dict:
    print("\n== 3. residual structure of a converged model ==")
    report["residuals"], models = {}, {}
    for sigma in SIGMAS:
        learner, hx, hy = converged_model(model, theta0, sigma, seeds, horizon)
        with torch.no_grad():
            r = (learner.predict(hx.unsqueeze(0))[0] - hy).numpy()
        variance = r.var(axis=0)
        corr = np.corrcoef(r, rowvar=False)
        rho1 = float(np.mean(np.diag(corr, k=1)))
        entry = {
            "rmse": float(np.sqrt((r**2).mean())),
            "mean_variance": float(variance.mean()),
            "variance_by_position": variance.tolist(),
            "variance_ratio": float(variance.max() / variance.min()),
            "variance_ratio_from_position_2": float(variance[1:].max() / variance[1:].min()),
            "rho1": rho1,
            "rho1_by_position": np.diag(corr, k=1).tolist(),
            "diagonal_R_ok": abs(rho1) < 0.2,
            "per_position_R": float(variance.max() / variance.min()) > 2.0,
        }
        report["residuals"][str(sigma)] = entry
        models[sigma] = learner
        print(
            f"  sigma={sigma:<5} rmse {entry['rmse']:.4f} (floor {sigma}) | rho1 {rho1:+.3f}"
            f" {'diag ok' if entry['diagonal_R_ok'] else 'NOT diagonal'} | variance ratio"
            f" {entry['variance_ratio']:.2f} ({entry['variance_ratio_from_position_2']:.2f} from pos 2)"
            f" | R centre {entry['mean_variance']:.3e}",
            flush=True,
        )
    return models


def lyapunov(beta: float, span: int = 3000) -> float:
    """Divergence rate of two histories 1e-8 apart, per time unit."""
    pair = mg.integrate(span, [0.9, 0.9 + 1e-8], beta=beta, burn_in=0.0)
    gap = np.abs(pair[0] - pair[1]) + 1e-300
    window = (gap > 1e-7) & (gap < 1e-3)
    idx = np.nonzero(window)[0]
    if idx.size < 50:
        return float(np.polyfit(np.arange(span), np.log(gap), 1)[0])
    return float(np.polyfit(idx, np.log(gap[idx]), 1)[0])


def beta_map(models, report, quick: bool) -> None:
    print("\n== 4. the beta map ==")
    learner = models[DAMAGE_SIGMA]
    report["beta_map"] = []
    base = None
    for beta in BETAS:
        rng = np.random.default_rng([999, int(beta * 1000)])
        clean = mg.integrate((50 if quick else 200) * LENGTH, mg.initial_histories(5, rng), beta=beta)
        z = mg.observe(mg.standardise(clean), DAMAGE_SIGMA, rng)
        x, y = mg.to_blocks(z, LENGTH)
        x = torch.tensor(x.reshape(-1, LENGTH - 1), dtype=DTYPE)
        y = torch.tensor(y.reshape(-1, LENGTH - 1), dtype=DTYPE)
        with torch.no_grad():
            rmse = float(((learner.predict(x.unsqueeze(0))[0] - y) ** 2).mean().sqrt())
        if abs(beta - mg.BETA) < 1e-9:
            base = rmse
        entry = {
            "beta": beta,
            "lyapunov": lyapunov(beta, 1000 if quick else 3000),
            "clean_mean": float(clean.mean()),
            "clean_std": float(clean.std()),
            "rmse_at_beta": rmse,
        }
        report["beta_map"].append(entry)
    for entry in report["beta_map"]:
        entry["damage"] = entry["rmse_at_beta"] - base
        print(
            f"  beta={entry['beta']:.2f}  lyapunov {entry['lyapunov']:+.4f}  mean {entry['clean_mean']:.3f}"
            f"  std {entry['clean_std']:.3f}  rmse {entry['rmse_at_beta']:.4f}  damage {entry['damage']:+.4f}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="a smoke run at a tiny horizon")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (leave the GPU sweep its core)")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    horizon = 60 if args.quick else HORIZON
    seeds = SEEDS[:3] if args.quick else SEEDS
    tune = TUNE_SEEDS[:1] if args.quick else TUNE_SEEDS
    if args.quick:
        for kind in LEARNING_RATES:
            LEARNING_RATES[kind] = LEARNING_RATES[kind][1:3]

    started = time.time()
    report: dict = {"horizon": horizon, "seeds": seeds, "quick": args.quick}
    print("== 1. standardisation constants ==")
    mean, std = mg.reference_moments()
    report["moments"] = {"mean": mean, "std": std}
    print(f"  mean {mean:.6f}  std {std:.6f}  (beta=0.2, 10 x 10 000 samples, seed 0)")

    model = CausalTransformer(dtype=DTYPE)
    theta0 = model.flatten(model.init_params(torch.Generator().manual_seed(0)))
    gate(model, theta0, horizon, seeds, tune, report)
    models = residuals(model, theta0, seeds, horizon, report)
    beta_map(models, report, args.quick)

    report["elapsed_min"] = (time.time() - started) / 60
    OUT.mkdir(parents=True, exist_ok=True)
    name = "report_quick.json" if args.quick else "report.json"
    (OUT / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT / name} in {report['elapsed_min']:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
