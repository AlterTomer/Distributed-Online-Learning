r"""The per-position R profile at the chosen law, for both models (D96, decision 28).

    python scripts/measure_r_profile.py                 # beta 0.22, sigma 0.1
    python scripts/measure_r_profile.py --beta 0.22 --sigma 0.1

M0 measured the residual profile at beta = 0.2, the law it trained at. The task
then settled on beta = 0.22 (the centre of the chaotic window), where the amplitude
is larger, so the profile is re-measured there before the R grid is built. The
tuned scale absorbs the level; this fixes the **shape**.

Two models, because they make different errors:

* the **Transformer**, trained offline to convergence exactly as M0 does;
* the **linear AR(31)**, fitted by least squares in closed form over every
  (block, position) pair -- its optimum, with no training noise at all.

Both are fitted on four seeds' ten agents and scored on the fifth. Writes
``results/m0_pilot/r_profile_b<beta>_s<sigma>.json``: the per-position variances
(the profile), their mean (the R centre), the adjacent-position correlation, and
persistence for scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pilot_mackey_glass import (  # noqa: E402
    DTYPE,
    HORIZON,
    LENGTH,
    OUT,
    SEEDS,
    agent_blocks,
    converged_model,
)

from dekf_bench.models.transformer import CausalTransformer  # noqa: E402


def summarise(residuals: np.ndarray) -> dict:
    variance = residuals.var(axis=0)
    corr = np.corrcoef(residuals, rowvar=False)
    return {
        "rmse": float(np.sqrt((residuals**2).mean())),
        "profile": variance.tolist(),
        "r_centre": float(variance.mean()),
        "variance_ratio": float(variance.max() / variance.min()),
        "rho1": float(np.mean(np.diag(corr, k=1))),
        "rho1_by_position": np.diag(corr, k=1).tolist(),
    }


def lagged(x: torch.Tensor) -> torch.Tensor:
    """Row i of each block is [x_i, x_{i-1}, ..., 1]: the linear AR's design."""
    context = x.shape[-1]
    padded = torch.nn.functional.pad(x, (context - 1, 0))
    lags = padded.unfold(-1, context, 1).flip(-1)
    return torch.cat([lags, torch.ones(*lags.shape[:-1], 1, dtype=x.dtype)], dim=-1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    report = {"beta": args.beta, "sigma": args.sigma}

    model = CausalTransformer(dtype=DTYPE)
    theta0 = model.flatten(model.init_params(torch.Generator().manual_seed(0)))
    learner, held_x, held_y = converged_model(
        model, theta0, args.sigma, SEEDS, HORIZON, beta=args.beta
    )
    with torch.no_grad():
        r = (learner.predict(held_x.unsqueeze(0))[0] - held_y).numpy()
    report["transformer"] = summarise(r)

    train = [agent_blocks(s, args.sigma, HORIZON, args.beta) for s in SEEDS[:-1]]
    x = torch.cat([d[0].reshape(-1, LENGTH - 1) for d in train])
    y = torch.cat([d[1].reshape(-1, LENGTH - 1) for d in train])
    design = lagged(x).reshape(-1, LENGTH)
    weights = torch.linalg.lstsq(design, y.reshape(-1, 1)).solution
    fitted = (lagged(held_x) @ weights).squeeze(-1)
    report["linear_ar"] = summarise((fitted - held_y).numpy())
    report["linear_ar"]["weights"] = weights.squeeze(-1).tolist()

    report["persistence_rmse"] = float(((held_y - held_x) ** 2).mean().sqrt())

    for name in ("transformer", "linear_ar"):
        entry = report[name]
        print(
            f"{name:<12} rmse {entry['rmse']:.4f} | R centre {entry['r_centre']:.4e} | "
            f"ratio {entry['variance_ratio']:.2f} | rho1 {entry['rho1']:+.3f}"
        )
    print(f"persistence  rmse {report['persistence_rmse']:.4f} | noise floor {args.sigma}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"r_profile_b{args.beta:g}_s{args.sigma:g}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
