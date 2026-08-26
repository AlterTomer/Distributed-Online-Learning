r"""X13 -- read the EKF tuning sweep.

Run this file directly.

    python scripts/report_ekf_sweep.py

Reports the **settled** error, the mean over the last fifth of the run, because
that is the quantity tuning is about: a filter that converges fast and then
tracks badly is not the one to carry into the diffusion version, and the whole
curve would let the early advantage hide the late failure.

Every cell is compared against `x13_baselines`, which ran the SGD learners under
the same drift with the same seeds. Given a seed the environment and
$\bm\theta_0$ do not depend on the learners list, so the comparison is **paired
by construction** rather than by seed -- the same subtraction every break in this
project is defined over (design note D54).

Diverged cells are printed as such rather than omitted. An empty square in the
grid means "not run"; a `div` means "run, and the answer was divergence", and
those are different facts (design note D61).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from run_ekf_sweep import (  # noqa: E402
    ALPHA,
    GAMMAS,
    HORIZON,
    LAMBDAS,
    PRIOR_SCALES,
    PROCESS_NOISES,
    STATUS,
    cell_name,
    cells,
)

#: The last fifth of the run. Matches the window X12's damage table used, so the
#: two are directly comparable.
SETTLED_FROM = int(0.8 * HORIZON)
RESULTS = ROOT / "results"


def settled_error(directory: str, learner: str | None = None) -> float | None:
    """Mean `current` error over the settled window, averaged across seeds."""
    path = RESULTS / directory
    files = sorted(path.glob("*.parquet"))
    if not files:
        return None
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    mask = (
        (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
        & (frame["t"] >= SETTLED_FROM)
    )
    if learner is not None:
        mask &= frame["learner"] == learner
    rows = frame[mask]
    return float(rows["value"].mean()) if len(rows) else None


def format_cell(value: float | None, note: str, best: float | None) -> str:
    if note.startswith("diverged"):
        return "   div"
    if value is None:
        return "     ."
    marker = "*" if best is not None and abs(value - best) < 1e-9 else " "
    return f"{value:.4f}{marker}"


def main() -> int:
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    if not status:
        print(f"No sweep results yet. Expected {STATUS}")
        return 1

    grid = cells()
    scores: dict[str, float | None] = {}
    for cell in grid:
        name = cell_name(cell)
        scores[name] = settled_error(name) if status.get(name) in ("ok", "cached") else None

    print(f"X13 -- centralised EKF under linear drift, alpha={ALPHA} deg/step, T={HORIZON}")
    print(f"settled error = mean `current` error over t >= {SETTLED_FROM}\n")

    print("baselines, same drift and seeds:")
    baseline_scores = {}
    for learner in ("centralized_sgd", "diffusion_sgd_atc", "frozen_atc"):
        value = settled_error("x13_baselines", learner)
        baseline_scores[learner] = value
        print(f"  {learner:<22} {value:.4f}" if value is not None else f"  {learner:<22} .")
    reference = baseline_scores.get("centralized_sgd")

    live = [v for v in scores.values() if v is not None]
    best = min(live) if live else None

    print(f"\n=== gamma family ===  ({len(GAMMAS)} gammas x {len(PROCESS_NOISES)} Q)")
    for prior in PRIOR_SCALES:
        print(f"\n  sigma_0^2 = {prior:g}")
        print("      Q ->  " + "".join(f"{q:>9g}" for q in PROCESS_NOISES))
        for gamma in GAMMAS:
            row = []
            for noise in PROCESS_NOISES:
                cell = {
                    "name": "centralized_ekf_gamma", "transition": "scalar", "gamma": gamma,
                    "process_noise_q": noise, "lambda_forget": 1.0, "prior_scale": prior,
                }
                name = cell_name(cell)
                row.append(f"{format_cell(scores.get(name), status.get(name, ''), best):>9}")
            print(f"  gamma {gamma:<7g}" + "".join(row))

    print(f"\n=== lambda family ===")
    print("   sigma_0^2 ->" + "".join(f"{p:>9g}" for p in PRIOR_SCALES))
    for lam in LAMBDAS:
        row = []
        for prior in PRIOR_SCALES:
            cell = {
                "name": "centralized_ekf_lambda", "transition": "identity", "gamma": 1.0,
                "process_noise_q": 0.0, "lambda_forget": lam, "prior_scale": prior,
            }
            name = cell_name(cell)
            row.append(f"{format_cell(scores.get(name), status.get(name, ''), best):>9}")
        print(f"  lambda {lam:<6g}" + "".join(row))

    ranked = sorted(
        ((value, name) for name, value in scores.items() if value is not None)
    )
    print("\n=== best five ===")
    for value, name in ranked[:5]:
        against = f"  ({reference - value:+.4f} vs centralized_sgd)" if reference else ""
        print(f"  {name:<36} {value:.4f}{against}")

    diverged = [name for name, note in status.items() if note.startswith("diverged")]
    print(f"\n{len(ranked)} cells ran, {len(diverged)} diverged")
    for name in diverged[:8]:
        print(f"  div  {name}")
    if len(diverged) > 8:
        print(f"  ... and {len(diverged) - 8} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
