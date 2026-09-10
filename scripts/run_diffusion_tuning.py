r"""X20 -- was the diffusion filter mis-tuned, and does a one-hop adapt repair it?

Run this file directly.

    python scripts/run_diffusion_tuning.py --tune    # the (q, sigma_0^2) grid
    python scripts/run_diffusion_tuning.py           # the corrected comparison
    python scripts/run_diffusion_tuning.py --fresh   # discard and redo

`--tune` must run first. X19's `run_diffusion_ekf.py --lr` supplies the \ac{sgd}
learning rates: the conditions and graphs are identical, so re-sweeping them here
would be the same thirty runs under different names. The main pass refuses
without either.

## Why this exists

X19 reported that diffusing the belief costs +0.0228 to +0.0411, growing with
drift severity, and that the filter loses to \ac{atc} under abrupt drift. Both
conclusions rest on two choices that were mine rather than the method's, and D79
records them as defects:

**The filter carried the centralised tuning.** Deliberate --- the X14 discipline,
so that a shortfall is attributable rather than confounded --- but the mechanism
D79 identifies names $q$ as the parameter a local adapt mis-scales, and gives the
direction. Writing $\bm\Omega=\bm P^{-1}$, the centralised filter accumulates
$\sum_{v}\bm\Delta_v$ per step while each diffusing agent accumulates
$\bm\Delta_v$; $q$ was chosen to balance the former against forgetting, and now
faces the latter. It should fall by roughly $N$.

**The baseline was not payload-matched.** X19's \ac{atc} carries momentum with
`mix_optimizer_state: momentum`, so it transmits $2p$ per link against the
filter's $p$ --- exactly the pairing D29 exists to prevent. `atc_plain` is the
matched arm and is added here.

## The grid spans wider than the argument predicts, on purpose

The scaling argument says $q\approx q_{\text{cent}}/N=6\times10^{-6}$. The grid
runs two decades either side of that, and includes the centralised value so the
re-tune can report "no change" if that is the truth. **A scaling argument that
predicts the answer is the worst possible reason to look only where it points**:
if the optimum lands at an edge the argument is wrong in a way a narrow grid
would have hidden, and X13 has already shown $\sigma_0^2$ to be the filter's most
sensitive hyperparameter, with values above $10^{-1}$ diverging rather than
merely slowing.

Tuned at the *harshest* condition, where D79's mechanism predicts the largest
effect and where X19's deficit is widest. Validated at all three, so a setting
that only helps where it was chosen is visible as such.

## What the main pass measures

Three questions, separable because the learners differ one axis at a time:

    is the deficit tuning?          diffusion_ekf at X13's q  vs  at the new q
    was the comparison unfair?      diffusion_ekf             vs  atc_plain (both p)
    does a one-hop adapt repair it? diffusion_ekf             vs  diffusion_ekf_onehop_mean

`diffusion_ekf_full` is **dropped**. X19 measured it at +0.0002 to +0.0007 over
mean-only for 2909x the bandwidth; carrying it again would cost 1.3 GiB of
covariance to re-confirm a null, and the memory is better spent on the one-hop
variant that addresses the actual deficit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_diffusion_ekf import (  # noqa: E402
    BASELINE,
    CONDITIONS,
    EVAL_EVERY,
    HORIZON,
    LOCAL,
    SEEDS,
    TOPOLOGY_NAMES,
    condition_name,
)
from run_ekf_generalization import run_one  # noqa: E402

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: Two decades either side of the scaling argument's 6e-6, with the centralised
#: 6e-5 included so "no change" is an expressible outcome.
PROCESS_NOISE = [6.0e-4, 6.0e-5, 6.0e-6, 6.0e-7, 6.0e-8]

#: X13 found values above 1e-1 diverge rather than merely slow, so 0.1 is the
#: ceiling; 0.001 is two decades below the centralised selection.
PRIOR_SCALES = [0.1, 0.03, 0.01, 0.003, 0.001]

GAMMA = 0.9995
TUNE_CONDITION = "every25_jump15"
TUNE_TOPOLOGY = "erdos_renyi"
TUNE_SEEDS = [0, 1, 2]

#: The matched arm X19 omitted. Plain SGD, nothing mixed, so its payload is
#: exactly p -- the same as `diffusion_ekf`'s.
PLAIN = {"name": "diffusion_sgd_atc_plain", "optimizer": "sgd", "momentum": 0.0,
         "mix_optimizer_state": "none"}

DEVICE = "auto"
DTYPE = "float64"
FRESH = False

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x20_status.json"
SELECTED = ROOT / "results" / "x20_selected.json"


def drift_for(condition: str) -> dict | None:
    return dict(CONDITIONS)[condition]


def tune_run_name(q: float, prior: float) -> str:
    return f"x20_tune_q{q:g}_s{prior:g}".replace(".", "p").replace("-", "m")


def cell_name(condition: str, topology: str, twin: bool = False) -> str:
    stem = condition_name(condition, topology)
    return f"x20_control_{stem}" if twin else f"x20_{stem}"


def settled(run: str, learner: str) -> float:
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    if not (directory / "_complete").exists():
        return float("inf")
    files = sorted(directory.glob("seed_*.parquet"))
    if not files:
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"])
         for f in files], ignore_index=True)
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == "error_rate")
                 & (frame["evalset"] == "current") & (frame["t"] >= int(0.8 * HORIZON))]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def filter_entry(name: str, q: float, prior: float) -> dict:
    return {"name": name, "transition": "scalar", "gamma": GAMMA, "lambda_forget": 1.0,
            "process_noise_q": q, "prior_scale": prior}


def config_for(name: str, drift: dict | None, topology: str, entries: list[dict],
               seeds: list[int] | None = None):
    from run_diffusion_ekf import TOPOLOGY_PARAMS  # noqa: PLC0415

    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": HORIZON, "eval_every": EVAL_EVERY,
                    "seeds": seeds or SEEDS, "device": DEVICE, "dtype": DTYPE},
            "graph": {"topology": topology, "params": dict(TOPOLOGY_PARAMS[topology])},
            "env": {"drift": block},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def selected_setting() -> dict | None:
    """The (q, sigma_0^2) the grid chose, or None if it has not run."""
    scored = []
    for q in PROCESS_NOISE:
        for prior in PRIOR_SCALES:
            value = settled(tune_run_name(q, prior), "diffusion_ekf")
            if value != float("inf"):
                scored.append((value, q, prior))
    if not scored:
        return None
    value, q, prior = min(scored)
    return {"process_noise_q": q, "prior_scale": prior, "settled": value,
            "cells": len(scored), "of": len(PROCESS_NOISE) * len(PRIOR_SCALES)}


def tune(train, test, fresh: bool) -> int:
    status = load_status()
    total = len(PROCESS_NOISE) * len(PRIOR_SCALES)
    print(f"X20 tuning: {len(PROCESS_NOISE)} q x {len(PRIOR_SCALES)} sigma_0^2 at "
          f"{len(TUNE_SEEDS)} seeds, on {TUNE_CONDITION}/{TUNE_TOPOLOGY}\n", flush=True)
    started, index = time.time(), 0
    for q in PROCESS_NOISE:
        for prior in PRIOR_SCALES:
            index += 1
            name = tune_run_name(q, prior)
            note = run_one(
                config_for(name, drift_for(TUNE_CONDITION), TUNE_TOPOLOGY,
                           [filter_entry("diffusion_ekf", q, prior)], seeds=TUNE_SEEDS),
                train, test, fresh)
            status[name] = note
            save_status(status)
            print(f"[{index}/{total}] {name:<34} {note:<28} "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\ntuning complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'q \\ sigma_0^2':>14}" + "".join(f"{p:>10g}" for p in PRIOR_SCALES))
    for q in PROCESS_NOISE:
        line = f"  {q:>14g}"
        for prior in PRIOR_SCALES:
            value = settled(tune_run_name(q, prior), "diffusion_ekf")
            line += f"{value:>10.4f}" if value != float("inf") else f"{'div':>10}"
        print(line)

    chosen = selected_setting()
    if chosen:
        SELECTED.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
        edge = (chosen["process_noise_q"] in (PROCESS_NOISE[0], PROCESS_NOISE[-1])
                or chosen["prior_scale"] in (PRIOR_SCALES[0], PRIOR_SCALES[-1]))
        print(f"\n  selected q={chosen['process_noise_q']:g}, "
              f"sigma_0^2={chosen['prior_scale']:g}  ({chosen['settled']:.4f})")
        print("  X13's centralised choice was q=6e-05, sigma_0^2=0.01")
        if edge:
            print("\n  !! the optimum is on a GRID EDGE. Widen PROCESS_NOISE or "
                  "PRIOR_SCALES and re-run;\n     an argmin at the boundary is a "
                  "truncated grid, not a selection.")
    return 0


def main(fresh: bool = FRESH, tune_only: bool = False) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)
    if tune_only:
        return tune(train, test, fresh)

    chosen = selected_setting()
    if chosen is None or chosen["cells"] < chosen["of"]:
        have = 0 if chosen is None else chosen["cells"]
        print(f"\nThe (q, sigma_0^2) grid has {have} of "
              f"{len(PROCESS_NOISE) * len(PRIOR_SCALES)} cells. Without it the filter\n"
              "would carry the centralised tuning again, which is the defect this\n"
              "experiment exists to remove (design note D79).\n"
              "  python scripts/run_diffusion_tuning.py --tune\n")
        return 1

    from run_diffusion_ekf import selected_rates  # noqa: PLC0415

    wanted = [(c, t) for c, _d in CONDITIONS for t in TOPOLOGY_NAMES]
    rates = {(c, t): selected_rates(c, t) for c, t in wanted}
    missing = sorted(condition_name(c, t) for (c, t), r in rates.items() if r is None)
    if missing:
        print(f"\nX19's learning-rate sweep is needed for {missing}:\n"
              "  python scripts/run_diffusion_ekf.py --lr\n")
        return 1

    q, prior = chosen["process_noise_q"], chosen["prior_scale"]
    print(f"\nX20: re-tuned filter at q={q:g}, sigma_0^2={prior:g}")
    print("     (X13's centralised choice was q=6e-05, sigma_0^2=0.01)\n")

    cells: list[tuple[str, dict | None, str, list[dict]]] = []
    for condition, drift in CONDITIONS:
        for topology in TOPOLOGY_NAMES:
            entries = [
                {"name": "centralized_ekf_gamma", "transition": "scalar", "gamma": GAMMA,
                 "lambda_forget": 1.0, "process_noise_q": 6.0e-5, "prior_scale": 0.01},
                filter_entry("diffusion_ekf", q, prior),
                filter_entry("diffusion_ekf_onehop_mean", q, prior),
                {**BASELINE, "lr": rates[(condition, topology)][BASELINE["name"]]},
                {**PLAIN, "lr": rates[(condition, topology)][BASELINE["name"]]},
                {**LOCAL, "lr": rates[(condition, topology)][LOCAL["name"]]},
            ]
            cells.append((cell_name(condition, topology), drift, topology, entries))
            if drift is not None:
                cells.append((cell_name(condition, topology, twin=True), None,
                              topology, entries))

    print(f"  {len(cells)} cells at {len(SEEDS)} seeds, T={HORIZON}")
    for name, _d, topology, _e in cells:
        print(f"    {name:<40}{topology}")
    print(flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (name, drift, topology, entries) in enumerate(cells, start=1):
        note = run_one(config_for(name, drift, topology, entries), train, test, fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(cells) - index)) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<40} {note:<28} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nX20 complete in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv, tune_only="--tune" in sys.argv))
