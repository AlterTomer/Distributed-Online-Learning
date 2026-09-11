r"""X22 -- how much of the network's evidence may an agent claim it has?

Run this file directly.

    python scripts/run_diffusion_extrapolation.py          # the alpha sweep
    python scripts/run_diffusion_extrapolation.py --fresh  # discard and redo

## The idea

The centralised filter accumulates $\sum_{u=1}^{N}\bm\Delta_u\approx
N\bar{\bm\Delta}$ per step; an agent holding $\lvert\mathcal M_v\rvert$ of them
has an unbiased estimator of that total after multiplying by
$c=N/\lvert\mathcal M_v\rvert$. Right mean, $c$ times the variance. $N$ is a
global constant an agent can be told even when it cannot know the graph.

`information_exponent` is $\alpha$ in $c=(N/\lvert\mathcal M_v\rvert)^{\alpha}$,
and **both** the information and the score are scaled by it --- scaling
$\bm\Delta$ alone would shrink $\bm P^{\psi}$ while leaving the step as gathered,
making every update $c$ times too small.

## Why an interior optimum is expected rather than $\alpha=1$

The two ends fail in opposite directions, which is the whole reason this is a
dial rather than a switch:

* $\alpha=0$ is what D80 measured as **under**-confident: \ac{ece} 0.085 at X19's
  tuning, the filter claiming 0.78 where it delivered 0.87.
* $\alpha=1$ asserts $N$ agents' certainty from one agent's batch. The estimator
  is unbiased but its variance is $N$ times higher, and an \ac{ekf} that believes
  a covariance smaller than its actual error is the over-confident regime D61
  documents --- where the Gauss-Newton step leaves the region its linearisation
  describes and does not come back.

So the question is not whether to extrapolate but how far, and the answer is
measured rather than argued.

## What this can and cannot conclude

⚠ The estimator is unbiased **only under exchangeable agents**. This sweep runs
\ac{iid} partitions, where that holds. Under Dirichlet skew it does not: an agent
holding three classes would claim the network's confidence about all ten, and the
bias would be in the direction that makes the filter confidently wrong. **P5.2 is
where this has to be re-measured**, and a good result here says nothing about
that case.

Run at the harshest condition and on \ac{er}, where X20 found the largest deficit
and the largest one-hop gain. Both adapt scopes are carried: under a local adapt
$c=N$, under one-hop $c=N/\lvert\mathcal M_v\rvert\approx N/3.2$, so the two
should want different $\alpha$ and carrying only one would answer half the
question.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ekf_generalization import run_one  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: 0 is what X20 measured. 1 is full extrapolation. The interior is where the
#: answer is expected, so it is sampled more finely than the ends.
EXPONENTS = [0.0, 0.25, 0.5, 0.75, 1.0]

#: Both scopes. A local adapt extrapolates by N; one-hop by N/|M_v|, which on
#: this graph is about N/3.2 -- so they should want different alpha, and the
#: sweep would answer half the question if it carried only one.
LEARNERS = ["diffusion_ekf", "diffusion_ekf_onehop_mean"]

#: X20's selection, held fixed. Changing two things at once would leave any
#: result unattributable, and alpha is the axis under test.
GAMMA, PROCESS_NOISE, PRIOR_SCALE = 0.9995, 6.0e-4, 1.0e-3

CONDITION = {"schedule": "recurring", "jump_every": 25, "jump_degrees": 15.0, "jump_seed": 0}
TOPOLOGY = ("erdos_renyi", {"p": 0.3})

HORIZON = 1500
EVAL_EVERY = 5
SEEDS = [0, 1, 2]

DEVICE = "auto"
DTYPE = "float64"
FRESH = False
DATASET = "mnist"

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x22_status.json"


def run_name(alpha: float) -> str:
    return f"x22_alpha{alpha:g}".replace(".", "p")


def settled(run: str, learner: str, metric: str = "error_rate") -> float:
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
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                 & (frame["t"] >= int(0.8 * HORIZON))]
    if metric == "error_rate":
        rows = rows[rows["evalset"] == "current"]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def config_for(alpha: float):
    topology, params = TOPOLOGY
    entries = [
        {"name": name, "transition": "scalar", "gamma": GAMMA, "lambda_forget": 1.0,
         "process_noise_q": PROCESS_NOISE, "prior_scale": PRIOR_SCALE,
         "information_exponent": alpha}
        for name in LEARNERS
    ]
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": run_name(alpha), "horizon": HORIZON, "eval_every": EVAL_EVERY,
                    "seeds": SEEDS, "device": DEVICE, "dtype": DTYPE},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": DATASET, "drift": dict(CONDITION)},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def main(fresh: bool = FRESH) -> int:
    if not dataset_is_cached(DATASET, DATA_ROOT):
        print(f"{DATASET} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(DATASET, DATA_ROOT, download=False)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    print(f"X22: {len(EXPONENTS)} values of alpha at {len(SEEDS)} seeds, "
          f"both adapt scopes in each cell")
    print(f"     q={PROCESS_NOISE:g}, sigma_0^2={PRIOR_SCALE:g}, gamma={GAMMA:g} "
          f"(X20's selection, held)\n", flush=True)

    started = time.time()
    for index, alpha in enumerate(EXPONENTS, start=1):
        note = run_one(config_for(alpha), train, test, fresh)
        status[run_name(alpha)] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(EXPONENTS)}] alpha={alpha:<5g} {note:<28} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nX22 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    r"""Error and the two quantities that say *why*: confidence and its calibration."""
    for metric, label in (("error_rate", "settled error"),
                          ("ece", "ECE"),
                          ("overconfidence", "overconfidence (+ = claims too much)"),
                          ("theta_mean_norm_sq", "|theta|^2")):
        print(f"  {label}")
        print(f"    {'alpha':>8}" + "".join(f"{name.replace('diffusion_ekf', 'diff'):>26}"
                                            for name in LEARNERS))
        for alpha in EXPONENTS:
            line = f"    {alpha:>8g}"
            for name in LEARNERS:
                directory = ROOT / "results" / run_name(alpha)
                if (directory / "_diverged").exists():
                    line += f"{'DIV':>26}"
                    continue
                value = settled(run_name(alpha), name, metric)
                line += f"{value:>26.4f}" if value != float("inf") else f"{'-':>26}"
            print(line)
        print()

    for name in LEARNERS:
        scored = [(settled(run_name(a), name), a) for a in EXPONENTS]
        scored = [(v, a) for v, a in scored if v != float("inf")]
        if not scored:
            continue
        value, alpha = min(scored)
        edge = alpha in (EXPONENTS[0], EXPONENTS[-1])
        print(f"  {name}: best alpha={alpha:g} -> {value:.4f}"
              + ("   (at an END of the range -- alpha is bounded by its meaning, "
                 "so this is a real answer, not a truncated grid)" if edge else ""))


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv))
