r"""X21 -- the knob X20 did not sweep: gamma, jointly with q.

Run this file directly.

    python scripts/run_diffusion_gamma.py          # the joint (gamma, q) grid
    python scripts/run_diffusion_gamma.py --fresh  # discard and redo

**A separate script rather than a mode on `run_diffusion_tuning.py`**, because
that script was executing when this one was written. Python imports a module
once, so editing a file under a running process produces errors about mismatches
that no longer exist on disk, with tracebacks pointing at the wrong lines
(`docs/howto.md`). Adding a `--gamma` mode there can happen once X20 is finished;
the experiment should not wait for that.

## Why

D80. The diffusion filter is the worst-calibrated method measured --- \ac{ece}
0.085 against \ac{atc}'s 0.015 --- and it is **under**-confident, claiming 0.78
where it delivers 0.87. The cause is not agent disagreement, which matches
\ac{atc}'s almost exactly. It is the parameter norm: 34.8 against \ac{atc}'s 79.6,
and confidence tracks $\lVert\bm\theta\rVert^2$ across every method on the page.

$\gamma$ explains that. At $\gamma=0.9995$ the mean is multiplied by
$\gamma^{1500}=0.47$ over a run absent information --- D26's point that $\gamma$ is
L2 weight decay in state-space form. The centralised filter shrugs it off and
reaches 84.5, having ten agents' information per step to push back with; the
diffusion filter, with $1/N$ of it, loses the tug-of-war. Same shrinkage,
different capacity to resist it --- a third observable of D79's deficit.

X20 re-tunes $q$ and $\sigma_0^2$ on exactly that argument and **holds $\gamma$ at
the value X13 chose for a filter seeing ten times the data**. So its setting is
tuned in two of three dimensions.

## Why the grid is joint, and not a $\gamma$ line at X20's $q$

$\gamma$ acts on both moments:

    m <- gamma * m                 P <- gamma^2 * P + Q

so $\gamma^2<1$ is a **contraction on the covariance**, and apart from the
information update it is the only one. Setting $\gamma=1$ removes it entirely.
X20's grid located the divergence cliff at $q=6\times10^{-3}$ *with* that
contraction present; without it the cliff moves down, and X20's selected
$q=6\times10^{-4}$ may sit the wrong side of it.

Sweeping $\gamma$ alone at a fixed $q$ would therefore answer a question nobody
asked: it would compare a stable configuration against an unstable one and
conclude that $\gamma<1$ is necessary, when what it measured was that *this* $q$
needs *that* contraction. The grid is crossed so the cliff can be located as a
function of $\gamma$ rather than assumed fixed.

$\sigma_0^2$ is held at $10^{-3}$. X20 measured that row as flat --- 0.1173 to
0.1192 across two decades, rising then falling rather than trending --- so it is
the one axis that has earned being fixed.

## What a result looks like

If $\gamma=1$ wins at a smaller $q$, then X20's large $q$ was compensating for
shrinkage rather than for a miscalibrated covariance, and the "nearly memoryless
filter" reading of D79 is wrong. If $\gamma<1$ still wins once $q$ is free to
move, the shrinkage is paying for itself and D80's mechanism is incomplete.
Divergence is a measurement either way (D61) and maps the cliff.
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

#: 1.0 is the driftless random walk -- no mean shrinkage, and no covariance
#: contraction either. 0.9995 is X13's and X20's value. 0.999 brackets the far
#: side: gamma^1500 = 0.22, shrinkage so strong it should be visibly worse if the
#: mechanism is right.
GAMMAS = [1.0, 0.9999, 0.9995, 0.999]

#: Spanning X20's selection and two decades below it, plus the cell where X20
#: found divergence at gamma = 0.9995. Whether 6e-3 still diverges at gamma = 1
#: -- and whether 6e-4 starts to -- is the interaction this grid exists to measure.
PROCESS_NOISE = [6.0e-3, 6.0e-4, 6.0e-5, 6.0e-6]

#: Fixed. X20 measured this row as flat across two decades, non-monotone, with a
#: total range of 0.0019 -- the one axis that has earned being held.
PRIOR_SCALE = 0.001

CONDITION = {"schedule": "recurring", "jump_every": 25, "jump_degrees": 15.0, "jump_seed": 0}
TOPOLOGY = ("erdos_renyi", {"p": 0.3})
LEARNER = "diffusion_ekf"

HORIZON = 1500
EVAL_EVERY = 5
SEEDS = [0, 1, 2]

DEVICE = "auto"
DTYPE = "float64"
FRESH = False

#: The dataset every cell consumes. A name, not an import (IMPLEMENTATION.md 15).
DATASET = "mnist"

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x21_status.json"


def run_name(gamma: float, q: float) -> str:
    return f"x21_g{gamma:g}_q{q:g}".replace(".", "p").replace("-", "m")


def settled(run: str, metric: str = "error_rate") -> float:
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
    rows = frame[(frame["learner"] == LEARNER) & (frame["metric"] == metric)
                 & (frame["t"] >= int(0.8 * HORIZON))]
    if metric == "error_rate":
        rows = rows[rows["evalset"] == "current"]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def config_for(name: str, gamma: float, q: float):
    topology, params = TOPOLOGY
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": HORIZON, "eval_every": EVAL_EVERY,
                    "seeds": SEEDS, "device": DEVICE, "dtype": DTYPE},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": DATASET, "drift": dict(CONDITION)},
            "learners": [{
                "name": LEARNER, "transition": "scalar", "gamma": gamma,
                "lambda_forget": 1.0, "process_noise_q": q, "prior_scale": PRIOR_SCALE,
            }],
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def main(fresh: bool = FRESH) -> int:
    if not dataset_is_cached(DATASET, DATA_ROOT):
        print(f"{DATASET} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(DATASET, DATA_ROOT, download=False)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    cells = [(g, q) for g in GAMMAS for q in PROCESS_NOISE]
    print(f"X21: {len(GAMMAS)} gamma x {len(PROCESS_NOISE)} q = {len(cells)} cells "
          f"at {len(SEEDS)} seeds, sigma_0^2 fixed at {PRIOR_SCALE:g}")
    print(f"     condition {CONDITION['jump_degrees']:g} deg every "
          f"{CONDITION['jump_every']} steps on {TOPOLOGY[0]}\n", flush=True)

    started = time.time()
    for index, (gamma, q) in enumerate(cells, start=1):
        name = run_name(gamma, q)
        note = run_one(config_for(name, gamma, q), train, test, fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(cells)}] {name:<26} {note:<28} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nX21 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    """Error, and the two quantities D80 says explain it."""
    for metric, label in (("error_rate", "settled error"),
                          ("theta_mean_norm_sq", "|theta|^2"),
                          ("ece", "ECE"),
                          ("mean_confidence", "mean confidence")):
        print(f"  {label}")
        print(f"    {'gamma \\ q':>12}" + "".join(f"{q:>12g}" for q in PROCESS_NOISE))
        for gamma in GAMMAS:
            line = f"    {gamma:>12g}"
            for q in PROCESS_NOISE:
                directory = ROOT / "results" / run_name(gamma, q)
                if (directory / "_diverged").exists():
                    line += f"{'DIV':>12}"
                    continue
                value = settled(run_name(gamma, q), metric)
                line += f"{value:>12.4f}" if value != float("inf") else f"{'-':>12}"
            print(line)
        print()

    best = [(settled(run_name(g, q)), g, q) for g in GAMMAS for q in PROCESS_NOISE]
    best = [(v, g, q) for v, g, q in best if v != float("inf")]
    if best:
        value, gamma, q = min(best)
        print(f"  best: gamma={gamma:g}, q={q:g} -> {value:.4f}")
        print("  X20 selected gamma=0.9995, q=0.0006 (that grid did not vary gamma)")
        if gamma in (GAMMAS[0], GAMMAS[-1]) or q in (PROCESS_NOISE[0], PROCESS_NOISE[-1]):
            print("\n  !! on a grid edge -- widen before believing it.")


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv))
