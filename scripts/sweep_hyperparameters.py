r"""Per-method tuning: learning rate, samples per step, and whether momentum helps.

Run this file directly.

    python scripts/sweep_hyperparameters.py

**Why this exists.** The X1 primary (SGD momentum 0.9 at lr 0.05) is unstable at
$n = 2$: the effective step is $\eta/(1-\beta) = 0.5$, and a single agent lands at
chance while the same agent at lr 0.005 reaches 0.188. Averaging over $N = 10$
agents cancels enough of that noise to keep the diffusion methods stable, so
`local_only` looked catastrophically worse than it is. Reporting that gap as the
value of cooperation would be reporting a tuning artefact. The giveaway was
`diffusion_sgd_atc_plain` finishing *ahead of* `centralized_sgd`, which pools
every agent's samples and cannot legitimately be beaten by a distributed method.

So each method is tuned on its own grid before any headline comparison is drawn.

## The grid

    lr         0.2, 0.05, 0.01, 0.005, 0.001
    n          2, 4, 6, 8, 10          samples per agent per step
    optimizer  sgd, sgd_momentum       swept, not assumed

`diffusion_sgd_atc` under `optimizer: sgd` is exactly `diffusion_sgd_atc_plain`,
so the optimizer axis subsumes that learner and it is not listed separately.

## Two constraints the grid is shaped by

**The shard budget.** $N n T \le 60000$ -- MNIST's training split, with no agent
seeing a sample twice. At $T = 1500$ that caps $n$ at 4, so the sweep runs at
$T = 600$, where $n = 10$ lands at exactly 60 000 and all five cells stay
epoch-free. The alternative -- `allow_epochs` for the large-$n$ cells -- would let
them train on repeated data while $n = 2$ did not, biasing the sweep in favour of
the very axis being measured.

**$n$ is swept at fixed $T$, not at fixed $nT$.** Larger $n$ therefore means more
total data, which is the intended question: $n$ is a property of the deployment
(how fast each agent samples) and $T$ is the horizon, so "does sampling faster
help, at a fixed number of communication rounds?" is what a practitioner is
choosing between. Holding $nT$ fixed would ask a different question -- bigger
chunks versus more rounds -- and is a separate sweep.

Selection is on the **held-out** `current` set over the last 100 steps, not on
the prequential stream: prequential error is what the tuning would be reported
against later, and picking on it selects for the noise in that estimate.

Results append to `results/sweep/cells.jsonl` as each cell finishes, so an
interrupted sweep resumes rather than restarting.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dekf_bench.data.mnist import load_mnist  # noqa: E402
from dekf_bench.env.environment import build_environment  # noqa: E402
from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: E402
from dekf_bench.learners.registry import build_learners  # noqa: E402
from dekf_bench.likelihoods.categorical import Categorical  # noqa: E402
from dekf_bench.models.registry import build_model_from_config  # noqa: E402
from dekf_bench.runner import simulate  # noqa: E402
from dekf_bench.utils.config import default_configs_dir, load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
LEARNING_RATES = [0.2, 0.05, 0.01, 0.005, 0.001]
SAMPLES_PER_STEP = [2, 4, 6, 8, 10]
OPTIMIZERS = ["sgd", "sgd_momentum"]
#: None keeps each config's own topology. A list sweeps it -- used for X3,
#: where the question is whether a denser graph tolerates a larger step: it
#: averages over more neighbours per round, which is the same noise-reduction
#: mechanism that let ATC survive an lr that killed local_only (D39).
TOPOLOGIES: list[str] | None = None
#: None keeps the config's own label availability. A list sweeps it -- used to
#: tune X4 per cell, where pi_lab changes each method's *effective* step: an
#: idle agent contributes its unchanged theta to the combine, so ATC's step is
#: eta * n_active/N while centralized takes the full eta (results.md 9.1).
LABEL_AVAILABILITY: list[float] | None = None
#: Which learners each cell runs. The environment does NOT depend on this list --
#: verified: two runs differing only in learners produce identical observation
#: streams -- so a learner added later can be swept alone under its own --tag and
#: merged, rather than re-running every cell.
LEARNERS = ["centralized_sgd", "diffusion_sgd_atc", "local_only"]
#: Tags a cell with the learner set that produced it. A cell records the
#: learners it ran, so changing LEARNERS without changing TAG would let the
#: resumption logic skip cells that never ran the new learner.
TAG = ""
SEEDS = [0, 1]
HORIZON = 600
SCORE_WINDOW = 100  # last N steps of the held-out curve
FRESH = False  # True discards results/sweep and starts over

OUT = ROOT / "results" / "sweep"
CELLS = OUT / "cells.jsonl"


def cell_key(
    lr: float,
    n: int,
    optimizer: str,
    seed: int,
    topology: str | None = None,
    label_availability: float | None = None,
) -> str:
    suffix = f"|tag={TAG}" if TAG else ""
    topo = f"|topo={topology}" if topology else ""
    pi = f"|pi={label_availability}" if label_availability is not None else ""
    return f"lr={lr}|n={n}|opt={optimizer}|seed={seed}{topo}{pi}{suffix}"


def completed() -> set[str]:
    if not CELLS.exists():
        return set()
    done = set()
    for line in CELLS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["key"])
    return done


def build_config(
    lr: float,
    n: int,
    optimizer: str,
    topology: str | None = None,
    label_availability: float | None = None,
):
    """One grid cell. Every learner takes the swept optimizer and lr.

    `mix_optimizer_state` follows the optimizer because the config rejects a
    stateful optimizer that is left unmixed -- Olshevskyi et al. Fig. 2a, where
    D-Adam keeps moments local and diverges.
    """
    mix = "momentum" if optimizer == "sgd_momentum" else "none"
    entries = [
        {
            "name": name,
            "optimizer": optimizer,
            "lr": lr,
            "momentum": 0.9 if optimizer == "sgd_momentum" else 0.0,
            "mix_optimizer_state": mix,
        }
        for name in LEARNERS
    ]
    overrides: dict = {
        "run": {"horizon": HORIZON},
        "env": {"samples_per_node_per_step": n},
        "learners": entries,
    }
    if label_availability is not None:
        overrides["env"]["label_availability"] = label_availability
    if topology is not None:
        from dekf_bench.env.graph import default_topology_params

        params = default_topology_params(10).get(topology)
        overrides["graph"] = {"topology": topology, "params": params or {}}
    return load_config("x1_stationary", overrides=overrides)


def score(records: list, config) -> dict[str, float]:
    """Held-out error per learner over the last SCORE_WINDOW steps.

    Counts-then-divide, so agents holding more samples are not down-weighted.
    """
    floor = config.run.horizon - SCORE_WINDOW
    totals: dict[str, list[float]] = {}
    for record in records:
        if record.step < floor:
            continue
        for row in record.rows:
            if row.get("evalset") != "current" or row["metric"] != "error_rate":
                continue
            bucket = totals.setdefault(record.learner, [0.0, 0.0])
            bucket[0] += row["n_correct"]
            bucket[1] += row["n_samples"]
    return {
        name: 1.0 - correct / samples for name, (correct, samples) in totals.items() if samples > 0
    }


def run_cell(
    lr: float,
    n: int,
    optimizer: str,
    seed: int,
    data,
    topology: str | None = None,
    label_availability: float | None = None,
) -> dict[str, float]:
    train, test = data
    config = build_config(lr, n, optimizer, topology, label_availability)
    environment = build_environment(config, seed, train)
    model = build_model_from_config(config)
    likelihood = Categorical(10)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    records = simulate.run(
        config,
        environment,
        learners,
        build_evalsets(config, environment, test),
        likelihood,
        theta0,
        verify_observations=False,  # recomputes every observation; not worth it here
    )
    return score(records, config)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if FRESH and CELLS.exists():
        CELLS.unlink()

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    data = load_mnist(default_configs_dir().parent / "data", download=False)

    topologies: list[str | None] = list(TOPOLOGIES) if TOPOLOGIES else [None]
    availabilities: list[float | None] = list(LABEL_AVAILABILITY) if LABEL_AVAILABILITY else [None]
    grid = list(
        itertools.product(
            LEARNING_RATES, SAMPLES_PER_STEP, OPTIMIZERS, SEEDS, topologies, availabilities
        )
    )
    done = completed()
    todo = [c for c in grid if cell_key(*c) not in done]
    print(f"grid: {len(grid)} cells, {len(done)} already done, {len(todo)} to run")
    if not todo:
        print("nothing to do -- set FRESH = True to rerun")
    started = time.perf_counter()

    for index, (lr, n, optimizer, seed, topology, pi_lab) in enumerate(todo, start=1):
        cell_started = time.perf_counter()
        try:
            errors = run_cell(lr, n, optimizer, seed, data, topology, pi_lab)
            failed = None
        except Exception as exception:  # a diverged cell must not kill the sweep
            errors, failed = {}, f"{type(exception).__name__}: {exception}"

        with CELLS.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "key": cell_key(lr, n, optimizer, seed, topology, pi_lab),
                        "lr": lr,
                        "n": n,
                        "optimizer": optimizer,
                        "seed": seed,
                        "topology": topology,
                        "label_availability": pi_lab,
                        "horizon": HORIZON,
                        "errors": errors,
                        "failed": failed,
                        "learners": LEARNERS,
                        "tag": TAG,
                        "seconds": round(time.perf_counter() - cell_started, 1),
                    }
                )
                + "\n"
            )

        elapsed = time.perf_counter() - started
        remaining = elapsed / index * (len(todo) - index)
        summary = (
            failed
            if failed
            else "  ".join(
                f"{k.replace('_sgd', '').replace('diffusion_', '')}={v:.3f}"
                for k, v in sorted(errors.items())
            )
        )
        where = f" {topology:<15}" if topology else ""
        where += f" pi={pi_lab}" if pi_lab is not None else ""
        print(
            f"[{index:>3}/{len(todo)}]{where} lr={lr:<6} n={n:<3} {optimizer:<13} seed={seed}  "
            f"{summary}   (eta {remaining / 60:.0f}m)"
        )

    print(f"\ndone in {(time.perf_counter() - started) / 60:.0f} min -> {CELLS}")
    print("summarise with: python scripts/sweep_hyperparameters.py --report")


def report() -> None:
    """The tuning table: best (lr, n) per learner and optimizer."""
    import pandas as pd

    if not CELLS.exists():
        print("no sweep results yet")
        return
    rows = []
    for line in CELLS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cell = json.loads(line)
        for learner, error in cell["errors"].items():
            rows.append(
                {
                    "learner": learner,
                    "optimizer": cell["optimizer"],
                    "lr": cell["lr"],
                    "n": cell["n"],
                    "seed": cell["seed"],
                    "error": error,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no completed cells")
        return

    averaged = frame.groupby(["learner", "optimizer", "lr", "n"], as_index=False).error.mean()

    print("\n=== best cell per learner x optimizer ===")
    best = averaged.loc[averaged.groupby(["learner", "optimizer"]).error.idxmin()]
    print(best.sort_values(["learner", "error"]).to_string(index=False))

    print("\n=== error vs lr, at the best n for each (learner, optimizer) ===")
    for (learner, optimizer), group in averaged.groupby(["learner", "optimizer"]):
        best_n = int(group.loc[group.error.idxmin(), "n"])
        slice_ = group[group.n == best_n].sort_values("lr")
        line = "  ".join(f"{r.lr}:{r.error:.3f}" for r in slice_.itertuples())
        print(f"  {learner:<20} {optimizer:<14} n={best_n:<3} {line}")

    print("\n=== error vs n, at the best lr for each (learner, optimizer) ===")
    for (learner, optimizer), group in averaged.groupby(["learner", "optimizer"]):
        best_lr = float(group.loc[group.error.idxmin(), "lr"])
        slice_ = group[group.lr == best_lr].sort_values("n")
        line = "  ".join(f"n={r.n}:{r.error:.3f}" for r in slice_.itertuples())
        print(f"  {learner:<20} {optimizer:<14} lr={best_lr:<7} {line}")


if __name__ == "__main__":
    # --tag / --learners let a second sweep (e.g. ATC vs CTA) share this file
    # without editing it, and keep its cells distinct from the main grid.
    if "--tag" in sys.argv:
        TAG = sys.argv[sys.argv.index("--tag") + 1]
    if "--learners" in sys.argv:
        LEARNERS = sys.argv[sys.argv.index("--learners") + 1].split(",")
    if "--topologies" in sys.argv:
        TOPOLOGIES = sys.argv[sys.argv.index("--topologies") + 1].split(",")
    if "--label-availability" in sys.argv:
        LABEL_AVAILABILITY = [
            float(x) for x in sys.argv[sys.argv.index("--label-availability") + 1].split(",")
        ]
    if "--samples" in sys.argv:
        SAMPLES_PER_STEP = [int(x) for x in sys.argv[sys.argv.index("--samples") + 1].split(",")]
    if "--lrs" in sys.argv:
        LEARNING_RATES = [float(x) for x in sys.argv[sys.argv.index("--lrs") + 1].split(",")]
    if "--horizon" in sys.argv:
        HORIZON = int(sys.argv[sys.argv.index("--horizon") + 1])
    if "--report" in sys.argv:
        report()
    else:
        main()
