r"""X3 — the topology sweep. Gap to the reference against connectivity.

Run this file directly.

    python scripts/run_topology_sweep.py
    python scripts/run_topology_sweep.py --fresh

One run per topology at the horizon and seed count of the headline experiments,
writing to `results/x3_<topology>/` so any reader can consume them like any
other experiment.

**Each topology runs at its own tuned learning rate**, from
`sweep_hyperparameters.py --tag topo`. A denser graph averages over more
neighbours per round, which is the same noise-reduction mechanism that let ATC
survive a step size that killed `local_only` (design note D39), so holding the
ring's rate fixed would leave F3 partly measuring "how well does the ring's lr
suit a star" rather than connectivity alone.

The measurement said the effect is real but small: six of seven topologies pick
the same cell as the ring. Only `complete` differs, and for a structural reason
— one combine step reaches full consensus there, so ATC *is* centralized (the X0
identity) and inherits its preference for a large plain step.

`centralized_sgd` and `local_only` are included in every run even though neither
reads the graph. They cost little, they give F3 its y-axis (the gap is measured
against centralized), and their invariance across topologies is a free
correctness check: at a fixed seed they should produce identical numbers
everywhere, and a difference would mean the graph is leaking into the data path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from dekf_bench.env.environment import build_environment  # noqa: E402
from dekf_bench.env.graph import build_graph, default_topology_params  # noqa: E402
from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: E402
from dekf_bench.learners.registry import build_learners  # noqa: E402
from dekf_bench.likelihoods.categorical import Categorical  # noqa: E402
from dekf_bench.models.registry import build_model_from_config  # noqa: E402
from dekf_bench.recording import recorder as rec  # noqa: E402
from dekf_bench.recording.schema import RunContext  # noqa: E402
from dekf_bench.runner import simulate  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402
from dekf_bench.utils.determinism import git_revision  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
#: Ordered by spectral gap, so a partial run still spans the axis.
TOPOLOGIES = [
    "path",
    "grid2d",
    "star",
    "ring",
    "watts_strogatz",
    "erdos_renyi",
    "complete",
]

#: From the topology tuning sweep. `complete` is the only departure, and it is
#: the X0 identity showing through rather than an anomaly.
TUNED: dict[str, tuple[str, float]] = {
    "path": ("sgd_momentum", 0.01),
    "grid2d": ("sgd_momentum", 0.01),
    "star": ("sgd_momentum", 0.01),
    "ring": ("sgd_momentum", 0.01),
    "watts_strogatz": ("sgd_momentum", 0.01),
    "erdos_renyi": ("sgd_momentum", 0.01),
    "complete": ("sgd", 0.20),
}

LEARNERS = ["centralized_sgd", "diffusion_sgd_atc", "diffusion_sgd_atc_plain", "local_only"]
FRESH = False
PROGRESS_EVERY = 0  # per-topology progress is noisy across seven runs

DATA_ROOT = ROOT / "data"


def config_for(topology: str):
    optimizer, lr = TUNED[topology]
    mix = "momentum" if optimizer == "sgd_momentum" else "none"
    entries = []
    for name in LEARNERS:
        if name == "diffusion_sgd_atc_plain":
            # Defined as the plain-SGD variant; it keeps its own tuned rate so
            # the p-per-link comparison stays honest per topology.
            entries.append({"name": name})
            continue
        if name == "local_only":
            entries.append({"name": name})  # ignores the graph; keeps its own tuning
            continue
        entries.append(
            {
                "name": name,
                "optimizer": optimizer,
                "lr": lr,
                "momentum": 0.9 if optimizer == "sgd_momentum" else 0.0,
                # Uniform: the config rejects a stateful optimizer left unmixed
                # for *any* learner, centralized included (WORKPLAN 3.4).
                "mix_optimizer_state": mix,
            }
        )
    params = default_topology_params(10).get(topology) or {}
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": f"x3_{topology}"},
            "graph": {"topology": topology, "params": params},
            "learners": entries,
        },
    )


def main(fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1

    train, test = load_mnist(DATA_ROOT, download=False)
    started = time.perf_counter()

    for index, topology in enumerate(TOPOLOGIES, start=1):
        config = config_for(topology)
        out_dir = ROOT / config.run.out_dir / config.run.name
        if fresh and out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)

        graph = build_graph(
            topology, config.graph.n_nodes, config.graph.weights, config.graph.params
        )
        optimizer, lr = TUNED[topology]
        print(
            f"\n[{index}/{len(TOPOLOGIES)}] {topology}: "
            f"1-rho={graph.spectral_gap:.4f}, diameter={graph.diameter}, "
            f"{optimizer} lr={lr}"
        )

        rec.write_metadata(out_dir, config, {"experiment": config.run.name})
        for seed in config.run.seeds:
            environment = build_environment(config, seed, train)
            model = build_model_from_config(config)
            likelihood = Categorical(config.model.output_dim)
            learners = build_learners(config, model, likelihood)
            theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
            sha, _dirty = git_revision(ROOT)
            recorder = rec.Recorder(
                out_dir,
                RunContext.from_config(
                    config, seed, environment.graph, run_id=rec.new_run_id(), git_sha=sha
                ),
                config,
            )
            simulate.run(
                config,
                environment,
                learners,
                build_evalsets(config, environment, test),
                likelihood,
                theta0,
                recorder=recorder,
                verify_observations=False,
                progress_every=PROGRESS_EVERY,
            )
            recorder.finalize()
            print(f"    seed {seed} done ({(time.perf_counter() - started) / 60:.0f} min elapsed)")

    print(f"\nX3 complete in {(time.perf_counter() - started) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh=FRESH or "--fresh" in sys.argv))
