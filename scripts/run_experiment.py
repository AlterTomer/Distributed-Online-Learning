r"""Run one experiment: every learner, every seed, results to parquet.

Run it from the IDE with no arguments — edit ``EXPERIMENT`` and go. Everything
stays in-process and serial, so a breakpoint anywhere in ``simulate.run`` stops
where you expect.

**Interrupted runs resume.** Results and a checkpoint are written every
``eval_every`` steps, so re-running the same experiment continues from the last
completed evaluation rather than starting over. Resumption is exact: the loop
consumes no randomness, so a resumed run reproduces an uninterrupted one exactly
(design note D38). ``FRESH = True`` forces a clean start.

From a terminal it also takes a name::

    python scripts/run_experiment.py x2_rotating
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dekf_bench.data.mnist import is_cached, load_mnist
from dekf_bench.env.environment import build_environment
from dekf_bench.evaluation.evalsets import build_evalsets
from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.metrics.communication import cost_for, ledger
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.recording import recorder as rec
from dekf_bench.recording.schema import RunContext
from dekf_bench.runner import simulate
from dekf_bench.utils.config import default_configs_dir, load_config
from dekf_bench.utils.determinism import git_revision

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
EXPERIMENT = "x1_stationary"
SEEDS: list[int] | None = None  # None uses the config's run.seeds
HORIZON: int | None = None  # None uses the config's run.horizon
FRESH = False  # True discards any existing results and starts over
PROGRESS_EVERY = 250  # 0 to silence

REPO = default_configs_dir().parent
DATA_ROOT = REPO / "data"


def overrides() -> dict:
    run: dict = {}
    if SEEDS is not None:
        run["seeds"] = SEEDS
    if HORIZON is not None:
        run["horizon"] = HORIZON
    return {"run": run} if run else {}


def run_seed(config, seed: int, train, test, out_dir: Path) -> tuple[int, float]:
    """One seed, resuming if a checkpoint is there. Returns (rows, seconds)."""
    environment = build_environment(config, seed, train)
    model = build_model_from_config(config)
    likelihood = Categorical(config.model.output_dim)
    learners = build_learners(config, model, likelihood)
    evalsets = build_evalsets(config, environment, test)

    # Every learner starts from the same theta_0: the filter needs a common
    # prior, and it removes a confound from the SGD comparison.
    #
    # Drawn on the CPU and moved, rather than drawn on the device: the same seed
    # then gives the same theta_0 on either, so a CUDA run and a CPU run of one
    # config are comparable rather than merely both valid.
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    theta0 = theta0.to(environment.train.images.device)

    sha, _dirty = git_revision(REPO)
    context = RunContext.from_config(
        config, seed, environment.graph, run_id=rec.new_run_id(), git_sha=sha
    )
    recorder = rec.Recorder(out_dir, context, config)

    started = time.perf_counter()
    simulate.run(
        config,
        environment,
        learners,
        evalsets,
        likelihood,
        theta0,
        recorder=recorder,
        progress_every=PROGRESS_EVERY,
    )
    recorder.finalize()
    return recorder.n_rows, time.perf_counter() - started


def write_ledger_for(config, out_dir: Path) -> None:
    """The Table-I-style summary, once per experiment."""
    from dekf_bench.env.graph import build_graph

    graph = build_graph(
        config.graph.topology, config.graph.n_nodes, config.graph.weights, config.graph.params
    )
    model = build_model_from_config(config)
    costs = [
        cost_for(
            entry,
            model.num_params,
            graph.n_edges,
            n_nodes=config.graph.n_nodes,
            samples_per_step=config.env.samples_per_node_per_step,
            input_dim=model.input_dim,
        )
        for entry in config.learners
    ]
    rows = ledger(costs, config.run.horizon)
    rec.write_ledger(out_dir, rows)

    print("\ncommunication ledger")
    print(f"  {'learner':<26}{'scalars/step':>14}{'total':>16}{'rel':>7}")
    print("  " + "-" * 61)
    for row in rows:
        relative = row["relative_to_cheapest_diffusing"]
        marker = f"{relative:.1f}x" if relative else "--"
        print(
            f"  {row['learner']:<26}{row['scalars_per_step']:>14,}"
            f"{row['total_scalars']:>16,}{marker:>7}"
        )


def main(experiment: str = EXPERIMENT, fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1

    config = load_config(experiment, overrides=overrides() or None)
    out_dir = REPO / config.run.out_dir / config.run.name

    if fresh and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
        print(f"discarded {out_dir}")

    train, test = load_mnist(DATA_ROOT, download=False)
    rec.write_metadata(out_dir, config, {"experiment": config.run.name})

    print(
        f"\n{config.run.name}: {len(config.learners)} learners, "
        f"{len(config.run.seeds)} seeds, T={config.run.horizon}, "
        f"N={config.graph.n_nodes} on a {config.graph.topology}\n"
        f"  learners: {', '.join(entry.name for entry in config.learners)}\n"
        f"  -> {out_dir}"
    )

    total_rows, started = 0, time.perf_counter()
    for seed in config.run.seeds:
        print(f"\nseed {seed}")
        rows, seconds = run_seed(config, seed, train, test, out_dir)
        total_rows += rows
        print(f"  {rows:,} rows in {seconds:.0f}s")

    write_ledger_for(config, out_dir)
    print(f"\n{total_rows:,} rows over {time.perf_counter() - started:.0f}s")
    print(f"read them with: pd.read_parquet(r'{out_dir}')")
    return 0


if __name__ == "__main__":
    # --fresh is reachable from the terminal because a config change invalidates
    # every checkpoint, and during a tuning pass that happens often.
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(
        main(
            args[0] if args else EXPERIMENT,
            fresh=FRESH or "--fresh" in sys.argv,
        )
    )
