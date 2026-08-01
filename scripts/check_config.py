"""Load a config, validate it, and print what the run will actually do.

Run it straight from the IDE: hit the run (or debug) button with no arguments and
it uses ``EXPERIMENT`` below. Change that constant to inspect a different config,
or set a breakpoint in ``dekf_bench.utils.config`` and step through composition.

From a terminal it also takes an optional name::

    python scripts/check_config.py x2_rotating
"""

from __future__ import annotations

import sys
from pathlib import Path

from dekf_bench.env.drift import build_drift
from dekf_bench.runner.seeding import STREAM_NAMES, Seeds
from dekf_bench.utils.config import MNIST_TRAIN_SIZE, Config, ConfigError, load_config

# ---------------------------------------------------------------------------
# Edit this, then run the file. No command-line arguments required.
# ---------------------------------------------------------------------------
EXPERIMENT = "x1_stationary"

# Set to a path to also write the fully resolved config, or leave as None.
DUMP_TO: str | Path | None = None


def describe(config: Config) -> str:
    """A human-readable summary of everything the config decides."""
    env, graph, run, model = config.env, config.graph, config.run, config.model

    shard = MNIST_TRAIN_SIZE // graph.n_nodes
    used = graph.n_nodes * env.samples_per_node_per_step * run.horizon
    max_horizon = shard // env.samples_per_node_per_step

    drift = build_drift(config)

    lines = [
        f"experiment      {run.name}",
        f"seeds           {run.seeds}",
        f"horizon T       {run.horizon}   (eval every K={run.eval_every})",
        f"dtype/device    {run.dtype} on {run.device}",
        "",
        f"graph           {graph.topology}, N={graph.n_nodes}, {graph.weights} weights",
        f"model           {model.name}, {model.input_size}x{model.input_size} input, "
        f"hidden={model.hidden}, p={model.num_params}",
        "",
        f"samples/node/t  n={env.samples_per_node_per_step}",
        f"label avail.    {env.label_availability}",
        f"partition       {env.partition.kind}"
        + (f" (beta={env.partition.beta})" if env.partition.kind == "dirichlet" else ""),
        "",
        f"drift           {env.drift.schedule}, scope={env.drift_scope}",
    ]

    if env.drift.schedule == "linear":
        lines.append(
            f"                alpha={drift.schedule.alpha:.4f} deg/step  "
            f"(derived: {env.drift.total_degrees} deg / {run.horizon} steps)"
        )
    elif env.drift.schedule == "piecewise":
        lines.append(
            f"                {env.drift.jump_degrees} deg jumps at {env.drift.change_points}"
        )
    elif env.drift.schedule == "sinusoidal":
        lines.append(
            f"                amplitude={env.drift.amplitude_degrees} deg, "
            f"period={env.drift.period} steps"
        )

    if env.drift.schedule != "stationary":
        checkpoints = [0, run.horizon // 4, run.horizon // 2, run.horizon]
        rotations = "  ".join(f"t={t}:{drift.rotation_at(t):+.1f}deg" for t in checkpoints)
        lines.append(f"                rotation   {rotations}")

    lines += [
        "",
        f"shard budget    {used} of {MNIST_TRAIN_SIZE} samples "
        f"({100 * used / MNIST_TRAIN_SIZE:.0f}%), {shard} per agent",
        f"                max horizon at this N and n: {max_horizon}"
        + ("  [epochs allowed]" if env.allow_epochs else ""),
        "",
        f"learners        {len(config.learners)}, sharing one environment",
    ]
    for learner in config.learners:
        detail = f"lr={learner.lr}, {learner.optimizer}"
        if learner.optimizer != "sgd":
            detail += f", mix={learner.mix_optimizer_state}"
        lines.append(f"                - {learner.name:<20} {detail}")

    lines += ["", f"evalsets        {config.eval.evalsets}"]

    seeds = Seeds.from_master(run.seeds[0])
    lines += [
        "",
        f"seed streams    from master {seeds.master} (first of {len(run.seeds)})",
    ]
    lines += [f"                {stream:<12} {seeds[stream]}" for stream in STREAM_NAMES]

    return "\n".join(lines)


def main(experiment: str = EXPERIMENT, dump_to: str | Path | None = DUMP_TO) -> int:
    try:
        config = load_config(experiment)
    except ConfigError as error:
        print(f"config error in {experiment!r}:\n\n{error}")
        return 1

    print(describe(config))

    if dump_to is not None:
        from dekf_bench.utils.config import dump_config

        written = dump_config(config, dump_to)
        print(f"\nresolved config written to {written}")
    return 0


if __name__ == "__main__":
    # Optional positional argument, so the terminal works too, but the IDE run
    # button needs nothing.
    name = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT
    raise SystemExit(main(name))
