"""Inspect the communication topologies: structure, mixing, and a picture.

Run it from the IDE with no arguments. ``TOPOLOGY = None`` compares every
topology in one table, which is the view that makes the price of connectivity
obvious; setting it to a name drills into that one.

Later phases extend this script with the stream and drift visualisations.
"""

from __future__ import annotations

import torch

from dekf_bench.env.graph import build_graph, default_topology_params
from dekf_bench.utils.config import load_config

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
N_NODES = 10
WEIGHTS = "metropolis"  # metropolis | relative_degree | uniform
TOPOLOGY: str | None = None  # None compares all of them
SHOW_PARTITIONS = True  # how the training set divides up, IID and Dirichlet
SHOW_DRIFT = True  # drift schedules and the image pipeline
SHOW_STREAMS = True  # who receives which samples, and exactly-once
SHOW_ENVIRONMENT = True  # the composed environment, end to end
SHOW_PLOT = False  # draw the graphs and the weight matrices
SEED = 0


def compare(n_nodes: int, rule: str, seed: int) -> None:
    """One row per topology: structure, connectivity and mixing."""
    header = (
        f"{'topology':<16}{'edges':>6}{'deg':>9}{'diam':>6}{'comp':>6}"
        f"{'2x-stoch':>10}{'gap':>9}{'mixing':>9}"
    )
    print(header)
    print("-" * len(header))

    for topology, params in default_topology_params(n_nodes).items():
        generator = torch.Generator().manual_seed(seed)
        graph = build_graph(topology, n_nodes, rule, params, generator)
        s = graph.summary()
        gap = f"{s['spectral_gap']:.4f}" if s["spectral_gap"] is not None else "undef"
        print(
            f"{topology:<16}{s['n_edges']:>6}"
            f"{f'{s['min_degree']}-{s['max_degree']}':>9}"
            f"{str(s['diameter']):>6}{s['n_components']:>6}"
            f"{str(s['is_doubly_stochastic']):>10}{gap:>9}{s['mixing_gap']:>9.4f}"
        )

    print(
        "\ngap    = 1 - ||A - 11^T/N||_2, the WORKPLAN definition. Undefined unless the\n"
        "         weights are doubly stochastic, in which case it equals mixing.\n"
        "mixing = 1 - SLEM. Always valid; higher means information spreads faster."
    )


def detail(topology: str, n_nodes: int, rule: str, seed: int) -> None:
    """Everything about one topology, including who talks to whom."""
    params = default_topology_params(n_nodes).get(topology, {})
    generator = torch.Generator().manual_seed(seed)
    graph = build_graph(topology, n_nodes, rule, params, generator)

    # Graph.weights is optional only because the data graph has none; anything
    # build_graph returns is populated.
    weights = graph.weights
    assert weights is not None

    print(f"topology     {topology}  (params {params or '{}'})")
    for key, value in graph.summary().items():
        if key != "topology":
            print(f"{key:<13}{value}")

    print("\nneighbourhoods (who each agent receives from)")
    for node in range(graph.n_nodes):
        neighbours = graph.neighbours(node).tolist()
        print(f"  {node:>2} <- {neighbours}   (self-weight {float(weights[node, node]):.4f})")

    print("\ncombination weights")
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)
    print(weights)

    consensus = torch.arange(graph.n_nodes, dtype=torch.float64).unsqueeze(1)
    spread = []
    mixed = consensus.clone()
    for step in range(6):
        spread.append((step, float(mixed.std())))
        mixed = weights @ mixed
    print("\ndisagreement under repeated combining (std across agents)")
    print("  " + "  ".join(f"t={s}:{v:.3f}" for s, v in spread))


def plot(n_nodes: int, rule: str, seed: int) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    topologies = default_topology_params(n_nodes)
    figure, axes = plt.subplots(2, len(topologies), figsize=(3 * len(topologies), 6))
    for column, (topology, params) in enumerate(topologies.items()):
        generator = torch.Generator().manual_seed(seed)
        graph = build_graph(topology, n_nodes, rule, params, generator)

        axis = axes[0][column]
        # Layout and draw in two explicit calls rather than via draw_circular:
        # networkx types that helper's styling keywords as a TypedDict with no
        # defaults, so the IDE flags every keyword not passed as "unfilled".
        drawable = graph.to_networkx()
        positions = nx.circular_layout(drawable)
        nx.draw_networkx_nodes(drawable, positions, ax=axis, node_size=90, node_color="#4477aa")
        nx.draw_networkx_edges(drawable, positions, ax=axis, width=0.8)
        axis.set_axis_off()
        gap = graph.mixing_gap
        axis.set_title(f"{topology}\nmixing {gap:.3f}", fontsize=9)

        axis = axes[1][column]
        assert graph.weights is not None
        axis.imshow(graph.weights, cmap="viridis", vmin=0.0)
        axis.set_xticks([])
        axis.set_yticks([])

    axes[1][0].set_ylabel("combination weights", fontsize=9)
    figure.tight_layout()
    plt.show()


def partitions(n_nodes: int, seed: int) -> None:
    """How the training set divides up, IID and at each Dirichlet concentration."""
    from dekf_bench.data.mnist import is_cached, load_split
    from dekf_bench.env.partition import build_partition

    if not is_cached():
        print("\n(MNIST not cached; run scripts/check_data.py to see the partition view)")
        return

    labels = load_split("train", download=False).labels
    print(f"\n\npartitions of {labels.numel()} training samples across {n_nodes} agents\n")
    header = f"{'partition':<18}{'sizes':>14}{'skew':>8}{'classes/agent':>16}"
    print(header)
    print("-" * len(header))

    for kind, beta in (("iid", None), ("dirichlet", 100.0), ("dirichlet", 1.0), ("dirichlet", 0.1)):
        generator = torch.Generator().manual_seed(seed)
        partition = build_partition(labels, n_nodes, kind, beta or 1.0, 10, generator)
        s = partition.summary(labels)
        name = kind if beta is None else f"{kind} beta={beta:g}"
        print(
            f"{name:<18}{f'{s['min_size']}-{s['max_size']}':>14}{s['skew']:>8.3f}"
            f"{f'{s['min_classes_present']}-{s['max_classes_present']}':>16}"
        )

    print(
        "\nskew = mean total-variation distance from the global class distribution.\n"
        "       0 means every agent mirrors the pooled data; it approaches 1 - 1/K = 0.9\n"
        "       when each agent holds a single class. Sizes stay equal by construction,\n"
        "       so the skew is composition alone -- see docs/design_notes.md D17."
    )


def drift_and_transforms() -> None:
    """What each schedule does, and what the pipeline does to a pixel."""
    import torch as _torch

    from dekf_bench.data.mnist import MNIST_STD, is_cached, load_split
    from dekf_bench.data.transforms import build_transform, downsample, rotate
    from dekf_bench.env.drift import build_drift
    from dekf_bench.utils.config import load_config

    print("\n\ndrift schedules\n")
    header = (
        f"{'schedule':<14}{'t=0':>9}{'t=375':>9}{'t=750':>9}{'t=1125':>9}{'t=1500':>9}{'travel':>9}"
    )
    print(header)
    print("-" * len(header))

    schedules = {
        "stationary": ("x1_stationary", {}),
        "linear": ("x2_rotating", {}),
        "piecewise": ("x5_abrupt_shift", {}),
        "sinusoidal": ("x2_rotating", {"include": {"env": "mnist_rotating_sinusoidal"}}),
    }
    for label, (experiment, overrides) in schedules.items():
        config = load_config(experiment, overrides=overrides or None)
        drift = build_drift(config)
        horizon = config.run.horizon
        cells = "".join(f"{drift.rotation_at(t):>9.1f}" for t in (0, 375, 750, 1125, horizon))
        print(f"{label:<14}{cells}{drift.schedule.total_travel(horizon):>9.1f}")

    print("\ntravel = the largest rotation reached, checked against the 45 deg cap.")

    print("\n\nper-node drift (drift_scope: per_node, spread 0.5)\n")
    config = load_config("x2_rotating", overrides={"env": {"drift_scope": "per_node"}})
    drift = build_drift(config)
    rotations = [drift.rotation_at(config.run.horizon, node) for node in range(N_NODES)]
    print("  rotation at T per agent: " + "  ".join(f"{r:.1f}" for r in rotations))
    print(
        f"  fastest {max(rotations):.1f} deg -- multipliers top out at 1, so no agent\n"
        "  is carried past the cap the schedule was validated against."
    )

    if not is_cached():
        print("\n(MNIST not cached; skipping the transform view)")
        return

    print("\n\nimage pipeline: rotate(28x28) -> downsample(14x14) -> normalize\n")
    train = load_split("train", download=False)
    pipeline = build_transform(train.images, size=14)
    for key, value in pipeline.summary().items():
        print(f"  {key:<12}{value}")
    print(
        f"\n  the published 28x28 std is {MNIST_STD}; pooling shrinks it to "
        f"{pipeline.std:.4f},\n  so reusing the published constants would over-scale the input."
    )

    sample = train.images[:200]
    corners = _torch.stack(
        [
            pipeline.apply(sample, 30.0)[:, 0, 0, 0],
            pipeline.apply(sample, 30.0)[:, 0, 0, -1],
            pipeline.apply(sample, 30.0)[:, 0, -1, 0],
            pipeline.apply(sample, 30.0)[:, 0, -1, -1],
        ]
    )
    deviation = float((corners - pipeline.background).abs().max())
    print(
        f"\n  corners exposed by a 30 deg rotation deviate from the background by at most\n"
        f"  {deviation:.2e} -- indistinguishable, which is what the ordering buys."
    )

    normalized_first = (sample - pipeline.mean) / pipeline.std
    wrong = downsample(rotate(normalized_first, 30.0), 14)
    wrong_corners = _torch.stack(
        [wrong[:, 0, 0, 0], wrong[:, 0, 0, -1], wrong[:, 0, -1, 0], wrong[:, 0, -1, -1]]
    )
    print(
        f"  normalizing first instead: corners land {float((wrong_corners - pipeline.background).abs().mean()):.4f}"
        f" away, i.e. {pipeline.mean / pipeline.std:.2f} sigma\n"
        "  -- a constant region present in every rotated image and no unrotated one."
    )


def streams(n_nodes: int, seed: int) -> None:
    """Who receives what, and whether the shard lasts the run."""
    from dekf_bench.data.mnist import is_cached, load_split
    from dekf_bench.env.partition import build_partition
    from dekf_bench.env.stream import build_stream
    from dekf_bench.utils.config import load_config

    if not is_cached():
        print("\n(MNIST not cached; skipping the stream view)")
        return

    config = load_config("x1_stationary")
    labels = load_split("train", download=False).labels
    partition = build_partition(labels, n_nodes, generator=torch.Generator().manual_seed(seed))

    print(
        f"\n\nsample streams (n={config.env.samples_per_node_per_step}, T={config.run.horizon})\n"
    )
    header = (
        f"{'label avail.':<14}{'realised':>10}{'consumed':>11}{'per agent':>14}{'shard left':>12}"
    )
    print(header)
    print("-" * len(header))

    for availability in (1.0, 0.5, 0.25):
        stream = build_stream(
            partition,
            horizon=config.run.horizon,
            samples_per_step=config.env.samples_per_node_per_step,
            label_availability=availability,
            generator=torch.Generator().manual_seed(seed),
        )
        s = stream.summary()
        shard = int(partition.sizes.min())
        print(
            f"{availability:<14}{s['empirical_label_rate']:>10.4f}{s['total_consumed']:>11}"
            f"{f'{s['min_required']}-{s['max_required']}':>14}"
            f"{shard - s['max_required']:>12}"
        )

    print(
        "\nAn idle step consumes nothing, so sparser labels leave more of the shard\n"
        "unused rather than shortening the run -- which is what keeps n and pi_lab\n"
        "independent axes in the sparsity sweep."
    )

    stream = build_stream(
        partition,
        horizon=config.run.horizon,
        samples_per_step=config.env.samples_per_node_per_step,
        label_availability=0.5,
        generator=torch.Generator().manual_seed(seed),
    )
    print("\nfirst 12 steps at pi_lab = 0.5   (. = idle, # = labelled)\n")
    for node in range(n_nodes):
        marks = "".join("#" if stream.is_labelled(node, t) else "." for t in range(12))
        first = stream.indices_at(node, 0).tolist()
        print(f"  agent {node:>2}  {marks}   step 0 -> {first if first else 'idle'}")

    seen = torch.cat([stream.consumed_by(node) for node in range(n_nodes)])
    print(
        f"\nover the whole run: {seen.numel()} samples served, "
        f"{len(torch.unique(seen))} distinct"
    )
    print("  equal counts = exactly-once holds; no sample reached two agents or one agent twice.")


def environment(seed: int) -> None:
    """The composed environment: what an agent actually receives at a step."""
    from dekf_bench.data.mnist import is_cached, load_split
    from dekf_bench.env.environment import build_environment, pool
    from dekf_bench.utils.config import load_config

    if not is_cached():
        print("\n(MNIST not cached; skipping the environment view)")
        return

    train = load_split("train", download=False)

    for experiment in ("x1_stationary", "x2_rotating"):
        env = build_environment(load_config(experiment), seed, train)
        print(f"\n\nenvironment: {experiment}\n")

        for step in (0, env.horizon // 2, env.horizon - 1):
            observations = env.step(step)
            xs, ys = pool(observations)
            sample = observations[0]
            print(
                f"  t={step:>5}  rotation {sample.rotation_degrees:>6.2f} deg   "
                f"agent 0: x{tuple(sample.x.shape)} y={sample.y.tolist()}   "
                f"pooled {tuple(xs.shape)}"
            )
            env.assert_unmodified(observations, step)

        print(
            f"  dtype {env.step(0)[0].x.dtype}, input_dim {env.transform.input_dim}, "
            f"graph {env.graph.topology}, data graph {env.graphs.data.n_edges} edges"
        )
        print("  assert_unmodified passed at every step shown.")

    env = build_environment(load_config("x1_stationary"), seed, train)
    observations = env.step(0)
    observations[3].x.add_(1.0)
    try:
        env.assert_unmodified(observations, 0)
    except Exception as error:  # noqa: BLE001 - demonstrating the guard
        print(f"\n  deliberately mutating agent 3's batch is caught:\n    {error}")


def main() -> int:
    print(f"N = {N_NODES}, weights = {WEIGHTS}, seed = {SEED}\n")
    if TOPOLOGY is None:
        compare(N_NODES, WEIGHTS, SEED)
    else:
        detail(TOPOLOGY, N_NODES, WEIGHTS, SEED)

    config = load_config("x1_stationary")
    print(
        f"\nx1_stationary uses: {config.graph.topology}, N={config.graph.n_nodes}, "
        f"{config.graph.weights} weights"
    )

    if SHOW_PARTITIONS:
        partitions(N_NODES, SEED)
    if SHOW_DRIFT:
        drift_and_transforms()
    if SHOW_STREAMS:
        streams(N_NODES, SEED)
    if SHOW_ENVIRONMENT:
        environment(SEED)
    if SHOW_PLOT:
        plot(N_NODES, WEIGHTS, SEED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
