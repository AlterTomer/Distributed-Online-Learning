"""Generate the presentation figures for the pre-Diff-EKF work.

Writes a numbered set of PNGs plus a summary into the ``preliminary work``
folder. Re-run it after each stage; it overwrites in place, so the folder is
always current.

Run from the IDE with no arguments. Set ``ONLY`` to a figure number to
regenerate just one while iterating on it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import torch

from dekf_bench.data.mnist import is_cached, load_split
from dekf_bench.data.transforms import build_transform, downsample, rotate
from dekf_bench.env.drift import build_drift
from dekf_bench.env.graph import build_graph, default_topology_params
from dekf_bench.env.partition import build_partition
from dekf_bench.utils.config import ModelConfig, load_config

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
OUT_DIR = Path(r"C:\Users\alter\OneDrive\Desktop\PhD\Distributed Online Learning\preliminary work")
ONLY: int | None = None  # e.g. 6 to regenerate only figure 06
N_NODES = 10
SEED = 0

# --- palette (validated: CVD dE 9.2, normal-vision dE 24.0, light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua -- fixed order
SEQUENTIAL = "Blues"
CRITICAL = "#d03b3b"

# Run-to-run noise in e*, in percentage points: five retrainings at 0 degrees,
# identical except for the seed, gave std 0.00163. See design note D34.
SEED_STD_PERCENT = 0.163


def style() -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted axis ink."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 9,
            "figure.dpi": 160,
        }
    )


def despine(axis, keep_left: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(keep_left)


def save(figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {path.name}")
    return path


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
# rotate/downsample come straight from data/transforms.py, so figure 06 is a
# live check on the shipped pipeline rather than an illustration of it. The
# normalization constants are the ones the transform fitted on canonical data.

_TRANSFORM = None


def transform_for(train_images: torch.Tensor):
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = build_transform(train_images, size=14)
    return _TRANSFORM


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def fig01_topologies() -> None:
    """Small multiples: the eight communication graphs and their weight matrices."""
    topologies = default_topology_params(N_NODES)
    figure, axes = plt.subplots(2, len(topologies), figsize=(2.05 * len(topologies), 4.9))

    for column, (name, params) in enumerate(topologies.items()):
        graph = build_graph(
            name, N_NODES, "metropolis", params, torch.Generator().manual_seed(SEED)
        )
        drawable = graph.to_networkx()
        positions = nx.circular_layout(drawable)

        axis = axes[0][column]
        nx.draw_networkx_edges(drawable, positions, ax=axis, width=1.2, edge_color=BASELINE)
        nx.draw_networkx_nodes(
            drawable, positions, ax=axis, node_size=70, node_color=SERIES[0], linewidths=0
        )
        axis.set_axis_off()
        diameter = graph.diameter
        axis.set_title(
            f"{name}\n{graph.n_edges} edges · diam {diameter if diameter else '\u221e'}",
            fontsize=8.5,
        )

        axis = axes[1][column]
        axis.imshow(graph.weights, cmap=SEQUENTIAL, vmin=0.0, vmax=1.0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_edgecolor(GRID)
        axis.set_xlabel(f"mixing {graph.mixing_gap:.3f}", fontsize=8, color=INK_SECONDARY)

    axes[1][0].set_ylabel("combination\nweights $a_{vu}$", fontsize=8.5, color=INK_SECONDARY)
    figure.suptitle(
        f"Communication topologies at N={N_NODES}, Metropolis weights",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    save(figure, "01_topologies.png")


def fig02_price_of_connectivity() -> None:
    """Horizontal bars: how fast each topology mixes. Previews figure F3."""
    topologies = default_topology_params(N_NODES)
    rows = []
    for name, params in topologies.items():
        graph = build_graph(
            name, N_NODES, "metropolis", params, torch.Generator().manual_seed(SEED)
        )
        rows.append((name, graph.mixing_gap, graph.diameter))
    rows.sort(key=lambda row: row[1])

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    names = [row[0] for row in rows]
    gaps = [row[1] for row in rows]
    positions = range(len(rows))

    axis.barh(list(positions), gaps, height=0.62, color=SERIES[0], zorder=3)
    for y, (_name, gap, diameter) in enumerate(rows):
        label = f"{gap:.3f}" + (f"   (diameter {diameter})" if diameter else "   (disconnected)")
        axis.text(gap + 0.012, y, label, va="center", fontsize=8.5, color=INK_SECONDARY)

    axis.set_yticks(list(positions))
    axis.set_yticklabels(names)
    axis.set_xlim(0, 1.18)
    axis.set_xlabel("mixing gap  $1 - \\mathrm{SLEM}$   (higher = information spreads faster)")
    axis.set_title("The price of connectivity")
    axis.xaxis.grid(True)
    axis.yaxis.grid(False)
    despine(axis)
    figure.text(
        0.5,
        -0.13,
        "Diameter and mixing rank differently: a star has diameter 2 but mixes worse than a "
        "ring of diameter 5,\nbecause every message funnels through one hub.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(figure, "02_price_of_connectivity.png")


def fig03_mnist_samples(train) -> None:
    """One column per class, three rows: the raw data before any transform."""
    figure, axes = plt.subplots(3, 10, figsize=(9.0, 3.0))
    for digit in range(10):
        matching = (train.labels == digit).nonzero(as_tuple=True)[0][:3]
        for row, index in enumerate(matching):
            axis = axes[row][digit]
            axis.imshow(train.images[index, 0], cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(str(digit), fontsize=10)
    figure.suptitle(
        "MNIST, raw intensities in [0,1] — 60 000 train / 10 000 test",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    save(figure, "03_mnist_samples.png")


def fig04_downsampling(train) -> None:
    """Why the input shrinks: the phase-5 covariance budget, made visual."""
    figure, axes = plt.subplots(2, 8, figsize=(8.4, 2.6))
    indices = [(train.labels == d).nonzero(as_tuple=True)[0][0] for d in range(8)]
    full = train.images[torch.tensor(indices)]
    small = downsample(full, 14)

    for column in range(8):
        for row, batch in enumerate((full, small)):
            axis = axes[row][column]
            axis.imshow(batch[column, 0], cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)

    big = ModelConfig(name="mlp", input_size=28, hidden=[128], output_dim=10)
    small_model = ModelConfig(name="mlp_small", input_size=14, hidden=[14], output_dim=10)
    axes[0][0].set_ylabel(
        f"28×28\n784–128–10\np = {big.num_params:,}",
        fontsize=8.5,
        rotation=0,
        ha="right",
        va="center",
        color=INK_SECONDARY,
        labelpad=8,
    )
    axes[1][0].set_ylabel(
        f"14×14\n196–14–10\np = {small_model.num_params:,}",
        fontsize=8.5,
        rotation=0,
        ha="right",
        va="center",
        color=INK_SECONDARY,
        labelpad=8,
    )
    figure.suptitle(
        "Input downsampling and the parameter budget", fontsize=12, fontweight="bold", y=1.06
    )
    figure.text(
        0.5,
        -0.12,
        "A dense Diff-EKF covariance is $p \\times p$. At $p\\approx10^5$ that is $10^{10}$ "
        "entries — impossible.\nShrinking the hidden layer cannot fix it (784 inputs force "
        "$h\\approx3$); the input has to come down.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(figure, "04_downsampling.png")


def fig05_rotation(train) -> None:
    """What drift looks like, across the capped range."""
    angles = [0, 15, 30, 45]
    digits = [(train.labels == d).nonzero(as_tuple=True)[0][0] for d in (3, 6, 9, 2, 7)]
    base = train.images[torch.tensor(digits)]

    figure, axes = plt.subplots(len(angles), len(digits), figsize=(5.4, 4.6))
    for row, angle in enumerate(angles):
        shown = downsample(rotate(base, float(angle)), 14)
        for column in range(len(digits)):
            axis = axes[row][column]
            axis.imshow(shown[column, 0], cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row][0].set_ylabel(
            f"{angle}°",
            rotation=0,
            ha="right",
            va="center",
            fontsize=10,
            color=INK_SECONDARY,
            labelpad=8,
        )

    figure.suptitle(
        "Rotating MNIST: the drift the benchmark tracks", fontsize=12, fontweight="bold", y=0.98
    )
    figure.text(
        0.5,
        0.02,
        "Total rotation is capped at 45°. Past that, accuracy falls for reasons unrelated to "
        "decentralization,\nand near 180° a 6 becomes a 9 — the reference error would then "
        "measure label ambiguity instead.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(figure, "05_rotation.png")


def fig06_normalization_order(train) -> None:
    """The ordering trap: normalize after rotating, never before.

    The artefact is a constant offset over the exposed corners, which is easy to
    miss on the images themselves -- so the third row shows the difference,
    where it is unmistakable.
    """
    digits = [(train.labels == d).nonzero(as_tuple=True)[0][0] for d in (0, 1, 4, 8)]
    base = train.images[torch.tensor(digits)]
    angle = 30.0

    pipeline = transform_for(train.images)

    # The correct path is the shipped transform itself, so this figure fails
    # loudly if data/transforms.py ever changes its ordering.
    correct = pipeline.apply(base, angle)
    normalized_first = (base - pipeline.mean) / pipeline.std
    wrong = downsample(rotate(normalized_first, angle), pipeline.size)

    difference = (wrong - correct).abs()
    offset = pipeline.mean / pipeline.std  # what the corners are wrong by, in sigma

    limits = dict(
        vmin=float(min(correct.min(), wrong.min())), vmax=float(max(correct.max(), wrong.max()))
    )

    figure, axes = plt.subplots(3, len(digits), figsize=(5.2, 4.5))
    rows = (
        (correct, "correct", "rotate → downsample → normalize", INK, "gray", limits),
        (wrong, "WRONG", "normalize → rotate → downsample", CRITICAL, "gray", limits),
        (
            difference,
            "difference",
            "|wrong − correct|",
            CRITICAL,
            SEQUENTIAL,
            dict(vmin=0.0, vmax=float(offset)),
        ),
    )
    for row, (batch, mark, title, colour, cmap, scale) in enumerate(rows):
        for column in range(len(digits)):
            axis = axes[row][column]
            axis.imshow(batch[column, 0], cmap=cmap, **scale)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row][0].set_ylabel(
            f"{mark}\n{title}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            color=colour,
            labelpad=8,
        )

    figure.suptitle("Why normalization comes last", fontsize=12, fontweight="bold", y=1.0)
    figure.text(
        0.5,
        0.055,
        f"The corners are wrong by exactly $\\mu/\\sigma = {offset:.2f}$ standard deviations —\n"
        "a large, clean, constant region present in every rotated image and no unrotated one.",
        ha="center",
        fontsize=8,
        color=INK_SECONDARY,
    )
    figure.text(
        0.5,
        -0.10,
        "Rotation fills exposed corners with zero. On raw intensities zero is the black "
        "background, so the fill is invisible.\nNormalize first and zero becomes mid-grey: the "
        "corners then carry a signal correlated with the drift state,\nwhich a classifier will "
        "happily learn instead of the digit.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    save(figure, "06_normalization_order.png")


def fig07_drift_schedules() -> None:
    """The three non-stationary regimes, on one axis."""
    horizon = 1500
    configs = {
        "linear (45° total)": load_config("x2_rotating"),
        "piecewise (15° at t=500)": load_config("x5_abrupt_shift"),
        "sinusoidal (±30°, period 500)": load_config(
            "x2_rotating",
            overrides={"include": {"env": "mnist_rotating_sinusoidal"}},
        ),
    }

    figure, axis = plt.subplots(figsize=(7.6, 3.9))
    steps = list(range(0, horizon + 1, 5))
    for index, (label, config) in enumerate(configs.items()):
        drift = build_drift(config)
        values = [drift.rotation_at(t) for t in steps]
        axis.plot(steps, values, color=SERIES[index], label=label, zorder=3)
        axis.text(horizon + 18, values[-1], label, fontsize=8.5, va="center", color=SERIES[index])

    axis.axhline(45, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    axis.text(10, 46.5, "45° well-posedness cap", fontsize=8, color=MUTED)
    axis.set_xlabel("step $t$")
    axis.set_ylabel("rotation applied to the data (degrees)")
    axis.set_xlim(0, horizon)
    axis.set_ylim(-38, 56)
    axis.set_title("Drift schedules")
    despine(axis)
    figure.text(
        0.5,
        -0.06,
        "The per-step rate is derived, never configured: $\\alpha = $ total_degrees $/\\,T$, so "
        "changing the horizon\ncannot silently change how far the distribution travels.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(figure, "07_drift_schedules.png")


def fig08_partition_skew(labels) -> None:
    """Per-agent class composition, IID against two Dirichlet concentrations."""
    settings = [
        ("IID", "iid", 1.0),
        ("Dirichlet β=1", "dirichlet", 1.0),
        ("Dirichlet β=0.1", "dirichlet", 0.1),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))

    for axis, (title, kind, beta) in zip(axes, settings, strict=True):
        partition = build_partition(
            labels, N_NODES, kind, beta, 10, torch.Generator().manual_seed(SEED)
        )
        distribution = partition.class_distribution(labels)
        image = axis.imshow(distribution, cmap=SEQUENTIAL, vmin=0.0, vmax=0.6, aspect="auto")
        axis.set_xticks(range(10))
        axis.set_yticks(range(N_NODES))
        axis.set_xlabel("class")
        axis.grid(False)
        axis.set_title(f"{title}\nskew {partition.skew(labels):.3f}", fontsize=10)
        for spine in axis.spines.values():
            spine.set_edgecolor(GRID)

    axes[0].set_ylabel("agent")
    figure.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="share of the agent's shard")
    figure.suptitle(
        "Label skew: what each agent actually sees", fontsize=12, fontweight="bold", y=1.04
    )
    save(figure, "08_partition_skew.png")


def fig09_shard_sizes(labels) -> None:
    """Why shard sizes are balanced: the classical construction starves agents."""
    beta = 0.1
    balanced = build_partition(
        labels, N_NODES, "dirichlet", beta, 10, torch.Generator().manual_seed(SEED)
    )
    classical = build_partition(
        labels,
        N_NODES,
        "dirichlet",
        beta,
        10,
        torch.Generator().manual_seed(SEED),
        balance_sizes=False,
    )

    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    x = torch.arange(N_NODES).numpy()
    width = 0.4
    axis.bar(
        x - width / 2,
        balanced.sizes.numpy(),
        width,
        color=SERIES[0],
        label="balanced sizes (ours)",
        zorder=3,
    )
    axis.bar(
        x + width / 2,
        classical.sizes.numpy(),
        width,
        color=SERIES[1],
        label="classical Dirichlet",
        zorder=3,
    )

    # The threshold goes in the legend rather than as an inline annotation: at
    # any height it would sit on top of a bar.
    threshold = axis.axhline(3000, color=CRITICAL, linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)
    threshold.set_label("3000 = what a default run consumes ($n\\,T$)")

    axis.set_xticks(x)
    axis.set_xlabel("agent")
    axis.set_ylabel("samples in shard")
    axis.set_title(f"Shard sizes under label skew (β={beta})")
    axis.legend(loc="upper left", ncol=1)
    axis.xaxis.grid(False)
    despine(axis)
    figure.text(
        0.5,
        -0.08,
        "The config-time budget check ($N\\,n\\,T \\leq 60000$) assumes equal shards. Under the "
        "classical construction\nevery seed tested leaves some agent below what the run "
        "consumes — it would exhaust mid-flight.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(figure, "09_shard_sizes.png")


def fig10_received_digits(train) -> None:
    """The phase-1 milestone: what each agent actually receives, over time.

    The same agents, the same steps, the same *indices* under both regimes --
    partition and stream are driven by their own seed streams and are untouched
    by drift, so the only difference between the two halves is the rotation.
    That is the figure's point: it isolates drift from everything else.

    Steps 0/100/500 are the milestone's; 1499 is added because the drift rate
    was recalibrated (design note D3) and the original three now span only
    0-15 degrees of the run's 45.
    """
    from dekf_bench.env.environment import build_environment

    steps = [0, 100, 500, 1499]
    regimes = [("stationary", "x1_stationary"), ("rotating", "x2_rotating")]
    environments = {
        label: build_environment(load_config(experiment), SEED, train)
        for label, experiment in regimes
    }

    columns = len(steps) * len(regimes)
    figure, axes = plt.subplots(N_NODES, columns, figsize=(0.78 * columns, 0.78 * N_NODES))

    for block, (label, _) in enumerate(regimes):
        env = environments[label]
        for index, step in enumerate(steps):
            column = block * len(steps) + index
            observations = env.step(step)
            rotation = observations[0].rotation_degrees

            for node in range(N_NODES):
                obs = observations[node]
                axis = axes[node][column]
                axis.imshow(obs.x[0, 0], cmap="gray")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                # Corner, not centre: over the stroke of the digit the label is
                # unreadable and obscures the thing it is labelling.
                axis.text(
                    0.06,
                    0.06,
                    str(int(obs.y[0])),
                    transform=axis.transAxes,
                    fontsize=6,
                    color="#ffd000",
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "#000000", "edgecolor": "none", "pad": 0.8, "alpha": 0.65},
                )
                if node == 0:
                    axis.set_title(f"t={step}\n{rotation:.0f}°", fontsize=7.5, pad=3)

    for node in range(N_NODES):
        axes[node][0].set_ylabel(
            f"agent {node}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=7.5,
            color=INK_SECONDARY,
            labelpad=4,
        )

    figure.subplots_adjust(hspace=0.12, wspace=0.08, top=0.90, bottom=0.03)
    figure.suptitle("What each agent receives", fontsize=12, fontweight="bold", y=1.02)

    # Headers centred over their own column block, not over the whole figure:
    # the row labels take up the left margin, so half-width would be off by an
    # inch and the labels would sit over the wrong columns.
    left = axes[0][0].get_position().x0
    right = axes[0][-1].get_position().x1
    for block, (label, _) in enumerate(regimes):
        centre = left + (right - left) * (block + 0.5) / len(regimes)
        figure.text(centre, 0.965, label, ha="center", fontsize=10, fontweight="bold", color=INK)
    figure.text(
        0.5,
        -0.012,
        "Identical indices in both halves -- the partition and the stream have their own seed "
        "streams and are untouched by drift,\nso the only difference is the rotation. Small "
        "figures are the labels; each agent draws from a disjoint shard, so no digit appears "
        "twice.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    save(figure, "10_received_digits.png")


def fig11_reference(train) -> None:
    """e* against rotation: the curve every gap is measured against.

    Two panels rather than one. The schedule coverage was originally drawn as
    bars inside the e* axes, where they read as data lying below the curve; a
    separate strip below says the same thing without pretending to be an error
    rate.
    """
    from dekf_bench.evaluation import reference as ref

    path = ref.cache_path(Path(__file__).resolve().parents[1] / "data", "shared_seed", "validation")
    if not path.is_file():
        print("  (no cached reference; skipping figure 11)")
        return

    reference = ref.load(path)
    rotations = list(reference.rotations)
    errors = [result.error_rate * 100 for result in reference.results]

    figure, (axis, strip) = plt.subplots(
        2, 1, figsize=(7.6, 4.6), sharex=True, height_ratios=[4, 1]
    )

    # A +/-1 sigma band from the seed-repeat diagnostic: five runs at 0 degrees,
    # identical except for the seed, gave std 0.00163. Drawn around every point
    # because it is the run-to-run noise each single draw carries.
    seed_std = SEED_STD_PERCENT
    axis.fill_between(
        rotations,
        [error - seed_std for error in errors],
        [error + seed_std for error in errors],
        color=SERIES[0],
        alpha=0.15,
        linewidth=0,
        zorder=2,
    )
    axis.plot(rotations, errors, color=SERIES[0], marker="o", markersize=5, zorder=3)

    # Each point is its own model, trained and tested at that rotation. Labelling
    # them makes that explicit -- the x-axis alone reads as one curve sampled.
    for index, (rotation, error) in enumerate(zip(rotations, errors, strict=True)):
        axis.annotate(
            f"{rotation:+.0f}°",
            (rotation, error),
            textcoords="offset points",
            xytext=(0, 10 if index % 2 == 0 else -16),
            ha="center",
            fontsize=7,
            color=INK_SECONDARY,
        )

    axis.set_ylabel("$e^*$   test error rate (%)")
    axis.set_title("The offline reference: one model per rotation, trained and tested there")
    axis.margins(y=0.18)
    despine(axis)

    for index, ((low, high), label) in enumerate(
        (((0.0, 45.0), "linear"), ((-30.0, 30.0), "sinusoidal"), ((0.0, 15.0), "piecewise"))
    ):
        y = -index
        strip.plot([low, high], [y, y], color=SERIES[1], linewidth=5, solid_capstyle="butt")
        strip.text(high + 1.6, y, label, fontsize=8, color=INK_SECONDARY, va="center")
    strip.set_ylim(-2.7, 0.7)
    strip.set_yticks([])
    strip.set_xticks(rotations[::2])
    strip.set_xlabel("rotation applied to the data (degrees)")
    strip.set_ylabel(
        "visited by", fontsize=8, color=MUTED, rotation=0, ha="right", va="center", labelpad=6
    )
    strip.grid(False)
    despine(strip, keep_left=False)

    symmetry = reference.summary()["symmetry_error"]
    caption = (
        "Each point is a SEPARATE offline model: the same 196-14-10 MLP the online methods use, "
        "trained on all 55k "
        + chr(10)
        + "training images rotated by that angle, then scored on the 10k test images rotated by "
        "the same angle. "
        + chr(10)
        + "It is the best this architecture can do with full access to the data, so every online "
        "result is reported "
        + chr(10)
        + "as the gap above this line rather than as a raw error rate."
        + chr(10)
        + chr(10)
        + "Band: +/-1 sigma of run-to-run noise, measured by retraining 0 degrees five times with "
        "different seeds (sigma = 0.16 pts)."
        + chr(10)
        + "The grid spread is "
        + f"{max(errors) - min(errors):.2f}"
        + " pts, about the size that "
        "noise alone would produce -- so most of the wiggle is not a rotation effect,"
        + chr(10)
        + "and whether any of it is cannot be settled from one run per point. Symmetry holds to "
        + f"{symmetry:.4f}"
        + "."
    )
    figure.text(0.5, -0.30, caption, ha="center", fontsize=8.5, color=MUTED)
    figure.tight_layout()
    save(figure, "11_reference.png")


def write_summary(labels) -> None:
    """A one-page factual companion to the figures."""
    config = load_config("x1_stationary")
    rows = []
    for name, params in default_topology_params(N_NODES).items():
        graph = build_graph(
            name, N_NODES, "metropolis", params, torch.Generator().manual_seed(SEED)
        )
        rows.append(
            f"| `{name}` | {graph.n_edges} | {int(graph.degrees.min())}–"
            f"{int(graph.degrees.max())} | {graph.diameter if graph.diameter else '∞'} | "
            f"{graph.mixing_gap:.3f} |"
        )

    skews = []
    for title, kind, beta in (
        ("IID", "iid", 1.0),
        ("β=100", "dirichlet", 100.0),
        ("β=1", "dirichlet", 1.0),
        ("β=0.1", "dirichlet", 0.1),
    ):
        partition = build_partition(
            labels, N_NODES, kind, beta, 10, torch.Generator().manual_seed(SEED)
        )
        present = partition.classes_present(labels)
        skews.append(
            f"| {title} | {int(partition.sizes.min())} | {partition.skew(labels):.3f} | "
            f"{int(present.min())}–{int(present.max())} |"
        )

    text = f"""# Preliminary work — status and figures

Generated by `scripts/make_preliminary_figures.py`. Re-run after each stage.

## What exists

A benchmark for **distributed online learning over a graph**: {N_NODES} agents on a
communication graph, each receiving {config.env.samples_per_node_per_step} labelled
samples per step, cooperatively learning one shared classifier online with no fusion
centre. **Phases 0 and 1 are complete**; the Diff-EKF is phase 5.

| Component | State |
|---|---|
| Repository, packaging, CI-ready tooling | done |
| Configuration system (composable YAML, validated) | done |
| Separable seed streams, determinism, provenance | done |
| MNIST loading and caching | done |
| Communication graphs, weights, diagnostics | done |
| Data partitioning (IID and Dirichlet skew) | done |
| Drift schedules (stationary / linear / piecewise / sinusoidal) | done |
| Image pipeline (rotate, downsample, normalize) | done |
| Per-agent sample streams, exactly-once consumption | done |
| The composed environment | done |
| Models, likelihoods, metrics, reference classifier | next (phase 2) |
| Learners (centralized / diffusion SGD / local-only) | phase 3 |
| **Diff-EKF** | phase 5 |

757 tests pass; `ruff`, `black` and `mypy` clean.

## The setting

- **N = {config.graph.n_nodes}** agents, **n = {config.env.samples_per_node_per_step}**
  samples per agent per step, horizon **T = {config.run.horizon}**.
- Shards are **disjoint**: 60 000 / N = 6 000 samples per agent, so the run consumes
  half the training set and no sample is seen twice.
- Model: **{config.model.input_size}×{config.model.input_size} input, 196–14–10 MLP,
  p = {config.model.num_params:,}** — small enough that a dense Diff-EKF covariance
  ({config.model.num_params}² ≈ {config.model.num_params ** 2 / 1e6:.1f}M entries) is affordable.

## Topologies (N={N_NODES}, Metropolis weights)

| Topology | Edges | Degrees | Diameter | Mixing gap |
|---|---|---|---|---|
{chr(10).join(rows)}

Mixing gap is $1-\\mathrm{{SLEM}}$: higher means information spreads faster. It is the
natural x-axis for "what does decentralization cost".

## Label skew (MNIST, N={N_NODES})

| Partition | Smallest shard | Skew | Classes per agent |
|---|---|---|---|
{chr(10).join(skews)}

Skew is the mean total-variation distance from the global class distribution. Shard
**sizes stay equal** under skew, so the non-IID experiment isolates composition rather
than confounding it with data volume.

## Decisions worth reporting

1. **14×14 input.** The phase-5 covariance budget caps p at ~3×10³. That cannot be met
   by narrowing the hidden layer — at 784 inputs it forces h≈3 — so the input dimension
   comes down instead. Phases 1–4 run the same model phase 5 will, so the comparison is
   like-for-like by construction.
2. **Total rotation capped at 45°, with α derived from the horizon.** Cumulative drift,
   not per-step drift, decides whether the task stays well-posed. The original 0.2°/step
   over 1500 steps is 300° of rotation, where a 6 is a 9.
3. **Normalization after rotation.** Otherwise every rotated image gains bright corners
   correlated with the drift state (figure 06).
4. **All learners share one environment instance.** The exactness check then holds by
   construction rather than by seed-matching, and the deviation-from-centralized metric
   becomes available immediately.
5. **Balanced shard sizes under label skew** (figure 09).
6. **The spectral gap in the plan is undefined for non-doubly-stochastic weights** — it
   returns a negative number on a star with relative-degree weights, ranking the
   fastest-mixing configuration as the worst. The code raises instead, and reports
   $1-\\mathrm{{SLEM}}$, which is always valid.

## Figures

| File | Shows |
|---|---|
| `01_topologies.png` | The eight communication graphs and their combination weights |
| `02_price_of_connectivity.png` | Mixing rate per topology — previews the headline sweep |
| `03_mnist_samples.png` | The raw data |
| `04_downsampling.png` | Why the input shrinks: the covariance budget |
| `05_rotation.png` | What drift looks like across the capped range |
| `06_normalization_order.png` | The ordering trap, and what it would have cost |
| `07_drift_schedules.png` | Linear, piecewise and sinusoidal regimes |
| `08_partition_skew.png` | What each agent actually sees, IID vs Dirichlet |
| `09_shard_sizes.png` | Why shard sizes are balanced |
| `10_received_digits.png` | The phase-1 milestone: what each agent receives, over time |
| `11_reference.png` | $e^\star$ per rotation: the curve every gap is measured against |
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path.name}")


def main() -> int:
    style()
    print(f"writing to {OUT_DIR}\n")

    if not is_cached():
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train = load_split("train", download=False)
    labels = train.labels

    figures = {
        1: lambda: fig01_topologies(),
        2: lambda: fig02_price_of_connectivity(),
        3: lambda: fig03_mnist_samples(train),
        4: lambda: fig04_downsampling(train),
        5: lambda: fig05_rotation(train),
        6: lambda: fig06_normalization_order(train),
        7: lambda: fig07_drift_schedules(),
        8: lambda: fig08_partition_skew(labels),
        9: lambda: fig09_shard_sizes(labels),
        10: lambda: fig10_received_digits(train),
        11: lambda: fig11_reference(train),
    }
    for number, build in figures.items():
        if ONLY is None or number == ONLY:
            build()

    if ONLY is None:
        write_summary(labels)
    print(f"\ndone — {len(list(OUT_DIR.glob('*')))} files in the folder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
