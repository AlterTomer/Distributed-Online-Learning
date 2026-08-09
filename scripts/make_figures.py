r"""The phase-3 figures: F1, F2, F5, F8.

Run this file directly.

    python scripts/make_figures.py                # collect, cache, and draw all
    python scripts/make_figures.py f1             # just one
    python scripts/make_figures.py --from-cache   # redraw without re-reading results
    python scripts/make_figures.py --dpi 300      # publication resolution

`IMPLEMENTATION.md` section 12 specifies the four; `docs/figures.md` explains
what each one is for and how to read it.

## Two stages, on purpose

**Collect** turns ~775 000 raw rows per experiment into the few hundred points a
figure actually draws, and writes them to `figure_data/<id>.parquet` next to the
PNGs. **Draw** renders from that. So changing a label, a colour, or the DPI is a
second of work against a small tidy table, and does not depend on `results/`
still being present -- which matters because the figure data is what gets carried
into a talk or a paper, long after a run has been superseded.

The cached table is the plotted series and nothing else:

    figure  panel  series  role  x  y  lo  hi

`role` separates a data line from a reference (e*, an off-axis baseline) so the
drawing code never has to re-derive which is which.

## Aggregation

**Counts-then-divide, never a mean of rates.** A step where one agent saw two
samples and another saw eight does not weigh those equally, and averaging the
per-agent rates would quietly reweight the result toward the agents holding the
least data. Every error rate here is
$1 - \sum n_\text{correct} / \sum n_\text{samples}$ over the rows being pooled.

Bands are +/-1 s.d. across seeds, computed on the per-seed aggregate, so a band
means "how far would a rerun move this curve".
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
OUT_DIR = Path(r"C:\Users\alter\OneDrive\Desktop\PhD\Distributed Online Learning\preliminary work")
DATA_DIR = OUT_DIR / "figure_data"
ONLY: str | None = None  # "f1" / "f2" / "f5" / "f8", or None for all
SMOOTH = 25  # rolling window in steps for error-rate curves; 0 disables
DPI = 160  # raise to 300 for print
FROM_CACHE = False  # skip collection and redraw from figure_data/

# --- palette -----------------------------------------------------------------
# Validated four-slot categorical: CVD dE 9.2 (deutan), normal-vision dE 27.6 on
# the light surface. The aqua sits at 2.74:1 against the surface, which the
# validator flags -- discharged by the direct labels every line carries.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Colour follows the *method*, never its rank in the plot, so a figure that
#: drops a learner does not repaint the survivors.
COLOURS = {
    "centralized_sgd": "#2a78d6",
    "diffusion_sgd_atc": "#eb6834",
    "diffusion_sgd_atc_plain": "#8a5cd6",
    "diffusion_sgd_cta": "#1baf7a",
    "local_only": "#1baf7a",
    "reference": MUTED,
}
LABELS = {
    "centralized_sgd": "centralized",
    "diffusion_sgd_atc": "ATC",
    "diffusion_sgd_atc_plain": "ATC (payload-matched)",
    "diffusion_sgd_cta": "CTA",
    "local_only": "local only",
    "reference": "$e^\\star$ offline",
}
ORDER = [
    "centralized_sgd",
    "diffusion_sgd_atc",
    "diffusion_sgd_atc_plain",
    "diffusion_sgd_cta",
    "local_only",
    "reference",
]

#: Which reference variant the figures quote. `shared_seed` holds theta0 fixed
#: across the rotation grid, so a difference between grid points is the rotation
#: and not the initialisation.
REFERENCE_STRATEGY = "shared_seed"

PANEL_TITLES = {
    "x1_stationary": "Stationary",
    "x2_rotating": "Rotating 45 degrees",
    "x5_abrupt_shift": "Abrupt 15-degree shift",
}


def style() -> None:
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
            "figure.dpi": DPI,
        }
    )


def despine(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


# =========================================================================== #
# collection
# =========================================================================== #


@dataclass
class Collected:
    """A figure's plotted data plus the facts its caption states."""

    frame: pd.DataFrame
    meta: dict


def available(experiment: str) -> bool:
    return bool(list((RESULTS / experiment).glob("seed_*.parquet")))


def load(experiment: str) -> pd.DataFrame:
    files = sorted((RESULTS / experiment).glob("seed_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no results for {experiment}. Run: python scripts/run_experiment.py {experiment}"
        )
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def learners_of(frame: pd.DataFrame) -> list[str]:
    present = set(frame.learner.unique())
    return [name for name in ORDER if name in present]


def error_rate(frame: pd.DataFrame, evalset: str) -> pd.DataFrame:
    """Pooled error rate per (learner, seed, t): counts summed, then divided."""
    rows = frame[(frame.evalset == evalset) & (frame.metric == "error_rate")]
    grouped = rows.groupby(["learner", "seed", "t"], as_index=False)[
        ["n_correct", "n_samples"]
    ].sum()
    grouped["error"] = 1.0 - grouped.n_correct / grouped.n_samples
    return grouped


def scalar_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """A network-level metric, from the row the runner already aggregated."""
    rows = frame[(frame.metric == metric) & (frame.node_id == "mean")]
    return rows[["learner", "seed", "t", "value"]].copy()


def band(series: pd.DataFrame, value: str, x: str = "t"):
    grouped = series.groupby(x)[value]
    return grouped.mean(), grouped.std().fillna(0.0)


def smooth(values: pd.Series, window: int = SMOOTH) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window, center=True, min_periods=1).mean()


def reference_curve(frame: pd.DataFrame) -> pd.Series | None:
    """$e^\\star$ at each step's drift state, or ``None`` if not yet trained.

    A *curve*, not a constant. Under rotation the offline reference is retrained
    per rotation, so one horizontal line would quote e* at a state most of the
    run is not in. Under a stationary schedule this collapses to a flat line on
    its own, which is the horizontal line the spec asks for.
    """
    from dekf_bench.evaluation import reference as ref
    from dekf_bench.utils.config import default_configs_dir

    path = ref.cache_path(default_configs_dir().parent / "data", REFERENCE_STRATEGY, "validation")
    if not path.is_file():
        return None
    reference = ref.load(path)
    return frame.groupby("t").drift_state.first().sort_index().map(reference.at)


def rows_for(panel, series, role, x, y, lo=None, hi=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel": panel,
            "series": series,
            "role": role,
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "lo": np.nan if lo is None else np.asarray(lo, dtype=float),
            "hi": np.nan if hi is None else np.asarray(hi, dtype=float),
        }
    )


def collect_error_panels(experiments: list[str], evalset: str = "prequential") -> Collected:
    """Shared by F1 and F2: the error curve per learner per panel, plus e*."""
    pieces, seeds = [], 0
    for experiment in experiments:
        frame = load(experiment)
        seeds = max(seeds, int(frame.seed.nunique()))
        rates = error_rate(frame, evalset)
        panel = PANEL_TITLES.get(experiment, experiment)

        ledger = frame.groupby(["learner", "seed", "t"], as_index=False).cum_scalars_tx.max()
        rates = rates.merge(ledger, on=["learner", "seed", "t"])

        for name in learners_of(frame):
            subset = rates[rates.learner == name]
            mean, spread = band(subset, "error")
            cost = subset.groupby("t").cum_scalars_tx.mean()
            piece = rows_for(
                panel,
                name,
                "line",
                mean.index,
                smooth(mean),
                smooth(mean - spread),
                smooth(mean + spread),
            )
            piece["cost"] = cost.reindex(mean.index).to_numpy(dtype=float)
            pieces.append(piece)

        star = reference_curve(frame)
        if star is not None:
            piece = rows_for(panel, "reference", "reference", star.index, star.values)
            piece["cost"] = np.nan
            pieces.append(piece)

    return Collected(pd.concat(pieces, ignore_index=True), {"seeds": seeds, "smooth": SMOOTH})


def collect_f1() -> Collected | None:
    panels = [e for e in ("x1_stationary", "x2_rotating") if available(e)]
    if not panels:
        return None
    return collect_error_panels(panels)


def collect_f2() -> Collected | None:
    return collect_f1()  # same series; F2 plots them against `cost`


def collect_f5() -> Collected | None:
    panels = [e for e in ("x1_stationary", "x2_rotating") if available(e)]
    if not panels:
        return None
    pieces, seeds = [], 0
    for experiment in panels:
        frame = load(experiment)
        seeds = max(seeds, int(frame.seed.nunique()))
        title = PANEL_TITLES.get(experiment, experiment)
        # e_cent_rel is e_cent divided by the squared norm of the mean parameter.
        #
        # The raw e_cent rises through the whole run, which reads as "diffusion
        # drifts further from centralized over time". Roughly half of that is
        # just the weights growing: ||theta||^2 nearly doubles (47.7 -> 79.5), so
        # two trajectories a fixed *relative* distance apart separate in absolute
        # terms simply by travelling further from the origin. Normalising
        # separates the two effects. The residual rise is real but costs almost
        # nothing in error (0.0749 vs 0.0762 at t=1499), because the loss surface
        # has flat directions -- e_cent measures parameter distance, not
        # disagreement about predictions.
        norms = scalar_metric(frame, "theta_mean_norm_sq")
        for metric in ("e_agree", "e_cent", "e_cent_rel"):
            source = "e_cent" if metric == "e_cent_rel" else metric
            series = scalar_metric(frame, source)
            for name in learners_of(frame):
                subset = series[series.learner == name]
                if subset.empty:
                    continue  # centralized has no e_cent against itself
                mean, _ = band(subset, "value")
                if metric == "e_cent_rel":
                    scale = norms[norms.learner == name].groupby("t").value.mean()
                    mean = (mean / scale.reindex(mean.index)).dropna()
                positive = mean[mean > 0]  # log axis: a zero is dropped, not clipped
                if positive.empty:
                    continue
                pieces.append(
                    rows_for(f"{title}|{metric}", name, "line", positive.index, positive.values)
                )
    if not pieces:
        return None
    return Collected(pd.concat(pieces, ignore_index=True), {"seeds": seeds, "smooth": 0})


def collect_f8() -> Collected | None:
    if not available("x1b_atc_vs_cta"):
        return None
    frame = load("x1b_atc_vs_cta")
    pieces = []
    rates = error_rate(frame, "prequential")
    for name in learners_of(frame):
        mean, spread = band(rates[rates.learner == name], "error")
        pieces.append(
            rows_for(
                "Error rate",
                name,
                "line",
                mean.index,
                smooth(mean),
                smooth(mean - spread),
                smooth(mean + spread),
            )
        )
    agree = scalar_metric(frame, "e_agree")
    for name in learners_of(frame):
        subset = agree[agree.learner == name]
        if subset.empty:
            continue
        mean, _ = band(subset, "value")
        positive = mean[mean > 0]
        if positive.empty:
            continue
        pieces.append(rows_for("Disagreement", name, "line", positive.index, positive.values))
    return Collected(
        pd.concat(pieces, ignore_index=True),
        {"seeds": int(frame.seed.nunique()), "smooth": SMOOTH},
    )


# =========================================================================== #
# the cache
# =========================================================================== #


def cache_write(figure_id: str, collected: Collected) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = collected.frame.copy()
    frame.insert(0, "figure", figure_id)
    frame.to_parquet(DATA_DIR / f"{figure_id}.parquet", index=False)
    (DATA_DIR / f"{figure_id}.meta.json").write_text(
        json.dumps(collected.meta, indent=2), encoding="utf-8"
    )


def cache_read(figure_id: str) -> Collected | None:
    path = DATA_DIR / f"{figure_id}.parquet"
    if not path.is_file():
        return None
    meta_path = DATA_DIR / f"{figure_id}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return Collected(pd.read_parquet(path), meta)


# =========================================================================== #
# drawing
# =========================================================================== #


#: Point size of a direct label, and the vertical room one needs including
#: leading. Two labels closer than this in display space collide.
LABEL_POINTS = 8.0
LABEL_ROOM = LABEL_POINTS * 1.55


def direct_labels(axis, entries: list[tuple[float, str, str]]) -> None:
    """Label each line just outside the right edge, nudged apart where needed.

    Converging curves -- the interesting case here -- would otherwise stack their
    labels on top of each other. The minimum gap is a real number of typographic
    points converted into data units via the axes height, rather than a fraction
    of a data range: the same fraction is generous on one panel and far too tight
    on another, which is what made the first drafts unreadable.

    Call after the axis limits are final; it reads them.
    """
    if not entries:
        return
    low, high = axis.get_ylim()
    span = high - low
    if span <= 0:
        return

    # Axes height in points, so the gap can be stated in points.
    height_points = axis.get_window_extent().height / axis.figure.dpi * 72.0
    gap = LABEL_ROOM / max(height_points, 1.0) * span

    ordered = sorted(entries, key=lambda item: item[0])
    placed: list[float] = []
    for value, _, _ in ordered:
        clamped = min(max(value, low), high)
        if placed and clamped - placed[-1] < gap:
            placed.append(placed[-1] + gap)
        else:
            placed.append(clamped)

    # If nudging pushed the stack past the top, slide the whole run down so the
    # labels stay inside the panel rather than climbing out of it.
    overflow = placed[-1] - high
    if overflow > 0:
        placed = [value - overflow for value in placed]

    transform = axis.get_yaxis_transform()
    for (value, text, colour), y in zip(ordered, placed, strict=True):
        axis.annotate(
            text,
            xy=(1.015, y),
            xycoords=transform,
            color=colour,
            fontsize=LABEL_POINTS,
            va="center",
            ha="left",
            annotation_clip=False,
        )
        if abs(y - value) > 0.15 * gap:  # nudged: a leader so it still reads
            axis.plot(
                [1.002, 1.012],
                [value, y],
                transform=transform,
                color=colour,
                linewidth=0.6,
                alpha=0.55,
                clip_on=False,
            )


def shared_legend(figure, names: list[str], ncol: int | None = None) -> None:
    """One legend for the whole figure, below it.

    Inside the axes it lands on the curves -- the convergence tail is the part
    worth reading, and a box over it is the thing that made the first drafts
    unreadable. Below the figure it can never collide, and one legend for a
    multi-panel figure also states that the colours mean the same thing in each.
    """
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=COLOURS[name],
            linewidth=2.0,
            linestyle="--" if name == "reference" else "-",
            label=LABELS[name],
        )
        for name in names
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol or min(len(handles), 5),
        columnspacing=2.6,
        handlelength=2.2,
        handletextpad=0.7,
        borderaxespad=0.0,
    )


def series_present(frame: pd.DataFrame) -> list[str]:
    present = set(frame.series.unique())
    return [name for name in ORDER if name in present]


def draw_f1(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    panels = list(dict.fromkeys(frame.panel))
    figure, axes = plt.subplots(
        1, len(panels), figsize=(max(6.8, 5.9 * len(panels)), 3.9), sharey=True
    )
    axes = np.atleast_1d(axes)

    for axis, panel in zip(axes, panels, strict=True):
        block = frame[frame.panel == panel]
        endpoints = []
        for name in series_present(block):
            line = block[block.series == name]
            if line.role.iloc[0] == "reference":
                axis.plot(line.x, line.y, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
                axis.annotate(
                    f"$e^\\star$ = {line.y.iloc[0]:.3f}",
                    xy=(0.02, line.y.iloc[0]),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 5),
                    textcoords="offset points",
                    color=INK_SECONDARY,
                    fontsize=8,
                )
                continue
            axis.fill_between(
                line.x, line.lo, line.hi, color=COLOURS[name], alpha=0.13, linewidth=0
            )
            axis.plot(line.x, line.y, color=COLOURS[name])
            endpoints.append((float(line.y.iloc[-1]), LABELS[name], COLOURS[name]))
        axis.set_title(panel)
        axis.set_xlabel("step $t$")
        despine(axis)
        direct_labels(axis, endpoints)

    axes[0].set_ylabel("prequential error rate")
    figure.subplots_adjust(right=0.86, wspace=0.30)
    shared_legend(figure, series_present(frame))
    figure.suptitle(
        f"F1  Error rate over time  ({meta.get('seeds', '?')} seeds, "
        f"band +/-1 s.d., {meta.get('smooth', SMOOTH)}-step smoothing)",
        y=1.04,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    save(figure, "12_f1_error_vs_time.png")


def draw_f2(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    panels = list(dict.fromkeys(frame.panel))
    figure, axes = plt.subplots(
        1, len(panels), figsize=(max(6.8, 5.9 * len(panels)), 3.9), sharey=True
    )
    axes = np.atleast_1d(axes)
    drawn: list[str] = []

    for axis, panel in zip(axes, panels, strict=True):
        block = frame[frame.panel == panel]
        for name in series_present(block):
            line = block[block.series == name]
            if line.role.iloc[0] == "reference" or line.cost.isna().all():
                continue
            if float(np.nanmax(line.cost)) == 0.0:
                # centralized and local_only sit at zero here for opposite
                # reasons: one pools every sample off-graph, the other never
                # speaks. A point at x=0 would read as "free and this good", so
                # both become horizontal references instead (design note D30).
                level = float(line.y.iloc[-1])
                axis.axhline(
                    level,
                    color=COLOURS[name],
                    linestyle=":" if name == "local_only" else "--",
                    linewidth=1.5,
                )
                # Labelled in place. Three different things use dashed
                # horizontals across F1 and F2 -- e*, centralized, local_only --
                # and an unlabelled one here reads as the offline bound.
                axis.annotate(
                    f"{LABELS[name]} (no cost on this axis)",
                    xy=(0.015, level),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 4),
                    textcoords="offset points",
                    color=COLOURS[name],
                    fontsize=7.5,
                )
                drawn.append(name)
                continue
            axis.plot(line.cost, line.y, color=COLOURS[name])
            drawn.append(name)
        axis.set_xscale("log")
        axis.set_title(panel)
        axis.set_xlabel("cumulative scalars transmitted")
        despine(axis)

    axes[0].set_ylabel("prequential error rate")
    shared_legend(figure, [n for n in ORDER if n in set(drawn)])
    figure.suptitle(
        f"F2  Error rate against communication cost  ({meta.get('seeds', '?')} seeds; "
        "dashed = no cost on this axis)",
        y=1.04,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    save(figure, "13_f2_error_vs_communication.png")


def draw_f5(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    keys = list(dict.fromkeys(frame.panel))
    columns = list(dict.fromkeys(key.split("|")[0] for key in keys))
    # The third row is the second one normalised. The raw E_cent rises through
    # the whole run, which invites "diffusion drifts further from centralized
    # over time" -- but roughly half of that is the weights themselves growing
    # (||theta||^2 nearly doubles), so a fixed *relative* separation shows up as
    # a growing absolute distance. Dividing by ||theta||^2 separates the two.
    metrics = [
        ("e_agree", "$E_{\\mathrm{agree}}$"),
        ("e_cent", "$E_{\\mathrm{cent}}$"),
        ("e_cent_rel", "$E_{\\mathrm{cent}}\\,/\\,\\Vert\\bar{\\theta}\\Vert^{2}$"),
    ]

    figure, axes = plt.subplots(
        len(metrics),
        len(columns),
        # A single column would otherwise render tall and narrow; the width floor
        # keeps one panel readable without distorting the two-panel case.
        figsize=(max(7.6, 5.9 * len(columns)), 2.9 * len(metrics)),
        sharex="col",
        squeeze=False,
    )
    for column, title in enumerate(columns):
        for row, (metric, label) in enumerate(metrics):
            axis = axes[row][column]
            block = frame[frame.panel == f"{title}|{metric}"]
            for name in series_present(block):
                line = block[block.series == name]
                axis.plot(line.x, line.y, color=COLOURS[name])
            axis.set_yscale("log")
            despine(axis)
            if column == 0:
                axis.set_ylabel(label)
            if row == 0:
                axis.set_title(title)
            if row == len(metrics) - 1:
                axis.set_xlabel("step $t$")

    shared_legend(figure, series_present(frame))
    figure.suptitle(
        f"F5  Disagreement and deviation from centralized  ({meta.get('seeds', '?')} seeds; "
        "log y, a zero is dropped not clipped)",
        y=1.0,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(figure, "14_f5_disagreement.png")


def draw_f8(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))

    block = frame[frame.panel == "Error rate"]
    endpoints = []
    for name in series_present(block):
        line = block[block.series == name]
        axes[0].fill_between(line.x, line.lo, line.hi, color=COLOURS[name], alpha=0.13, linewidth=0)
        axes[0].plot(line.x, line.y, color=COLOURS[name])
        endpoints.append((float(line.y.iloc[-1]), LABELS[name], COLOURS[name]))
    axes[0].set_ylabel("prequential error rate")
    axes[0].set_xlabel("step $t$")
    axes[0].set_title("Error rate")
    despine(axes[0])
    direct_labels(axes[0], endpoints)

    block = frame[frame.panel == "Disagreement"]
    endpoints = []
    for name in series_present(block):
        line = block[block.series == name]
        axes[1].plot(line.x, line.y, color=COLOURS[name])
        endpoints.append((float(line.y.iloc[-1]), LABELS[name], COLOURS[name]))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("$E_{\\mathrm{agree}}$")
    axes[1].set_xlabel("step $t$")
    axes[1].set_title("Disagreement (log y)")
    despine(axes[1])
    direct_labels(axes[1], endpoints)

    figure.subplots_adjust(right=0.88, wspace=0.34)
    shared_legend(figure, series_present(frame))
    figure.suptitle(
        f"F8  ATC vs CTA at identical communication cost  ({meta.get('seeds', '?')} seeds)",
        y=1.04,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    save(figure, "15_f8_atc_vs_cta.png")


def save(figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_DIR / name, bbox_inches="tight", dpi=DPI)
    plt.close(figure)
    print(f"  wrote {name}")


# =========================================================================== #

# =========================================================================== #
# F3 -- the price of connectivity
# =========================================================================== #

#: The windows F3 reports, and what the names claim.
#:
#: `settled` means the gap has stopped changing -- not merely "the end of the
#: run". The test is whether a line fitted over the last 500 steps has a slope
#: exceeding the seed s.d.; measured, no topology does, and the two with
#: meaningful gaps (path, star) stop moving by t ~ 1075:
#:
#:     topology        final gap   seed sd   slope/500
#:     path            0.0035      0.0023    -0.0021
#:     star            0.0077      0.0037    -0.0026
#:     ring            0.0014      0.0012    -0.0008
#:     complete        0.0000      0.0000    +0.0000
#:
#: `transient` is a window while the gap is still falling for every topology,
#: which is where connectivity matters most -- the spread across topologies is
#: several times larger there than at the end.
#:
#: `test_figures.py` asserts the settled window really is flat, so a change to
#: run.horizon cannot silently leave this label describing something else.
F3_WINDOWS = [("transient", 150, 300), ("settled", 1400, 1500)]


def topology_runs() -> list[str]:
    return sorted(p.name[3:] for p in RESULTS.glob("x3_*") if list(p.glob("seed_*.parquet")))


def paired_gap(frame: pd.DataFrame, lo: int, hi: int) -> tuple[float, float]:
    """Mean and s.d. of $e_\\text{ATC} - e_\\text{cent}$, paired within seed.

    **Paired, not a difference of means.** Both learners run on the same
    environment at the same seed, so most of the run-to-run variation is common
    to them and cancels in the difference. Measured on X3: the individual error
    rates carry a seed s.d. of 0.0035 while the paired gap carries 0.0012 -- the
    difference between a connectivity axis that is mostly noise and one that
    separates at 3 sigma.
    """
    rows = frame[
        (frame.evalset == "current")
        & (frame.metric == "error_rate")
        & (frame.t >= lo)
        & (frame.t < hi)
    ]
    per_seed = rows.groupby(["learner", "seed"])[["n_correct", "n_samples"]].sum()
    per_seed["err"] = 1.0 - per_seed.n_correct / per_seed.n_samples
    try:
        gaps = per_seed.loc["diffusion_sgd_atc", "err"] - per_seed.loc["centralized_sgd", "err"]
    except KeyError:
        return float("nan"), float("nan")
    return float(gaps.mean()), float(gaps.std())


def collect_f3() -> Collected | None:
    import torch

    from dekf_bench.env.graph import build_graph, default_topology_params

    topologies = topology_runs()
    if not topologies:
        return None

    defaults = default_topology_params(10)
    pieces, seeds = [], 0
    for topology in topologies:
        frame = load(f"x3_{topology}")
        seeds = max(seeds, int(frame.seed.nunique()))
        graph = build_graph(
            topology, 10, "metropolis", defaults.get(topology), torch.Generator().manual_seed(0)
        )
        try:
            spectral = graph.spectral_gap
        except Exception:  # noqa: BLE001 - undefined for non-doubly-stochastic weights
            spectral = float("nan")

        for name, lo, hi in F3_WINDOWS:
            mean, spread = paired_gap(frame, lo, hi)
            self_weight = float(graph.weights.double().diag().mean())
            for axis_name, axis_value in (
                ("spectral", spectral),
                ("selfweight", self_weight),
            ):
                piece = rows_for(
                    f"{name}|{axis_name}",
                    topology,
                    "point",
                    [axis_value],
                    [mean],
                    [mean - spread],
                    [mean + spread],
                )
                piece["cost"] = float("nan")
                pieces.append(piece)

    return Collected(pd.concat(pieces, ignore_index=True), {"seeds": seeds, "smooth": 0})


def draw_f3(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    windows = [w for w, _, _ in F3_WINDOWS]
    # Two genuinely different predictors. The first version of this figure plotted
    # the spectral gap against the mixing gap, which are the *same quantity* under
    # symmetric (Metropolis) weights -- two panels of one plot.
    #
    # Both correlate in the expected direction. The mean self-weight fits better
    # (Spearman +0.96, p=0.0005, against -0.79, p=0.036) and is the one that ranks
    # `star` correctly: 7th of 7 rather than 3rd. See results.md 7.3.
    axes_kinds = [
        ("spectral", "spectral gap  $1-\\rho$   (Spearman $-0.79$)"),
        ("selfweight", "mean self-weight  $\\bar{a}_{vv}$   (Spearman $+0.96$)"),
    ]

    figure, axes = plt.subplots(len(windows), 2, figsize=(11.0, 4.0 * len(windows)), squeeze=False)
    for row, window in enumerate(windows):
        for column, (kind, label) in enumerate(axes_kinds):
            axis = axes[row][column]
            block = frame[frame.panel == f"{window}|{kind}"].sort_values("x")
            axis.errorbar(
                block.x,
                block.y,
                yerr=(block.y - block.lo).abs(),
                fmt="o",
                markersize=6,
                capsize=3,
                linewidth=1.2,
                color=COLOURS["diffusion_sgd_atc"],
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
            )
            for point in block.itertuples():
                axis.annotate(
                    point.series,
                    xy=(point.x, point.y),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=INK_SECONDARY,
                )
            axis.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=0)
            # The spectral gap spans 0.03 to 1.0 and reads better on a log axis;
            # the self-weight is a fraction on [0, 1], where log adds nothing.
            axis.set_xscale("log" if kind == "spectral" else "linear")
            axis.set_xlabel(label)
            despine(axis)
            if column == 0:
                axis.set_ylabel(f"$e_{{\\mathrm{{ATC}}}}-e_{{\\mathrm{{cent}}}}$  ({window})")

    figure.suptitle(
        f"F3  The price of connectivity  ({meta.get('seeds', '?')} seeds; gap paired "
        "within seed, bars +/-1 s.d.)",
        y=0.99,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save(figure, "16_f3_price_of_connectivity.png")


# =========================================================================== #
# F4 -- per-agent spread
# =========================================================================== #


def collect_f4() -> Collected | None:
    panels = [e for e in ("x1_stationary", "x2_rotating") if available(e)]
    if not panels:
        return None
    pieces, seeds = [], 0
    for experiment in panels:
        frame = load(experiment)
        seeds = max(seeds, int(frame.seed.nunique()))
        title = PANEL_TITLES.get(experiment, experiment)
        rows = frame[
            (frame.evalset == "current")
            & (frame.metric == "error_rate")
            & (frame.node_id != "mean")
        ]
        for name in learners_of(frame):
            subset = rows[rows.learner == name]
            if subset.empty:
                continue
            # Per agent first, then the spread across agents -- averaged over
            # seeds. This is agent-to-agent disagreement in *performance*, which
            # is a different thing from E_agree's disagreement in parameters.
            per_node = subset.groupby(["t", "node_id"])[["n_correct", "n_samples"]].sum()
            per_node["err"] = 1.0 - per_node.n_correct / per_node.n_samples
            by_step = per_node.groupby("t").err
            mean, low, high = by_step.mean(), by_step.min(), by_step.max()
            piece = rows_for(title, name, "line", mean.index, mean.values, low.values, high.values)
            piece["cost"] = float("nan")
            pieces.append(piece)
    if not pieces:
        return None
    return Collected(pd.concat(pieces, ignore_index=True), {"seeds": seeds, "smooth": 0})


def draw_f4(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    panels = list(dict.fromkeys(frame.panel))
    figure, axes = plt.subplots(
        1, len(panels), figsize=(max(6.8, 5.9 * len(panels)), 3.9), sharey=True
    )
    axes = np.atleast_1d(axes)

    for axis, panel in zip(axes, panels, strict=True):
        block = frame[frame.panel == panel]
        endpoints = []
        for name in series_present(block):
            line = block[block.series == name].sort_values("x")
            axis.fill_between(
                line.x, line.lo, line.hi, color=COLOURS[name], alpha=0.16, linewidth=0
            )
            axis.plot(line.x, line.y, color=COLOURS[name])
            endpoints.append((float(line.y.iloc[-1]), LABELS[name], COLOURS[name]))
        axis.set_title(panel)
        axis.set_xlabel("step $t$")
        despine(axis)
        direct_labels(axis, endpoints)

    axes[0].set_ylabel("held-out error rate")
    figure.subplots_adjust(right=0.86, wspace=0.30)
    shared_legend(figure, series_present(frame))
    figure.suptitle(
        f"F4  Per-agent spread  ({meta.get('seeds', '?')} seeds; line = mean over agents, "
        "band = min to max)",
        y=1.04,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    save(figure, "17_f4_per_agent_spread.png")


# =========================================================================== #
# F6a / F6b -- the sparsity plane
# =========================================================================== #

#: One hue, light to dark, for a magnitude that has a natural zero.
SEQUENTIAL = "Blues"
#: Two hues either side of a neutral midpoint, for a signed quantity.
DIVERGING = "RdBu_r"

X4_SAMPLES = [1, 2, 4, 8]
X4_PI = [0.25, 0.5, 1.0]


def _sweep_cells(*tags: str) -> pd.DataFrame:
    """Tuning cells for one or more tags, concatenated.

    Several tags because a learner added after a sweep can be swept alone
    under its own tag and merged: the environment does not depend on which
    learners run (verified -- identical observation streams), so the cells
    are directly comparable. That turns a 2.7-hour re-run into 45 minutes.
    """
    path = RESULTS / "sweep" / "cells.jsonl"
    if not path.is_file():
        return pd.DataFrame()
    wanted = set(tags)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cell = json.loads(line)
        if cell.get("tag") not in wanted:
            continue
        for learner, error in cell["errors"].items():
            rows.append(
                {
                    "learner": learner,
                    "n": cell["n"],
                    "pi": cell["label_availability"],
                    "optimizer": cell["optimizer"],
                    "lr": cell["lr"],
                    "seed": cell["seed"],
                    "error": error,
                }
            )
    return pd.DataFrame(rows)


def x4_tuned() -> pd.DataFrame:
    """Each learner's best (optimizer, lr) in each $(n, \\pi)$ cell.

    Per-cell rather than one global setting, because $\\pi_\\text{lab}$ changes
    each method's *effective* step size by a different factor: an idle agent
    contributes its unchanged theta to the combine, so ATC steps by
    $\\eta\\, n_\\text{active}/N$ while centralized takes the full $\\eta$. At
    $\\pi = 0.25$ that is 4x, and comparing them at one lr compares step sizes
    rather than methods (`results.md` 9.1).
    """
    frame = _sweep_cells("x4", "x4plain")
    if frame.empty:
        return frame

    # The payload-matched variant is tuned WITHIN the plain-SGD arm only.
    #
    # Both names map to one class, and the sweep sets the optimizer for every
    # learner it runs -- so left unconstrained, `atc_plain` picks momentum and
    # becomes numerically identical to ATC. Its tuned "payload cost" then comes
    # out as exactly 0.000 in all twelve cells, which is not a small effect but
    # a definition being overridden: carrying no optimizer state is precisely
    # what makes it p per link rather than 2p.
    plain = frame.learner == "diffusion_sgd_atc_plain"
    frame = frame[~plain | (frame.optimizer == "sgd")]

    averaged = frame.groupby(["learner", "n", "pi", "optimizer", "lr"], as_index=False).error.mean()
    return averaged.loc[averaged.groupby(["learner", "n", "pi"]).error.idxmin()]


def x4_headline() -> pd.DataFrame:
    """Each learner's error at its HEADLINE lr, taken from the tuning sweep.

    Deliberately from the sweep and not from the X4 experiment runs, even though
    those have five seeds against the sweep's two. F6b subtracts this from the
    per-cell best, and the two terms have to be the *same estimator*: mixing a
    five-seed number with a two-seed one produced penalties as low as -0.04,
    which is impossible for a quantity defined as "headline minus the minimum
    over a grid containing the headline".

    The five-seed X4 runs remain the numbers quoted everywhere else; they are
    simply not what this particular difference should be built from.
    """
    from dekf_bench.utils.config import load_config

    headline = {e.name: (e.optimizer, e.lr) for e in load_config("x1_stationary").learners}
    frame = _sweep_cells("x4", "x4plain")
    if frame.empty:
        return frame
    averaged = frame.groupby(["learner", "n", "pi", "optimizer", "lr"], as_index=False).error.mean()
    keep = [
        (row.learner in headline) and (row.optimizer, row.lr) == headline[row.learner]
        for row in averaged.itertuples()
    ]
    return averaged[keep][["learner", "n", "pi", "error"]].rename(columns={"error": "fixed"})


def x4_fixed_from_runs() -> pd.DataFrame:
    """The headline-tuning error from the X4 experiment runs (five seeds).

    Used for the reported tables, not for F6b's difference -- see x4_headline.
    """
    rows = []
    for n in X4_SAMPLES:
        for pi in X4_PI:
            name = f"x4_n{n}_pi{pi}"
            if not available(name):
                continue
            frame = load(name)
            horizon = int(frame.t.max()) + 1
            block = frame[
                (frame.evalset == "current")
                & (frame.metric == "error_rate")
                & (frame.t >= horizon - 100)
            ]
            grouped = block.groupby("learner")[["n_correct", "n_samples"]].sum()
            for learner, row in grouped.iterrows():
                rows.append(
                    {
                        "learner": learner,
                        "n": n,
                        "pi": pi,
                        "error": 1.0 - row.n_correct / row.n_samples,
                    }
                )
    return pd.DataFrame(rows)


def _grid(frame: pd.DataFrame, learner: str, value: str = "error") -> np.ndarray:
    """A (pi, n) matrix, NaN where a cell has not been measured."""
    out = np.full((len(X4_PI), len(X4_SAMPLES)), np.nan)
    subset = frame[frame.learner == learner]
    for row in subset.itertuples():
        if row.n in X4_SAMPLES and row.pi in X4_PI:
            out[X4_PI.index(row.pi), X4_SAMPLES.index(row.n)] = getattr(row, value)
    return out


#: Type sizes for the heatmaps, which are set larger than the rest of the
#: figures on purpose. These two figures go on slides, where the layout gives a
#: wide image the width of the frame and whatever height the notes leave -- a
#: four-panel figure is therefore scaled to about 0.7x. At the default 7.5pt the
#: cell values land near 5pt on the projected slide, which is not readable from
#: a room. Sized so they survive the scaling; standalone they look generous.
HEATMAP_TITLE_PT = 12
HEATMAP_CELL_PT = 10.5

#: Half-width figure size for the two-panel slide variants. Chosen for an aspect
#: near 1.7, which puts the slide layout in its *image-left, notes-right* branch
#: rather than its *image-across-the-top* one -- the wide branch caps a figure at
#: whatever height the notes leave (~2.9in) and shrinks it to 0.64x, while the
#: tall branch gives it 7.5in of width and leaves it near 1:1. Same figure, same
#: type, more than twice the projected size.
HEATMAP_HALF_SIZE = (7.6, 4.4)


def _heatmap(axis, matrix, title, cmap, vmin=None, vmax=None, fmt="{:.3f}"):
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    axis.set_xticks(range(len(X4_SAMPLES)), [str(v) for v in X4_SAMPLES])
    axis.set_yticks(range(len(X4_PI)), [str(v) for v in X4_PI])
    axis.set_xlabel("$n$  (samples per agent per step)")
    axis.set_title(title, fontsize=HEATMAP_TITLE_PT)
    axis.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                axis.text(
                    j, i, "--", ha="center", va="center", fontsize=HEATMAP_CELL_PT, color=MUTED
                )
                continue
            # Ink on light cells, paper on dark ones -- decided from the cell's
            # *actual* mapped luminance rather than from where the value sits in
            # the range. Those differ for a diverging map, where both ends are
            # dark and the middle is light: a range-based rule put paper text on
            # the pale midtones and ink on the dark extremes, i.e. exactly
            # backwards, and made half of F6a's third panel unreadable.
            rgba = image.cmap(image.norm(matrix[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                j,
                i,
                fmt.format(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=HEATMAP_CELL_PT,
                color=INK if luminance > 0.55 else SURFACE,
            )
    return image


def collect_f6a() -> Collected | None:
    tuned = x4_tuned()
    if tuned.empty:
        return None
    pieces = []
    for row in tuned.itertuples():
        piece = rows_for(f"{row.learner}|tuned", row.learner, "cell", [row.n], [row.error])
        piece["pi"] = row.pi
        piece["lr"] = row.lr
        piece["optimizer"] = row.optimizer
        piece["cost"] = float("nan")
        pieces.append(piece)
    return Collected(pd.concat(pieces, ignore_index=True), {"seeds": 2, "smooth": 0})


@dataclass
class Panel:
    """One heatmap in a multi-panel figure: enough to draw it anywhere."""

    matrix: np.ndarray
    title: str
    cmap: Any
    vmin: float | None = None
    vmax: float | None = None


def _panel_figure(
    panels: list[Panel],
    suptitle: str,
    filename: str,
    *,
    figsize: tuple[float, float] | None = None,
    shared_bar: str | None = None,
) -> None:
    """Render `panels` side by side and save.

    Split out so the four-panel figure the docs use and the two-panel halves the
    slides use are drawn by *one* code path from *one* panel list. Drawing them
    separately would let the halves drift from the whole -- and the halves carry
    pre-computed `vmin`/`vmax`, so a scale that silently differed between a slide
    and the document would be invisible in both.
    """
    width = figsize or (4.2 * len(panels), 4.0)
    figure, axes = plt.subplots(1, len(panels), figsize=width, constrained_layout=True)
    axes = np.atleast_1d(axes)

    image = None
    for axis, panel in zip(axes, panels, strict=True):
        image = _heatmap(axis, panel.matrix, panel.title, panel.cmap, panel.vmin, panel.vmax)
        despine(axis)
        if shared_bar is None:
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if shared_bar is not None:
        # One bar, because every panel is on one scale -- which is the point of
        # the figure. Per-panel bars would autoscale each to its own range and
        # suggest the opposite.
        figure.colorbar(image, ax=list(axes), fraction=0.03, pad=0.02, label=shared_bar)

    axes[0].set_ylabel("$\\pi_{\\mathrm{lab}}$")
    figure.suptitle(suptitle, y=1.10, fontsize=11, color=INK, weight="bold")
    save(figure, filename)


def _f6a_panels() -> list[Panel]:
    tuned = x4_tuned()
    atc = _grid(tuned, "diffusion_sgd_atc")
    local = _grid(tuned, "local_only")
    central = _grid(tuned, "centralized_sgd")
    plain = _grid(tuned, "diffusion_sgd_atc_plain")
    limit = float(np.nanmax(np.abs(atc - central))) or 0.01
    return [
        Panel(atc, "ATC error", SEQUENTIAL),
        Panel(local - atc, "cooperation gap  (local $-$ ATC)", SEQUENTIAL),
        Panel(atc - central, "pooling gap  (ATC $-$ centralized)", DIVERGING, -limit, limit),
        # What the second p scalars per link are worth. Sequential rather than
        # diverging -- properly tuned it is positive in all twelve cells, so a
        # neutral midpoint would imply a sign change that does not occur.
        Panel(plain - atc, "payload cost  (payload-matched $-$ ATC)", SEQUENTIAL),
    ]


def draw_f6a(collected: Collected) -> None:
    panels = _f6a_panels()
    title = "F6a  The sparsity plane, each method at its own tuned lr per cell"
    _panel_figure(panels, title, "18_f6a_sparsity_tuned.png")

    # Slide halves. Each panel carries its own colourbar here, so a split costs
    # nothing -- unlike F6b, where the shared scale has to be carried across.
    _panel_figure(
        panels[:2],
        "F6a (1/2)  What diffusion achieves, and what cooperation is worth",
        "18a_f6a_achievement.png",
        figsize=HEATMAP_HALF_SIZE,
    )
    _panel_figure(
        panels[2:],
        "F6a (2/2)  What it costs against pooled data, and what halving the message costs",
        "18b_f6a_costs.png",
        figsize=HEATMAP_HALF_SIZE,
    )


def collect_f6b() -> Collected | None:
    tuned, fixed = x4_tuned(), x4_headline()
    if tuned.empty or fixed.empty:
        return None
    merged = fixed.merge(tuned[["learner", "n", "pi", "error"]], on=["learner", "n", "pi"])
    merged["penalty"] = merged.fixed - merged.error
    pieces = []
    for row in merged.itertuples():
        piece = rows_for(f"{row.learner}|penalty", row.learner, "cell", [row.n], [row.penalty])
        piece["pi"] = row.pi
        piece["cost"] = float("nan")
        pieces.append(piece)
    return Collected(pd.concat(pieces, ignore_index=True), {"seeds": 3, "smooth": 0})


def draw_f6b(collected: Collected) -> None:
    tuned, fixed = x4_tuned(), x4_headline()
    merged = fixed.merge(tuned[["learner", "n", "pi", "error"]], on=["learner", "n", "pi"])
    merged["penalty"] = merged.fixed - merged.error

    names = [n for n in ORDER if n in set(merged.learner)]
    # The limit is taken over EVERY panel and then handed to each one, so the
    # slide halves stay on the whole figure's scale. Letting a half rescale to
    # its own two panels would make ATC's near-blank plane look like
    # centralized's -- destroying the only comparison this figure makes, and
    # doing it invisibly, since each half would still look internally sensible.
    limit = max(float(np.nanmax(np.abs(_grid(merged, name, "penalty")))) for name in names)
    panels = [
        Panel(_grid(merged, name, "penalty"), LABELS[name], SEQUENTIAL, 0.0, limit or 0.01)
        for name in names
    ]

    title = "F6b  The cost of not re-tuning:  error(headline lr) $-$ error(best lr for this cell)"
    _panel_figure(panels, title, "19_f6b_cost_of_not_retuning.png", shared_bar="penalty")

    # Slide halves, ordered so the headline contrast -- centralized's 0.183
    # against ATC's 0.028 -- lands on one slide rather than being split across
    # two. Both halves say "shared scale" in their title because a reader who
    # sees only one of them has no way to know.
    half = len(panels) // 2
    _panel_figure(
        panels[:half],
        "F6b (1/2)  Centralized against ATC  (shared colour scale with 2/2)",
        "19a_f6b_headline.png",
        figsize=HEATMAP_HALF_SIZE,
        shared_bar="penalty",
    )
    _panel_figure(
        panels[half:],
        "F6b (2/2)  The two cheaper methods  (shared colour scale with 1/2)",
        "19b_f6b_baselines.png",
        figsize=HEATMAP_HALF_SIZE,
        shared_bar="penalty",
    )


# =========================================================================== #
# F7 -- the adaptation transient
# =========================================================================== #

#: The window the spec asks for, around the change point.
F7_BEFORE, F7_AFTER = 50, 300


def collect_f7() -> Collected | None:
    if not available("x5_abrupt_shift"):
        return None
    frame = load("x5_abrupt_shift")

    # The change point is read from the data rather than hardcoded: the drift
    # state is recorded per step, so the shift is wherever it actually changes.
    states = frame.groupby("t").drift_state.first().sort_index()
    changed = states.ne(states.shift()).to_numpy()
    changed[0] = False
    points = list(states.index[changed])
    if not points:
        return None
    star = int(points[0])

    lo, hi = max(0, star - F7_BEFORE), star + F7_AFTER
    rates = error_rate(frame, "current")
    window = rates[(rates.t >= lo) & (rates.t < hi)]

    pieces = []
    for name in learners_of(frame):
        subset = window[window.learner == name]
        mean, spread = band(subset, "error")
        # NOT smoothed. The `current` evalset is scored every `eval_every`
        # steps (25), not every step, so a rolling window of w points spans
        # 25w steps. A 9-point window here erased the entire transient: the
        # shift shows up in a *single* evaluation point (0.097 -> 0.174 -> 0.136)
        # and averaging over 225 steps flattened it into a smooth decline.
        piece = rows_for("transient", name, "line", mean.index, mean, mean - spread, mean + spread)
        piece["cost"] = float("nan")
        pieces.append(piece)
    return Collected(
        pd.concat(pieces, ignore_index=True),
        {"seeds": int(frame.seed.nunique()), "smooth": 0, "change_point": star},
    )


def draw_f7(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    star = meta.get("change_point", 500)
    figure, axis = plt.subplots(figsize=(7.6, 4.2))

    endpoints = []
    for name in series_present(frame):
        line = frame[frame.series == name].sort_values("x")
        axis.fill_between(line.x, line.lo, line.hi, color=COLOURS[name], alpha=0.14, linewidth=0)
        axis.plot(line.x, line.y, color=COLOURS[name])
        endpoints.append((float(line.y.iloc[-1]), LABELS[name], COLOURS[name]))

    axis.axvline(star, color=INK_SECONDARY, linestyle="--", linewidth=1.2, zorder=1)
    axis.annotate(
        f"15$^\\circ$ shift at $t={star}$",
        xy=(star, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )
    axis.set_xlabel("step $t$")
    axis.set_ylabel("held-out error rate")
    despine(axis)
    direct_labels(axis, endpoints)
    figure.subplots_adjust(right=0.82)
    shared_legend(figure, series_present(frame))
    figure.suptitle(
        f"F7  Adaptation after an abrupt shift  ({meta.get('seeds', '?')} seeds)",
        y=1.02,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    save(figure, "20_f7_adaptation_transient.png")


# =========================================================================== #
# F9 -- non-IID
# =========================================================================== #

X6_BETAS = [0.1, 1.0, 100.0]


def collect_f9() -> Collected | None:
    pieces, seeds = [], 0
    for beta in X6_BETAS:
        name = f"x6_beta{beta}"
        if not available(name):
            continue
        frame = load(name)
        seeds = max(seeds, int(frame.seed.nunique()))
        horizon = int(frame.t.max()) + 1
        block = frame[
            (frame.evalset == "current")
            & (frame.metric == "error_rate")
            & (frame.t >= horizon - 100)
        ]
        per_seed = block.groupby(["learner", "seed"])[["n_correct", "n_samples"]].sum()
        per_seed["err"] = 1.0 - per_seed.n_correct / per_seed.n_samples
        for learner in learners_of(frame):
            values = per_seed.loc[learner, "err"]
            piece = rows_for(
                "error",
                learner,
                "point",
                [beta],
                [float(values.mean())],
                [float(values.mean() - values.std())],
                [float(values.mean() + values.std())],
            )
            piece["cost"] = float("nan")
            pieces.append(piece)
        # The cooperation gap, paired within seed so common noise cancels.
        gaps = per_seed.loc["local_only", "err"] - per_seed.loc["diffusion_sgd_atc", "err"]
        piece = rows_for(
            "gap",
            "cooperation",
            "point",
            [beta],
            [float(gaps.mean())],
            [float(gaps.mean() - gaps.std())],
            [float(gaps.mean() + gaps.std())],
        )
        piece["cost"] = float("nan")
        pieces.append(piece)
    if not pieces:
        return None
    return Collected(pd.concat(pieces, ignore_index=True), {"seeds": seeds, "smooth": 0})


def draw_f9(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    block = frame[frame.panel == "error"]
    for name in series_present(block):
        line = block[block.series == name].sort_values("x")
        axes[0].errorbar(
            line.x,
            line.y,
            yerr=(line.y - line.lo).abs(),
            fmt="o-",
            markersize=5,
            capsize=3,
            linewidth=1.6,
            color=COLOURS[name],
            label=LABELS[name],
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Dirichlet $\\beta$   (smaller = more label skew)")
    axes[0].set_ylabel("held-out error rate")
    axes[0].set_title("Error under label skew")
    despine(axes[0])

    gap = frame[frame.panel == "gap"].sort_values("x")
    axes[1].errorbar(
        gap.x,
        gap.y,
        yerr=(gap.y - gap.lo).abs(),
        fmt="o-",
        markersize=6,
        capsize=3,
        linewidth=1.8,
        # Neutral ink, NOT a palette hue. This is a *derived* quantity -- the
        # difference between two rates -- and colour in this project follows the
        # method. Drawing it in local_only's aqua said "this is local_only's
        # error rate", which is the one thing it is not.
        color=INK_SECONDARY,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Dirichlet $\\beta$")
    axes[1].set_ylabel("cooperation gap  (local $-$ ATC)")
    axes[1].set_title("What cooperation is worth  (a difference, not a rate)")
    despine(axes[1])

    figure.subplots_adjust(wspace=0.28)
    shared_legend(figure, series_present(block))
    figure.suptitle(
        f"F9  Non-IID: cooperation matters most when agents see different classes  "
        f"({meta.get('seeds', '?')} seeds)",
        y=1.03,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    save(figure, "21_f9_non_iid.png")


# =========================================================================== #
# F10 -- forgetting
# =========================================================================== #


def collect_f10() -> Collected | None:
    """Current vs backward error under the sinusoidal schedule.

    The `backward` evalset scores the model at a rotation it visited *earlier*
    and has since left. Its gap to `current` is forgetting: how much worse the
    model is on a state it used to handle than on the one in front of it.

    Only meaningful where the schedule revisits states. Under `linear` the
    rotation never returns, so the probe would be asking about a state the model
    will never face again; under `stationary` there is no earlier state at all.
    The probe is defined for 97% of steps here against 67% and 0%.
    """
    if not available("x7_sinusoidal"):
        return None
    frame = load("x7_sinusoidal")
    pieces = []
    for evalset in ("current", "backward"):
        rates = error_rate(frame, evalset)
        if rates.empty:
            continue
        for name in learners_of(frame):
            subset = rates[rates.learner == name]
            mean, spread = band(subset, "error")
            piece = rows_for(evalset, name, "line", mean.index, mean, mean - spread, mean + spread)
            piece["cost"] = float("nan")
            pieces.append(piece)

    # The forgetting gap, paired within seed so common noise cancels.
    cur = error_rate(frame, "current").set_index(["learner", "seed", "t"]).error
    back = error_rate(frame, "backward").set_index(["learner", "seed", "t"]).error
    gap = (back - cur).dropna().reset_index()
    for name in learners_of(frame):
        subset = gap[gap.learner == name]
        if subset.empty:
            continue
        mean, spread = band(subset, "error")
        piece = rows_for("gap", name, "line", mean.index, mean, mean - spread, mean + spread)
        piece["cost"] = float("nan")
        pieces.append(piece)

    if not pieces:
        return None
    return Collected(
        pd.concat(pieces, ignore_index=True),
        {"seeds": int(frame.seed.nunique()), "smooth": 0},
    )


def draw_f10(collected: Collected) -> None:
    frame, meta = collected.frame, collected.meta
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))

    axis = axes[0]
    for name in series_present(frame[frame.panel == "current"]):
        for evalset, style_ in (("current", "-"), ("backward", "--")):
            line = frame[(frame.panel == evalset) & (frame.series == name)].sort_values("x")
            if line.empty:
                continue
            axis.plot(
                line.x,
                line.y,
                color=COLOURS[name],
                linestyle=style_,
                linewidth=1.9 if evalset == "current" else 1.3,
                alpha=1.0 if evalset == "current" else 0.75,
            )
    axis.set_xlabel("step $t$")
    axis.set_ylabel("held-out error rate")
    axis.set_title("solid = current rotation,  dashed = a rotation left behind")
    despine(axis)

    axis = axes[1]
    endpoints = []
    block = frame[frame.panel == "gap"]
    for name in series_present(block):
        line = block[block.series == name].sort_values("x")
        axis.fill_between(line.x, line.lo, line.hi, color=COLOURS[name], alpha=0.13, linewidth=0)
        axis.plot(line.x, line.y, color=COLOURS[name])
        endpoints.append((float(line.y.iloc[-1]), LABELS[name], COLOURS[name]))
    axis.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=0)

    # The cycle average, drawn because the instantaneous gap is dominated by
    # *phase*, not by forgetting: it swings +/-0.05 while the whole-period mean
    # is near zero. Averaged over a fifth of a period the sign even flips
    # (+0.016 against -0.0035), so a scalar summary here is only meaningful over
    # a whole number of periods. Without this line the peaks read as forgetting.
    period = 500
    for name in series_present(block):
        line = block[block.series == name].sort_values("x")
        cycles = line[line.x >= line.x.max() - 2 * period]
        if cycles.empty:
            continue
        level = float(cycles.y.mean())
        axis.plot(
            [cycles.x.min(), cycles.x.max()],
            [level, level],
            color=COLOURS[name],
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            zorder=3,
        )
    axis.annotate(
        "dashed: mean over the last two full periods",
        xy=(0.03, 0.04),
        xycoords="axes fraction",
        fontsize=8,
        color=INK_SECONDARY,
    )
    axis.set_xlabel("step $t$")
    axis.set_ylabel("forgetting  (backward $-$ current)")
    axis.set_title("Forgetting, paired within seed")
    despine(axis)
    direct_labels(axis, endpoints)

    figure.subplots_adjust(right=0.86, wspace=0.26)
    shared_legend(figure, series_present(block))
    figure.suptitle(
        f"F10  Forgetting under a revisiting schedule  ({meta.get('seeds', '?')} seeds, "
        "sinusoidal, amplitude 30 degrees)",
        y=1.03,
        fontsize=11,
        color=INK,
        weight="bold",
    )
    save(figure, "22_f10_forgetting.png")


FIGURES = {
    "f1": (collect_f1, draw_f1),
    "f2": (collect_f2, draw_f2),
    "f5": (collect_f5, draw_f5),
    "f8": (collect_f8, draw_f8),
    "f3": (collect_f3, draw_f3),
    "f4": (collect_f4, draw_f4),
    "f6a": (collect_f6a, draw_f6a),
    "f6b": (collect_f6b, draw_f6b),
    "f7": (collect_f7, draw_f7),
    "f9": (collect_f9, draw_f9),
    "f10": (collect_f10, draw_f10),
}


def main() -> None:
    global DPI, FROM_CACHE

    argv = [a for a in sys.argv[1:]]
    if "--dpi" in argv:
        DPI = int(argv[argv.index("--dpi") + 1])
    from_cache = FROM_CACHE or "--from-cache" in argv
    wanted = next((a.lower() for a in argv if a.lower() in FIGURES), ONLY)

    style()
    selected = {wanted: FIGURES[wanted]} if wanted else FIGURES
    print(f"figures -> {OUT_DIR}")
    print(f"data    -> {DATA_DIR}   ({'reading' if from_cache else 'refreshing'})\n")

    for figure_id, (collect, draw) in selected.items():
        print(f"{figure_id.upper()}:")
        if from_cache:
            collected = cache_read(figure_id)
            if collected is None:
                print("  no cached data -- run without --from-cache first")
                continue
        else:
            collected = collect()
            if collected is None:
                print("  skipped: results not available yet")
                continue
            cache_write(figure_id, collected)
        draw(collected)


if __name__ == "__main__":
    main()
