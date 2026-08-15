r"""X11 as a picture: the recovery grid, and the transients behind it.

Run this file directly.

    python scripts/plot_recurring.py
    python scripts/plot_recurring.py --learner local_only

**Three heatmaps and two transient panels.** The heatmaps say *what* happens
across the grid; the transients say *why*, by showing the shape the summary
numbers are computed from. A grid alone would leave "recovered 0.2" ambiguous
between a small wound that heals slowly and a large one that heals fast.

**Dark is worse in every panel.** ``rise`` and ``standing`` are costs, so a
plain sequential ramp already reads that way. ``recovered`` is a *good* thing,
so its ramp is reversed rather than its numbers negated -- the annotated value
stays the same quantity ``report_recurring.py`` prints, and a reader comparing
the two never has to reconcile a sign.

Cell values are printed as well as shaded, so the grid is readable without
relying on colour, and the ordered $t'$ and $J$ series in the transient panels
step through one hue rather than taking categorical colours -- they are
magnitudes, not identities.
"""

from __future__ import annotations

import glob
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dekf_bench.metrics import recovery  # noqa: E402
from dekf_bench.metrics.breaks import error_by_step  # noqa: E402
from dekf_bench.utils.paths import figures_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
JUMP_EVERY = [25, 50, 100, 200]
JUMP_DEGREES = [5.0, 15.0, 30.0]
LEARNER = "diffusion_sgd_atc"
#: The row and column held fixed in the two transient panels.
TRANSIENT_AT_J = 15.0
TRANSIENT_AT_EVERY = 50
DPI = 200

SEQUENTIAL = "Blues"
CELL_PT = 10.0
MUTED = "#666666"
INK = "#222222"


def cell_name(jump_every: int, jump_degrees: float) -> str:
    return f"x11_every{jump_every}_jump{jump_degrees:g}"


def load(experiment: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(ROOT / "results" / experiment / "seed_*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def heatmap(axis, matrix, title, reverse: bool = False) -> None:
    cmap = f"{SEQUENTIAL}_r" if reverse else SEQUENTIAL
    finite = matrix[np.isfinite(matrix)]
    image = axis.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        vmin=float(finite.min()) if finite.size else 0.0,
        vmax=float(finite.max()) if finite.size else 1.0,
    )
    axis.set_xticks(range(len(JUMP_DEGREES)), [f"{value:g}" for value in JUMP_DEGREES])
    axis.set_yticks(range(len(JUMP_EVERY)), [str(value) for value in JUMP_EVERY])
    axis.set_xlabel("shift size J (degrees)", fontsize=9)
    axis.set_ylabel("steps between shifts  t'", fontsize=9)
    axis.set_title(title, fontsize=10, color=INK)

    for row, column in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        value = matrix[row, column]
        if not np.isfinite(value):
            axis.text(column, row, "--", ha="center", va="center", fontsize=CELL_PT, color=MUTED)
            continue
        # Label in whichever ink stays legible on this cell's shade, so the
        # numbers survive the darkest end of the ramp.
        rgba = image.cmap(image.norm(value))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=CELL_PT,
            color="white" if luminance < 0.55 else INK,
        )


def transients(axis, results, fixed, vary, title, label) -> None:
    """Aligned recovery curves, one line per value of the varying axis."""
    ramp = plt.get_cmap(SEQUENTIAL)
    shades = np.linspace(0.42, 0.95, max(len(vary), 2))
    for shade, value in zip(shades, vary, strict=False):
        profile = results.get(fixed(value))
        if profile is None or profile[1] is None:
            continue
        curve = profile[1]
        axis.plot(
            curve.offset.to_numpy(),
            curve["mean"].to_numpy(),
            color=ramp(shade),
            lw=1.7,
            label=label(value),
        )
    axis.axhline(0.0, color="#999999", lw=0.8, ls="--")
    axis.set_xlabel("steps since the shift", fontsize=9)
    axis.set_ylabel("error above the pre-shift level", fontsize=9)
    axis.set_title(title, fontsize=10, color=INK)
    axis.grid(alpha=0.25, lw=0.5)
    axis.legend(fontsize=8, frameon=False)


def main() -> int:
    learner = LEARNER
    if "--learner" in sys.argv:
        learner = sys.argv[sys.argv.index("--learner") + 1]

    results: dict[tuple[int, float], tuple[object, object]] = {}
    for jump_every, jump_degrees in itertools.product(JUMP_EVERY, JUMP_DEGREES):
        frame = load(cell_name(jump_every, jump_degrees))
        if frame is None:
            continue
        errors = error_by_step(frame, "current", by_seed=True)
        horizon = int(errors.t.max()) + 1
        try:
            summary = recovery.recovery_profile(errors, learner, jump_every, horizon)
            curve = recovery.aligned_profile(errors, learner, jump_every, horizon)
        except recovery.RecoveryError as error:
            print(f"  ({cell_name(jump_every, jump_degrees)}: {error})")
            continue
        results[(jump_every, jump_degrees)] = (summary, curve)

    if not results:
        raise SystemExit(
            "no X11 results yet. Run them first:\n    python scripts/run_recurring_sweep.py"
        )
    missing = len(JUMP_EVERY) * len(JUMP_DEGREES) - len(results)
    if missing:
        print(f"note: {missing} of {len(JUMP_EVERY) * len(JUMP_DEGREES)} cells not yet run\n")

    def grid(field: str) -> np.ndarray:
        matrix = np.full((len(JUMP_EVERY), len(JUMP_DEGREES)), np.nan)
        for row, jump_every in enumerate(JUMP_EVERY):
            for column, jump_degrees in enumerate(JUMP_DEGREES):
                found = results.get((jump_every, jump_degrees))
                if found is not None:
                    matrix[row, column] = getattr(found[0], field)
        return matrix

    # Margins set on the gridspec rather than by tight_layout, which cannot
    # honour an explicit hspace/wspace and warns that its result may be wrong.
    figure = plt.figure(figsize=(13.0, 8.0))
    spec = figure.add_gridspec(
        2, 6, hspace=0.42, wspace=0.75, top=0.88, bottom=0.075, left=0.055, right=0.985
    )

    heatmap(
        figure.add_subplot(spec[0, 0:2]),
        grid("recovered"),
        "Fraction of shifts recovered from\nbefore the next (dark = fails to keep up)",
        reverse=True,
    )
    heatmap(
        figure.add_subplot(spec[0, 2:4]),
        grid("rise"),
        "Error rise above the pre-shift level\n(dark = bigger wound)",
    )
    heatmap(
        figure.add_subplot(spec[0, 4:6]),
        grid("standing"),
        "Standing error, second half\n(dark = worse)",
    )

    transients(
        figure.add_subplot(spec[1, 0:3]),
        results,
        lambda value: (value, TRANSIENT_AT_J),
        JUMP_EVERY,
        f"Same shift size (J = {TRANSIENT_AT_J:g}), varying how often",
        lambda value: f"every {value} steps",
    )
    transients(
        figure.add_subplot(spec[1, 3:6]),
        results,
        lambda value: (TRANSIENT_AT_EVERY, value),
        JUMP_DEGREES,
        f"Same interval (t' = {TRANSIENT_AT_EVERY}), varying how big",
        lambda value: f"{value:g} degrees",
    )

    seeds = "?"
    any_frame = load(cell_name(*next(iter(results))))
    if any_frame is not None:
        seeds = str(any_frame.seed.nunique())
    figure.suptitle(
        f"X11 — recovery under repeated abrupt shifts, {learner}, {seeds} seeds. "
        "Every shift is scored against the error just before it.",
        fontsize=12,
    )

    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"24_x11_recovery_{learner}.png"
    figure.savefig(path, dpi=DPI)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
