r"""Drift damage against drift rate, with the located breaks marked.

Run this file directly.

    python scripts/plot_breaks.py
    python scripts/plot_breaks.py x9_rate_ramp x9_control

**The x-axis is the drift rate, not the step.** Under a ramp the step is only
an index into a rate sweep, and plotting against it would invite reading the
break as "it survived 1100 steps" when the claim is "it survived up to 0.038
degrees per step". Everything here is stated in the units the threshold is a
threshold on (design note D47).

The y-axis is the paired excess: the drifting run's error minus its stationary
twin's, seed for seed. Zero means the drift cost nothing. The band is one
standard error of the seed mean, which is also what the break test compares
against.

**Two panels, because one hides the interesting part.** The frozen baseline
ends an order of magnitude above everything else, so on a shared axis the
adapting methods are a flat line at the bottom. The right panel drops it.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dekf_bench.env.drift import build_drift  # noqa: E402
from dekf_bench.metrics.breaks import excess_break, paired_excess, pooled_sem  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402
from dekf_bench.utils.paths import figures_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
DRIFTING = "x9_rate_ramp"
CONTROL = "x9_control"
BASELINE = "frozen_atc"
NOISE_MULTIPLE = 3.0
PERSISTENCE = 3
#: Marked with a vertical line: the rate at which report_breaks.py compares
#: methods head to head, so the figure and the table point at the same place.
MATCHED_RATE = 0.10
DPI = 200

LABELS = {
    "centralized_sgd": "Centralized",
    "diffusion_sgd_atc": "ATC (momentum)",
    "diffusion_sgd_atc_plain": "ATC (plain, payload-matched)",
    "local_only": "Local only",
    "frozen_atc": "Frozen at t=300",
}
COLOURS = {
    "centralized_sgd": "#444444",
    "diffusion_sgd_atc": "#1f77b4",
    "diffusion_sgd_atc_plain": "#17becf",
    "local_only": "#d62728",
    # Not a second grey: on the shared axis it sits next to `centralized_sgd`
    # and the two were indistinguishable, which is the one comparison a reader
    # must not get backwards -- the best method against the one that stopped.
    "frozen_atc": "#9467bd",
}
#: The baseline is dashed as well as differently coloured, so it survives a
#: greyscale print of the deck.
STYLES = {"frozen_atc": "--"}


def load(experiment: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "results" / experiment / "seed_*.parquet")))
    if not files:
        raise SystemExit(f"no results for {experiment!r}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def draw(axis, excess, drift, horizon, learners, title, noise) -> None:
    for learner in learners:
        rows = excess[excess.learner == learner]
        grouped = rows.groupby("t").excess
        mean = grouped.mean().sort_index()
        counts = rows.groupby("t").seed.nunique().sort_index()
        sem = (grouped.std().sort_index() / counts.pow(0.5)).fillna(0.0)
        rates = [drift.schedule.rate_at(int(step)) for step in mean.index]

        colour = COLOURS.get(learner, "#333333")
        axis.plot(
            rates,
            mean.to_numpy(),
            color=colour,
            lw=1.6,
            ls=STYLES.get(learner, "-"),
            label=LABELS.get(learner, learner),
        )
        axis.fill_between(
            rates,
            (mean - sem).to_numpy(),
            (mean + sem).to_numpy(),
            color=colour,
            alpha=0.18,
            linewidth=0,
        )

        point = excess_break(
            excess, learner, drift, horizon, NOISE_MULTIPLE, PERSISTENCE, noise=noise
        )
        if point.broke:
            axis.plot(
                [point.rate_at_break],
                [point.value_at_break],
                marker="o",
                ms=6,
                mfc="white",
                mec=colour,
                mew=1.8,
                zorder=5,
            )

    axis.axhline(0.0, color="#999999", lw=0.8, ls="--")
    if drift.schedule.peak_rate(horizon) >= MATCHED_RATE:
        axis.axvline(MATCHED_RATE, color="#bbbbbb", lw=0.9, ls=":", zorder=0)
        axis.annotate(
            f"compared at {MATCHED_RATE:g}",
            xy=(MATCHED_RATE, axis.get_ylim()[1]),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=7,
            color="#777777",
        )
    axis.set_xlabel("drift rate at that step (degrees / step)")
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.25, lw=0.5)


def main() -> int:
    drifting_name = sys.argv[1] if len(sys.argv) > 1 else DRIFTING
    control_name = sys.argv[2] if len(sys.argv) > 2 else CONTROL

    config = load_config(drifting_name)
    drift = build_drift(config)
    horizon = config.run.horizon
    excess = paired_excess(load(drifting_name), load(control_name))

    everything = [name for name in LABELS if name in set(excess.learner)]
    adapting = [name for name in everything if name != BASELINE]

    # The same pooled bar report_breaks.py uses, and pooled over the same
    # learners: the baseline is an order of magnitude noisier, so including it
    # would raise the bar for everyone to accommodate a learner not competing.
    # If the figure used per-learner noise its markers would disagree with the
    # table, and a reader comparing them would have no way to tell which was right.
    noise = pooled_sem(excess, adapting)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    draw(axes[0], excess, drift, horizon, everything, "All methods", noise)
    draw(axes[1], excess, drift, horizon, adapting, "Adapting methods only", noise)
    axes[0].set_ylabel("drift damage: error minus the stationary twin's")
    for axis in axes:
        axis.legend(fontsize=8, frameon=False, loc="upper left")

    figure.suptitle(
        f"Where tracking breaks — {drifting_name} against {control_name}, "
        f"{excess.seed.nunique()} seeds. Circles mark the located break.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"23_breaks_{drifting_name}.png"
    figure.savefig(path, dpi=DPI)
    print(f"wrote {path}")

    print("\nbreaks, for reference (same pooled bar as report_breaks.py)")
    for learner in everything:
        point = excess_break(
            excess, learner, drift, horizon, NOISE_MULTIPLE, PERSISTENCE, noise=noise
        )
        where = (
            f"rate {point.rate_at_break:.4f} deg/step at t={point.step}"
            if point.broke
            else f"no break up to {point.max_rate_probed:.4f} deg/step"
        )
        print(f"  {LABELS.get(learner, learner):30s} {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
