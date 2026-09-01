r"""How much of the filter's advantage is tracking, and how much is just fitting?

Run this file directly.

    python scripts/plot_ekf_advantage.py

X13's headline is +0.0288 under drift. The filter is a **second-order method**,
so some of that would appear on a stationary benchmark with no drift anywhere,
and only the rest is what this project is actually about. Every X13 cell has a
paired stationary twin, so the split is measured rather than argued:

    total      = baseline drift error - filter drift error
    fitting    = baseline stationary  - filter stationary       (no drift involved)
    tracking   = baseline damage      - filter damage           (what drift cost each)

and the two components sum to the total exactly, because damage is defined as
drift minus stationary. The figure exists because "+0.0288 under drift" invites
the question and it is better answered on a slide than from the floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dekf_bench.utils.paths import figures_dir  # noqa: E402

DPI = 200
SETTLED = 1200
BEST = "x13_g_s0p01_g0p9995_q6em05"
FILTER_LEARNER = "centralized_ekf_gamma"

FIT = "#8fb8d8"      # the part that has nothing to do with drift
TRACK = "#c1440e"    # the part that does
INK = "#333333"


def settled(directory: str, learner: str) -> float:
    files = sorted((ROOT / "results" / directory).glob("seed_*.parquet"))
    if not files:
        raise SystemExit(f"missing results/{directory}")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = frame[
        (frame["learner"] == learner)
        & (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
        & (frame["t"] >= SETTLED)
    ]
    return float(rows["value"].mean())


def abrupt_split() -> dict[str, tuple[float, float]]:
    """The same split at X14's worst condition, or ``{}`` if it has not run.

    Uses X14's *own* control and its *own* re-tuned baselines: the learning rate
    ATC prefers differs between the two conditions (0.01 here against 0.02 under
    smooth drift), so borrowing X13's numbers would mix two baselines.
    """
    drift, control = "x14_every25_jump30", "x14_control_n4_T1500"
    if not (ROOT / "results" / drift).exists():
        return {}
    out = {}
    for learner, label in (
        ("centralized_sgd", "Centralized SGD"),
        ("diffusion_sgd_atc", "ATC (momentum)"),
    ):
        try:
            base_still = settled(control, learner)
            base_drift = settled(drift, learner)
            filt_still = settled(control, FILTER_LEARNER)
            filt_drift = settled(drift, FILTER_LEARNER)
        except SystemExit:
            return {}
        out[label] = (
            base_still - filt_still,
            (base_drift - base_still) - (filt_drift - filt_still),
        )
    return out


def main() -> int:
    ekf = (settled(f"{BEST}_control", FILTER_LEARNER), settled(f"{BEST}_full", FILTER_LEARNER))
    baselines = {
        "Centralized SGD": (
            settled("x13_baselines_control", "centralized_sgd"),
            settled("x13_baselines_full", "centralized_sgd"),
        ),
        "ATC (momentum)": (
            settled("x13_baselines_control", "diffusion_sgd_atc"),
            settled("x13_baselines_full", "diffusion_sgd_atc"),
        ),
    }

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # -- (a) the four errors the split is computed from --------------------- #
    axis = axes[0]
    names = list(baselines) + ["EKF (best cell)"]
    still = [baselines[n][0] for n in baselines] + [ekf[0]]
    drift = [baselines[n][1] for n in baselines] + [ekf[1]]
    x = np.arange(len(names))
    width = 0.36
    axis.bar(x - width / 2, still, width, color="#bbbbbb", label="stationary twin", zorder=3)
    axis.bar(x + width / 2, drift, width, color=TRACK, label="under drift", zorder=3)
    for index, (a, b) in enumerate(zip(still, drift)):
        axis.annotate(f"{a:.4f}", (index - width / 2, a), ha="center", va="bottom", fontsize=7.5)
        axis.annotate(f"{b:.4f}", (index + width / 2, b), ha="center", va="bottom", fontsize=7.5)
    axis.set_xticks(x)
    axis.set_xticklabels(names, fontsize=9)
    axis.set_ylabel("settled error")
    axis.set_ylim(0, max(drift) * 1.22)
    # Names its condition, because panel (b) now carries two and an unlabelled
    # worked example would read as applying to both.
    axis.set_title(
        "(a) How the split is computed, at linear 0.025°/step", fontsize=10, loc="left"
    )
    axis.legend(fontsize=8, frameon=False, loc="upper right")
    axis.grid(alpha=0.25, linewidth=0.5, axis="y")
    axis.set_axisbelow(True)

    # -- (b) the split, at both conditions ---------------------------------- #
    #
    # X14's abrupt cell is included when it exists, because the *comparison* is
    # the point: the split is a property of the condition, not of the filter, and
    # one column alone invites reading it as the latter.
    axis = axes[1]
    labels, fitting, tracking = [], [], []
    for name, (base_still, base_drift) in baselines.items():
        labels.append(f"vs {name}\nlinear 0.025°/step")
        fitting.append(base_still - ekf[0])
        tracking.append((base_drift - base_still) - (ekf[1] - ekf[0]))

    abrupt = abrupt_split()
    for name, (fit, track) in abrupt.items():
        labels.append(f"vs {name}\nevery 25, jump 30°")
        fitting.append(fit)
        tracking.append(track)

    positions = np.arange(len(labels))
    axis.barh(positions, fitting, 0.55, color=FIT, label="already there with NO drift", zorder=3)
    axis.barh(positions, tracking, 0.55, left=fitting, color=TRACK,
              label="from tracking the drift", zorder=3)

    for index, (f, t) in enumerate(zip(fitting, tracking)):
        total = f + t
        axis.annotate(f"{f / total:.0%}", (f / 2, index), ha="center", va="center",
                      fontsize=8.5, color=INK, fontweight="bold")
        axis.annotate(f"{t / total:.0%}", (f + t / 2, index), ha="center", va="center",
                      fontsize=8.5, color="white", fontweight="bold")
        axis.annotate(f"  {total:+.4f}", (total, index), va="center", fontsize=8, color=INK)

    if abrupt:
        # The line separates conditions, not methods: above it the drift is
        # abrupt, below it smooth, and the same filter setting produced both.
        axis.axhline(len(baselines) - 0.5, color="#999999", linewidth=0.8, linestyle=":")

    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlim(0, max(f + t for f, t in zip(fitting, tracking)) * 1.30)
    axis.set_xlabel("advantage in settled error")
    title = (
        "(b) The tracking share triples where drift hurts"
        if abrupt else "(b) Most of the headline is not about drift"
    )
    axis.set_title(title, fontsize=10, loc="left")
    axis.legend(fontsize=8, frameon=False, loc="lower right")
    axis.grid(alpha=0.25, linewidth=0.5, axis="x")
    axis.set_axisbelow(True)

    figure.suptitle(
        "The filter is a better optimiser and a better tracker — which dominates depends on the drift"
        if abrupt else
        "The filter is a better optimiser and a better tracker — mostly the former, at this rate",
        fontsize=11, y=1.00,
    )
    figure.tight_layout()

    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "29_ekf_advantage_split.png"
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"wrote {path}")

    for name, f, t in zip(labels, fitting, tracking):
        print(f"  {name:<24} total {f + t:+.4f}  fitting {f:+.4f} ({f / (f + t):.0%})  "
              f"tracking {t:+.4f} ({t / (f + t):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
