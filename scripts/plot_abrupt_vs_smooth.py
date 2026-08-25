r"""Abrupt against smooth drift: the matched comparison, and the range beyond it.

Run this file directly.

    python scripts/plot_abrupt_vs_smooth.py

**Two panels because two different windows are in play, and mixing them in one
axis would be dishonest.**

*Left* is the controlled comparison: both regimes read over the same step
window, so a learner 200 steps in is never compared with one 1000 steps in.
Only the four speeds where a smooth counterpart exists appear.

*Right* drops the smooth side and shows every X11 cell over its own second half,
which is a consistent definition across those cells because they all run to
T=1500. It exists to show the range: past 0.15 deg/step there is no smooth
counterpart at all, because constant drift leaves the 45-degree band. That gap
is the finding, not missing data.

Marker size carries how many shifts the window contains. The two points at 0.15
rest on one transient each, and drawing them the same size as points averaged
over four would overstate them.
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

from dekf_bench.metrics.breaks import paired_excess  # noqa: E402
from dekf_bench.utils.paths import figures_dir  # noqa: E402

LEARNER = "diffusion_sgd_atc"
SMOOTH = {
    0.025: ("x12_linear_a0p025", "x12_control_a0p025", 1500),
    0.05: ("x12_linear_a0p05", "x12_control_a0p05", 900),
    0.10: ("x12_linear_a0p1", "x12_control_a0p1", 450),
    0.15: ("x12_linear_a0p15", "x12_control_a0p15", 300),
}
ABRUPT_CONTROL = "x11_control"
JUMP_EVERY = [25, 50, 100, 200]
JUMP_DEGREES = [5.0, 15.0, 30.0]

SMOOTH_INK = "#444444"
SEQUENTIAL = "Blues"
INK = "#222222"
DPI = 200


def load(name: str):
    files = sorted(glob.glob(str(ROOT / "results" / name / "seed_*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def damage(run: str, control: str, lo: int, hi: int) -> float | None:
    drifting, twin = load(run), load(control)
    if drifting is None or twin is None:
        return None
    excess = paired_excess(drifting, twin)
    rows = excess[(excess.learner == LEARNER) & (excess.t >= lo) & (excess.t < hi)]
    return float(rows.excess.mean()) if len(rows) else None


def shade_for(jump_degrees: float) -> tuple:
    ramp = plt.get_cmap(SEQUENTIAL)
    position = JUMP_DEGREES.index(jump_degrees) / max(len(JUMP_DEGREES) - 1, 1)
    return ramp(0.45 + 0.5 * position)


def main() -> int:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # -- left: matched windows ------------------------------------------- #
    speeds, smooth_values = [], []
    for speed in sorted(SMOOTH):
        run, control, horizon = SMOOTH[speed]
        value = damage(run, control, horizon // 2, horizon)
        if value is None:
            continue
        speeds.append(speed)
        smooth_values.append(value)
    axes[0].plot(
        speeds, smooth_values, color=SMOOTH_INK, lw=1.8, marker="o", ms=6, label="smooth (linear)"
    )

    seen = set()
    for speed in sorted(SMOOTH):
        _run, _control, horizon = SMOOTH[speed]
        lo, hi = horizon // 2, horizon
        for jump_every in JUMP_EVERY:
            for jump_degrees in JUMP_DEGREES:
                if abs(jump_degrees / jump_every - speed) > 1e-9:
                    continue
                cell = f"x11_every{jump_every}_jump{jump_degrees:g}"
                value = damage(cell, ABRUPT_CONTROL, lo, hi)
                if value is None:
                    continue
                shifts = sum(1 for s in range(jump_every, hi, jump_every) if lo <= s < hi)
                label = f"abrupt, J={jump_degrees:g}deg"
                colour = shade_for(jump_degrees)
                # Hollow when the window holds too few shifts to average, rather
                # than encoding the count in marker *size*: the legend swatch
                # takes its size from the first point drawn, so a size encoding
                # makes the key silently disagree with the data.
                solid = shifts >= 2
                axes[0].plot(
                    [speed],
                    [value],
                    marker="s",
                    ms=8,
                    color=colour,
                    mfc=colour if solid else "white",
                    mec=colour,
                    mew=1.6,
                    ls="none",
                    label=None if label in seen else label,
                    zorder=4,
                )
                seen.add(label)
                if not solid:
                    axes[0].annotate(
                        f"{shifts} shift",
                        xy=(speed, value),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=7,
                        color="#888888",
                        ha="center",
                    )

    axes[0].set_title("Matched window: both regimes read over the same steps", fontsize=10)
    axes[0].set_ylabel("drift damage (error minus a stationary twin)")
    # Headroom, or the largest point sits on the frame and reads as clipped.
    axes[0].margins(x=0.12, y=0.18)
    axes[0].annotate(
        "hollow = one shift in the window,\nso a single transient rather than an average",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        color="#888888",
    )

    # -- right: the whole abrupt range ------------------------------------ #
    for jump_degrees in JUMP_DEGREES:
        xs, ys = [], []
        for jump_every in JUMP_EVERY:
            cell = f"x11_every{jump_every}_jump{jump_degrees:g}"
            value = damage(cell, ABRUPT_CONTROL, 750, 1500)
            if value is None:
                continue
            xs.append(jump_degrees / jump_every)
            ys.append(value)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        axes[1].plot(
            [xs[i] for i in order],
            [ys[i] for i in order],
            marker="s",
            ms=6,
            lw=1.5,
            color=shade_for(jump_degrees),
            label=f"abrupt, J={jump_degrees:g}deg",
        )

    limit = max(SMOOTH)
    axes[1].axvspan(limit, 1.4, color="#f2f2f2", zorder=0)
    # "Leaves the band" overstated it: smooth drift at these speeds is not
    # impossible, it is too *short* to compare. A constant rate reaches the 45
    # degree cap at T = 45/alpha, so 0.3 allows 150 steps and 1.2 allows 37 --
    # shorter than one interval of the abrupt cell it would be compared with.
    fastest = max(jump_degrees / jump_every for jump_every in JUMP_EVERY for jump_degrees in [30.0])
    # Short in-panel, with the reasoning carried by the caption below the
    # figure: the annotation has to survive being read at a glance, and the
    # arithmetic behind it does not fit anywhere a glance would look.
    # Low and centred inside the shaded band. Top-right is where the J=30 line
    # ends, and the label sat on top of its final marker.
    axes[1].annotate(
        "smooth drift cannot be\nmeasured here — see caption",
        xy=(0.74, 0.06),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#777777",
    )
    axes[1].set_title("Every abrupt cell, over its own second half", fontsize=10)

    from matplotlib.ticker import FuncFormatter

    for index, axis in enumerate(axes):
        axis.set_xscale("log")
        axis.set_xlabel("average speed (degrees / step)")
        axis.grid(alpha=0.25, lw=0.5)
        # Lower right on the left panel: the abrupt points climb into the upper
        # left there, and a legend on top of the crossover would hide it.
        axis.legend(fontsize=8, frameon=False, loc="lower right" if index == 0 else "upper left")
        axis.tick_params(labelsize=9)
        # Plain decimals: 3x10^-2 is harder to match against "0.03 deg/step" in
        # the text than 0.03 is.
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        axis.xaxis.set_minor_formatter(FuncFormatter(lambda v, _pos: ""))

    figure.suptitle(
        f"Abrupt against smooth drift at matched average speed -- {LEARNER}, 5 seeds",
        fontsize=11,
        color=INK,
    )
    figure.text(
        0.5,
        0.035,
        "Shaded: smooth drift cannot be measured here. The 45° cap allows only T = 45/α steps, "
        f"so a constant-rate run at these speeds ends before the learner converges\n"
        f"({45 / fastest:.0f} steps at {fastest:g}°/step, where it is still 41 % of the way from "
        "random initialisation). Abrupt shifts stay inside the band and have no such limit.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))

    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "25_abrupt_vs_smooth.png"
    figure.savefig(path, dpi=DPI)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
