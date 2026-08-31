r"""Which drift conditions actually hurt, ranked by measured damage.

Run this file directly.

    python scripts/plot_drift_damage.py

The deck already shows *where* the baselines break on an accelerating ramp, and
what abruptness costs at matched speed. This answers a third question that the
filter work forced: **across every condition the project has run, which ones do
the most damage?** It is the figure that chose X13's tuning rate and X14's
conditions, and it carries a warning that is easy to miss otherwise.

**Damage, not error.** Each condition's settled error minus its own paired
stationary twin's, so a learner's own convergence cancels and only what the drift
cost remains (design note D54).

**Two ceilings, and one of them invalidates half the smooth rates.** A constant
rate reaches the 45-degree well-posedness cap at $T=45/\alpha$, so faster smooth
drift runs *shorter*: 299 steps at 0.15 deg/step against 1499 at 0.025. Damage
there is part drift and part "had less time to converge", and the two cannot be
separated afterwards. The tell is drawn on the figure -- at 0.15 the *frozen*
baseline's damage equals ATC's exactly, because `freeze_after` is 300 and the run
ends at 299, so that bar describes one algorithm twice.

Recurring shifts reflect at the cap and so sustain any rate for a full 1500
steps. That is why they, and not the fast smooth rates, are the conditions the
filter is tested against.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dekf_bench.utils.paths import figures_dir  # noqa: E402

DPI = 200
LEARNER = "diffusion_sgd_atc"
FROZEN = "frozen_atc"

#: (label, drifting run, control run, kind). Every X11 cell shares one control;
#: each X12 speed has its own, because their horizons differ and a control at a
#: different horizon is a different experiment.
CONDITIONS = [
    ("linear 0.025", "x12_linear_a0p025", "x12_control_a0p025", "smooth"),
    ("linear 0.05", "x12_linear_a0p05", "x12_control_a0p05", "smooth"),
    ("linear 0.10", "x12_linear_a0p1", "x12_control_a0p1", "smooth"),
    ("linear 0.15", "x12_linear_a0p15", "x12_control_a0p15", "smooth"),
    *[
        (f"every {every}, jump {jump}", f"x11_every{every}_jump{jump}", "x11_control", "abrupt")
        for every in (25, 50, 100, 200)
        for jump in (5, 15, 30)
    ],
]

COLOURS = {"smooth": "#1f77b4", "abrupt": "#c1440e"}
#: The rate X13 tuned the filter at. Marked because it is one of the mildest
#: conditions here, which is the thing a reader should leave knowing.
TUNED_AT = "linear 0.025"


def load(name: str) -> pd.DataFrame | None:
    files = sorted((ROOT / "results" / name).glob("seed_*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def settled(frame: pd.DataFrame, learner: str, horizon: int) -> float | None:
    rows = frame[
        (frame["learner"] == learner)
        & (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
        & (frame["t"] >= int(0.8 * horizon))
    ]
    return float(rows["value"].mean()) if len(rows) else None


def measure() -> list[dict]:
    records = []
    for label, drift_name, control_name, kind in CONDITIONS:
        drifting, control = load(drift_name), load(control_name)
        if drifting is None or control is None:
            continue
        shared = set(drifting["t"].unique()) & set(control["t"].unique())
        horizon = int(max(shared)) + 1
        entry = {"label": label, "kind": kind, "horizon": horizon}
        for who, key in ((LEARNER, "atc"), (FROZEN, "frozen")):
            a = settled(drifting, who, horizon)
            b = settled(control, who, horizon)
            entry[key] = None if a is None or b is None else a - b
        if entry["atc"] is not None:
            records.append(entry)
    return records


def main() -> int:
    records = measure()
    if not records:
        print("No X11/X12 results found.")
        return 1
    records.sort(key=lambda r: r["atc"])
    print(f"{len(records)} conditions measured")

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    positions = range(len(records))
    axis.barh(
        list(positions), [r["atc"] for r in records],
        color=[COLOURS[r["kind"]] for r in records], height=0.68, zorder=3,
    )

    # The frozen baseline as a marker on each bar. Where it sits *on* the bar the
    # run was too short for freezing to mean anything.
    for index, record in enumerate(records):
        if record["frozen"] is None:
            continue
        collapsed = abs(record["frozen"] - record["atc"]) < 1e-9
        axis.scatter(
            [record["frozen"]], [index], marker="|", s=140, linewidth=2.0,
            color="#7d2c09" if collapsed else "#666666", zorder=5,
        )
        if collapsed:
            axis.annotate(
                "  frozen never froze: T=299 < freeze_after=300",
                (record["atc"], index), fontsize=7.5, va="center", color="#7d2c09",
                fontweight="bold",
            )

    labels = []
    for record in records:
        mark = "  <- X13 tuned here" if record["label"] == TUNED_AT else ""
        labels.append(f"{record['label']}  (T={record['horizon']}){mark}")
    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels, fontsize=8.5)
    for tick, record in zip(axis.get_yticklabels(), records):
        if record["label"] == TUNED_AT:
            tick.set_color("#c1440e")
            tick.set_fontweight("bold")

    handles = [
        plt.Line2D([], [], color=COLOURS["abrupt"], linewidth=8, label="abrupt (recurring shifts)"),
        plt.Line2D([], [], color=COLOURS["smooth"], linewidth=8, label="smooth (constant rate)"),
        plt.Line2D([], [], color="#666666", marker="|", linestyle="none", markersize=11,
                   markeredgewidth=2, label="frozen baseline's damage"),
    ]
    axis.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")

    axis.set_xlabel("damage: settled error minus its own stationary twin's  (ATC)")
    axis.set_title(
        "Where the baselines are actually hurt\n"
        "abrupt shifts dominate, and they run a full horizon while fast smooth drift cannot",
        fontsize=11, loc="left",
    )
    axis.grid(alpha=0.25, linewidth=0.5, axis="x")
    axis.set_axisbelow(True)
    figure.tight_layout()

    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "26_drift_damage_by_condition.png"
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"wrote {path}")

    print(f"\n{'condition':<24}{'T':>6}{'ATC damage':>12}{'frozen':>10}")
    for record in reversed(records):
        frozen = f"{record['frozen']:.4f}" if record["frozen"] is not None else "."
        print(f"  {record['label']:<22}{record['horizon']:>6}{record['atc']:>12.4f}{frozen:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
