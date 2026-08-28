r"""X13 pilot in three panels: the win, the knob, and the fairness check.

Run this file directly.

    python scripts/plot_ekf_pilot.py

Three questions, three panels, in the order a reader asks them:

**(a) Does the filter win?** Error against step, EKF versus the SGD baselines and
the frozen control. The gap is large enough to read without a scale bar, which is
the point of showing the curves rather than only the settled numbers.

**(b) Which hyperparameter carries it?** Best cell per level on each axis, all
four on one shared vertical scale so their spans are directly comparable. That
shared scale *is* the finding: $Q$ and $\lambda$ move the error by more than
0.02 while $\gamma$ moves it by 0.003 and $\sigma_0^2$ by less than the seed
noise. Plotted against a normalised axis position because the four axes have
incomparable units -- the shape and the span are what transfer, not the abscissa.

**(c) Were the baselines given their best shot?** Settled error against learning
rate, with the shipped value marked. Answers the objection that a drift-tuned
filter is being compared with a stationary-tuned baseline (design note D65).

The seed-noise band (median $|{\rm seed}_0-{\rm seed}_1|$ over all 80 cells,
0.0021) is drawn wherever a difference is being read, so no panel invites a
comparison finer than the data supports.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_ekf_sweep import (  # noqa: E402
    BASELINE_LRS,
    GAMMAS,
    LAMBDAS,
    PRIOR_SCALES,
    PROCESS_NOISES,
    baseline_name,
    cell_name,
    cells,
)

from dekf_bench.utils.paths import figures_dir  # noqa: E402

SETTLED = 1200
HORIZON = 1500
DPI = 200
BEST_CELL = "x13_g_s0p1_g0p999_q0p0001"

#: Median |seed0 - seed1| across all 80 pilot cells. Drawn, not asserted.
SEED_NOISE = 0.0021

LABELS = {
    "centralized_ekf_gamma": "EKF (gamma family)",
    "centralized_sgd": "Centralized SGD",
    "diffusion_sgd_atc": "ATC (momentum)",
    "frozen_atc": "Frozen at t=300",
}
COLOURS = {
    "centralized_ekf_gamma": "#c1440e",
    "centralized_sgd": "#444444",
    "diffusion_sgd_atc": "#1f77b4",
    "frozen_atc": "#9467bd",
}
STYLES = {"frozen_atc": "--"}

#: One hue per axis, and the two that matter get the saturated pair. Colour
#: carries the axis identity; every line is also directly labelled, so identity
#: never rests on colour alone.
AXIS_COLOURS = {
    "Q": "#c1440e",
    "lambda": "#1f77b4",
    "gamma": "#7f7f7f",
    "sigma_0^2": "#bbbbbb",
}


def load(directory: str) -> pd.DataFrame:
    files = sorted((ROOT / "results" / directory).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no results in results/{directory}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def error_curve(frame: pd.DataFrame, learner: str) -> pd.Series:
    rows = frame[
        (frame["learner"] == learner)
        & (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
    ]
    return rows.groupby("t")["value"].mean()


def settled(directory: str, learner: str) -> float | None:
    frame = load(directory)
    rows = frame[
        (frame["learner"] == learner)
        & (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
        & (frame["t"] >= SETTLED)
    ]
    return float(rows["value"].mean()) if len(rows) else None


def cell_scores() -> dict[str, float]:
    scores = {}
    for cell in cells():
        name = cell_name(cell)
        if not (ROOT / "results" / name).exists():
            continue
        value = settled(name, cell["name"])
        if value is not None:
            scores[name] = value
    return scores


def best_per_level(scores: dict[str, float]) -> dict[str, tuple[list, list[float]]]:
    """Best settled error at each level of each axis, for the marginals panel."""
    grid = cells()
    marginals: dict[str, tuple[list, list[float]]] = {}

    def best_where(predicate) -> float:
        values = [
            scores[cell_name(cell)]
            for cell in grid
            if cell_name(cell) in scores and predicate(cell)
        ]
        return min(values) if values else float("nan")

    gamma_only = lambda cell: cell["name"].endswith("gamma")  # noqa: E731
    lambda_only = lambda cell: cell["name"].endswith("lambda")  # noqa: E731

    marginals["Q"] = (
        PROCESS_NOISES,
        [best_where(lambda c, q=q: gamma_only(c) and c["process_noise_q"] == q)
         for q in PROCESS_NOISES],
    )
    marginals["lambda"] = (
        LAMBDAS,
        [best_where(lambda c, v=v: lambda_only(c) and c["lambda_forget"] == v) for v in LAMBDAS],
    )
    marginals["gamma"] = (
        GAMMAS,
        [best_where(lambda c, v=v: gamma_only(c) and c["gamma"] == v) for v in GAMMAS],
    )
    marginals["sigma_0^2"] = (
        PRIOR_SCALES,
        [best_where(lambda c, v=v: gamma_only(c) and c["prior_scale"] == v)
         for v in PRIOR_SCALES],
    )
    return marginals


def panel_curves(axis, scores: dict[str, float]) -> None:
    cell = load(BEST_CELL)
    baselines = load("x13_baselines")

    for learner, frame in (
        ("centralized_ekf_gamma", cell),
        ("centralized_sgd", baselines),
        ("diffusion_sgd_atc", baselines),
        ("frozen_atc", baselines),
    ):
        curve = error_curve(frame, learner)
        axis.plot(
            curve.index, curve.values,
            color=COLOURS[learner], linestyle=STYLES.get(learner, "-"),
            linewidth=2.0, label=LABELS[learner],
        )

    axis.axvspan(SETTLED, HORIZON, color="#000000", alpha=0.05, lw=0)
    # Inside the band, in the corridor between the flat SGD curves below and the
    # rising frozen one above. Blended coordinates -- data in x, axes-fraction in
    # y -- because the data limits are not settled when this is drawn, and
    # reading get_ylim() here puts the label off the figure entirely.
    axis.text(
        (SETTLED + HORIZON) / 2, 0.42, "settled\nwindow",
        transform=axis.get_xaxis_transform(),
        ha="center", va="center", fontsize=7, color="#888888",
    )
    axis.set_yscale("log")
    axis.set_xlabel("step")
    axis.set_ylabel("error rate (current evalset, log scale)")
    axis.set_title("(a) The filter tracks; the frozen control does not", fontsize=10, loc="left")
    axis.legend(fontsize=7.5, frameon=False, loc="upper right")
    axis.grid(alpha=0.25, linewidth=0.5)


def panel_marginals(axis, scores: dict[str, float]) -> None:
    marginals = best_per_level(scores)
    for name, (levels, values) in marginals.items():
        # Normalised position, because the four axes have incomparable units and
        # only the shape and the vertical span transfer. The abscissa carries no
        # shared meaning and is labelled so -- "more forgetting" would be true
        # for Q and lambda, backwards for neither, and simply false for
        # sigma_0^2, which is not a forgetting knob at all.
        positions = np.linspace(0, 1, len(levels))
        span = max(values) - min(values)
        heavy = span > 2 * SEED_NOISE
        axis.plot(
            positions, values,
            color=AXIS_COLOURS[name], linewidth=2.4 if heavy else 1.4,
            marker="o", markersize=5 if heavy else 3.5,
            label=f"{name}   span {span:.4f}" + ("" if heavy else "  (within noise)"),
            zorder=3 if heavy else 2,
        )
        # Only the axes that actually move the error get their optimum labelled.
        # On a flat axis the argmin is a noise draw, and printing it would invite
        # exactly the over-reading the span annotation exists to prevent.
        if heavy:
            best = int(np.argmin(values))
            # Above the marker, not below: Q's optimum is the lowest point on the
            # panel, and a label under it lands on the x-axis caption.
            axis.annotate(
                f"best {levels[best]:g}",
                (positions[best], values[best]), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=7.5, color=AXIS_COLOURS[name], fontweight="bold",
            )

    floor = min(min(v) for _, v in marginals.values())
    band = axis.axhspan(floor, floor + SEED_NOISE, color="#000000", alpha=0.07, lw=0)
    # The band is identified in the legend rather than by floating text. Four
    # curves, two optimum annotations and an axis caption leave no region of this
    # panel reliably empty, and a label that has to dodge the data is a label in
    # the wrong place.
    band.set_label(f"seed noise, {SEED_NOISE:.4f}")

    axis.set_xticks([])
    axis.set_xlabel("level, each axis on its own scale (see labels)", fontsize=8)
    axis.set_ylabel("best settled error at that level")
    axis.set_title("(b) Q and lambda carry it; gamma and the prior do not",
                   fontsize=10, loc="left")
    axis.legend(fontsize=7.5, frameon=False, loc="upper right")
    axis.set_ylim(top=floor + 1.30 * (max(max(v) for _, v in marginals.values()) - floor))
    axis.grid(alpha=0.25, linewidth=0.5, axis="y")


def panel_learning_rate(axis, scores: dict[str, float]) -> None:
    for learner in ("centralized_sgd", "diffusion_sgd_atc"):
        values, rates = [], []
        for lr in BASELINE_LRS:
            if not (ROOT / "results" / baseline_name(lr)).exists():
                continue
            value = settled(baseline_name(lr), learner)
            if value is not None:
                rates.append(lr)
                values.append(value)
        axis.plot(rates, values, color=COLOURS[learner], linewidth=2.0,
                  marker="o", markersize=5, label=LABELS[learner])

    best_ekf = min(scores.values())
    axis.axhline(best_ekf, color=COLOURS["centralized_ekf_gamma"], linewidth=2.0)
    axis.text(
        BASELINE_LRS[0], best_ekf, " best EKF cell", va="bottom", fontsize=7.5,
        color=COLOURS["centralized_ekf_gamma"],
    )
    axis.axvline(0.01, color="#666666", linewidth=1.0, linestyle=":")
    axis.text(0.0105, 0.55, "shipped lr,\ntuned on\nstationary data",
              fontsize=7, color="#666666", va="top")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("learning rate")
    axis.set_ylabel("settled error")
    axis.set_title("(c) The baselines had their best shot", fontsize=10, loc="left")
    axis.legend(fontsize=7.5, frameon=False, loc="upper left")
    axis.grid(alpha=0.25, linewidth=0.5)


def main() -> int:
    scores = cell_scores()
    if not scores:
        print("No pilot results found. Run scripts/run_ekf_sweep.py first.")
        return 1
    print(f"{len(scores)} pilot cells, best {min(scores.values()):.4f}")

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    panel_curves(axes[0], scores)
    panel_marginals(axes[1], scores)
    panel_learning_rate(axes[2], scores)

    figure.suptitle(
        "X13 pilot -- centralised EKF under linear drift, "
        f"alpha=0.025 deg/step, T={HORIZON}, 2 seeds",
        fontsize=11, y=1.00,
    )
    figure.tight_layout()

    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "24_ekf_pilot.png"
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
