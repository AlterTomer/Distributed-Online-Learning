r"""The claims a figure's *labels* make about the data.

A figure can be drawn correctly and still lie in its axis titles. F3 reports a
window called `settled`; that word asserts the gap has stopped changing, which is
a measurable claim and was originally just an assumption -- the window was chosen
as "the last 100 steps of the run" and labelled after the fact.

These tests check the labels against the data. They are skipped when the results
are absent, so a fresh clone does not fail on them.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results"
TOPOLOGIES = ("path", "grid2d", "star", "ring", "watts_strogatz", "erdos_renyi", "complete")


def gap_over_time(topology: str) -> tuple[pd.Series, pd.Series]:
    """Mean and seed s.d. of the ATC-minus-centralized gap at each step."""
    files = glob.glob(str(RESULTS / f"x3_{topology}" / "seed_*.parquet"))
    if not files:
        pytest.skip(f"no X3 results for {topology}")
    frame = pd.concat([pd.read_parquet(f) for f in files])
    rows = frame[(frame.evalset == "current") & (frame.metric == "error_rate")]
    grouped = rows.groupby(["learner", "seed", "t"])[["n_correct", "n_samples"]].sum()
    grouped["err"] = 1.0 - grouped.n_correct / grouped.n_samples
    gap = (
        grouped.loc["diffusion_sgd_atc", "err"] - grouped.loc["centralized_sgd", "err"]
    ).reset_index()
    per_step = gap.groupby("t").err
    return per_step.mean(), per_step.std()


@pytest.mark.needs_data
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_settled_window_is_actually_settled(topology: str) -> None:
    """No residual slope at the end of the run, for any topology.

    "Settled" is a claim about the data, not a synonym for "last 100 steps". A
    line fitted over the final 500 steps must have a slope smaller than the seed
    s.d. over the same span -- otherwise the gap is still converging and F3's
    right-hand column is reporting a moving target.

    This is what would catch a change to `run.horizon` that left the window
    sitting inside the transient.
    """
    from make_figures import F3_WINDOWS

    settled_lo = next(lo for name, lo, _ in F3_WINDOWS if name == "settled")
    mean, spread = gap_over_time(topology)

    tail = mean.loc[settled_lo - 400 :]
    slope_per_500 = float(np.polyfit(tail.index, tail.values, 1)[0]) * 500
    noise = float(spread.loc[settled_lo:].mean())

    assert abs(slope_per_500) <= max(noise, 1e-4), (
        f"{topology}: the gap still moves {slope_per_500:+.4f} per 500 steps at the end, "
        f"against a seed s.d. of {noise:.4f} -- the 'settled' window is not settled"
    )


@pytest.mark.needs_data
def test_the_transient_window_precedes_the_settled_one() -> None:
    """Ordering, which is cheap to get wrong when editing the constant."""
    from make_figures import F3_WINDOWS

    windows = {name: (lo, hi) for name, lo, hi in F3_WINDOWS}
    assert windows["transient"][1] <= windows["settled"][0]


@pytest.mark.needs_data
def test_the_transient_window_shows_a_wider_spread_than_the_settled_one() -> None:
    """The reason F3 reports two windows at all.

    Connectivity matters most while information is still propagating. If the
    spread across topologies were the same in both, the transient column would
    be redundant and the figure should drop it.
    """
    from make_figures import F3_WINDOWS

    spreads = {}
    for name, lo, hi in F3_WINDOWS:
        values = []
        for topology in TOPOLOGIES:
            mean, _ = gap_over_time(topology)
            window = mean.loc[lo : hi - 1]
            if not window.empty:
                values.append(float(window.mean()))
        spreads[name] = max(values) - min(values)

    assert spreads["transient"] > spreads["settled"], (
        f"transient spread {spreads['transient']:.4f} is not wider than settled "
        f"{spreads['settled']:.4f}; the two-window design buys nothing"
    )


# =========================================================================== #
# F6a / F6b -- quantities with a sign that is fixed by construction
# =========================================================================== #


def test_the_retuning_penalty_is_never_negative() -> None:
    """F6b's penalty is "headline lr minus the best lr", over a grid containing it.

    A minimum over a set cannot exceed a member of that set, so a negative cell
    is a bug rather than a finding. Two produced them: lr 0.2 was missing from
    the sweep grid, and the two terms were drawn from different estimators (a
    five-seed X4 error minus a two-seed sweep minimum), which reached -0.042.
    Both are invisible in the figure -- a negative penalty just renders as the
    palest cell -- so they are checked here instead.
    """
    from make_figures import x4_headline, x4_tuned

    tuned, headline = x4_tuned(), x4_headline()
    if tuned.empty or headline.empty:
        pytest.skip("no X4 sweep results")

    merged = headline.merge(tuned[["learner", "n", "pi", "error"]], on=["learner", "n", "pi"])
    merged["penalty"] = merged.fixed - merged.error
    bad = merged[merged.penalty < -1e-9]
    assert bad.empty, "negative re-tuning penalties:\n" + bad.to_string(index=False)


def test_the_payload_matched_variant_is_not_tuned_into_being_atc() -> None:
    """Its defining property is carrying no optimizer state.

    Both names map to one class and the sweep sets the optimizer for every
    learner it runs, so an unconstrained tuning picks momentum for the
    payload-matched variant and it becomes numerically identical to ATC. The
    payload cost then reads exactly 0.000 in all twelve cells, which looks like
    a strong null result and is actually a definition being overridden.
    """
    from make_figures import x4_tuned

    tuned = x4_tuned()
    if tuned.empty:
        pytest.skip("no X4 sweep results")

    plain = tuned[tuned.learner == "diffusion_sgd_atc_plain"]
    assert not plain.empty, "the payload-matched variant is missing from the sweep"
    assert (plain.optimizer == "sgd").all(), (
        "the payload-matched variant was tuned onto a stateful optimizer: "
        f"{sorted(set(plain.optimizer))}"
    )

    atc = tuned[tuned.learner == "diffusion_sgd_atc"]
    cost = plain.merge(atc, on=["n", "pi"], suffixes=("_plain", "_atc"))
    cost["payload"] = cost.error_plain - cost.error_atc
    assert (cost.payload > 0).all(), (
        "the payload cost is non-positive somewhere, which at matched tuning "
        f"means the two arms have collapsed:\n{cost[['n', 'pi', 'payload']].to_string(index=False)}"
    )


# =========================================================================== #
# document integrity
# =========================================================================== #


def test_no_document_contains_a_mangled_latex_escape() -> None:
    r"""Control characters in the docs, which is what a broken `\times` looks like.

    Writing LaTeX-heavy prose through a shell heredoc into a non-raw Python
    string silently converts `\times` to TAB+"imes", `\alpha` to BEL+"lpha" and
    `\approx` to BEL+"pprox". The result renders as a plausible-looking sentence
    with a character missing, and has slipped through three times.

    Cheap to check, so checked rather than remembered.
    """
    bad = []
    for path in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.split("\n"), start=1):
            if any(char in line for char in "\x07\x08\x09\x0b\x0c\x1b"):
                bad.append(f"{path.name}:{number}: {line.strip()[:70]!r}")
    assert not bad, "control characters in documentation:\n  " + "\n  ".join(bad)
