r"""What a drift regime costs, and whether cooperation still pays under it.

Run this file directly.

    python scripts/report_cooperation.py
    python scripts/report_cooperation.py x8_per_node_drift x8_global

Two readings of one paired experiment.

**Damage** is the per-learner cost of the treatment: its error minus the
control's, seed for seed. It says how much the regime hurt.

**The cooperation gap** is ``local_only`` minus a diffusing method *within* each
run. It says what communication was worth there. This is the reading X8 exists
for: under global drift the neighbours have adapted to the same state, so
combining is pure variance reduction and can essentially only help; under
per-node drift they sit at different states, so combining also drags each agent
toward one that is not its own. If the gap shrinks or reverses, that is
consensus ceasing to pay -- and it is the claim a *diffusion* filter rests on,
as distinct from a local one.

The gap is read inside each run rather than across the pair on purpose: both
learners there share one environment, one graph and one $\bm\theta_0$, so the
difference is paired by construction rather than by seed alone.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dekf_bench.metrics.breaks import assert_paired_runs, error_by_step, paired_excess  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
TREATMENT = "x8_per_node_drift"
CONTROL = "x8_global"
#: Which pair of learners the cooperation gap is measured between.
ALONE = "local_only"
TOGETHER = "diffusion_sgd_atc"
#: The settled window, as a fraction of the horizon.
SETTLED_FROM = 0.5
EVALSET = "current"


def load(name: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(ROOT / "results" / name / "seed_*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def settled(frame: pd.DataFrame, evalset: str) -> pd.Series:
    """Counts-then-divide error per learner over the settled window."""
    errors = error_by_step(frame, evalset)
    horizon = int(errors.t.max()) + 1
    late = errors[errors.t >= SETTLED_FROM * horizon]
    return late.groupby("learner").apply(
        lambda rows: 1.0 - rows.n_correct.sum() / rows.n_samples.sum(), include_groups=False
    )


def gap_with_error(frame: pd.DataFrame, evalset: str) -> tuple[float, float]:
    """Cooperation gap and its seed s.e.m., paired within each seed."""
    errors = error_by_step(frame, evalset, by_seed=True)
    horizon = int(errors.t.max()) + 1
    late = errors[errors.t >= SETTLED_FROM * horizon]
    wide = late.pivot_table(index=["seed", "t"], columns="learner", values="error")
    if ALONE not in wide.columns or TOGETHER not in wide.columns:
        return float("nan"), float("nan")
    per_seed = (wide[ALONE] - wide[TOGETHER]).groupby("seed").mean()
    return float(per_seed.mean()), float(per_seed.std() / len(per_seed) ** 0.5)


def main() -> int:
    treatment_name = sys.argv[1] if len(sys.argv) > 1 else TREATMENT
    control_name = sys.argv[2] if len(sys.argv) > 2 else CONTROL

    treatment, control = load(treatment_name), load(control_name)
    for name, frame in ((treatment_name, treatment), (control_name, control)):
        if frame is None:
            raise SystemExit(f"no results for {name!r}")

    assert_paired_runs(ROOT / "results" / treatment_name, ROOT / "results" / control_name)
    excess = paired_excess(treatment, control, EVALSET)
    horizon = int(excess.t.max()) + 1
    late = excess[excess.t >= SETTLED_FROM * horizon]
    damage = late.groupby("learner").excess.mean()
    spread = late.groupby(["learner", "seed"]).excess.mean().groupby("learner").std() / (
        excess.seed.nunique() ** 0.5
    )

    treated = settled(treatment, EVALSET)
    untreated = settled(control, EVALSET)

    print(f"{treatment_name}  against  {control_name}")
    print(f"  {EVALSET} set, settled window from {SETTLED_FROM:.0%} of T={horizon}\n")
    header = f"{'learner':26s} {'control':>9} {'treated':>9} {'damage':>9} {'+- sem':>8}"
    print(header)
    print("-" * len(header))
    for learner in sorted(treated.index):
        print(
            f"{learner:26s} {untreated.get(learner, float('nan')):>9.4f} "
            f"{treated.get(learner, float('nan')):>9.4f} "
            f"{damage.get(learner, float('nan')):>9.4f} "
            f"{spread.get(learner, float('nan')):>8.4f}"
        )

    print(f"\n  cooperation gap: {ALONE} minus {TOGETHER}, measured inside each run")
    control_gap, control_sem = gap_with_error(control, EVALSET)
    treated_gap, treated_sem = gap_with_error(treatment, EVALSET)
    print(f"    {control_name:24s} {control_gap:>8.4f} +- {control_sem:.4f}")
    print(f"    {treatment_name:24s} {treated_gap:>8.4f} +- {treated_sem:.4f}")
    change = treated_gap - control_gap
    verdict = "shrinks" if change < 0 else "widens"

    # The change in the gap is exactly damage(alone) - damage(together), and
    # measuring it that way is far more sensitive: the two runs share a seed,
    # hence a graph and a theta_0, so the paired subtraction cancels noise that
    # comparing two independently-computed gaps leaves in. Differencing the two
    # gap standard errors instead can call a real effect noise -- on X8 it did,
    # by a factor of seven.
    per_seed = (
        late[late.learner == ALONE].groupby(["seed", "t"]).excess.mean()
        - late[late.learner == TOGETHER].groupby(["seed", "t"]).excess.mean()
    )
    by_seed = per_seed.groupby("seed").mean()
    paired_change = float(by_seed.mean())
    paired_sem = float(by_seed.std() / len(by_seed) ** 0.5)

    loose = (control_sem**2 + treated_sem**2) ** 0.5
    print(f"    change {change:+.4f} ({verdict})")
    print(
        f"      as a paired difference: {paired_change:+.4f} +- {paired_sem:.4f} "
        f"(3 s.e.m. {3 * paired_sem:.4f}) -- "
        f"{'clear of the noise' if abs(paired_change) > 3 * paired_sem else 'inside the noise'}"
    )
    print(
        f"      unpaired, for contrast:  3 s.e.m. {3 * loose:.4f} -- the weaker estimate, "
        "and not the one to read"
    )

    if "current_mean" in {str(v) for v in treatment.evalset.unique()}:
        mean_gap, mean_sem = gap_with_error(treatment, "current_mean")
        print(
            f"\n  the same gap scored at the network-mean state: {mean_gap:.4f} +- {mean_sem:.4f}"
            "\n    (a state no agent occupies; it separates 'this agent learned worse'"
            "\n     from 'this agent sits further from the mean')"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
