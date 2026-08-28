r"""X14 -- read the generalisation experiment.

Run this file directly.

    python scripts/report_ekf_generalization.py        # the conditions
    python scripts/report_ekf_generalization.py --lr   # the baseline re-tune

Reports **damage** -- each condition's settled error minus the shared stationary
control's -- rather than raw error, because raw error confounds "this regime is
hard" with "this method handles it badly". Damage is the quantity every break in
this project is defined over (design note D54), and it is what the conditions
were selected on in the first place.

The question is not whether the filter wins somewhere. X13 established that at
one mild rate. It is whether **one setting**, tuned at linear 0.025, still holds
at conditions up to 5x more damaging -- and in particular under abrupt shifts,
which the filter's process-noise model should suit and which no part of the
tuning ever saw.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from run_ekf_generalization import (  # noqa: E402
    BASELINES,
    CONDITIONS,
    LR_CONDITION,
    STATUS,
    control_name,
    lr_name,
)
from run_ekf_sweep import BASELINE_LRS  # noqa: E402

FILTERS = ("centralized_ekf_gamma", "centralized_ekf_lambda")
SHORT = {
    "centralized_ekf_gamma": "EKF gamma",
    "centralized_ekf_lambda": "EKF lambda",
    "centralized_sgd": "Central",
    "diffusion_sgd_atc": "ATC",
    "frozen_atc": "Frozen",
}
#: Measured on the X13 pilot, 80 cells at two seeds. Five seeds here, so the
#: floor is tighter, but quoting the looser number keeps the bar conservative.
SEED_NOISE = 0.0021


def settled(directory: str, learner: str, horizon: int) -> float | None:
    """Mean `current` error over the last fifth of a run of length ``horizon``.

    The window is derived from the run's own horizon rather than fixed, because
    the low-n conditions are twice as long -- a constant cut-off would read the
    settled error of one and the mid-run transient of the other.
    """
    files = sorted((ROOT / "results" / directory).glob("*.parquet"))
    if not files:
        return None
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = frame[
        (frame["learner"] == learner)
        & (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
        & (frame["t"] >= int(0.8 * horizon))
    ]
    return float(rows["value"].mean()) if len(rows) else None


def report_lr() -> int:
    horizon = next(t for label, _, _, t in CONDITIONS if label == LR_CONDITION)
    print(f"X14 baseline re-tune, on {LR_CONDITION} -- the most damaging condition")
    print(f"settled error over the last fifth of T={horizon}\n")
    print("      lr  " + "".join(f"{SHORT[n]:>12}" for n in BASELINES))
    rows = []
    for lr in BASELINE_LRS:
        values = [settled(lr_name(lr), name, horizon) for name in BASELINES]
        if all(v is None for v in values):
            continue
        rows.append((lr, values))
        print(f"  {lr:>6g}  " + "".join(
            f"{v:>12.4f}" if v is not None else f"{'.':>12}" for v in values
        ))
    if not rows:
        print("\n  nothing run yet: python scripts/run_ekf_generalization.py --lr")
        return 1

    index = BASELINES.index("diffusion_sgd_atc")
    scored = [(row[1][index], row[0]) for row in rows if row[1][index] is not None]
    value, lr = min(scored)
    print(f"\n  ATC prefers lr {lr:g} ({value:.4f}) under abrupt shifts.")
    x13 = [v for v, rate in scored if rate == 0.02]
    if x13:
        print(f"  X13 chose 0.02 under smooth drift, which gives {x13[0]:.4f} here "
              f"({x13[0] - value:+.4f}).")
    return 0


def main() -> int:
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    if not status:
        print(f"No results yet. Expected {STATUS}")
        return 1

    learners = list(FILTERS) + BASELINES

    # One control per (n, T). Each condition subtracts the one matching its own
    # setting: a control at a different horizon or sample count would produce
    # perfectly well-formed rows whose difference means nothing.
    settings_used = sorted({(samples, horizon) for _, _, samples, horizon in CONDITIONS})
    controls: dict[tuple[int, int], dict[str, float | None]] = {}
    for samples, horizon in settings_used:
        controls[(samples, horizon)] = {
            name: settled(control_name(samples, horizon), name, horizon) for name in learners
        }

    if all(c.get("diffusion_sgd_atc") is None for c in controls.values()):
        print("No stationary control has run; damage cannot be formed.")
        return 1

    print("X14 -- one tuned setting per family, across regimes it never saw")
    print("damage = condition minus its own (n, T) control, over the last fifth\n")

    for (samples, horizon), control in controls.items():
        print(f"stationary control, n={samples} T={horizon}:")
        for name in learners:
            value = control[name]
            print(f"  {SHORT[name]:<12} {value:.4f}" if value is not None
                  else f"  {SHORT[name]:<12} .")

    header = "".join(f"{SHORT[n]:>12}" for n in learners)
    print(f"\n=== damage by condition ==={'':<10}{header}")
    rows = []
    for label, _, samples, horizon in CONDITIONS:
        name = f"x14_{label}"
        control = controls[(samples, horizon)]
        values = []
        for learner in learners:
            value = settled(name, learner, horizon)
            base = control[learner]
            values.append(None if value is None or base is None else value - base)
        if all(v is None for v in values):
            continue
        rows.append((f"{label}  (n={samples})", values))

    for label, values in sorted(rows, key=lambda r: r[1][learners.index("diffusion_sgd_atc")] or 0):
        cells_text = "".join(
            f"{v:>12.4f}" if v is not None else f"{'.':>12}" for v in values
        )
        print(f"  {label:<32}{cells_text}")

    print("\n=== filter advantage: ATC damage minus filter damage ===")
    print(f"  positive means the filter is less damaged. Noise floor {SEED_NOISE:.4f}.\n")
    atc = learners.index("diffusion_sgd_atc")
    for label, values in sorted(rows, key=lambda r: r[1][atc] or 0):
        if values[atc] is None:
            continue
        parts = []
        for name in FILTERS:
            value = values[learners.index(name)]
            if value is None:
                parts.append(f"{SHORT[name]}: .")
                continue
            gain = values[atc] - value
            mark = "" if abs(gain) > SEED_NOISE else "  (within noise)"
            parts.append(f"{SHORT[name]} {gain:+.4f}{mark}")
        print(f"  {label:<32} ATC damage {values[atc]:.4f}   " + "   ".join(parts))

    held = [
        label for label, values in rows
        if values[atc] is not None
        and any(
            values[learners.index(n)] is not None
            and values[atc] - values[learners.index(n)] > SEED_NOISE
            for n in FILTERS
        )
    ]
    print(f"\n{len(held)} of {len(rows)} conditions where a filter is measurably "
          f"less damaged than ATC.")
    diverged = [name for name, note in status.items() if note.startswith("diverged")]
    for name in diverged:
        print(f"  div  {name}")
    return 0


if __name__ == "__main__":
    if "--lr" in sys.argv:
        raise SystemExit(report_lr())
    raise SystemExit(main())
