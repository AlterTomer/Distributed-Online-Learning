r"""X15 -- is the gamma family's win a mechanism, or is lambda just mistuned?

Run this file directly.

    python scripts/run_ekf_retune.py          # 10 cells + 2 twins, 5 seeds (~7h)
    python scripts/run_ekf_retune.py --fresh  # discard and redo

## The question

X14 ran eighteen drift conditions and the gamma family came out ahead: it beat a
tuned ATC in 18/18, and beat the lambda family in 10 conditions to lambda's 4,
with 4 ties. Averaged into rate bands the gamma family leads everywhere --

    slow  (< 0.5 deg/step)   gamma 0.0143   lambda 0.0209
    mid   (0.5 - 1.5)        gamma 0.0292   lambda 0.0296
    fast  (>= 1.5)           gamma 0.0532   lambda 0.0607

-- which reads as a clean statement that additive inflation tracks better than
multiplicative forgetting. **That reading is not supported by how X14 was run.**

Both settings came from X13, which tuned at a *single* condition: linear drift at
0.025 deg/step, the mildest cell in the whole sweep. Neither was re-tuned for the
other seventeen. So X14 compared two settings, each chosen for a condition that
neither was then tested on.

The lambda family is the one that suffers from this, because its single knob has
a direct interpretation as a memory length -- lambda = 0.996 keeps roughly
1/(1-lambda) = 250 steps -- and X14's conditions place their shifts anywhere from
1.2 to 125 per memory window:

    t' = 200:    1.2 shifts per window   settles, then forgets with nothing new
    t' =  25:   10.0
    t' =   2:  125.0                     averages over the whole angle band

Its two worst cells are exactly those two extremes: `every200_jump15`, the only
place in X14 where a filter loses to ATC at all (-0.0049), and `every2_jump5`,
where it trails the gamma family by 0.0376. A knob that fails at both ends of an
axis it is not being adjusted along is a mistuning signature, not a mechanism.

So this sweep re-tunes both families at one fast condition and compares each at
its own optimum. If the gamma family still wins there, the mechanism claim is
earned. If they converge, X14's family ordering was a tuning artefact and should
be reported as one.

## Why `every2_jump5`, and not the fast cell the J=30 column offers

The obvious candidate was `every25_jump30`, and it is a bad one. Jumps reflect
inside [-45, +45], so the reachable angles go as 2*floor(45/J)+1 -- and above
J = 22.5 no two jumps can chain in one direction, which forces the walk back to
zero after every excursion:

    J = 5    12 angles     5% of steps at zero rotation
    J = 15    6 angles    13%
    J = 20    5 angles    37%
    J = 30    3 angles    50%   <- half the run is not rotated at all

Every J = 30 condition is therefore an oscillation between the unrotated
distribution and a single displaced pair, whatever its nominal rate. Tuning a
forgetting factor against a two-state alternation would measure recall of one
angle, not tracking.

`every2_jump5` is the fast cell with the least of that problem: 19 angles, 5% at
zero, 2.50 deg/step. It also carries by far the largest gamma-lambda gap in X14
(0.0575 against 0.0951), which is where the tuning hypothesis makes its sharpest
prediction -- a mistuned knob has the most room to recover.

Its one liability is that t' = 2 leaves only two steps between shifts, so no
transient completes. That cuts both ways and is worth stating plainly: it is the
condition where a *short* memory should help most, which is the same as saying it
is where lambda = 0.996 is most obviously the wrong number.

## What a cell contains, and what it is scored on

One EKF learner. ATC and the stationary references already exist in
`x14_every2_jump5` and `x14_control_n4_T1500` and are read from there rather than
recomputed.

**The config and runner are X14's own**, imported rather than reimplemented. That
is not tidiness: `config_for` stamps `jump_seed = JUMP_SEED` into every recurring
block, so a locally rebuilt config would draw a *different realisation of the same
drift law* and every comparison against the X14 references would be off by an
unmodelled term. Sharing the constructor makes the pairing hold by construction,
which is the same argument that lets the baselines be read instead of rerun.

**Cells are ranked on drift error, not on damage.** Damage subtracts a stationary
twin at the *same* setting, so ten cells would need ten twins and the sweep would
cost twice what it does. X13 settled this: tune on the drifting number, then run
twins for the survivors only, which is what the final block here does for the
best cell in each family. The damage comparison -- the one that answers the
question -- is made between those two winners.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from run_ekf_generalization import (  # noqa: E402
    HORIZON, SAMPLES, SEEDS, config_for, control_name, run_one, tuned_settings,
)

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
#: The condition both families are re-tuned at. See the module docstring for why
#: this one and not a J=30 cell. The label must match X14's, because the ATC
#: reference is read out of that run directory.
CONDITION_LABEL = "every2_jump5"
CONDITION = {"schedule": "recurring", "jump_every": 2, "jump_degrees": 5.0}

# --- the gamma family's axis ----------------------------------------------- #
#
# Only Q moves. X13 measured gamma as worth 0.0030 across its whole range and
# sigma_0^2 as flat to within the noise for this family, while Q was worth
# 0.0232 -- so a re-tune that varied all three would spend most of its cells
# confirming two flat axes.
#
#: X13's optimum was 6e-5 at 0.025 deg/step. This condition is a hundred times
#: faster, and Q is the per-step variance the filter believes theta accumulates,
#: so the optimum should move up. Four points over 1.5 decades, anchored at the
#: old value so "no change" is a result the grid can return.
GAMMA_PROCESS_NOISES = [6.0e-5, 2.0e-4, 6.0e-4, 2.0e-3]

# --- the lambda family's axes ---------------------------------------------- #
#
#: Extended far below X13's floor. Its note recorded 0.993 as the lowest stable
#: value *at 0.025 deg/step*; at 2.50 deg/step the memory that matches the shift
#: rate is an order of magnitude shorter, and 1/(1-lambda) for these four values
#: is 250, 100, 50 and 20 steps -- against a shift every 2 steps. Some of these
#: may inflate without bound, which is recorded and not treated as a failure.
LAMBDAS = [0.996, 0.99, 0.98, 0.95]

#: sigma_0^2 does not wash out under multiplicative forgetting -- P <- P/lambda
#: preserves scale, so the prior persists in a way it does not under additive Q
#: (design note D62). X13 measured this family as monotonically preferring a
#: smaller prior. Faster forgetting should sharpen that preference, so the two
#: shortest memories get a lower prior as well: a two-point check on whether the
#: axes interact, rather than a full second sweep.
LAMBDA_LOW_PRIOR = 1.0e-3
LAMBDA_LOW_PRIOR_VALUES = [0.98, 0.95]

FRESH = False

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x15_status.json"

#: Where the already-computed references live.
DRIFT_REFERENCE = f"x14_{CONDITION_LABEL}"
STILL_REFERENCE = control_name(SAMPLES, HORIZON)


# ---------------------------------------------------------------------------


def anchors() -> tuple[dict, dict]:
    r"""X14's two competing settings, read back rather than restated.

    The grid below is written as "X13's optimum, plus points around it", which is
    only true if the anchor values really are what X14 ran. Restating them as
    literals would let this sweep drift out of step with the experiment it is
    auditing -- the re-tune would silently become a comparison against something
    X14 never used, and the "no change is a result" reading of the first cell
    would be false without anything failing.
    """
    chosen = {s["name"]: s for s in tuned_settings()}
    missing = {"centralized_ekf_gamma", "centralized_ekf_lambda"} - set(chosen)
    if missing:
        raise SystemExit(f"X13 has no selected setting for {sorted(missing)}; run it first.")
    return chosen["centralized_ekf_gamma"], chosen["centralized_ekf_lambda"]


def cells() -> list[dict]:
    """Every grid point, gamma family first.

    Ordered family-major rather than most-informative-first, because with ten
    cells the whole sweep completes in one sitting and a readable ordering is
    worth more than a partial-result ordering. Within the lambda family the
    memory shortens monotonically, so a divergence boundary shows up as a
    contiguous tail rather than scattered cells.
    """
    gamma_anchor, lambda_anchor = anchors()
    if gamma_anchor["process_noise_q"] not in GAMMA_PROCESS_NOISES:
        raise SystemExit(
            f"X14 ran the gamma family at Q={gamma_anchor['process_noise_q']:g}, which is "
            f"not in GAMMA_PROCESS_NOISES. The grid no longer brackets what it audits."
        )
    if lambda_anchor["lambda_forget"] not in LAMBDAS:
        raise SystemExit(
            f"X14 ran the lambda family at lambda={lambda_anchor['lambda_forget']:g}, which "
            f"is not in LAMBDAS. The grid no longer brackets what it audits."
        )

    grid = [{**gamma_anchor, "process_noise_q": noise} for noise in GAMMA_PROCESS_NOISES]
    grid += [{**lambda_anchor, "lambda_forget": lam} for lam in LAMBDAS]
    grid += [
        {**lambda_anchor, "lambda_forget": lam, "prior_scale": LAMBDA_LOW_PRIOR}
        for lam in LAMBDA_LOW_PRIOR_VALUES
    ]
    return grid


def cell_name(cell: dict) -> str:
    """A directory name that can be read back into the settings it came from."""
    def tag(value: float) -> str:
        return f"{value:g}".replace(".", "p").replace("-", "m")

    if cell["name"].endswith("gamma"):
        return f"x15_g_q{tag(cell['process_noise_q'])}"
    return f"x15_l_l{tag(cell['lambda_forget'])}_s{tag(cell['prior_scale'])}"


def settled(directory: str, learner: str) -> float:
    """One learner's settled error in one run directory, or +inf if absent."""
    import pandas as pd

    files = sorted((ROOT / "results" / directory).glob("*.parquet"))
    if not files:
        return float("inf")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    rows = frame[
        (frame["learner"] == learner)
        & (frame["metric"] == "error_rate")
        & (frame["evalset"] == "current")
        & (frame["t"] >= int(0.8 * HORIZON))
    ]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def load_status() -> dict:
    if STATUS.exists():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    for reference in (DRIFT_REFERENCE, STILL_REFERENCE):
        if not (ROOT / "results" / reference / "_complete").exists():
            print(f"X15 reads its ATC and stationary references from {reference},\n"
                  f"which has not completed. Run X14 first:\n"
                  f"  python scripts/run_ekf_generalization.py --only {CONDITION_LABEL}")
            return 1
    train, test = load_mnist(DATA_ROOT, download=False)

    print("X13's selected settings, which this sweep brackets:")
    grid = cells()
    status = load_status()
    atc = settled(DRIFT_REFERENCE, "diffusion_sgd_atc")
    rate = CONDITION["jump_degrees"] / CONDITION["jump_every"]
    print(f"\nX15 re-tune at {CONDITION_LABEL}: {len(grid)} cells at {len(SEEDS)} seeds")
    print(f"  {CONDITION['jump_degrees']:g} deg every {CONDITION['jump_every']} steps"
          f" -> {rate:.2f} deg/step, T={HORIZON}, n={SAMPLES}")
    print(f"  ATC settles at {atc:.4f} here (read from {DRIFT_REFERENCE})\n", flush=True)

    started = time.time()
    ran = 0
    for index, cell in enumerate(grid, start=1):
        name = cell_name(cell)
        note = run_one(
            config_for(name, CONDITION, [cell], SEEDS), train, test, fresh
        )
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(grid) - index)) if ran else 0.0
        score = settled(name, cell["name"]) if note in ("ok", "cached") else float("inf")
        shown = f"{score:.4f}" if score != float("inf") else "--"
        print(f"[{index}/{len(grid)}] {name:<26} {note:<30} settled {shown:<9}"
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    # --- stationary twins, for the winner of each family only --------------- #
    #
    # Damage needs a twin at the SAME setting, so a twin per cell would double the
    # sweep. The comparison that answers the question is between the two family
    # optima, and those are the two that get one.
    print("\nstationary twins for the best cell in each family", flush=True)
    for family, prefix in (("centralized_ekf_gamma", "x15_g_"),
                           ("centralized_ekf_lambda", "x15_l_")):
        scored = [
            (settled(cell_name(c), family), c) for c in grid
            if cell_name(c).startswith(prefix)
            and status.get(cell_name(c), "") in ("ok", "cached")
        ]
        scored = [(value, c) for value, c in scored if value != float("inf")]
        if not scored:
            print(f"    {family}: every cell diverged, no twin to run", flush=True)
            continue
        best_score, best = min(scored, key=lambda pair: pair[0])
        name = f"{cell_name(best)}_control"
        print(f"    {family}: {cell_name(best)} at {best_score:.4f}", flush=True)
        note = run_one(config_for(name, None, [best], SEEDS), train, test, fresh)
        status[name] = note
        save_status(status)
        twin = settled(name, family)
        shown = f"{twin:.4f}" if twin != float("inf") else "--"
        print(f"      twin {note}, stationary {shown}, "
              f"damage {best_score - twin:+.4f}", flush=True)

    diverged = sum(1 for note in status.values() if note.startswith("diverged"))
    print(f"\nX15 complete in {(time.time() - started) / 60:.1f} min")
    print(f"{len(status) - diverged} ran, {diverged} diverged")
    print(f"status: {STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv))
