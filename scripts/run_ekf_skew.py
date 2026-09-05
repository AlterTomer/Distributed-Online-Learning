r"""X17 -- the centralised filter under Dirichlet label skew, still and drifting.

Run this file directly.

    python scripts/run_ekf_skew.py          # 7 runs, ~3h
    python scripts/run_ekf_skew.py --fresh  # discard and redo

Every filter result so far -- X13 through X16, the 21/21 generalisation, the
gamma/lambda decision, the break rate -- was measured with an **IID** partition.
Of 242 completed runs only three use Dirichlet skew, and all three are X6, which
predates the filter. So "the filter has never been tested under label skew" was
true and unstated, which is the kind of gap that survives until someone asks.

## Why the obvious version of this experiment would measure nothing

The centralised filter trains on the union $\bigcup_v \mathcal D_t^v$, and
Dirichlet skew is a statement about how labels are split *across agents*. Pooling
puts them back together. X6's own numbers show how completely, at the settled
error across $\beta\in\{0.1,1,100\}$:

    centralized SGD  (pooled)      0.0788  0.0777  0.0797    spread 0.0020
    ATC              (per-agent)   0.1021  0.0812  0.0809    spread 0.0212
    ATC plain        (per-agent)   0.1188  0.0931  0.0939    spread 0.0257
    local only       (per-agent)   0.6301  0.2757  0.1419    spread 0.4882

Against a 0.0013 threshold the pooled learner barely moves while local-only moves
by half. Skew is a property of the *distribution* of data, and a pooled method
undoes the distribution.

## What is worth measuring anyway, and it is not "does it survive"

Two things, and neither is the question X6 asks of the distributed learners.

**(a) The curvature hypothesis.** Under skew the per-step pooled batch of $Nn=40$
samples is drawn from ten shards with different label mixes, so its
*composition* has higher variance than IID even though the marginal is right.
\ac{sgd} averages gradients and is insensitive to that. The filter estimates
curvature $\bm H^{\trans}\bm\Lambda\bm H$ from the same batch, and
$\bm\Lambda=\diag(\bm\pi)-\bm\pi\bm\pi^{\trans}$ depends on the predicted class
distribution -- so there is a mechanism by which the filter could be *more*
skew-sensitive than pooled \ac{sgd} despite seeing the same marginal. If the
filter's spread across $\beta$ matches centralized SGD's 0.0020, the mechanism is
absent and the null is worth recording. If it is several times larger, that is a
property of second-order methods on heterogeneous data and it is new.

**(b) Skew crossed with drift**, which is where the project's claim lives. The
fitting half of the advantage is what skew would plausibly damage; the tracking
half is measured against a paired twin and might not care. Only a drifting cell
separates them, so one is run: strong skew at $\beta=0.1$ with the recurring
shifts of `every25_jump15`, plus its own stationary twin at the same $\beta$.

## What is deliberately not varied

The setting is the one X13 selected and X14/X15 validated, read from the sweep
rather than restated: $\gamma=0.9995$, $q=6\times10^{-5}$, $\sigma_0^2=10^{-2}$.
Re-tuning per $\beta$ would answer a different question -- "can the filter be
made to cope" rather than "does the setting we chose cope" -- and the second is
the one that matters for a setting we intend to carry into the diffusion filter.
If the skew cells come out badly, re-tuning is the follow-up, not the first move.
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
from dekf_bench.utils.config import load_config  # noqa: E402
from run_ekf_generalization import run_one, tuned_settings  # noqa: E402

#: X6's own axis and horizon, so the stationary cells are comparable to it.
BETAS = [0.1, 1.0, 100.0]
HORIZON = 1500
SEEDS = [0, 1, 2, 3, 4]
EVAL_EVERY = 5

#: Two drift cells at the SAME rate and opposite abruptness, so the comparison
#: is "which kind of motion hurts more alongside skew" rather than "does drift
#: hurt". Both are J/t' = 0.60 deg/step, and measured over the run they cover
#: near-identical ground -- 885 degrees of total travel against 899 -- but:
#:
#:     every25_jump15   59 moves of 15.00 deg, still on 96% of steps, 6 angles
#:     every1_jump0p6   1499 moves of  0.60 deg, moving every step,   63 angles
#:
#: 0.60 deg/step also sits mid-range on both axes, so an interaction with skew
#: has room to show in either direction rather than being clipped by a condition
#: that already dominates the error.
#:
#: **They differ in state count as well as abruptness**, and that is unavoidable:
#: abruptness *is* the J/t' ratio at fixed rate, so holding the angle count fixed
#: too is not possible (design note D74). The near-equal travel is what makes the
#: pair fair; the angle counts are what stops it being read as a pure abruptness
#: experiment.
#:
#: There is a prior to beat. X11 against X12 found abrupt shifts cost *less* than
#: smooth drift at matched speed, which was not the expectation. If skew reverses
#: that ordering, the interaction is the result rather than the drift.
DRIFT_BETA = 0.1
DRIFTS = {
    "abrupt": {"schedule": "recurring", "jump_every": 25,
               "jump_degrees": 15.0, "jump_seed": 0},
    "smooth": {"schedule": "recurring", "jump_every": 1,
               "jump_degrees": 0.6, "jump_seed": 0},
}

#: The baselines ride along in every cell. Without them a skew effect on the
#: filter could not be told apart from a skew effect on the *task*, which is
#: large -- X6 moves local-only by 0.49.
BASELINES = ["centralized_sgd", "diffusion_sgd_atc", "local_only"]
BASELINE_LR = 0.05

DEVICE = "auto"
DTYPE = "float64"
FRESH = False

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x17_status.json"


def learners(setting: dict) -> list[dict]:
    return [setting] + [
        {"name": name, "optimizer": "sgd_momentum", "lr": BASELINE_LR, "momentum": 0.9}
        for name in BASELINES
    ]


def config_for(name: str, beta: float, setting: dict, drift: dict | None):
    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": name,
                "horizon": HORIZON,
                "eval_every": EVAL_EVERY,
                "seeds": SEEDS,
                "device": DEVICE,
                "dtype": DTYPE,
            },
            "env": {"partition": {"kind": "dirichlet", "beta": beta}, "drift": block},
            "learners": learners(setting),
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)

    print("X13's selected setting, carried here unchanged:")
    setting = next(s for s in tuned_settings() if s["name"] == "centralized_ekf_gamma")

    cells: list[tuple[str, float, dict | None]] = [
        (f"x17_still_beta{beta:g}", beta, None) for beta in BETAS
    ]
    # One control serves both drift cells: damage is drift minus twin, the twin is
    # stationary at the same beta and seeds, and neither drift block reaches it.
    cells += [
        (f"x17_{kind}_beta{DRIFT_BETA:g}", DRIFT_BETA, block)
        for kind, block in DRIFTS.items()
    ]
    cells += [(f"x17_drift_control_beta{DRIFT_BETA:g}", DRIFT_BETA, None)]

    print(f"\nX17: {len(cells)} cells at {len(SEEDS)} seeds, T={HORIZON}")
    print(f"  stationary at beta in {BETAS}")
    print(f"  two drift cells at beta={DRIFT_BETA:g}, both 0.60 deg/step, sharing one twin:")
    for kind, block in DRIFTS.items():
        print(f"    {kind:<8} {block['jump_degrees']:g} deg every "
              f"{block['jump_every']} step(s)")
    print(f"  filter + {', '.join(BASELINES)}\n", flush=True)

    status = load_status()
    started = time.time()
    ran = 0
    for index, (name, beta, drift) in enumerate(cells, start=1):
        note = run_one(config_for(name, beta, setting, drift), train, test, fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(cells) - index)) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<32} {note:<30} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nX17 complete in {(time.time() - started) / 60:.1f} min")
    print(f"status: {STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv))
