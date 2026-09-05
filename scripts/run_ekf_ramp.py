r"""X16 -- the centralised filter on X9's rate ramp, for the break figure.

Run this file directly.

    python scripts/run_ekf_ramp.py          # two runs, ~1h
    python scripts/run_ekf_ramp.py --fresh  # discard and redo

X9 locates the drift rate at which each method's tracking gives way, by sweeping
the rate within a single run: the ramp climbs from 0 to 0.18 deg/step while
ending at exactly 45 degrees, so the step at which damage takes off names a
*rate* rather than a step count. It answered that question for the gradient
baselines in phase 4, before the filter existed, so its figure has no \ac{ekf}
curve on it and cannot be given one by editing the plot.

**Two runs rather than a re-run of X9.** Given the same seed, the environment and
$\bm\theta_0$ do not depend on the learners list, so a filter-only run at X9's
seeds is paired with X9 *by construction* rather than by seed -- the same
argument that lets X13 compute its baselines once instead of eighty times. That
leaves the phase-4 results untouched, which matters because the existing figures
and documents cite them, and it avoids recomputing five SGD arms that would
return exactly what they returned before.

**Ring, not Erdos-Renyi, and it costs nothing.** X9 is on ring, and the standing
topology standard is Erdos-Renyi(0.3) (D52). The centralised filter pools every
agent's batch and transmits nothing, so its trajectory is *identical* on any
topology; matching X9's graph keeps the pairing exact and changes no number.

**The setting is the one X13 selected and X14/X15 validated**, read from the
sweep rather than restated here: gamma < 1 with additive Q, tuned at linear
0.025 deg/step. The ramp reaches 0.18 deg/step, well inside the range over which
that setting was shown to generalise (D74, D76), so this is a
tuned-elsewhere-reported-here comparison rather than a fitted one.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402
from run_ekf_generalization import run_one, tuned_settings  # noqa: E402

#: X9's own settings, so the pairing holds. `eval_every` is denser than the
#: default because the whole point is to locate a takeoff in t, and the
#: resolution of that estimate is the evaluation cadence.
HORIZON = 1500
EVAL_EVERY = 10
SEEDS = [0, 1, 2, 3, 4]
DEVICE = "auto"
DTYPE = "float64"
FRESH = False

DATA_ROOT = ROOT / "data"


def config_for(name: str, *, drifting: bool, learner: dict):
    """X9's config with one learner swapped in, and nothing else changed."""
    return load_config(
        "x9_rate_ramp" if drifting else "x9_control",
        overrides={
            "run": {
                "name": name,
                "horizon": HORIZON,
                "eval_every": EVAL_EVERY,
                "seeds": SEEDS,
                "device": DEVICE,
                "dtype": DTYPE,
            },
            "learners": [learner],
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def main(fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    for reference in ("x9_rate_ramp", "x9_control"):
        if not (ROOT / "results" / reference).exists():
            print(f"X16 pairs against {reference}, which has not been run.")
            return 1
    train, test = load_mnist(DATA_ROOT, download=False)

    print("X13's selected setting, carried here unchanged:")
    setting = next(s for s in tuned_settings() if s["name"] == "centralized_ekf_gamma")
    print(f"\nX16: the filter on X9's ramp, {len(SEEDS)} seeds, T={HORIZON}")
    print(f"  eval_every {EVAL_EVERY}, {DTYPE}, paired against x9_rate_ramp/x9_control\n",
          flush=True)

    started = time.time()
    for name, drifting in (("x9_rate_ramp_ekf", True), ("x9_control_ekf", False)):
        note = run_one(config_for(name, drifting=drifting, learner=setting),
                       train, test, fresh)
        print(f"  {name:<22} {note:<30} {(time.time() - started) / 60:.0f} min",
              flush=True)

    print(f"\nX16 complete in {(time.time() - started) / 60:.1f} min")
    print("Read it with: python scripts/report_breaks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv))
