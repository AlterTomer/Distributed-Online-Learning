r"""X18 -- sawtooth drift: is a momentum term a liability when direction resets?

Run this file directly.

    python scripts/run_sawtooth.py --lr     # re-tune the baselines, first
    python scripts/run_sawtooth.py          # the cells themselves
    python scripts/run_sawtooth.py --fresh  # discard and redo

`--lr` must run first; the main pass refuses without it (X17, design note D77).

## The question, which is not "does the filter win"

Every condition run so far shows the filter and the gradient baselines moving in
parallel: at a shift they take the same hit -- +0.0594 against +0.0635 -- because
the rise is set by the shift and no causal method can pre-empt one. The filter
wins on the floor it returns to. Asking a third drift shape to break that pattern
would be a fourth attempt at the same question.

This asks a different one, about a mechanism none of the existing schedules
isolates. **Momentum is a directional memory** -- at $\beta=0.9$, roughly ten
steps of velocity. On a monotone stretch that is an asset: the velocity points
where the distribution is going, so the method effectively anticipates. At a
reset it is maximally wrong and must unwind before it helps again. A filter
carries no directional state at all: $\bm Q=q\bm I$ is isotropic and the
covariance says "I am uncertain", never "I was moving that way".

So the prediction is sharp and falsifiable, and it is about the baselines rather
than the filter:

    diffusion_sgd_atc_plain (no momentum) should gain on
    diffusion_sgd_atc (momentum 0.9) as the reset period shortens,

and the crossover should sit near momentum's own horizon of $1/(1-\beta)\approx10$
steps. If it does, the mechanism is demonstrated and it *explains* why a filter
is structurally advantaged here rather than merely showing that it is. If the two
track each other at every period, the directional-memory story is wrong and the
filter's advantage here is the same one it has everywhere.

## Why the period axis moves two things at once, and why that is unavoidable

A sawtooth of amplitude $A$ and period $P$ ramps at $A/P$ and resets by $A$. With
$A$ pinned at the $45^{\circ}$ cap, shortening $P$ shortens the monotone stretch
*and* raises the rate together. Measured over the 1500-step horizon:

    period   ramp deg/step   resets   mean rate   total travel
       300        0.15          4        0.30          404 deg
       100        0.45         14        0.89         1292
        50        0.90         29        1.76         2602
        30        1.50         49        2.90         4306
        20        2.25         74        4.28         6370

Holding the ramp rate fixed instead would need $A=\text{rate}\times P$, which
passes the cap by $P=300$ at any interesting rate. The cap binds, so the axes
cannot be separated -- the same bind design note D74 records for $J$ and $t'$.

**Three consequences, all of which limit what this can conclude.**

*There is no fast monotone control, and cannot be.* The fastest legal monotone
drift is the cap spread over the horizon, $0.03^{\circ}$/step, which does no
damage to anything. So "sawtooth against linear at a matched rate" is unavailable
at every rate that does any damage at all. A $P=1500$ cell would be monotone --
one ramp, no in-run reset -- but at $0.03^{\circ}$/step it is also barely drift,
so a tie there is equally explained by "no reset" and by "nothing happened"; it
was dropped for that reason rather than to save time. Being able to sustain
motion a cap forbids is what a sawtooth is *for*, and it is the same reason the
comparison it would want does not exist.

*So the estimator is the trend, not a difference.* The readout is the within-cell
gap `atc_plain - atc` measured at each period and read *across* periods. The
prediction is that the gap moves monotonically toward `atc_plain` as $P$ falls
past momentum's ten-step horizon. A monotone trend over five periods is the
evidence; no single cell carries it, which is why the grid is spent on resolving
the crossover rather than on anchoring the ends.

*And $P=20$ is likely off the end.* Its mean rate of $4.28^{\circ}$/step is above
the $\approx3^{\circ}$/step ceiling X14 established, even though its *ramp* is
only 2.25 -- the resets supply the rest of the travel. If every method collapses
to chance there, that cell measures the ceiling rather than momentum. $P=30$ is
the fastest cell that should still read: mean rate 2.90, just inside the ceiling,
and a 30-step stretch is three times momentum's horizon. It is where the
crossover is expected, and it is the cell the conclusion will rest on.

## What is held fixed

The filter runs at the setting X13 selected and X14/X15 validated, unchanged, as
everywhere else. Baselines keep the optimiser X6 tuned for each and have their
learning rate re-swept per condition, each taking its own argmin -- the procedure
D77 exists to enforce. Partition is \ac{iid}: X17 established that skew barely
reaches a pooled learner, and mixing it in here would confound the axis under
test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ekf_generalization import run_one, tuned_settings  # noqa: E402

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: The cap. A sawtooth reaches A(P-1)/P, so this stays inside 45 degrees.
AMPLITUDE = 45.0

#: Spent on resolving the crossover rather than anchoring the ends: 30 sits just
#: inside the rate ceiling at mean 2.90 deg/step and is where momentum's ten-step
#: horizon predicts the sign to flip. 20 is past the ceiling and is kept to see
#: the collapse, not because it is expected to read (see the module docstring).
PERIODS = [300, 100, 50, 30, 20]

HORIZON = 1500
SEEDS = [0, 1, 2, 3, 4]
EVAL_EVERY = 5

#: `atc_plain` is the point of the experiment rather than a payload-matched
#: also-ran: it is `atc` with the momentum removed, so the pair isolates the one
#: term under test. X6's tuned optimisers are kept; only the rate is re-swept.
BASELINES = {
    "centralized_sgd": {"optimizer": "sgd_momentum", "momentum": 0.9},
    "diffusion_sgd_atc": {"optimizer": "sgd_momentum", "momentum": 0.9},
    "diffusion_sgd_atc_plain": {"optimizer": "sgd", "momentum": 0.0},
}

LEARNING_RATES = [0.2, 0.05, 0.01, 0.005, 0.001]
LR_SEEDS = [0, 1]

DEVICE = "auto"
DTYPE = "float64"
FRESH = False

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x18_status.json"


def drift_block(period: int) -> dict:
    return {"schedule": "sawtooth", "amplitude_degrees": AMPLITUDE, "period": period}


def condition_name(period: int) -> str:
    return f"p{period}"


def lr_run_name(period: int, rate: float) -> str:
    return f"x18_lr_{condition_name(period)}_lr{f'{rate:g}'.replace('.', 'p')}"


def rate_signature(rates: dict[str, float]) -> str:
    """A twin is identified by the rates it carries, not by which cell wanted it.

    Damage needs the twin to differ from its drifting run in the drift alone,
    learning rates included -- but several periods may select the same three
    rates, and then one twin serves them all. Naming by signature makes that
    sharing automatic instead of running the same stationary cell five times.
    """
    return "_".join(f"{f'{rates[n]:g}'.replace('.', 'p')}" for n in BASELINES)


def settled(run: str, learner: str) -> float:
    import pandas as pd  # noqa: PLC0415

    files = sorted((ROOT / "results" / run).glob("seed_*.parquet"))
    if not files:
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"])
         for f in files], ignore_index=True)
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == "error_rate")
                 & (frame["evalset"] == "current") & (frame["t"] >= int(0.8 * HORIZON))]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def selected_rates(period: int) -> dict[str, float] | None:
    rates: dict[str, float] = {}
    for name in BASELINES:
        scored = [(settled(lr_run_name(period, rate), name), rate)
                  for rate in LEARNING_RATES]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(name: str, drift: dict | None, entries: list[dict],
               seeds: list[int] | None = None):
    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": name, "horizon": HORIZON, "eval_every": EVAL_EVERY,
                "seeds": seeds or SEEDS, "device": DEVICE, "dtype": DTYPE,
            },
            "env": {"drift": block},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(train, test, fresh: bool) -> int:
    status = load_status()
    total = len(PERIODS) * len(LEARNING_RATES)
    print(f"X18 lr re-tune: {len(PERIODS)} periods x {len(LEARNING_RATES)} rates "
          f"at {len(LR_SEEDS)} seeds\n", flush=True)
    started, index = time.time(), 0
    for period in PERIODS:
        for rate in LEARNING_RATES:
            index += 1
            name = lr_run_name(period, rate)
            entries = [{"name": n, "lr": rate, **o} for n, o in BASELINES.items()]
            note = run_one(config_for(name, drift_block(period), entries, seeds=LR_SEEDS),
                           train, test, fresh)
            status[name] = note
            save_status(status)
            print(f"[{index}/{total}] {name:<28} {note:<12} "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nlr re-tune complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'period':>8}" + "".join(f"{n.replace('diffusion_sgd_', ''):>16}"
                                       for n in BASELINES))
    for period in PERIODS:
        rates = selected_rates(period)
        if rates:
            print(f"  {period:>8}" + "".join(f"{rates[n]:>16g}" for n in BASELINES))
    return 0


def main(fresh: bool = FRESH, tune_only: bool = False) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)
    if tune_only:
        return tune(train, test, fresh)

    print("X13's selected setting, carried here unchanged:")
    setting = next(s for s in tuned_settings() if s["name"] == "centralized_ekf_gamma")

    rates = {p: selected_rates(p) for p in PERIODS}
    missing = sorted(p for p, r in rates.items() if r is None)
    if missing:
        print(f"\nThe learning-rate sweep has not run for periods {missing}. Without it\n"
              "the baselines would carry a rate chosen for another condition, which is\n"
              "how X17's first attempt put a baseline at chance and inverted the damage\n"
              "ordering (design note D77).\n"
              "  python scripts/run_sawtooth.py --lr\n")
        return 1

    cells: list[tuple[str, dict | None, list[dict]]] = []
    seen_twins: set[str] = set()
    for period in PERIODS:
        entries = [setting] + [{"name": n, "lr": rates[period][n], **o}
                               for n, o in BASELINES.items()]
        cells.append((f"x18_saw_p{period}", drift_block(period), entries))
        signature = rate_signature(rates[period])
        if signature not in seen_twins:
            seen_twins.add(signature)
            cells.append((f"x18_control_{signature}", None, entries))

    print(f"\nX18: {len(cells)} cells at {len(SEEDS)} seeds, T={HORIZON}, "
          f"amplitude {AMPLITUDE:g} deg")
    print(f"  {'period':>8}{'ramp deg/step':>15}{'resets':>9}"
          + "".join(f"{n.replace('diffusion_sgd_', ''):>16}" for n in BASELINES))
    for period in PERIODS:
        note = "  (= linear drift)" if period >= HORIZON else ""
        print(f"  {period:>8}{AMPLITUDE / period:>15.2f}{HORIZON // period - 1:>9}"
              + "".join(f"{rates[period][n]:>16g}" for n in BASELINES) + note)
    print(f"  {len(seen_twins)} distinct stationary twin(s) after sharing by rate\n",
          flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (name, drift, entries) in enumerate(cells, start=1):
        note = run_one(config_for(name, drift, entries), train, test, fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(cells) - index)) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<34} {note:<28} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nX18 complete in {(time.time() - started) / 60:.1f} min")
    print(f"status: {STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv, tune_only="--lr" in sys.argv))
