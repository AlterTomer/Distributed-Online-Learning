r"""X17 -- the centralised filter under Dirichlet label skew, still and drifting.

Run this file directly.

    python scripts/run_ekf_skew.py --lr     # re-tune the baselines, first
    python scripts/run_ekf_skew.py          # the cells themselves
    python scripts/run_ekf_skew.py --fresh  # discard and redo

`--lr` must run first. Without it the main pass refuses to start rather than
substituting a rate chosen elsewhere -- which is exactly what the first
version of this experiment did, and what invalidated its cross-method numbers.

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

## The baselines are tuned per condition, and per learner

The first version of this experiment gave all three baselines one optimiser
and one learning rate -- momentum 0.9 at lr 0.05, carried over from a
2.50 deg/step drift condition. On stationary data that is far too large, and
for `local_only` it is catastrophic: the effective step eta/(1-beta) = 0.5 put
it at 0.859 against X6's 0.630, essentially chance. Every filter-versus-
baseline number it produced was measured against a handicapped opponent.

That is the failure `sweep_hyperparameters.py` was written to prevent and the
rule design note D39 exists to enforce, arrived at again from the other side.
So: each baseline keeps the optimiser X6 selected for it -- momentum for the
pooled and diffusing methods, plain \ac{sgd} for `local_only` -- and the
learning rate is re-swept for every condition, each learner taking its own
argmin. A drifting cell and its twin share the selection, because damage is
only meaningful when the two differ in the drift alone.

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

from run_ekf_generalization import run_one, tuned_settings  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

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
#:
#: **Each carries its own optimiser, and that is a tuned choice rather than a
#: default.** `sweep_hyperparameters.py` exists because it is not: momentum at
#: lr 0.05 gives an effective step of eta/(1-beta) = 0.5, at which a single agent
#: lands at chance while the same agent at lr 0.005 reaches 0.188. Averaging over
#: N = 10 cancels enough of that for the diffusion methods to survive it, so a
#: shared setting makes `local_only` look catastrophically worse than it is and
#: reports a tuning artefact as the value of cooperation. X6 settled these; only
#: the learning rate is re-swept per condition below.
BASELINES = {
    "centralized_sgd": {"optimizer": "sgd_momentum", "momentum": 0.9},
    "diffusion_sgd_atc": {"optimizer": "sgd_momentum", "momentum": 0.9},
    "local_only": {"optimizer": "sgd", "momentum": 0.0},
}

#: `sweep_hyperparameters.py`'s grid, unchanged so the selections are comparable
#: with X6's. Spans two decades: 0.2 is where `atc_plain` lives and 0.001 is
#: where a starved `local_only` does.
LEARNING_RATES = [0.2, 0.05, 0.01, 0.005, 0.001]

#: Two seeds to locate the optimum, five to measure at it -- the pilot answers a
#: different question ("roughly where") and two seeds answer it.
LR_SEEDS = [0, 1]

DEVICE = "auto"
DTYPE = "float64"
FRESH = False

#: The dataset every cell in this sweep consumes. A name, not an import,
#: so a second dataset is a one-line change here rather than a new script
#: (IMPLEMENTATION.md section 15).
DATASET = "mnist"

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x17_status.json"


#: (label, beta, drift block). The label is what an lr is selected *for*: a
#: condition, not a cell, since a drifting run and its twin must differ only in
#: the drift and therefore share a learning rate.
CONDITIONS: list[tuple[str, float, dict | None]] = [
    (f"still_beta{beta:g}", beta, None) for beta in BETAS
] + [("abrupt", DRIFT_BETA, DRIFTS["abrupt"]), ("smooth", DRIFT_BETA, DRIFTS["smooth"])]


def lr_run_name(condition: str, rate: float) -> str:
    return f"x17_lr_{condition}_lr{f'{rate:g}'.replace('.', 'p')}"


def baseline_learners(rates: dict[str, float]) -> list[dict]:
    """One entry per baseline at the rate selected for it."""
    return [{"name": name, "lr": rates[name], **opts} for name, opts in BASELINES.items()]


def learners(setting: dict, rates: dict[str, float]) -> list[dict]:
    return [setting] + baseline_learners(rates)


def settled(run: str, learner: str) -> float:
    """One learner's settled error, or +inf when the run is absent."""
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


def selected_rates(condition: str) -> dict[str, float] | None:
    """Each baseline's own argmin over `LEARNING_RATES`, or None if unswept.

    Per learner rather than one shared rate: the methods differ in how many
    samples each update sees, so their optima are not the same number, and X6
    selected 0.01 for the pooled and diffusing methods against 0.05 for
    `local_only`. One rate for all three is what produced the artefact this
    re-tune exists to remove.
    """
    rates: dict[str, float] = {}
    for name in BASELINES:
        scored = [(settled(lr_run_name(condition, rate), name), rate)
                  for rate in LEARNING_RATES]
        scored = [(value, rate) for value, rate in scored if value != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(name: str, beta: float, drift: dict | None, entries: list[dict],
               seeds: list[int] | None = None):
    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": name,
                "horizon": HORIZON,
                "eval_every": EVAL_EVERY,
                "seeds": seeds or SEEDS,
                "device": DEVICE,
                "dtype": DTYPE,
            },
            "env": {"partition": {"kind": "dirichlet", "beta": beta}, "drift": block},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def tune(train, test, fresh: bool) -> int:
    """Sweep the learning rate per condition, baselines only, two seeds.

    Baselines only: the filter is not re-tuned here on purpose (see the module
    docstring), and leaving it out makes each cell cheap enough to sweep five
    rates across five conditions.
    """
    status = load_status()
    total = len(CONDITIONS) * len(LEARNING_RATES)
    print(f"X17 lr re-tune: {len(CONDITIONS)} conditions x {len(LEARNING_RATES)} rates "
          f"at {len(LR_SEEDS)} seeds\n", flush=True)

    started, index = time.time(), 0
    for condition, beta, drift in CONDITIONS:
        for rate in LEARNING_RATES:
            index += 1
            name = lr_run_name(condition, rate)
            entries = [{"name": n, "lr": rate, **o} for n, o in BASELINES.items()]
            note = run_one(config_for(name, beta, drift, entries, seeds=LR_SEEDS),
                           train, test, fresh)
            status[name] = note
            save_status(status)
            elapsed = (time.time() - started) / 60
            print(f"[{index}/{total}] {name:<34} {note:<12} {elapsed:.0f} min",
                  flush=True)

    print(f"\nlr re-tune complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'condition':<16}" + "".join(f"{n.split('_')[0]:>18}" for n in BASELINES))
    for condition, _beta, _drift in CONDITIONS:
        rates = selected_rates(condition)
        if rates is None:
            continue
        print(f"  {condition:<16}" + "".join(f"{rates[n]:>18g}" for n in BASELINES))
    return 0


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(fresh: bool = FRESH, tune_only: bool = False) -> int:
    if not dataset_is_cached(DATASET, DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(DATASET, DATA_ROOT, download=False)
    if tune_only:
        return tune(train, test, fresh)

    print("X13's selected setting, carried here unchanged:")
    setting = next(s for s in tuned_settings() if s["name"] == "centralized_ekf_gamma")

    rates = {c: selected_rates(c) for c, _b, _d in CONDITIONS}
    missing = sorted(c for c, r in rates.items() if r is None)
    if missing:
        print(
            f"\nThe learning-rate sweep has not run for {missing}, so this pass would\n"
            "compare the filter against baselines carrying a rate chosen for a\n"
            "different condition. That is the mistake the first version of this\n"
            "experiment made: one rate across stationary and drifting cells put\n"
            "`local_only` at chance and inflated every advantage measured against it.\n"
            "  python scripts/run_ekf_skew.py --lr\n"
        )
        return 1

    # A drifting run and its twin must differ only in the drift, learning rates
    # included, so each drift condition gets a twin at its own rates. They are
    # separate runs only when the rates actually differ; when the sweep selects
    # the same three, one twin serves both and the second is skipped as cached.
    cells: list[tuple[str, float, dict | None, dict[str, float]]] = [
        (f"x17_still_beta{beta:g}", beta, None, rates[f"still_beta{beta:g}"])
        for beta in BETAS
    ]
    for kind in DRIFTS:
        cells.append((f"x17_{kind}_beta{DRIFT_BETA:g}", DRIFT_BETA, DRIFTS[kind],
                      rates[kind]))
        cells.append((f"x17_control_{kind}_beta{DRIFT_BETA:g}", DRIFT_BETA, None,
                      rates[kind]))

    print(f"\nX17: {len(cells)} cells at {len(SEEDS)} seeds, T={HORIZON}")
    print(f"  {'condition':<16}" + "".join(f"{n.split('_')[0]:>18}" for n in BASELINES))
    for condition, _b, _d in CONDITIONS:
        print(f"  {condition:<16}"
              + "".join(f"{rates[condition][n]:>18g}" for n in BASELINES))
    print(flush=True)

    status = load_status()
    started = time.time()
    ran = 0
    for index, (name, beta, drift, condition_rates) in enumerate(cells, start=1):
        note = run_one(
            config_for(name, beta, drift, learners(setting, condition_rates)),
            train, test, fresh,
        )
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(cells) - index)) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<34} {note:<30} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nX17 complete in {(time.time() - started) / 60:.1f} min")
    print(f"status: {STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv, tune_only="--lr" in sys.argv))
