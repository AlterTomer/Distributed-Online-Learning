r"""X19 -- the diffusion filter's first measurement: what does diffusing a belief cost?

Run this file directly.

    python scripts/run_diffusion_ekf.py --lr     # re-tune the SGD baseline, first
    python scripts/run_diffusion_ekf.py          # the cells themselves
    python scripts/run_diffusion_ekf.py --fresh  # discard and redo

`--lr` must run first; the main pass refuses without it (design note D77).

## The question

Every filter result so far is centralised: one belief over the pooled batch. That
is the reference, not the method --- the project's claim is about a *distributed*
filter, and until now nothing has measured one. Two things are asked at once, and
they are separable because the two diffusion variants differ in exactly one line
of the combine step:

    how much of the centralised filter does diffusion recover?
        centralized_ekf_gamma  minus  diffusion_ekf_full

    what does not shipping covariances cost?
        diffusion_ekf_full     minus  diffusion_ekf

The second is the deployable question --- 68 MB per link per step against 23 kB
--- and the first is what bounds it. Measuring the cheap variant alone would
leave any shortfall ambiguous between "diffusion does not recover the centralised
filter here" and "mean-only sharing was the part that cost us", and no amount of
later tuning separates those.

## Why the complete graph is a condition here and was not for \ac{sgd}

For \ac{sgd}, `complete` is the X0 fixture: \ac{atc} reproduces centralised \ac{sgd}
exactly there, so as an *experiment* it measures nothing. For the filter it is a
genuine condition, because the identity needs a **one-hop** adapt step and these
learners adapt locally. On a complete graph each agent still sees only its own
$1/N$ of the data and then averages; the gap that remains is what the *local
adapt* costs, with graph sparsity removed entirely. Against the \ac{er} cells, the
difference between the two graphs is then what *sparsity* costs on top.

That decomposition is the reason both graphs are run, and it is only available
because `diffusion_ekf_onehop` established (in `tests/test_exactness.py`, at a
residual of 4.4e-16) that the identity does hold when the adapt step reaches
every agent. Without that anchor a shortfall on the complete graph would be
indistinguishable from a bug.

## What is held fixed

The filter runs at X13's selected setting, unchanged, as everywhere since X14 ---
$\gamma=0.9995$, $q=6\times10^{-5}$, $\sigma_0^2=0.01$. Each agent sees $1/N$ of
the information at the same forgetting rate, so if the diffusion beliefs settle
too diffuse, $\sigma_0^2$ is the first thing to suspect and the signature is a
damage dominated by the *fitting* term. Re-tune then, report both, and say so ---
the X15 pattern. Do not re-tune pre-emptively: a setting chosen for the
centralised filter and never adjusted is what makes a shortfall attributable.

The \ac{sgd} baseline has its learning rate re-swept per condition, per D77.

## Memory, which is what actually binds

Each diffusion variant holds $N=10$ covariances at 64.5 MiB. Full sharing needs a
second set live during the mix, because every $\bm P^{\psi}_u$ must survive until
the last $\bm P_{v,t|t}$ is written. A smoke run carrying both variants plus the
centralised filter peaked at **3.3 GiB of 8**. A third diffusion variant in the
same process would not fit, which is why `diffusion_ekf_onehop` stays a test
fixture rather than riding along here.
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

#: Stationary, plus one mild and one harsh drift.
#:
#: **The harsh one cannot be monotone, and the cap is why.** Over T = 1500 the
#: 45-degree cap makes 0.03 deg/step the fastest *linear* drift there is, and X14
#: put the gradient methods' break at 0.038-0.044 -- so no legal monotone rate
#: reaches even the mild end of where methods start failing. A harsh condition has
#: to be delivered as jumps, which are not cap-limited because they revisit
#: angles rather than accumulating. This is the same bind X18 documents.
#:
#: `every25_jump15` is chosen for the harsh end rather than something new: it runs
#: at 0.60 deg/step, and X11 and X17 have both already characterised it, so a
#: surprise here is attributable to the method rather than to an unfamiliar
#: condition.
CONDITIONS: list[tuple[str, dict | None]] = [
    ("still", None),
    ("linear_a0p03", {"schedule": "linear", "total_degrees": 45.0}),
    ("every25_jump15", {"schedule": "recurring", "jump_every": 25,
                        "jump_degrees": 15.0, "jump_seed": 0}),
]

#: Both graphs, for the decomposition the docstring describes: `complete` isolates
#: what the local adapt costs, `erdos_renyi` adds what sparsity costs on top.
#:
#: Carried with their params, not just their names. `x1_stationary` includes
#: `graph: ring`, which has no params, so overriding the topology alone leaves
#: `erdos_renyi` without its p -- and the config validates happily, because the
#: requirement is enforced by the *graph builder* rather than by the schema.
TOPOLOGIES: list[tuple[str, dict]] = [
    ("complete", {}),
    ("erdos_renyi", {"p": 0.3}),
]

HORIZON = 1500
SEEDS = [0, 1, 2, 3, 4]
EVAL_EVERY = 5

#: The two subjects, and the references they are read against. `local_only` is
#: carried because the cooperation gap is a question again: a filter that holds a
#: belief per agent might not need its neighbours as much as SGD does.
DIFFUSION = ["diffusion_ekf_full", "diffusion_ekf"]
BASELINE = {"name": "diffusion_sgd_atc", "optimizer": "sgd_momentum", "momentum": 0.9}
LOCAL = {"name": "local_only", "optimizer": "sgd", "momentum": 0.0}

LEARNING_RATES = [0.2, 0.05, 0.01, 0.005, 0.001]
LR_SEEDS = [0, 1]

DEVICE = "auto"
DTYPE = "float64"
FRESH = False

TOPOLOGY_PARAMS = dict(TOPOLOGIES)
TOPOLOGY_NAMES = [name for name, _p in TOPOLOGIES]

#: The dataset every cell in this sweep consumes. A name, not an import,
#: so a second dataset is a one-line change here rather than a new script
#: (IMPLEMENTATION.md section 15).
DATASET = "mnist"

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x19_status.json"


def condition_name(condition: str, topology: str) -> str:
    return f"{condition}_{topology}"


def lr_run_name(condition: str, topology: str, rate: float) -> str:
    stem = condition_name(condition, topology)
    return f"x19_lr_{stem}_lr{f'{rate:g}'.replace('.', 'p')}"


def settled(run: str, learner: str) -> float:
    import pandas as pd  # noqa: PLC0415

    files = sorted((ROOT / "results" / run).glob("seed_*.parquet"))
    if not files or not (ROOT / "results" / run / "_complete").exists():
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"])
         for f in files], ignore_index=True)
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == "error_rate")
                 & (frame["evalset"] == "current") & (frame["t"] >= int(0.8 * HORIZON))]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def selected_rates(condition: str, topology: str) -> dict[str, float] | None:
    """One rate per SGD learner per cell, each taking its own argmin."""
    rates: dict[str, float] = {}
    for entry in (BASELINE, LOCAL):
        scored = [(settled(lr_run_name(condition, topology, rate), entry["name"]), rate)
                  for rate in LEARNING_RATES]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[entry["name"]] = min(scored)[1]
    return rates


def config_for(name: str, drift: dict | None, topology: str, entries: list[dict],
               seeds: list[int] | None = None):
    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    params = dict(TOPOLOGY_PARAMS[topology])
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": name, "horizon": HORIZON, "eval_every": EVAL_EVERY,
                "seeds": seeds or SEEDS, "device": DEVICE, "dtype": DTYPE,
            },
            "graph": {"topology": topology, "params": params},
            "env": {"dataset": DATASET, "drift": block},
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
    """The SGD baselines only. The filter carries X13's setting by design."""
    status = load_status()
    cells = [(c, t) for c, _d in CONDITIONS for t in TOPOLOGY_NAMES]
    total = len(cells) * len(LEARNING_RATES)
    print(f"X19 lr re-tune: {len(cells)} cells x {len(LEARNING_RATES)} rates "
          f"at {len(LR_SEEDS)} seeds\n", flush=True)
    started, index = time.time(), 0
    drifts = dict(CONDITIONS)
    for condition, topology in cells:
        for rate in LEARNING_RATES:
            index += 1
            name = lr_run_name(condition, topology, rate)
            entries = [{**BASELINE, "lr": rate}, {**LOCAL, "lr": rate}]
            note = run_one(
                config_for(name, drifts[condition], topology, entries, seeds=LR_SEEDS),
                train, test, fresh)
            status[name] = note
            save_status(status)
            print(f"[{index}/{total}] {name:<34} {note:<12} "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nlr re-tune complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'cell':>24}{'atc':>10}{'local_only':>13}")
    for condition, topology in cells:
        rates = selected_rates(condition, topology)
        if rates:
            print(f"  {condition_name(condition, topology):>24}"
                  f"{rates[BASELINE['name']]:>10g}{rates[LOCAL['name']]:>13g}")
    return 0


def main(fresh: bool = FRESH, tune_only: bool = False) -> int:
    if not dataset_is_cached(DATASET, DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(DATASET, DATA_ROOT, download=False)
    if tune_only:
        return tune(train, test, fresh)

    print("X13's selected setting, carried here unchanged:")
    setting = next(s for s in tuned_settings() if s["name"] == "centralized_ekf_gamma")
    filter_fields = {k: v for k, v in setting.items() if k != "name"}

    cells_wanted = [(c, t) for c, _d in CONDITIONS for t in TOPOLOGY_NAMES]
    rates = {(c, t): selected_rates(c, t) for c, t in cells_wanted}
    missing = sorted(condition_name(c, t) for (c, t), r in rates.items() if r is None)
    if missing:
        print(f"\nThe learning-rate sweep has not run for {missing}. Without it the\n"
              "baselines would carry a rate chosen for another condition, which is how\n"
              "X17's first attempt put a baseline at chance and inverted the damage\n"
              "ordering (design note D77).\n"
              "  python scripts/run_diffusion_ekf.py --lr\n")
        return 1

    drifts = dict(CONDITIONS)
    cells: list[tuple[str, dict | None, str, list[dict]]] = []
    for condition, topology in cells_wanted:
        entries = (
            [{"name": "centralized_ekf_gamma", **filter_fields}]
            + [{"name": n, **filter_fields} for n in DIFFUSION]
            + [{**BASELINE, "lr": rates[(condition, topology)][BASELINE["name"]]},
               {**LOCAL, "lr": rates[(condition, topology)][LOCAL["name"]]}]
        )
        stem = condition_name(condition, topology)
        cells.append((f"x19_{stem}", drifts[condition], topology, entries))
        if drifts[condition] is not None:
            # Damage needs a twin differing in the drift alone -- the topology and
            # every learning rate included, so each drifting cell gets its own.
            cells.append((f"x19_control_{stem}", None, topology, entries))

    print(f"\nX19: {len(cells)} cells at {len(SEEDS)} seeds, T={HORIZON}")
    print(f"  {'cell':>28}{'topology':>10}{'atc lr':>9}{'local lr':>10}")
    for name, _d, topology, entries in cells:
        by_name = {e["name"]: e for e in entries}
        print(f"  {name:>28}{topology:>10}{by_name['diffusion_sgd_atc']['lr']:>9g}"
              f"{by_name['local_only']['lr']:>10g}")
    print("\n  memory: two diffusion variants hold 2 x 10 x 64.5 MiB of covariance,\n"
          "  and full sharing needs a second set live during the mix (~3.3 GiB peak)\n",
          flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (name, drift, topology, entries) in enumerate(cells, start=1):
        note = run_one(config_for(name, drift, topology, entries), train, test, fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(cells) - index)) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<30} {note:<28} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nX19 complete in {(time.time() - started) / 60:.1f} min")
    print(f"status: {STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(fresh="--fresh" in sys.argv, tune_only="--lr" in sys.argv))
