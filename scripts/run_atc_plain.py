r"""X26 -- `atc_plain` at its own learning rate, which it has never had.

    python scripts/run_atc_plain.py --lr    # select a rate per condition, first
    python scripts/run_atc_plain.py         # the cells themselves

## Why

`diffusion_sgd_atc_plain` is the **bandwidth-matched** baseline: plain \ac{sgd},
nothing mixed but the parameters, so its payload is exactly $\psi$ --- the same as
`diffusion_ekf`'s. It is the arm that answers "is the filter better than diffusion
\ac{sgd} at equal communication", which is the claim M2 was built for and the one a
reviewer presses hardest.

It has never been tuned. X20 built it as `{**PLAIN, "lr": rates[BASELINE]}`, giving
it the **momentum** arm's selected rate, while `local_only` in the same cell got its
own. Plain \ac{sgd} at $\eta=0.01$ takes an effective step of 0.01; momentum 0.9 at
the same $\eta$ takes $\eta/(1-\beta)=0.10$ --- ten times larger (D90). X20 noticed
the symptom, recording that `atc_plain` was "weak enough that the matched margin
flatters us", and reported the $2\psi$ comparison instead. The symptom was real and
the cause was a learning rate, not the method.

X25 then carried no `atc_plain` at all, so the matched-bandwidth comparison **under
skew** does not exist --- and skew is exactly where D89 finds the filter losing to
the $2\psi$ arm.

## Why separate cells rather than re-running the originals

The data stream does not depend on which learners are attached: X25's `lr` cells
reproduce its main cells' baseline numbers to twelve decimals on shared seeds. So
`atc_plain` can run alone, at matching conditions and seeds, and be compared paired
against everything already on disk --- minutes of \ac{sgd} instead of re-running
hours of filters.

## The two condition families

**IID** (X19/X20's): the headline cells, five seeds, where the filter's claim to
beat \ac{atc} at half the bandwidth lives.
**Skewed** (X25's): where the filter loses to the $2\psi$ arm and the matched
question becomes sharp.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

PLAIN = {"name": "diffusion_sgd_atc_plain", "optimizer": "sgd", "momentum": 0.0,
         "mix_optimizer_state": "none"}

#: `sweep_hyperparameters.py`'s grid, so the selection is comparable with every
#: other baseline's. Plain SGD wants the *larger* end of it -- that is the point.
LEARNING_RATES = [0.2, 0.05, 0.01, 0.005, 0.001]
LR_SEEDS = [0, 1]

TOPOLOGY = ("erdos_renyi", {"p": 0.3})
EVAL_EVERY = 5

#: (label, partition, drift, seeds). Seeds match the cells each will be paired
#: against: five for the IID family, three for the skewed one.
IID = {"kind": "iid", "beta": 1.0}
CONDITIONS: list[tuple[str, dict, dict | None, list[int]]] = [
    ("still", IID, None, [0, 1, 2, 3, 4]),
    ("linear_a0p03", IID, {"schedule": "linear", "total_degrees": 45.0}, [0, 1, 2, 3, 4]),
    ("every25_jump15", IID, {"schedule": "recurring", "jump_every": 25,
                             "jump_degrees": 15.0, "jump_seed": 0}, [0, 1, 2, 3, 4]),
    ("skew_b0p1", {"kind": "dirichlet", "beta": 0.1}, None, [0, 1, 2]),
    ("skew_b1", {"kind": "dirichlet", "beta": 1.0}, None, [0, 1, 2]),
    ("skew_b100", {"kind": "dirichlet", "beta": 100.0}, None, [0, 1, 2]),
    ("skew_abrupt", {"kind": "dirichlet", "beta": 0.1},
     {"schedule": "recurring", "jump_every": 25, "jump_degrees": 15.0,
      "jump_seed": 0}, [0, 1, 2]),
    ("skew_smooth", {"kind": "dirichlet", "beta": 0.1},
     {"schedule": "recurring", "jump_every": 1, "jump_degrees": 0.6,
      "jump_seed": 0}, [0, 1, 2]),
]

#: Defaults for the CLI; --seeds is overridden per condition by design.
HORIZON, SEEDS = 1500, [0, 1, 2, 3, 4]

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x26_status.json"


def lr_run_name(condition: str, rate: float) -> str:
    return f"x26_lr_{condition}_lr{rate:g}".replace(".", "p")


def cell_name(condition: str) -> str:
    return f"x26_{condition}"


def settled(run: str, learner: str = PLAIN["name"]) -> float:
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    files = sorted(directory.glob("seed_*.parquet"))
    if not (directory / "_complete").exists() or not files:
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"])
         for f in files], ignore_index=True)
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == "error_rate")
                 & (frame["evalset"] == "current")
                 & (frame["t"] >= int(0.8 * frame["t"].max()))]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def selected_rate(condition: str) -> float | None:
    scored = [(settled(lr_run_name(condition, r)), r) for r in LEARNING_RATES]
    scored = [(v, r) for v, r in scored if v != float("inf")]
    return min(scored)[1] if scored else None


def config_for(args, name: str, partition: dict, drift: dict | None,
               entries: list[dict], seeds: list[int]):
    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    topology, params = TOPOLOGY
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds, "device": args.device, "dtype": args.dtype},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": args.dataset, "partition": dict(partition),
                    "drift": block},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, train, test) -> int:
    status = load_status()
    total = len(CONDITIONS) * len(LEARNING_RATES)
    print(f"X26 lr: {len(CONDITIONS)} conditions x {len(LEARNING_RATES)} rates "
          f"at {len(LR_SEEDS)} seeds\n", flush=True)
    started, index = time.time(), 0
    for condition, partition, drift, _seeds in CONDITIONS:
        for rate in LEARNING_RATES:
            index += 1
            name = lr_run_name(condition, rate)
            note = run_one(config_for(args, name, partition, drift,
                                      [{**PLAIN, "lr": rate}], LR_SEEDS),
                           train, test, args.fresh)
            status[name] = note
            save_status(status)
            print(f"[{index}/{total}] {name:<34} {note:<14} "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'condition':>16}{'selected':>10}{'X20 gave it':>14}")
    for condition, _p, _d, _s in CONDITIONS:
        rate = selected_rate(condition)
        note = "0.01 (momentum arm's)" if not condition.startswith("skew") else "-"
        print(f"  {condition:>16}{rate if rate else '-':>10}   {note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="select a rate per condition instead of running the cells")
    parser.add_argument("--report-only", action="store_true",
                        help="print what is already on disk")
    args = parser.parse_args(argv)

    if args.report_only:
        report()
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)
    if args.lr:
        return tune(args, train, test)

    rates = {c: selected_rate(c) for c, _p, _d, _s in CONDITIONS}
    missing = sorted(c for c, r in rates.items() if r is None)
    if missing:
        print(f"\nNo rate selected for {missing}. Giving this arm a rate chosen for a\n"
              "method with ten times its effective step is the defect it exists to\n"
              "correct (D90).\n  python scripts/run_atc_plain.py --lr\n")
        return 1

    status = load_status()
    print(f"\nX26: {len(CONDITIONS)} cells, `atc_plain` alone at its own rate")
    for condition, _p, _d, seeds in CONDITIONS:
        print(f"  {cell_name(condition):<26} lr {rates[condition]:<8g} "
              f"{len(seeds)} seeds")
    print(flush=True)

    started = time.time()
    for index, (condition, partition, drift, seeds) in enumerate(CONDITIONS, start=1):
        name = cell_name(condition)
        note = run_one(config_for(args, name, partition, drift,
                                  [{**PLAIN, "lr": rates[condition]}], seeds),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(CONDITIONS)}] {name:<26} {note:<14} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nX26 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    r"""Tuned against mis-tuned, and the matched-bandwidth comparison it unlocks."""
    print(f"  {'condition':>16}{'rate':>7}{'tuned':>10}{'X20 mis-tuned':>15}"
          f"{'diffusion_ekf':>15}{'matched gap':>13}")
    paired = {
        "still": ("x20_still_erdos_renyi", "x20_still_erdos_renyi"),
        "linear_a0p03": ("x20_linear_a0p03_erdos_renyi", "x20_linear_a0p03_erdos_renyi"),
        "every25_jump15": ("x20_every25_jump15_erdos_renyi",
                           "x20_every25_jump15_erdos_renyi"),
        "skew_b0p1": (None, "x25_still_b0.1"),
        "skew_b1": (None, "x25_still_b1"),
        "skew_b100": (None, "x25_still_b100"),
        "skew_abrupt": (None, "x25_abrupt"),
        "skew_smooth": (None, "x25_smooth"),
    }
    for condition, _p, _d, _s in CONDITIONS:
        rate = selected_rate(condition)
        tuned = settled(cell_name(condition))
        old_run, filter_run = paired[condition]
        old = settled(old_run, PLAIN["name"]) if old_run else float("inf")
        filt = settled(filter_run, "diffusion_ekf") if filter_run else float("inf")
        gap = filt - tuned if tuned != float("inf") and filt != float("inf") else float("nan")
        print(f"  {condition:>16}{rate if rate else 0:>7g}{tuned:>10.4f}"
              + (f"{old:>15.4f}" if old != float("inf") else f"{'-':>15}")
              + (f"{filt:>15.4f}" if filt != float("inf") else f"{'-':>15}")
              + f"{gap:>13.4f}")
    print("\n  matched gap = diffusion_ekf minus atc_plain, both sending exactly psi.")
    print("  Negative means the filter wins at equal bandwidth.")


if __name__ == "__main__":
    raise SystemExit(main())
