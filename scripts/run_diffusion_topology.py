r"""P5.3 -- topology and the spectral gap: does the covariance-sharing gap widen as
connectivity falls?

    python scripts/run_diffusion_topology.py --lr   # re-tune the baselines, first
    python scripts/run_diffusion_topology.py        # the cells themselves
    python scripts/run_diffusion_topology.py --report-only

`--lr` must run first and the main pass refuses without it (D77).

## The question

`phase5_plan.md` P5.3: whether the covariance-sharing gap widens as connectivity
falls. Mean-only sharing loses more when information has to travel further, so the
two variants should separate here if they separate anywhere.

## Why this is now also a test of D100

X28 measured full sharing as worth nothing on top of a one-hop adapt -- but at
\ac{er} $p=0.3$, where a one-hop neighbourhood already reaches much of a ten-node
network. The mechanism offered was that fresh neighbour evidence, once inside the
adapt step, leaves the neighbours' accumulated uncertainty nothing to add. **On a
ring each agent reaches two neighbours**, so that mechanism is at its weakest, and
this sweep is where D100's null should break if it breaks anywhere. Group B
therefore runs at *every* topology, not only at the extremes.

## The receiver point, natively

X20--X26 ran one-hop at the sender point; D99 adopted the receiver point and D100
confirmed the sharing result there. This is the first sweep to carry it by default,
so its one-hop numbers are **not** comparable cell-for-cell with X25's -- they are
the better variant, by 0.0005 to 0.0091 on the five conditions X27 paired. It is
also the cheaper one: 3 696 scalars per link per step against the sender's 6 604,
which puts one-hop *below* momentum \ac{atc}'s 5 816 for the first time.

## Baselines re-tuned per topology, the filter carried

A denser graph averages over more neighbours per round -- the same noise-reduction
mechanism that let \ac{atc} survive a step size which killed `local_only` (D39) --
so a rate held across topologies would leave this sweep partly measuring "how well
does the ring's rate suit a complete graph". X3 measured that effect as real but
small: six of seven topologies picked the same cell, and only `complete` differed,
because one combine step reaches consensus there and \ac{atc} *is* centralized.

The **filter** carries X20's selection unchanged, confirmed jointly by X23. That is
the X14 discipline: one setting, chosen once, so a shortfall here is attributable to
connectivity rather than confounded with tuning. ⚠ Those settings were selected on
\ac{er} 0.3 at the *sender* point; if the ring behaves oddly, that is the first
thing to suspect.

## A correctness check that costs nothing

`centralized_ekf_gamma`, `centralized_sgd` and `local_only` never read the graph. At
a fixed seed they must produce identical numbers at every topology, and any
difference would mean the graph is leaking into the data path. The report checks it,
as X3's did.
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
from run_atc_plain import PLAIN  # noqa: E402
from run_diffusion_skew import (  # noqa: E402
    BASELINES,
    CENTRALIZED,
    FILTER,
    LEARNING_RATES,
    LR_SEEDS,
    settled,
)
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import paired, per_seed  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: (label, topology, params), ordered sparse to dense so a partial run still spans
#: the axis. The spectral gap is the real x-axis and is read off in analysis.
#:
#: A builder takes only the params it names -- `_ring` and `_complete` take none --
#: so a stale key inherited from the composed config is silently IGNORED rather than
#: rejected. Do not read an accepted `params` as a validated one.
#: ⚠ The sparse point was ER $p=0.15$, and that was wrong at this size. The
#: connectivity threshold is $\ln(n)/n = 0.230$, so 0.15 sits below it: the builder
#: resamples for a connected draw and gives up after 20 attempts, which it did
#: mid-sweep. The deeper fault is that a draw which *does* succeed is conditioned on
#: a rare event and is no longer a sample from ER(10, 0.15) -- the cell would not
#: mean what its label said, so the crash was the lucky outcome.
#:
#: `path` replaces it: connected by construction, and genuinely sparser than the ring
#: in spectral gap, a ring being a path with its ends joined. The axis still spans
#: four points, monotone in gap, in X3's order.
TOPOLOGIES: list[tuple[str, str, dict]] = [
    ("path", "path", {}),
    ("ring", "ring", {}),
    ("er030", "erdos_renyi", {"p": 0.3}),
    ("complete", "complete", {}),
]

#: Mean-only filters and every gradient baseline share a cell.
GROUP_A = ["diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
#: Full sharing gets its own process: X24 measured that variant at 3.43 GiB of 8.
GROUP_B = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 5

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "p53_status.json"

#: The ledger, per link per step, for the bandwidth column of the report.
SCALARS = {"diffusion_ekf_onehop_mean_receiver": 3696, "diffusion_ekf": 2908,
           "diffusion_sgd_atc": 5816, PLAIN["name"]: 2908}


def lr_run_name(label: str, rate: float) -> str:
    return f"p53_lr_{label}_lr{rate:g}".replace(".", "p")


def cell_name(label: str, group: str) -> str:
    return f"p53_{label}_{group}"


def tuned_baselines() -> dict[str, dict]:
    """The three X25 baselines plus `atc_plain`, all tuned per topology here."""
    return {**BASELINES, PLAIN["name"]: {k: v for k, v in PLAIN.items() if k != "name"}}


def selected_rates(label: str) -> dict[str, float] | None:
    """Each baseline's own argmin at this topology, or None when it is unswept."""
    rates: dict[str, float] = {}
    for name in tuned_baselines():
        scored = [(settled(lr_run_name(label, r), name), r) for r in LEARNING_RATES]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, topology: str, params: dict, entries: list[dict],
               seeds: list[int] | None = None):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": args.dataset,
                    "drift": {"schedule": "stationary", "total_degrees": 0.0}},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def entries_for(group: str, rates: dict[str, float]) -> list[dict]:
    if group == "b":
        return [{"name": n, **FILTER, "combine_exponent": 1.0} for n in GROUP_B]
    learners = [{"name": "centralized_ekf_gamma", **CENTRALIZED}]
    learners += [{"name": n, **FILTER} for n in GROUP_A]
    learners += [{"name": n, "lr": rates[n], **o} for n, o in tuned_baselines().items()]
    return learners


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, train, test) -> int:
    """The gradient baselines only; the filter carries its own selection by design."""
    status = load_status()
    total = len(TOPOLOGIES) * len(LEARNING_RATES)
    print(f"P5.3 lr: {len(TOPOLOGIES)} topologies x {len(LEARNING_RATES)} rates at "
          f"{len(LR_SEEDS)} seeds\n", flush=True)
    started, index = time.time(), 0
    for label, topology, params in TOPOLOGIES:
        for rate in LEARNING_RATES:
            index += 1
            name = lr_run_name(label, rate)
            entries = [{"name": n, "lr": rate, **o} for n, o in tuned_baselines().items()]
            note = run_one(config_for(args, name, topology, params, entries, seeds=LR_SEEDS),
                           train, test, args.fresh)
            status[name] = note
            save_status(status)
            print(f"[{index}/{total}] {name:<34} {note:<28} "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'topology':>10}" + "".join(f"{n:>26}" for n in tuned_baselines()))
    for label, _t, _p in TOPOLOGIES:
        rates = selected_rates(label)
        if rates:
            print(f"  {label:>10}" + "".join(f"{rates[n]:>26g}" for n in tuned_baselines()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="re-tune the gradient baselines instead of running the sweep")
    parser.add_argument("--report-only", action="store_true")
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

    missing = [label for label, _t, _p in TOPOLOGIES if selected_rates(label) is None]
    if missing:
        print(f"no selected rates for {missing}: run --lr first.\n"
              "A rate carried across topologies is the mistake D77 exists to record --\n"
              "it put a baseline at chance and inverted a damage ordering.")
        return 1

    cells = [(label, topology, params, group)
             for label, topology, params in TOPOLOGIES for group in ("a", "b")]
    print(f"\nP5.3: {len(cells)} cells at {len(args.seeds)} seeds, T={args.horizon}")
    for label, _t, _p, group in cells:
        print(f"  {cell_name(label, group):<26} group {group}")
    print(flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (label, topology, params, group) in enumerate(cells, start=1):
        name = cell_name(label, group)
        entries = entries_for(group, selected_rates(label))
        note = run_one(config_for(args, name, topology, params, entries), train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<26} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nP5.3 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    """Settled error by topology, the sharing gap, and the graph-blind check."""
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    labels = [label for label, _t, _p in TOPOLOGIES]

    print("  settled error by topology (sparse to dense)")
    every = ["centralized_ekf_gamma", *GROUP_A, *GROUP_B, *tuned_baselines()]
    print(f"    {'learner':<36}" + "".join(f"{label:>12}" for label in labels))
    for learner in every:
        row = f"    {learner:<36}"
        for label in labels:
            values = [per_seed(cell_name(label, g), learner) for g in ("a", "b")]
            values = [v for v in values if v]
            row += f"{mean(values[0]):>12.4f}" if values else f"{'-':>12}"
        print(row)

    print("\n  does covariance sharing pay more as connectivity falls?")
    print("    full minus mean-only, paired by seed; negative = sharing helps")
    for label in labels:
        for full, base in (("diffusion_ekf_full", "diffusion_ekf"),
                           ("diffusion_ekf_onehop_receiver",
                            "diffusion_ekf_onehop_mean_receiver")):
            diff, t, n = paired(per_seed(cell_name(label, "b"), full),
                                per_seed(cell_name(label, "a"), base))
            stem = full.replace("diffusion_ekf_", "")
            print(f"    {label:<10} {stem:<24} {diff:+.4f}  t={t:>6.2f}  n={n}")

    print("\n  one-hop minus local adapt, paired (does one-hop's value scale with degree?)")
    for label in labels:
        diff, t, n = paired(per_seed(cell_name(label, "a"), "diffusion_ekf_onehop_mean_receiver"),
                            per_seed(cell_name(label, "a"), "diffusion_ekf"))
        print(f"    {label:<10} {diff:+.4f}  t={t:>6.2f}  n={n}")

    print("\n  against the cheapest baseline, per link per step")
    print(f"    one-hop receiver {SCALARS['diffusion_ekf_onehop_mean_receiver']} scalars, "
          f"atc_plain {SCALARS[PLAIN['name']]}, momentum ATC {SCALARS['diffusion_sgd_atc']}")
    for label in labels:
        diff, t, n = paired(per_seed(cell_name(label, "a"), "diffusion_ekf_onehop_mean_receiver"),
                            per_seed(cell_name(label, "a"), PLAIN["name"]))
        print(f"    {label:<10} one-hop minus atc_plain {diff:+.4f}  t={t:>6.2f}  n={n}")

    print("\n  graph-blind check: these never read the graph, so every topology must agree")
    for learner in ("centralized_ekf_gamma", "centralized_sgd", "local_only"):
        seeds = [per_seed(cell_name(label, "a"), learner) for label in labels]
        shared = set.intersection(*[set(s) for s in seeds]) if all(seeds) else set()
        worst = max((max(s[k] for s in seeds) - min(s[k] for s in seeds) for k in shared),
                    default=float("nan"))
        print(f"    {learner:<24} max spread across topologies {worst:.1e} over {len(shared)} seeds")


if __name__ == "__main__":
    raise SystemExit(main())
