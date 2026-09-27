r"""M9 -- topology on Mackey--Glass: path, ring and complete beside M6's ER 0.3.

    python scripts/run_m9_topology.py --lr --device cuda   # re-tune the diffusion baselines, first
    python scripts/run_m9_topology.py --device cuda        # the cells
    python scripts/run_m9_topology.py --report-only
    python scripts/run_m9_topology.py --lr --smoke         # then --smoke: the path, not the numbers

`--lr` must run first and the main pass refuses without it (D77).

`docs/mackey_glass_plan.md` M9, the P5.3 analogue ([[D103]]). On MNIST both of P5.3's
hypotheses were refuted: the covariance-sharing gap did not widen as connectivity fell,
and one-hop's value over local adapt was non-monotone in degree -- negative on the ring
and at ER 0.3, null on the path, nominally *positive* on the complete graph. M9 asks the
same questions of a second task, a second architecture and a regression likelihood.

## The axis

Path, ring, ER $p=0.3$, complete: P5.3's four points, sparse to dense. A sparse ER was
the plan's first choice and is not used, for P5.3's reason -- at $N=10$ the
connectivity threshold is $\ln N/N=0.230$, and a draw conditioned on connectivity below
it is not a sample from the family its label names.

**ER 0.3 is not re-run: it is M6's `m6_stationary_a/b`.** A fresh ER cell would be a
byte-for-byte duplicate of it (D101). The three new topologies are therefore built
**from M6's recorded configs** -- `results/m6_stationary_{a,b}/config.yaml`, fully
resolved -- with the graph replaced and nothing else touched, so law, sensor, $\boldsymbol R$
scale, horizon, cadence, dtype and seeds match by construction.

## What is carried and what is re-tuned

The filters carry M6's entries verbatim -- M4's centralised selection, M5's diffusion
selection -- the X14 discipline P5.3 followed: one setting, chosen once, so a shortfall
is attributable to connectivity. ⚠ Those were selected on ER 0.3; if the path behaves
oddly, that is the first thing to suspect.

Of the gradient baselines, only the three that **read the graph** are re-tuned per
topology, on M3's grids at M3's seeds (0--1): `diffusion_sgd_atc`, `atc_plain` and
`diffusion_atc_adamw`. The other four never read the graph, so their M3 curves *are*
their curves at every topology; re-tuning them would re-run M3. They carry M6's entries.
The $\gamma$ reference arms stay out, as in M8: the question is the graph, not $\gamma$.

## Two gates

1. **Graph-blind:** `centralized_ekf_walk`, `centralized_sgd`, `local_only`,
   `centralized_adamw` and `local_adamw` must match per seed across the three new cells
   *and* `m6_stationary_a`, within $10^{-9}$. It checks that the graph does not leak
   into the data path, and it is also what licenses reading M6's cell as this sweep's
   ER column. The confirmatory tables are withheld if it fails.
2. **Rates off the edge:** `--lr` flags a selection on either end of the grid.

## Named before the run (confirmatory under D118)

Three families, each over the three **new** topologies. ER 0.3 is printed beside them
for reference, but its numbers were read in [[D106]], so it is in no family.

* **Q1. Does one-hop beat local adapt at every topology?** One-hop minus local adapt,
  per seed, Holm across three, **predicted negative** everywhere. M6 measured it at
  $-0.0035$ on ER 0.3. The complete graph is where the prediction is at risk -- MNIST
  went $+0.0031$ (ns) there.
* **Q2. Is full sharing worth nothing at every topology?** Full minus mean-only, both
  adapt scopes, per seed: **TOST at $\pm$`SHARING_MARGIN`**, Holm across six. M6's six
  such pairs all fell within $\pm0.0004$. On the complete graph the two one-hop variants
  coincide by construction, so that row is exact rather than measured.
* **Q3. Does one-hop beat the strongest diffusion baseline at every topology?** One-hop
  minus ATC AdamW, per seed, Holm across three, **predicted negative**.

Exploratory: each diffusion learner's distance to its centralised counterpart against the
mixing gap (does decentralisation cost more on a path, and for which family most?);
one-hop against `atc_plain`; calibration by topology; each cell's wall-clock.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, ADAMW_RATES, SGD_FAMILY, SGD_RATES, settled  # noqa: E402
from run_m8_tau_heterogeneity import ALPHA, STAT_HEADER, _by_seed, _stat_columns  # noqa: E402

from dekf_bench.env.graph import build_graph  # noqa: E402
from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: The new topologies, sparse to dense, so a partial run still spans the axis.
TOPOLOGIES: list[str] = ["path", "ring", "complete"]
#: M6's cells, read as the ER 0.3 column and as the source every new cell is built from.
SOURCE = {"a": "m6_stationary_a", "b": "m6_stationary_b"}
REFERENCE_LABEL = "er030"

#: The baselines that read the graph, and so the only ones re-tuned per topology.
GRAPH_READING = ["diffusion_sgd_atc", "diffusion_sgd_atc_plain", "diffusion_atc_adamw"]
#: M6's gamma = 0.9995 reference arms, left out here as in M8.
DROPPED = {"centralized_ekf_gamma", "diffusion_ekf_onehop_mean_receiver_gamma"}
GRAPH_BLIND = ["centralized_ekf_walk", "centralized_sgd", "local_only",
               "centralized_adamw", "local_adamw"]
BLIND_TOLERANCE = 1e-9

ONEHOP, LOCAL = "diffusion_ekf_onehop_mean_receiver", "diffusion_ekf"
ONEHOP_FULL, LOCAL_FULL = "diffusion_ekf_onehop_receiver", "diffusion_ekf_full"
ATC_ADAMW = "diffusion_atc_adamw"
#: Q2's equivalence margin: a quarter of the one-hop effect M6 resolved (0.0035), and
#: 2.5x the largest full-minus-mean pair M6 measured (0.0004).
SHARING_MARGIN = 1e-3

LR_SEEDS = [0, 1]
SMOKE_HORIZON, SMOKE_EVAL_EVERY = 60, 20
STATUS = ROOT / "results" / "m9_status.json"


def lr_run_name(topology: str, index: int, suffix: str = "") -> str:
    return f"m9_lr_{topology}_r{index}{suffix}"


def cell_name(topology: str, group: str, suffix: str = "") -> str:
    if topology == REFERENCE_LABEL:
        return SOURCE[group]
    return f"m9_{topology}_{group}{suffix}"


def recorded(group: str) -> dict:
    return yaml.safe_load((ROOT / "results" / SOURCE[group] / "config.yaml")
                          .read_text(encoding="utf-8"))


def rate_of(learner: str, index: int) -> float:
    return (ADAMW_RATES if learner in ADAMW_FAMILY else SGD_RATES)[index]


def config_for(group: str, name: str, topology: str, entries: list[dict], args, suffix: str,
               seeds: list[int] | None = None):
    """M6's recorded config with the graph, the learners and the name replaced."""
    source = recorded(group)
    run = {"name": name, "seeds": seeds or source["run"]["seeds"], "device": args.device}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "eval_every": SMOKE_EVAL_EVERY, "seeds": [0]})
    config = load_config(ROOT / "results" / SOURCE[group] / "config.yaml",
                         overrides={"run": run, "graph": {"topology": topology},
                                    "learners": entries})
    # Overrides deep-merge, so `params: {}` would keep ER's `p: 0.3` under a path. The
    # builders ignore keys they do not name, but the recorded config would then carry a
    # parameter the graph never had. Cleared on the resolved config instead.
    config = replace(config, graph=replace(config.graph, params={}))
    if config.graph.params:
        raise SystemExit(f"{name}: the ER params survived the override: {config.graph.params}")
    return config


def lr_entries(index: int) -> list[dict]:
    families = {**SGD_FAMILY, **ADAMW_FAMILY}
    return [{"name": n, "lr": rate_of(n, index), **families[n]} for n in GRAPH_READING]


def selected_rates(topology: str, suffix: str = "") -> dict[str, dict] | None:
    """Each graph-reading baseline's argmin at this topology, or None when unswept."""
    chosen = {}
    for learner in GRAPH_READING:
        scored = [(settled(lr_run_name(topology, k, suffix), learner), k)
                  for k in range(len(SGD_RATES))]
        finite = [(v, k) for v, k in scored if v != float("inf")]
        if not finite:
            return None
        value, index = min(finite)
        chosen[learner] = {"lr": rate_of(learner, index), "settled_rmse": value,
                           "edge": index in (0, len(SGD_RATES) - 1)}
    return chosen


def entries_for(group: str, rates: dict[str, dict]) -> list[dict]:
    """M6's recorded entries, gamma arms dropped, graph-reading rates replaced."""
    out = []
    for entry in recorded(group)["learners"]:
        if entry["name"] in DROPPED:
            continue
        entry = dict(entry)
        if entry["name"] in GRAPH_READING:
            entry["lr"] = rates[entry["name"]]["lr"]
        out.append(entry)
    return out


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tune(args, suffix: str) -> int:
    status = load_status()
    cells = [(t, k) for t in TOPOLOGIES for k in range(len(SGD_RATES))]
    print(f"M9 lr{' SMOKE' if suffix else ''}: {len(cells)} cells, "
          f"{', '.join(GRAPH_READING)}\n", flush=True)
    started = time.time()
    for index, (topology, k) in enumerate(cells, start=1):
        name = lr_run_name(topology, k, suffix)
        note = run_one(config_for("a", name, topology, lr_entries(k), args, suffix,
                                  seeds=LR_SEEDS), None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<22} sgd {SGD_RATES[k]:<7g} adamw "
              f"{ADAMW_RATES[k]:<6g} {note:<24} {(time.time() - started) / 60:.0f} min",
              flush=True)
    print()
    for topology in TOPOLOGIES:
        rates = selected_rates(topology, suffix)
        if not rates:
            continue
        print(f"  {topology:<9}" + "  ".join(
            f"{n} {p['lr']:g}{' <- EDGE' if p['edge'] else ''}" for n, p in rates.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2, 3, 4])
    parser.add_argument("--lr", action="store_true", help="re-tune the graph-reading baselines")
    parser.add_argument("--smoke", action="store_true", help="60 rounds, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0
    missing = [c for c in SOURCE.values() if not (ROOT / "results" / c / "_complete").exists()]
    if missing:
        print(f"  REFUSED: M9 is built from M6's recorded cells, and {missing} is not finished.")
        return 1
    if args.lr:
        return tune(args, suffix)

    unswept = [t for t in TOPOLOGIES if selected_rates(t, suffix) is None]
    if unswept:
        print(f"  REFUSED: no selected rates for {unswept}. Run --lr first (D77).")
        return 1

    cells = [(t, g) for t in TOPOLOGIES for g in ("a", "b")]
    print(f"M9{' SMOKE' if suffix else ''}: {len(cells)} cells, device {args.device}; "
          f"ER 0.3 is {SOURCE['a']}/{SOURCE['b']}\n", flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (topology, group) in enumerate(cells, start=1):
        name = cell_name(topology, group, suffix)
        learners = entries_for(group, selected_rates(topology, suffix))
        cell_started = time.time()
        note = run_one(config_for(group, name, topology, learners, args, suffix),
                       None, None, args.fresh)
        status[name] = note
        if note != "cached":
            ran += 1
            status[f"{name}_minutes"] = round((time.time() - cell_started) / 60, 1)
        save_status(status)
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<22} {len(learners):>2} learners  {note:<14}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM9{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _mixing_gap(topology: str) -> str:
    if topology == REFERENCE_LABEL:
        return "per draw"
    return f"{build_graph(topology, 10).mixing_gap:.3f}"


def _family(rows: list[tuple[str, dict, dict]], margin: float | None = None) -> None:
    """One Holm family; `margin` makes it a TOST family (equivalence), else a paired t."""
    print(f"\n    {'':<30}{STAT_HEADER}" + ("   equivalent" if margin else ""))
    results = [compare(left, right) for _label, left, right in rows]
    pvalues = [(r.tost_p(margin) if margin else r.p) if r.n >= 2 else float("nan")
               for r in results]
    for (label, _l, _r), result, p_adj in zip(rows, results, holm(pvalues), strict=True):
        if result.n < 2:
            print(f"    {label:<30}{'-':>9}")
            continue
        verdict = ("   yes" if p_adj < ALPHA else "   not shown") if margin else ""
        print(f"    {label:<30}{_stat_columns(result, p_adj)}{verdict}")


def report(suffix: str = "") -> None:
    columns = [*TOPOLOGIES] if suffix else ["path", "ring", REFERENCE_LABEL, "complete"]
    if suffix:
        print("  SMOKE: 60 rounds at one seed. ER 0.3 (M6's cell) is not shown: it ran")
        print("  the full horizon. The numbers mean nothing; the path is what is checked.\n")
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731

    print("  settled RMSE on the held-out current set, sparse to dense\n")
    print(f"    {'mixing gap':<40}" + "".join(f"{_mixing_gap(c):>12}" for c in columns))
    every = [*GRAPH_BLIND[:1], LOCAL, ONEHOP, LOCAL_FULL, ONEHOP_FULL,
             *SGD_FAMILY, *ADAMW_FAMILY]
    for learner in every:
        group = "b" if learner in (LOCAL_FULL, ONEHOP_FULL) else "a"
        row = f"    {learner:<40}"
        for topology in columns:
            value = mean(_by_seed(cell_name(topology, group, suffix), learner))
            row += f"{value:>12.4f}" if value == value else f"{'-':>12}"
        print(row)

    print("\n  gate 1, graph-blind: these never read the graph, so every topology must")
    print(f"  agree per seed within {BLIND_TOLERANCE:g}")
    blind_ok = True
    for learner in GRAPH_BLIND:
        seeds = [_by_seed(cell_name(t, "a", suffix), learner) for t in columns]
        shared = set.intersection(*[set(s) for s in seeds]) if all(seeds) else set()
        worst = max((max(s[k] for s in seeds) - min(s[k] for s in seeds) for k in shared),
                    default=float("nan"))
        ok = bool(shared) and worst <= BLIND_TOLERANCE
        blind_ok &= ok
        print(f"    {learner:<24} max spread {worst:.1e} over {len(shared)} seed(s)  "
              f"{'ok' if ok else 'FAILS'}")
    if suffix:
        return
    if not blind_ok:
        print("\n  CONFIRMATORY TABLES WITHHELD: the graph reaches the data path, or a")
        print("  new cell does not reproduce M6's. Nothing across topologies is paired.")
        return

    def rows(left: str, right: str, left_group: str = "a", right_group: str = "a"):
        return [(t, _by_seed(cell_name(t, left_group), left),
                 _by_seed(cell_name(t, right_group), right)) for t in TOPOLOGIES]

    print("\n  CONFIRMATORY (D128). Holm within each family, over the three new topologies;")
    print(f"  a verdict needs p_holm < {ALPHA}. ER 0.3 is reported in D106 and below.")
    print("\n  Q1. one-hop minus local adapt; negative = one-hop better. Predicted negative.")
    _family(rows(ONEHOP, LOCAL))
    print(f"\n  Q2. full minus mean-only; TOST at +/-{SHARING_MARGIN:g}, p_holm is the equivalence p.")
    print("  Predicted equivalent. On the complete graph the one-hop pair is exact.")
    q2 = [(f"{t} {scope}", left, right)
          for scope, full, base in (("local", LOCAL_FULL, LOCAL), ("one-hop", ONEHOP_FULL, ONEHOP))
          for t, left, right in rows(full, base, "b", "a")]
    _family(q2, margin=SHARING_MARGIN)
    print("\n  Q3. one-hop minus ATC AdamW; negative = the filter better. Predicted negative.")
    _family(rows(ONEHOP, ATC_ADAMW))

    print("\n  reference, ER 0.3 (M6, read in D106; in no family)")
    for label, left, right in (("one-hop - local", ONEHOP, LOCAL),
                               ("one-hop - ATC AdamW", ONEHOP, ATC_ADAMW)):
        got = compare(_by_seed(SOURCE["a"], left), _by_seed(SOURCE["a"], right))
        print(f"    {label:<30}{got.mean:>+9.4f}  t={got.t:>+6.2f}  n={got.n}")

    print("\n  EXPLORATORY. Distance to the centralised counterpart, per seed, by topology;")
    print("  positive = decentralising costs. Does the path cost more, and whom most?")
    pairs = [(LOCAL, "centralized_ekf_walk"), (ONEHOP, "centralized_ekf_walk"),
             ("diffusion_sgd_atc", "centralized_sgd"), (ATC_ADAMW, "centralized_adamw")]
    print(f"    {'':<46}" + "".join(f"{c:>12}" for c in columns))
    for learner, centre in pairs:
        row = f"    {learner + ' - ' + centre:<46}"
        for topology in columns:
            got = compare(_by_seed(cell_name(topology, "a"), learner),
                          _by_seed(cell_name(topology, "a"), centre))
            row += f"{got.mean:>+12.4f}" if got.n else f"{'-':>12}"
        print(row)
    row = f"    {'one-hop - atc_plain':<46}"
    for topology in columns:
        got = compare(_by_seed(cell_name(topology, "a"), ONEHOP),
                      _by_seed(cell_name(topology, "a"), "diffusion_sgd_atc_plain"))
        row += f"{got.mean:>+12.4f}" if got.n else f"{'-':>12}"
    print(row)

    status = load_status()
    print("\n  wall-clock, minutes (cells run this session; M6's from D106)")
    for topology in TOPOLOGIES:
        print(f"    {topology:<9}" + "  ".join(
            f"{g}: {status.get(cell_name(topology, g) + '_minutes', '-')}" for g in ("a", "b")))


if __name__ == "__main__":
    raise SystemExit(main())
