r"""M13 -- data rate on Mackey--Glass: blocks per round x label availability.

    python scripts/run_m13_data_rate.py --lr --device cuda   # re-tune the baselines per cell, first
    python scripts/run_m13_data_rate.py --device cuda        # the cells
    python scripts/run_m13_data_rate.py --report-only
    python scripts/run_m13_data_rate.py --lr --smoke         # then --smoke: the path, not the numbers

`--lr` must run first and the main pass refuses without it (D77).

`docs/mackey_glass_plan.md` M13, the P5.4 analogue ([[D122]]). At $\pi_{\text{lab}}<1$ an
agent's sensor drops a round's block: it predicts, takes part in the combine, and its
belief keeps widening under the time update until the next observed block.

## The grid (decided with the user 2026-09-27)

$n_b\in\{1,2,4\}\times\pi_{\text{lab}}\in\{1,0.25\}$: the two ends of P5.4's $\pi$ axis at
each of its three data rates. P5.4's three named questions all read the change from
$\pi=1$ to 0.25 at each $n$, so they carry over whole; only its exploratory $\pi=0.5$
column is dropped. ~25 GPU-h, against the plan's ~8, which predates every diffusion
variant in every comparison.

**$n_b=1$, $\pi=1$ is M6's `m6_stationary_a/b`**, not a re-run. Every other cell is built
from M6's recorded config with `env.series.n_blocks`, `env.label_availability`, the
learners and the name replaced -- $T=1500$, as M6 and as M10 argue (Mackey--Glass has no
data budget, and a short horizon reads a transient). The dropout mask is drawn from its
own seed stream, so at a given $\pi$ the *same rounds* drop at every $n_b$.

## Tuned and carried

Filters carry M6's entries (M4, M5), tuned at $n_b=1$, $\pi=1$ -- a caveat for the sparse
corner, as P5.4's was. The seven baselines are re-tuned per cell, at M3's seeds, on M3's
grids **extended one step down**: the score sums over every observed block, so at
$n_b=4$ the SGD family's optimum should fall about fourfold, which would put ATC (1e-5 at
$n_b=1$) on the old floor. The sparse corner pushes the other way (an idle agent
contributes its unchanged $\bm\theta$, so ATC's effective step shrinks by $\pi$, X4), and
M3's grid already reaches 3e-4 above. An argmin on either edge is flagged.

## A pairing gate

Each $\pi=0.25$ cell must differ from its $\pi=1$ partner in `env.label_availability`,
the baselines' rates and the name alone (at $n_b=1$ the partner is M6's, which also
carries the $\gamma$ arms). Checked on the recorded configs; the confirmatory tables
are withheld for any $n_b$ that fails.

## Named before the run (confirmatory under D118) -- P5.4's, verbatim

1. `one-hop - local adapt`, the change from $\pi=1$ to 0.25, per seed, at each $n_b$;
   Holm across three. **Predicted negative or null**: one-hop assimilates about
   $|\mathcal M_v|\approx3.9$ times local adapt's data at any $\pi$, and at 0.25 it
   still updates on $1-0.75^{3.9}\approx67\%$ of rounds against local adapt's 25%.
2. Each mean-only filter's gap to the centralised filter, the same change, at each
   $n_b$; Holm across six. No direction.
3. `diffusion_ekf - diffusion_sgd_atc`, the same change, at each $n_b$; Holm across
   three. No direction.

Exploratory: full sharing, the AdamW family, one-hop against ATC AdamW, and the
filters' `variance_ratio` by cell -- whether a belief left widening between blocks stays
calibrated.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, ADAMW_RATES, SGD_FAMILY, SGD_RATES, settled  # noqa: E402
from run_m8_tau_heterogeneity import ALPHA, STAT_HEADER, _by_seed, _stat_columns  # noqa: E402

from dekf_bench.metrics.paired import differences, holm, summarise  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

BLOCKS = [1, 2, 4]
DENSE, SPARSE = 1.0, 0.25
SOURCE = {"a": "m6_stationary_a", "b": "m6_stationary_b"}
GROUPS = ["a", "b"]

#: M3's grids, one step further down each (see the docstring); paired by index.
TUNE_SGD = [1e-6, *SGD_RATES]
TUNE_ADAMW = [3e-4, *ADAMW_RATES]
BASELINES = {**SGD_FAMILY, **ADAMW_FAMILY}
DROPPED = {"centralized_ekf_gamma", "diffusion_ekf_onehop_mean_receiver_gamma"}

CENTRAL = "centralized_ekf_walk"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
ATC, ATC_ADAMW = "diffusion_sgd_atc", "diffusion_atc_adamw"
#: What a sparse cell may differ from its dense partner in.
#: `run.device` too: M6 recorded `cuda`, and the n_b = 1 partner is M6's cell, so a launch
#: under `auto` on the same GPU must not fail the gate over a label.
ALLOWED = {"env.label_availability", "learners", "run.name", "run.device"}

LR_SEEDS = [0, 1]
SMOKE_HORIZON, SMOKE_EVAL_EVERY = 20, 5
STATUS = ROOT / "results" / "m13_status.json"


def label(n_blocks: int, availability: float) -> str:
    return f"n{n_blocks}_p{int(round(availability * 100))}"


def is_reused(n_blocks: int, availability: float) -> bool:
    return n_blocks == 1 and availability == DENSE


def cells() -> list[tuple[int, float]]:
    """Dense then sparse at each n_b; the reused M6 cell included, for the report."""
    return [(n, pi) for n in BLOCKS for pi in (DENSE, SPARSE)]


def new_cells() -> list[tuple[int, float]]:
    return [c for c in cells() if not is_reused(*c)]


def cell_name(n_blocks: int, availability: float, group: str, suffix: str = "") -> str:
    if is_reused(n_blocks, availability) and not suffix:
        return SOURCE[group]
    return f"m13_{label(n_blocks, availability)}_{group}{suffix}"


def lr_run_name(n_blocks: int, availability: float, index: int, suffix: str = "") -> str:
    return f"m13_lr_{label(n_blocks, availability)}_r{index}{suffix}"


def rate_of(learner: str, index: int) -> float:
    return (TUNE_ADAMW if learner in ADAMW_FAMILY else TUNE_SGD)[index]


def recorded(group: str) -> dict:
    return yaml.safe_load((ROOT / "results" / SOURCE[group] / "config.yaml")
                          .read_text(encoding="utf-8"))


def config_for(group: str, name: str, n_blocks: int, availability: float,
               entries: list[dict], args, suffix: str, seeds: list[int] | None = None):
    """M6's recorded config with the data rate, the learners and the name replaced."""
    run = {"name": name, "seeds": seeds or recorded(group)["run"]["seeds"],
           "device": args.device}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "eval_every": SMOKE_EVAL_EVERY, "seeds": [0]})
    return load_config(
        ROOT / "results" / SOURCE[group] / "config.yaml",
        overrides={"run": run, "learners": entries,
                   "env": {"label_availability": availability,
                           "series": {"n_blocks": n_blocks}}},
    )


def selected_rates(n_blocks: int, availability: float, suffix: str = "") -> dict | None:
    """Each baseline's argmin in this cell; M3's own stationary pick for the reused one."""
    if is_reused(n_blocks, availability) and not suffix:
        picks = json.loads((ROOT / "results" / "m3_rates.json").read_text(encoding="utf-8"))
        return {n: {"lr": p["lr"], "edge": p["edge"]} for n, p in picks["stationary"].items()}
    chosen = {}
    for learner in BASELINES:
        scored = [(settled(lr_run_name(n_blocks, availability, k, suffix), learner), k)
                  for k in range(len(TUNE_SGD))]
        finite = [(v, k) for v, k in scored if v != float("inf")]
        if not finite:
            return None
        value, index = min(finite)
        chosen[learner] = {"lr": rate_of(learner, index), "settled_rmse": value,
                           "edge": index in (0, len(TUNE_SGD) - 1)}
    return chosen


def entries_for(group: str, rates: dict) -> list[dict]:
    """M6's recorded entries, gamma arms dropped, baselines at this cell's rates."""
    out = []
    for entry in recorded(group)["learners"]:
        if entry["name"] in DROPPED:
            continue
        entry = dict(entry)
        if entry["name"] in BASELINES:
            entry["lr"] = rates[entry["name"]]["lr"]
        out.append(entry)
    return out


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def tuning_cells(suffix: str) -> list[tuple[int, float]]:
    # The smoke has no M6 cell of its own length to reuse, so it tunes all six.
    return cells() if suffix else new_cells()


def tune(args, suffix: str) -> int:
    status = load_status()
    grid = [(n, pi, k) for n, pi in tuning_cells(suffix) for k in range(len(TUNE_SGD))]
    print(f"M13 lr{' SMOKE' if suffix else ''}: {len(grid)} runs, seven baselines each\n",
          flush=True)
    started = time.time()
    for index, (n, pi, k) in enumerate(grid, start=1):
        name = lr_run_name(n, pi, k, suffix)
        entries = [{"name": b, "lr": rate_of(b, k), **o} for b, o in BASELINES.items()]
        note = run_one(config_for("a", name, n, pi, entries, args, suffix, seeds=LR_SEEDS),
                       None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(grid)}] {name:<24} sgd {TUNE_SGD[k]:<7g} adamw "
              f"{TUNE_ADAMW[k]:<6g} {note:<24} {(time.time() - started) / 60:.0f} min",
              flush=True)
    print()
    for n, pi in tuning_cells(suffix):
        rates = selected_rates(n, pi, suffix)
        if rates:
            edge = [b for b, r in rates.items() if r["edge"]]
            print(f"  {label(n, pi):<8} " + "  ".join(f"{b} {r['lr']:g}" for b, r in rates.items())
                  + (f"   <- grid edge: {', '.join(edge)}" if edge else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2, 3, 4])
    parser.add_argument("--lr", action="store_true", help="re-tune the baselines per cell")
    parser.add_argument("--smoke", action="store_true", help="20 rounds, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0
    missing = [c for c in SOURCE.values() if not (ROOT / "results" / c / "_complete").exists()]
    if missing:
        print(f"  REFUSED: M13 is built from M6's recorded cells, and {missing} is not finished.")
        return 1
    if args.lr:
        return tune(args, suffix)
    unswept = [label(*c) for c in tuning_cells(suffix) if selected_rates(*c, suffix) is None]
    if unswept:
        print(f"  REFUSED: no selected rates for {unswept}. Run --lr first (D77).")
        return 1

    run_cells = [(n, pi, g) for n, pi in tuning_cells(suffix) for g in GROUPS]
    print(f"M13{' SMOKE' if suffix else ''}: {len(run_cells)} cells, device {args.device}; "
          f"n1_p100 is {SOURCE['a']}/{SOURCE['b']}\n", flush=True)
    status = load_status()
    started, ran = time.time(), 0
    for index, (n, pi, group) in enumerate(run_cells, start=1):
        name = cell_name(n, pi, group, suffix)
        learners = entries_for(group, selected_rates(n, pi, suffix))
        cell_started = time.time()
        note = run_one(config_for(group, name, n, pi, learners, args, suffix),
                       None, None, args.fresh)
        status[name] = note
        if note != "cached":
            ran += 1
            status[f"{name}_minutes"] = round((time.time() - cell_started) / 60, 1)
        save_status(status)
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(run_cells) - index) if ran else 0.0
        print(f"[{index}/{len(run_cells)}] {name:<22} {len(learners):>2} learners  {note:<14}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nM13{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _flatten(config: dict, prefix: str = "") -> dict[str, object]:
    out = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


def pairing_differences(n_blocks: int, group: str, suffix: str) -> set[str] | None:
    """The config keys where the sparse cell differs from its dense partner."""
    paths = [ROOT / "results" / cell_name(n_blocks, pi, group, suffix) / "config.yaml"
             for pi in (DENSE, SPARSE)]
    if not all(p.exists() for p in paths):
        return None
    # Resolved, not raw: M6's file predates fields added since (history_prefix, readout,
    # the belief settings), and a raw comparison would flag their defaults as differences.
    dense, sparse = (_flatten(asdict(load_config(p))) for p in paths)
    keys = set(dense) | set(sparse)
    return {k for k in keys if dense.get(k) != sparse.get(k)}


def _seeds(n_blocks: int, availability: float, learner: str, suffix: str,
           metric: str = "rmse") -> dict[int, float]:
    for group in GROUPS:
        if metric == "rmse":
            got = _by_seed(cell_name(n_blocks, availability, group, suffix), learner)
        else:
            got = _metric_by_seed(cell_name(n_blocks, availability, group, suffix), learner, metric)
        if got:
            return got
    return {}


def _metric_by_seed(cell: str, learner: str, metric: str) -> dict[int, float]:
    import pandas as pd  # noqa: PLC0415

    out = {}
    for path in sorted((ROOT / "results" / cell).glob("seed_*.parquet")):
        frame = pd.read_parquet(path)
        rows = frame[(frame.learner == learner) & (frame.metric == metric)
                     & (frame.evalset == "current")]
        if len(rows):
            rows = rows[rows.t >= int(0.8 * rows.t.max())]
            out[int(path.stem.split("_")[1])] = float(rows.value.mean())
    return out


def _change(n_blocks: int, left: str, right: str, suffix: str) -> dict[int, float]:
    """(left - right) at pi = 0.25 minus the same at pi = 1, per seed."""
    sparse = differences(_seeds(n_blocks, SPARSE, left, suffix),
                         _seeds(n_blocks, SPARSE, right, suffix))
    dense = differences(_seeds(n_blocks, DENSE, left, suffix),
                        _seeds(n_blocks, DENSE, right, suffix))
    return differences(sparse, dense)


def _family(rows: list[tuple[str, dict[int, float]]]) -> None:
    results = [(name, summarise(values)) for name, values in rows]
    live = [(name, r) for name, r in results if r.n >= 2]
    print(f"\n    {'':<34}{STAT_HEADER}")
    for (name, result), p_adj in zip(live, holm([r.p for _n, r in live]), strict=True):
        print(f"    {name:<34}{_stat_columns(result, p_adj)}")
    for name, result in results:
        if result.n < 2:
            print(f"    {name:<34}{'-':>9}")


def report(suffix: str = "") -> None:
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    shown = cells()
    if suffix:
        print("  SMOKE: 20 rounds at one seed, all six cells run fresh. The numbers mean")
        print("  nothing; the path is what is checked.\n")

    print("  settled RMSE on the held-out current set\n")
    print(f"    {'learner':<40}" + "".join(f"{label(*c):>10}" for c in shown))
    for learner in [CENTRAL, LOCAL, ONEHOP, *FULL_SHARING, *BASELINES]:
        row = f"    {learner:<40}"
        for n, pi in shown:
            value = mean(_seeds(n, pi, learner, suffix))
            row += f"{value:>10.4f}" if value == value else f"{'-':>10}"
        print(row)

    print("\n  pairing gate: each sparse cell against its dense partner; allowed to differ")
    print(f"  in {sorted(ALLOWED)} only")
    paired = {}
    for n in BLOCKS:
        ok = True
        for group in GROUPS:
            found = pairing_differences(n, group, suffix)
            if found is None:
                ok = False
                print(f"    n{n} {group}: not run")
                continue
            extra = sorted(k for k in found if not any(k == a or k.startswith(a + ".")
                                                       for a in ALLOWED))
            ok &= not extra
            print(f"    n{n} {group}: " + ("paired" if not extra else f"DIFFERS in {extra}"))
        paired[n] = ok
    if suffix:
        return

    gated = [n for n in BLOCKS if paired[n]]
    if len(gated) < len(BLOCKS):
        print(f"\n  withheld (pairing gate): n_b = {[n for n in BLOCKS if not paired[n]]}")
    print("\n  CONFIRMATORY (D131). Each row is a change from pi = 1 to 0.25, per seed.")
    print("  Q1. one-hop - local adapt; predicted negative or null. Holm across n_b.")
    _family([(f"n_b={n}", _change(n, ONEHOP, LOCAL, "")) for n in gated])
    print("\n  Q2. each mean-only filter's gap to the centralised filter; no direction.")
    print("  Holm across the filters and n_b.")
    _family([(f"{name}, n_b={n}", _change(n, learner, CENTRAL, ""))
             for n in gated for name, learner in (("local", LOCAL), ("one-hop", ONEHOP))])
    print("\n  Q3. diffusion_ekf - diffusion_sgd_atc; no direction. Holm across n_b.")
    _family([(f"n_b={n}", _change(n, LOCAL, ATC, "")) for n in gated])

    print("\n  EXPLORATORY. The same change for other pairs (mean, t):")
    for name, left, right in (("one-hop - ATC AdamW", ONEHOP, ATC_ADAMW),
                              ("ATC - centralised SGD", ATC, "centralized_sgd"),
                              ("ATC AdamW - centralised AdamW", ATC_ADAMW, "centralized_adamw"),
                              ("full - local (mean-only)", FULL_SHARING[0], LOCAL),
                              ("one-hop full - one-hop", FULL_SHARING[1], ONEHOP)):
        row = f"    {name:<34}"
        for n in BLOCKS:
            got = summarise(_change(n, left, right, ""))
            row += f"  n_b={n} {got.mean:+.4f} (t={got.t:+.2f})" if got.n >= 2 else f"  n_b={n} -"
        print(row)
    print("\n  EXPLORATORY. variance_ratio, settled, on current (1 = calibrated)")
    print(f"    {'filter':<40}" + "".join(f"{label(*c):>10}" for c in shown))
    for learner in [CENTRAL, LOCAL, ONEHOP, *FULL_SHARING]:
        row = f"    {learner:<40}"
        for n, pi in shown:
            value = mean(_seeds(n, pi, learner, "", metric="variance_ratio"))
            row += f"{value:>10.3f}" if value == value else f"{'-':>10}"
        print(row)
    print(f"\n  * = p_holm < {ALPHA} (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
