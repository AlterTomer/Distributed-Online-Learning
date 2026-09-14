r"""X25 -- the diffusion filter under Dirichlet label skew (P5.2).

    python scripts/run_diffusion_skew.py --lr   # re-tune the baselines, first
    python scripts/run_diffusion_skew.py        # the cells themselves

`--lr` must run first and the main pass refuses without it (D77).

## Why this is the row that matters

Every diffusion result so far -- X19 through X24 -- used an **IID** partition, and
[[D77]] says plainly that X17 transfers nothing: the centralised filter pools, so
Dirichlet skew barely reaches it, and "the filter is robust to heterogeneity"
really means "pooling defused the skew". X6 measured the gap that pooling hides --
across $\beta_{\mathrm{dir}}\in\{0.1,1,100\}$ the pooled learner moves 0.0020
while `local_only` moves 0.4882.

Here each agent holds its own skewed shard and the **combine step has to reconcile
beliefs formed from different label distributions**. That is the first time the
combine is asked to do something hard, and it is the first question in phase 5
whose answer is not implied by D79.

## The second hypothesis, which X24 handed us

X24 found $\beta=1$ -- `lem:conservative`, which assumes the neighbours' errors
*coincide* -- beating $\beta=2$ by up to 0.2181 at $t=42.5$. The bound is
approximately tight, so on IID shards the agents' errors really do nearly
coincide.

**Skew is precisely what should break that.** Agents holding different label
distributions make genuinely different errors, so the correlation $\beta=1$
assumes is what a Dirichlet partition destroys. If the conservative bound is tight
at $\beta_{\mathrm{dir}}=100$ and loose at $0.1$, then $\beta>1$ becomes live in
exactly the regime where cooperation is hardest -- which is a far better result
than "we swept $\beta$ and it did nothing". The full-sharing pair therefore rides
along at both ends of the skew axis, at $\beta\in\{1,2\}$.

## Why its own learning-rate pass

X17 swept these same conditions and its selections are on disk, but it ran on
`x1_stationary`'s **ring**; every diffusion experiment runs on \ac{er} $p=0.3$.
Carrying a rate across topologies is the mistake D77 exists to record -- it put a
baseline at chance and inverted a damage ordering -- so the rates are re-selected
here even though the conditions are X17's.

## Memory, and why the learners are split across cells

Four diffusion filters do not fit in one process. Group A carries the two
deployable (mean-only) variants with the centralised filter and the \ac{sgd}
baselines; group B carries one full-sharing variant per cell, which X24 measured
at 3.43 GiB of 8.00. Splitting by learner rather than by condition keeps every
arm's cells paired by seed against the same data stream.
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

#: X6's and X17's axis, unchanged so the stationary cells stay comparable to both.
#: 0.1 is severe skew, 100 is IID in all but name.
SKEWS = [0.1, 1.0, 100.0]

#: X17's matched pair: same 0.60 deg/step, opposite abruptness, both at the
#: severe skew. Which *kind* of motion hurts more alongside skew, not whether
#: drift hurts.
DRIFT_SKEW = 0.1
DRIFTS = {
    "abrupt": {"schedule": "recurring", "jump_every": 25, "jump_degrees": 15.0,
               "jump_seed": 0},
    "smooth": {"schedule": "recurring", "jump_every": 1, "jump_degrees": 0.6,
               "jump_seed": 0},
}

TOPOLOGY = ("erdos_renyi", {"p": 0.3})

#: The X20 selection, confirmed jointly by X23 and left alone by X22 and X24.
FILTER = {"transition": "scalar", "gamma": 0.9995, "lambda_forget": 1.0,
          "process_noise_q": 6.0e-4, "prior_scale": 1.0e-3}
CENTRALIZED = {"transition": "scalar", "gamma": 0.9995, "lambda_forget": 1.0,
               "process_noise_q": 6.0e-5, "prior_scale": 0.01}

GROUP_A = ["diffusion_ekf", "diffusion_ekf_onehop_mean"]
GROUP_B = ["diffusion_ekf_full", "diffusion_ekf_onehop"]
COMBINE_EXPONENTS = [1.0, 2.0]

#: Group B only visits the ends of the skew axis: the hypothesis is that the
#: conservative bound is tight at 100 and loose at 0.1, and the midpoint cannot
#: separate those.
GROUP_B_SKEWS = [0.1, 100.0]

#: X17's, unchanged, each with its own optimiser for the reason X6 settled.
BASELINES = {
    "centralized_sgd": {"optimizer": "sgd_momentum", "momentum": 0.9},
    "diffusion_sgd_atc": {"optimizer": "sgd_momentum", "momentum": 0.9},
    "local_only": {"optimizer": "sgd", "momentum": 0.0},
}
LEARNING_RATES = [0.2, 0.05, 0.01, 0.005, 0.001]
LR_SEEDS = [0, 1]

#: Defaults for the CLI (`--horizon`, `--seeds`); see scripts/_args.py.
HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2], 5

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x25_status.json"

#: (label, skew, drift). A drifting cell and its twin share a label, because they
#: must differ in the drift alone -- the learning rate included.
CONDITIONS: list[tuple[str, float, dict | None]] = [
    (f"still_b{s:g}", s, None) for s in SKEWS
] + [(name, DRIFT_SKEW, block) for name, block in DRIFTS.items()]


def lr_run_name(condition: str, rate: float) -> str:
    return f"x25_lr_{condition}_lr{rate:g}".replace(".", "p")


def cell_name(condition: str, twin: bool = False) -> str:
    return f"x25_control_{condition}" if twin else f"x25_{condition}"


def group_b_name(learner: str, skew: float, beta: float) -> str:
    stem = learner.replace("diffusion_ekf_", "")
    return f"x25_{stem}_b{skew:g}_beta{beta:g}".replace(".", "p")


def settled(run: str, learner: str, metric: str = "error_rate") -> float:
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    files = sorted(directory.glob("seed_*.parquet"))
    if not (directory / "_complete").exists() or not files:
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"])
         for f in files], ignore_index=True)
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                 & (frame["t"] >= int(0.8 * frame["t"].max()))]
    if metric == "error_rate":
        rows = rows[rows["evalset"] == "current"]
    return float(rows["value"].mean()) if len(rows) else float("inf")


def selected_rates(condition: str) -> dict[str, float] | None:
    """Each baseline's own argmin, or None when the condition is unswept."""
    rates: dict[str, float] = {}
    for name in BASELINES:
        scored = [(settled(lr_run_name(condition, r), name), r) for r in LEARNING_RATES]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, skew: float, drift: dict | None, entries: list[dict],
               seeds: list[int] | None = None):
    block = {"schedule": "stationary", "total_degrees": 0.0} if drift is None else dict(drift)
    topology, params = TOPOLOGY
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": args.horizon, "eval_every": EVAL_EVERY,
                    "seeds": seeds or args.seeds, "device": args.device,
                    "dtype": args.dtype},
            "graph": {"topology": topology, "params": dict(params)},
            "env": {"dataset": args.dataset,
                    "partition": {"kind": "dirichlet", "beta": skew},
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
    """The SGD baselines only; the filter carries its own selection by design."""
    status = load_status()
    total = len(CONDITIONS) * len(LEARNING_RATES)
    print(f"X25 lr: {len(CONDITIONS)} conditions x {len(LEARNING_RATES)} rates at "
          f"{len(LR_SEEDS)} seeds, on {TOPOLOGY[0]}\n", flush=True)
    started, index = time.time(), 0
    for condition, skew, drift in CONDITIONS:
        for rate in LEARNING_RATES:
            index += 1
            name = lr_run_name(condition, rate)
            entries = [{"name": n, "lr": rate, **o} for n, o in BASELINES.items()]
            note = run_one(config_for(args, name, skew, drift, entries, seeds=LR_SEEDS),
                           train, test, args.fresh)
            status[name] = note
            save_status(status)
            print(f"[{index}/{total}] {name:<38} {note:<28} "
                  f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    print(f"  {'condition':>16}" + "".join(f"{n:>20}" for n in BASELINES))
    for condition, _s, _d in CONDITIONS:
        rates = selected_rates(condition)
        if rates:
            print(f"  {condition:>16}" + "".join(f"{rates[n]:>20g}" for n in BASELINES))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="re-tune the SGD baselines instead of running the sweep")
    parser.add_argument("--report-only", action="store_true",
                        help="print the surface from cells already on disk")
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

    rates = {c: selected_rates(c) for c, _s, _d in CONDITIONS}
    missing = sorted(c for c, r in rates.items() if r is None)
    if missing:
        print(f"\nThe learning-rate sweep has not run for {missing}. Without it a\n"
              "baseline would carry a rate chosen for another condition, which put a\n"
              "baseline at chance and inverted a damage ordering once already (D77).\n"
              "  python scripts/run_diffusion_skew.py --lr\n")
        return 1

    cells: list[tuple[str, float, dict | None, list[dict]]] = []
    for condition, skew, drift in CONDITIONS:
        entries = (
            [{"name": "centralized_ekf_gamma", **CENTRALIZED}]
            + [{"name": n, **FILTER} for n in GROUP_A]
            + [{"name": n, "lr": rates[condition][n], **o} for n, o in BASELINES.items()]
        )
        cells.append((cell_name(condition), skew, drift, entries))
        if drift is not None:
            # Damage needs a twin differing in the drift alone -- same skew, same
            # rates, same learners.
            cells.append((cell_name(condition, twin=True), skew, None, entries))

    for learner in GROUP_B:
        for skew in GROUP_B_SKEWS:
            for beta in COMBINE_EXPONENTS:
                entries = [{"name": learner, **FILTER, "combine_exponent": beta}]
                cells.append((group_b_name(learner, skew, beta), skew, None, entries))

    print(f"\nX25: {len(cells)} cells at {len(args.seeds)} seeds, T={args.horizon}")
    for name, skew, drift, entries in cells:
        kind = "still" if drift is None else drift["schedule"]
        print(f"  {name:<34} skew {skew:<6g} {kind:<10} {len(entries)} learners")
    print(flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (name, skew, drift, entries) in enumerate(cells, start=1):
        note = run_one(config_for(args, name, skew, drift, entries), train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = (elapsed / ran * (len(cells) - index)) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<34} {note:<28} "
              f"{elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nX25 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    r"""The skew curve, the damage pair, and whether skew makes $\beta$ live."""
    names = ["centralized_ekf_gamma", *GROUP_A, *BASELINES]
    print("  settled error by skew (still cells)")
    print(f"    {'skew':>8}" + "".join(f"{n.replace('diffusion_', 'd_'):>26}" for n in names))
    for skew in SKEWS:
        run = cell_name(f"still_b{skew:g}")
        line = f"    {skew:>8g}"
        for learner in names:
            value = settled(run, learner)
            line += f"{value:>26.4f}" if value != float("inf") else f"{'-':>26}"
        print(line)

    print("\n  spread across the skew axis (X6: pooled 0.0020, local_only 0.4882)")
    for learner in names:
        vals = [settled(cell_name(f"still_b{s:g}"), learner) for s in SKEWS]
        vals = [v for v in vals if v != float("inf")]
        if len(vals) == len(SKEWS):
            print(f"    {learner:<26}{max(vals) - min(vals):.4f}")

    print("\n  damage under drift at skew 0.1 (drifting minus its still twin)")
    print(f"    {'condition':>10}" + "".join(f"{n.replace('diffusion_', 'd_'):>26}"
                                             for n in names))
    for condition in DRIFTS:
        line = f"    {condition:>10}"
        for learner in names:
            drifting = settled(cell_name(condition), learner)
            twin = settled(cell_name(condition, twin=True), learner)
            if drifting == float("inf") or twin == float("inf"):
                line += f"{'-':>26}"
            else:
                line += f"{drifting - twin:>26.4f}"
        print(line)

    print("\n  does skew make beta live? (full sharing, still cells)")
    print(f"    {'learner':>22}{'skew':>8}" + "".join(f"{'beta=' + f'{b:g}':>12}"
                                                      for b in COMBINE_EXPONENTS))
    for learner in GROUP_B:
        for skew in GROUP_B_SKEWS:
            line = f"    {learner:>22}{skew:>8g}"
            for beta in COMBINE_EXPONENTS:
                run = group_b_name(learner, skew, beta)
                if (ROOT / "results" / run / "_diverged").exists():
                    line += f"{'DIV':>12}"
                    continue
                value = settled(run, learner)
                line += f"{value:>12.4f}" if value != float("inf") else f"{'-':>12}"
            print(line)
    print("\n  X24 measured beta=2 as ruinous on IID shards. If it is merely bad at")
    print("  skew 100 and competitive at skew 0.1, the conservative bound is tight")
    print("  only while the agents' errors coincide -- which is what skew destroys.")


if __name__ == "__main__":
    raise SystemExit(main())
