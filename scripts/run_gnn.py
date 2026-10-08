r"""D3c -- the GNN on the coupled Mackey--Glass network, and its FP64 twin (D139).

    python scripts/run_gnn.py --reference            # the GNN reference: R and e*
    python scripts/run_gnn.py --tune-filters         # M4/M5-style filter grids
    python scripts/run_gnn.py --lr                   # the gradient learners' rates, per cell
    python scripts/run_gnn.py                        # the cells: stationary and linear beta
    python scripts/run_gnn.py --twin --reference     # the twin: its own R ...
    python scripts/run_gnn.py --twin --tune-filters  # ... its own filter settings ...
    python scripts/run_gnn.py --twin                 # ... and the ablations
    python scripts/run_gnn.py [--twin] --report-only
    python scripts/run_gnn.py [--twin] --smoke --reference   (and so on: the path, on CPU)

Each stage refuses until the one before it is on disk.

## The design (decided 2026-10-06 to 10-09, D139)

**The task**: the coupled network ($\kappa=0.6$, M8's delays, the communication graph's
Metropolis weights), $N=10$ on the runs' ER graphs, $T=1500$, five seeds; stationary
and linear $\beta$ drift (`m_coupled`, `m_coupled_linear`). **The model**: the MG causal
decoder per node as the encoder, two aggregate-then-transform graph layers, a
per-position head, $p=20\,001$, **FP32** (the user's exception to the FP64 rule).
**The learners**: the centralised filter **exact** (the whole network linearised) and
**truncated** (every node's packed batch) side by side; local adapt and the GNN one-hop
(mean-only) in group a with the SGD and AdamW families; both full-sharing variants in
group b. Every filter on **neuron blocks**; the distributed ones truncated.

**R** is measured, not assumed: M2's protocol on network snapshots, one run's budget,
the per-position residual variance pooled over nodes (`--reference`). **The filters**
are tuned on the stationary cell, 2 seeds, M4's and M5's grids with q one decade lower,
and carried to the drift cell; **the gradient learners** per cell on M3's grids, SGD one
decade lower.

**The twin**: $d=12$, $d_{\text{ff}}=24$, two graph layers ($p\approx2\,000$), FP64,
stationary, five seeds, its own reference and filter tuning; the filters under the
full, neuron and layer covariance, and local adapt and one-hop also **exact** (the
multi-hop recomputation at $\boldsymbol\theta_v$) under the full covariance.

## Named before the run (confirmatory under D118, Holm within each)

1. **Truncation cost**: the truncated minus the exact centralised filter, per cell.
   Predicted $\ge 0$ (truncation costs). The question as first put said "exact minus
   truncated, predicted >= 0", a sign slip; the meaning approved was that truncation
   costs, and this is the contrast that is non-negative when it does.
2. **Decentralisation gap**: one-hop and local adapt minus the exact centralised
   filter, per cell.
3. **Filter vs gradient**: one-hop minus ATC AdamW and minus ATC momentum, per cell.
   Predicted negative, as on the Transformer (M6).
4. **Blocks cost** (the twin): neuron blocks minus the full covariance, per filter.
   No direction predicted; the size read against 0.002.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_inner_steps import per_seed  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, SGD_FAMILY, settled  # noqa: E402

from dekf_bench.metrics.paired import differences, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

CELLS = {"stationary": "m_coupled", "linear": "m_coupled_linear"}
TWIN_MODEL = {"d_model": 12, "d_ff": 24}
EXACT_C, TRUNC_C = "centralized_ekf_gamma", "centralized_ekf_truncated"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
FILTERS = [EXACT_C, TRUNC_C, LOCAL, ONEHOP, *FULL_SHARING]
STRUCTURES = ["full", "neuron", "layer"]

GAMMAS = [1.0, 0.9995]
CENTRAL_QS = [6e-4, 6e-5, 6e-6, 6e-7]
DIFFUSION_QS = [6e-4, 6e-5, 6e-6, 6e-7, 6e-8]
PRIORS = [0.01, 0.001]
R_SCALES = [1.0, 2.0]
SGD_RATES = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
ADAMW_RATES = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LR_SEEDS, TUNE_SEEDS = [0, 1], [0, 1]
REPORT_SEEDS = [0, 1, 2, 3, 4]
HORIZON, EVAL_EVERY = 1500, 25
SMOKE_HORIZON, SMOKE_EVAL = 6, 3


# --------------------------------------------------------------------------- #
# names, files, configs
# --------------------------------------------------------------------------- #

def tag(args) -> str:
    return ("gnt" if args.twin else "gnn")


def suffix_of(args) -> str:
    return "_smoke" if args.smoke else ""


def results_file(args, what: str) -> Path:
    return ROOT / "results" / f"{tag(args)}_{what}{suffix_of(args)}.json"


def read(args, what: str) -> dict:
    path = results_file(args, what)
    if not path.exists():
        raise SystemExit(f"  REFUSED: {path.name} is not on disk -- run the stage that writes it")
    return json.loads(path.read_text(encoding="utf-8"))


def config_for(args, name: str, cell: str, learners: list[dict], seeds: list[int],
               r_scale: float = 1.0, reference: dict | None = None):
    model: dict = dict(TWIN_MODEL) if args.twin else {}
    if reference is not None:
        model["observation_variances"] = [v * r_scale for v in reference["observation_variances"]]
        model["observation_variance"] = reference["observation_variance"] * r_scale
    run = {"name": name, "horizon": args.horizon, "seeds": seeds, "device": args.device,
           "dtype": "float64" if args.twin else "float32",
           "eval_every": SMOKE_EVAL if args.smoke else EVAL_EVERY}
    overrides = {"run": run, "learners": learners, "model": model}
    if args.smoke:
        overrides["env"] = {"series": {"burn_in": 100.0, "eval_blocks": 4, "eval_trajectories": 2}}
    return load_config(CELLS[cell], overrides=overrides)


def filter_entry(name: str, gamma: float, q: float, prior: float, structure: str,
                 **extra) -> dict:
    return {"name": name, "transition": "scalar", "gamma": gamma, "forgetting": "process_noise",
            "process_noise_q": q, "lambda_forget": 1.0, "prior_scale": prior,
            "covariance_blocks": structure, **extra}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def reference_stage(args) -> int:
    from dekf_bench.env.graph import build_graphs  # noqa: PLC0415
    from dekf_bench.evaluation.series_reference import (  # noqa: PLC0415
        ReferenceSettings,
        train_gnn_reference,
    )
    from dekf_bench.runner.seeding import Seeds  # noqa: PLC0415

    config = config_for(args, "reference", "stationary", [{"name": "centralized_sgd"}], [0])
    graphs = [build_graphs(config, Seeds.from_master(s).torch_generator("graph")).comm
              for s in REPORT_SEEDS]
    settings = (ReferenceSettings(train_trajectories=2, blocks_per_trajectory=50, val_blocks=100,
                                  test_blocks=100, epochs=2, patience=1)
                if args.smoke else ReferenceSettings())
    started = time.time()
    result = train_gnn_reference(config, graphs, settings)
    path = results_file(args, "reference")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  reference ({'twin' if args.twin else 'GNN'}): test RMSE {result['rmse']:.4f}, "
          f"persistence {result['persistence_rmse']:.4f}, epoch {result['best_epoch']}, "
          f"R mean {result['observation_variance']:.5f}  ({(time.time() - started) / 60:.1f} min)")
    return 0


def tune_filters_stage(args) -> int:
    reference = read(args, "reference")
    structure = "full" if args.twin else "neuron"
    seeds = [0] if args.smoke else TUNE_SEEDS
    pick = (lambda grid: [grid[0], grid[-1]]) if args.smoke else (lambda grid: grid)
    runs = []
    for g, q, s, r in itertools.product(pick(GAMMAS), pick(CENTRAL_QS), pick(PRIORS),
                                        pick(R_SCALES)):
        runs.append(("central", {"gamma": g, "q": q, "prior": s, "r": r},
                     filter_entry(EXACT_C, g, q, s, structure)))
    for g, q, s in itertools.product(pick(GAMMAS), pick(DIFFUSION_QS), pick(PRIORS)):
        runs.append(("diffusion", {"gamma": g, "q": q, "prior": s, "r": 1.0},
                     filter_entry(ONEHOP, g, q, s, structure)))
    print(f"filter tuning ({tag(args)}): {len(runs)} runs at {len(seeds)} seed(s)\n", flush=True)
    scored: dict[str, list] = {"central": [], "diffusion": []}
    for index, (family, point, entry) in enumerate(runs, start=1):
        name = (f"{tag(args)}_tune_{family}_g{point['gamma']:g}_q{point['q']:g}_s{point['prior']:g}"
                f"_r{point['r']:g}").replace(".", "p").replace("-", "m") + suffix_of(args)
        note = run_one(config_for(args, name, "stationary", [entry], seeds, point["r"], reference),
                       None, None, args.fresh)
        value = settled(name, entry["name"])
        scored[family].append((value, point))
        print(f"[{index}/{len(runs)}] {name:<58} {note:<14} {value:.4f}", flush=True)
    selection = {}
    for family, rows in scored.items():
        finite = [(v, p) for v, p in rows if math.isfinite(v)]
        if not finite:
            raise SystemExit(f"  every {family} grid point diverged")
        value, point = min(finite, key=lambda vp: vp[0])
        edges = [axis for axis, grid in (("q", CENTRAL_QS if family == "central" else DIFFUSION_QS),
                                         ("prior", PRIORS))
                 if point[axis if axis != "prior" else "prior"] in (grid[0], grid[-1])
                 and len(grid) > 2]
        selection[family] = {**point, "settled_rmse": value, "edges": edges}
        print(f"\n  {family}: {point}  rmse {value:.4f}"
              + (f"   <- on the grid's edge: {edges}" if edges else ""))
    results_file(args, "filters").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return 0


def lr_stage(args) -> int:
    reference = read(args, "reference")
    seeds = [0] if args.smoke else LR_SEEDS
    indices = [0, len(SGD_RATES) - 1] if args.smoke else list(range(len(SGD_RATES)))
    runs = [(cell, i) for cell in CELLS for i in indices]
    print(f"rates: {len(runs)} runs at {len(seeds)} seed(s)\n", flush=True)
    for index, (cell, i) in enumerate(runs, start=1):
        learners = [{"name": n, "lr": SGD_RATES[i], **o} for n, o in SGD_FAMILY.items()]
        if i < len(ADAMW_RATES):
            learners += [{"name": n, "lr": ADAMW_RATES[i], **o} for n, o in ADAMW_FAMILY.items()]
        name = f"gnn_lr_{cell}_r{i}{suffix_of(args)}"
        note = run_one(config_for(args, name, cell, learners, seeds, 1.0, reference),
                       None, None, args.fresh)
        print(f"[{index}/{len(runs)}] {name:<28} {note}", flush=True)
    rates = {cell: {} for cell in CELLS}
    for cell in CELLS:
        for learner, grid in [*((n, SGD_RATES) for n in SGD_FAMILY),
                              *((n, ADAMW_RATES) for n in ADAMW_FAMILY)]:
            scored = [(settled(f"gnn_lr_{cell}_r{i}{suffix_of(args)}", learner), i)
                      for i in indices if i < len(grid)]
            finite = [(v, i) for v, i in scored if math.isfinite(v)]
            if finite:
                value, i = min(finite)
                rates[cell][learner] = {"lr": grid[i], "settled_rmse": value,
                                        "edge": i in (0, len(grid) - 1)}
    results_file(args, "rates").write_text(json.dumps(rates, indent=2), encoding="utf-8")
    for cell, chosen in rates.items():
        print(f"\n  {cell}: " + ", ".join(f"{n} {c['lr']:g}{' EDGE' if c['edge'] else ''}"
                                        for n, c in chosen.items()))
    return 0


def settings_of(selection: dict, structure: str) -> tuple[dict, dict, float]:
    c, d = selection["central"], selection["diffusion"]
    central = filter_entry("", c["gamma"], c["q"], c["prior"], structure)
    diffusion = filter_entry("", d["gamma"], d["q"], d["prior"], structure)
    central.pop("name"), diffusion.pop("name")
    return central, diffusion, c["r"]


def cells_stage(args) -> int:
    reference, selection = read(args, "reference"), read(args, "filters")
    seeds = [0] if args.smoke else REPORT_SEEDS
    runs = []
    if args.twin:
        for structure in STRUCTURES:
            central, diffusion, r = settings_of(selection, structure)
            runs.append((f"gnt_{structure}_a", "stationary", r,
                         [{"name": EXACT_C, **central}, {"name": TRUNC_C, **central},
                          {"name": LOCAL, **diffusion}, {"name": ONEHOP, **diffusion}]))
            runs.append((f"gnt_{structure}_b", "stationary", r,
                         [{"name": n, **diffusion, "combine_exponent": 1.0} for n in FULL_SHARING]))
        central, diffusion, r = settings_of(selection, "full")
        runs.append(("gnt_exact", "stationary", r,
                     [{"name": n, **diffusion, "gnn_jacobian": "exact"} for n in (LOCAL, ONEHOP)]))
    else:
        rates = read(args, "rates")
        central, diffusion, r = settings_of(selection, "neuron")
        for cell in CELLS:
            gradient = [{"name": n, "lr": rates[cell][n]["lr"], **o}
                        for n, o in {**SGD_FAMILY, **ADAMW_FAMILY}.items() if n in rates[cell]]
            runs.append((f"gnn_{cell}_a", cell, r,
                         [{"name": EXACT_C, **central}, {"name": TRUNC_C, **central},
                          {"name": LOCAL, **diffusion}, {"name": ONEHOP, **diffusion},
                          *gradient]))
            runs.append((f"gnn_{cell}_b", cell, r,
                         [{"name": n, **diffusion, "combine_exponent": 1.0} for n in FULL_SHARING]))
    print(f"{tag(args)} cells: {len(runs)} at {len(seeds)} seed(s), T={args.horizon}\n",
          flush=True)
    started = time.time()
    for index, (name, cell, r, learners) in enumerate(runs, start=1):
        cell_started = time.time()
        note = run_one(config_for(args, name + suffix_of(args), cell, learners, seeds, r,
                                  reference), None, None, args.fresh)
        print(f"[{index}/{len(runs)}] {name + suffix_of(args):<24} {len(learners):>2} learners "
              f"{note:<14} {(time.time() - cell_started) / 60:.1f} min this cell, "
              f"{(time.time() - started) / 60:.0f} total", flush=True)
    report(args)
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

ALPHA = 0.05
STAT_HEADER = f"{'diff':>9}  {'95% CI':^20}  {'t':>7}  {'p_holm':>7}"


def _table(title: str, lines: list[str], rows: list[tuple[str, object]]) -> None:
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
    width = max(len(r[0]) for r in rows) + 2
    print(f"\n    {'':<{width}}{STAT_HEADER}")
    adjusted = holm([r.p if r is not None else float("nan") for _l, r in rows])
    for (lab, result), p_adjusted in zip(rows, adjusted, strict=True):
        if result is None:
            print(f"    {lab:<{width}}{'-':>9}   (cells missing or diverged)")
            continue
        lo, hi = result.ci(0.95)
        star = " *" if p_adjusted < ALPHA else ""
        print(f"    {lab:<{width}}{result.mean:>+9.4f}  {f'[{lo:+.4f}, {hi:+.4f}]':<20}"
              f"  {result.t:>+7.2f}  {p_adjusted:>7.3f}{star}")


def _finite(values: dict) -> dict:
    return {s: v for s, v in values.items() if math.isfinite(v)}


def report(args) -> None:
    sfx = suffix_of(args)
    if args.smoke:
        print("\n  SMOKE: a few steps at one seed. These numbers mean nothing.\n")

    def seeds_of(cell_name: str, learner: str) -> dict:
        return _finite(per_seed(cell_name + sfx, learner))

    def contrast(a: dict, b: dict) -> object:
        result = compare(a, b) if a and b else None
        return result if result is not None and result.n >= 2 else None

    if args.twin:
        print("\n  settled RMSE, the twin (stationary), by covariance structure\n")
        print(f"    {'filter':<36}" + "".join(f"{s:>9}" for s in STRUCTURES) + f"{'exact':>9}")
        cell = lambda s, f: f"gnt_{s}_{'b' if f in FULL_SHARING else 'a'}"  # noqa: E731
        for f in FILTERS:
            row = [seeds_of(cell(s, f), f) for s in STRUCTURES]
            exact = seeds_of("gnt_exact", f) if f in (LOCAL, ONEHOP) else {}
            shown = [f"{sum(v.values()) / len(v):>9.4f}" if v else f"{'-':>9}" for v in [*row, exact]]
            print(f"    {f:<36}" + "".join(shown))
        _table("Q4 [confirmatory]: blocks cost -- neuron blocks minus the full covariance",
               ["no direction predicted; read the size against 0.002."],
               [(f, contrast(seeds_of(cell("neuron", f), f), seeds_of(cell("full", f), f)))
                for f in FILTERS])
        _table("exploratory: layer blocks minus the full covariance", [],
               [(f, contrast(seeds_of(cell("layer", f), f), seeds_of(cell("full", f), f)))
                for f in FILTERS])
        _table("exploratory: truncation's cost -- truncated minus exact, full covariance",
               ["positive = truncation costs accuracy."],
               [(f, contrast(seeds_of("gnt_full_a", f), seeds_of("gnt_exact", f)))
                for f in (LOCAL, ONEHOP)]
               + [(f"{TRUNC_C} - {EXACT_C}", contrast(seeds_of("gnt_full_a", TRUNC_C),
                                                      seeds_of("gnt_full_a", EXACT_C)))])
        print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")
        return

    every = [*FILTERS, *SGD_FAMILY, *ADAMW_FAMILY]
    for cell in CELLS:
        print(f"\n  settled RMSE, {cell} (last 20%, current set)\n")
        for learner in every:
            group = "b" if learner in FULL_SHARING else "a"
            values = seeds_of(f"gnn_{cell}_{group}", learner)
            shown = f"{sum(values.values()) / len(values):.4f}" if values else "-"
            print(f"    {learner:<36}{shown:>9}")

    def gap(first: str, second: str, cell: str) -> object:
        group = lambda n: "b" if n in FULL_SHARING else "a"  # noqa: E731
        return contrast(seeds_of(f"gnn_{cell}_{group(first)}", first),
                        seeds_of(f"gnn_{cell}_{group(second)}", second))

    _table("Q1 [confirmatory]: truncation cost -- truncated minus exact centralised filter",
           ["positive = truncation costs the centralised filter. Predicted >= 0."],
           [(cell, gap(TRUNC_C, EXACT_C, cell)) for cell in CELLS])
    _table("Q2 [confirmatory]: decentralisation gap -- distributed minus exact centralised",
           ["positive = the distributed filter trails the centralised one."],
           [(f"{d} {cell}", gap(d, EXACT_C, cell)) for d in (ONEHOP, LOCAL) for cell in CELLS])
    _table("Q3 [confirmatory]: filter vs gradient -- one-hop minus ATC",
           ["negative = the filter wins. Predicted negative, as M6."],
           [(f"vs {g} {cell}", gap(ONEHOP, g, cell))
            for g in ("diffusion_atc_adamw", "diffusion_sgd_atc") for cell in CELLS])
    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=REPORT_SEEDS)
    parser.add_argument("--twin", action="store_true", help="the FP64 d=12 twin's ablations")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--reference", action="store_true")
    stage.add_argument("--tune-filters", action="store_true")
    stage.add_argument("--lr", action="store_true")
    stage.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="a few steps at one seed, on CPU")
    args = parser.parse_args(argv)
    if args.smoke:
        args.horizon = SMOKE_HORIZON
    if args.report_only:
        report(args)
        return 0
    if args.reference:
        return reference_stage(args)
    if args.tune_filters:
        return tune_filters_stage(args)
    if args.lr:
        if args.twin:
            print("  the twin has no gradient learners: its ablations are the filters'")
            return 1
        return lr_stage(args)
    return cells_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
