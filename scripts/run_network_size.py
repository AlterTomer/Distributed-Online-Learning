r"""N>10 -- network size at fixed connectivity: does decentralisation cost more as N grows?

    python scripts/run_network_size.py --probe-only       # the memory smoke, first, GPU idle
    python scripts/run_network_size.py --lr               # re-tune the baselines per (N, condition)
    python scripts/run_network_size.py                    # the cells themselves
    python scripts/run_network_size.py --full-sharing     # also full sharing at N = 20, 30
    python scripts/run_network_size.py --report-only
    python scripts/run_network_size.py --lr --smoke       # then --smoke, to prove the path

`--lr` must run first and the main pass refuses without it (D77).

## The question

`schedule.md` Track A, tier 1: every result so far is at $N=10$, and a reviewer will
ask whether the picture survives a larger network. Specifically, whether the
diffusion filter's gap to the centralised one grows with $N$ -- the pooled filter
sees $Nn$ samples a step, each agent only its own $n$ plus what diffusion brings --
and whether cooperation pays more when there are more agents to cooperate with.

## Two design decisions, taken 2026-09-25 with the numbers below

**Connectivity is held, not the edge probability.** At a fixed ER $p=0.3$ the graph
gets denser as $N$ grows: mean degree 2.9, 5.8, 8.7 and mixing gap 0.12, 0.20, 0.29
at $N=10, 20, 30$, so a size effect would be confounded with a connectivity effect,
and P5.3 showed connectivity alone moves these results ([[D103]]). Instead $p$ is
chosen per $N$ to hold the **mixing gap** at its $N=10$ value, 0.119 over 120 draws,
measured with this repository's own builder and Metropolis weights: $p=0.224$ at
$N=20$ and $p=0.167$ at $N=30$ (mean degree 4.3 and 4.8). Both sit above the
$\ln N/N$ connectivity threshold, 0.150 and 0.113, so no draw is conditioned on a
rare event -- the failure that removed ER 0.15 from P5.3.

Matching $p$ matches the gap only *on average*, and one draw varies a lot (s.d.
0.04--0.05): the first probe's five seeds gave realised means 0.162, 0.140, 0.119 at
$N=10, 20, 30$. So each seed's draw is also **conditioned into 0.119 +/- 0.02**, with
at most three draws and the third kept regardless (decided with the user,
2026-09-26). About 35% of ER draws land in the band at every $N$, so the cap rarely
binds, and it guarantees the conditioning never searches for a rare graph. The
pre-flight prints each seed's realised gap and whether it landed in the band.

**One horizon for every $N$, because MNIST has 60 000 images.** Shards are disjoint
and consumed exactly once, so $NnT\le60\,000$ (D5), and $N=10$ at $n=4$, $T=1500$
already uses all of it. Rather than epochs (which would feed the filter the same
sample twice, the temporal form of data incest) or a per-agent rate that falls with
$N$ (which would confound $N$ with P5.4's data-rate axis), every size runs at
$n=4$, $T=500$ -- the most $N=30$ allows. $N=10$ is re-run at that horizon as the
in-experiment reference, so only $N$ and $p$ differ along the axis. `EVAL_EVERY`
drops to 10 so the settled window still holds ten evaluations.

## One diffusion filter per process

Group A holds the centralised filter and the four gradient baselines; **each
diffusion filter has its own cell**, mean-only and full sharing alike. Every cell
of a (size, condition) shares its seed, hence its data stream and graph, so
splitting by learner keeps every comparison paired. The first probe forced it: with
both mean-only filters in one process, $N=30$ reserved **8.64 GiB on an 8.00 GiB
card**, two sets of thirty 64.5 MiB covariances plus the centralised filter's.

## Full sharing: on at $N=10$, opt-in above it

All four variants stay in every comparison (decided 2026-09-15), so full sharing
runs at $N=10$ by default and at $N=20, 30$ behind `--full-sharing`. The first probe
found it cheaper than X24's figure suggested: alone in a process it needed 3.3 and
4.6 GiB at $N=20$, and 4.6 and 6.5 GiB at $N=30$ -- the last against a 6.55 GiB
budget, too thin a margin to trust over a long run.

## The memory smoke

`--probe-only` runs each planned (size, group) cell for a few steps at one seed and
compares the peak the CUDA allocator reserved -- plus the evaluation-set cache the
full horizon will grow that the probe did not (D112) -- with 95% of *free* device
memory. A probe that diverges is inconclusive, never a pass: it may stop before the
peak. The main pass re-probes unless `--skip-probe`, and refuses if any planned cell
does not fit. Run it with the GPU otherwise idle.

⚠ **An out-of-memory error is not a reliable signal on this machine.** The Windows
driver can spill allocations past the card into shared system memory instead of
raising -- which is how the first probe *reserved* 8.64 GiB on an 8 GiB card and
completed. Slow, silent, and not a fit. The verdict therefore rests on the budget
comparison; a caught error is only the extreme case of it.
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
    DRIFTS,
    FILTER,
    LEARNING_RATES,
    LR_SEEDS,
    settled,
)
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import per_seed  # noqa: E402
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import differences, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: (label, N, ER edge probability), small to large so a partial run still spans the
#: axis. p holds the mixing gap at the N=10 value -- see the module docstring.
SIZES: list[tuple[str, int, float]] = [
    ("n10", 10, 0.3),
    ("n20", 20, 0.224),
    ("n30", 30, 0.167),
]
#: The N=10 mixing gap the other sizes are matched to (120 draws, 2026-09-25).
TARGET_MIXING_GAP = 0.119
#: Each seed's draw is conditioned into this band: at most BAND_DRAWS draws, the
#: last kept regardless (graph.build_graph; decided 2026-09-26).
GAP_BAND = [TARGET_MIXING_GAP - 0.02, TARGET_MIXING_GAP + 0.02]
BAND_DRAWS = 3

#: Stationary, and X17's recurring abrupt schedule -- a 15-degree jump every 25
#: steps -- which keeps its shape at any horizon and simply holds fewer jumps.
#: IID partition throughout (X25 ran abrupt at skew 0.1), so N is the only axis.
CONDITIONS: dict[str, dict] = {
    "stationary": {"schedule": "stationary", "total_degrees": 0.0},
    "abrupt": dict(DRIFTS["abrupt"]),
}

SAMPLES_PER_AGENT = 4
MNIST_TRAIN = 60_000
HORIZON, SEEDS, EVAL_EVERY = 500, [0, 1, 2, 3, 4], 10

#: Group label -> the one diffusion filter its cell carries, each in its own
#: process: two filters' covariance sets together do not fit at N=30.
MEAN_ONLY = {"local": "diffusion_ekf", "onehop": "diffusion_ekf_onehop_mean_receiver"}
FULL_SHARING = {"full": "diffusion_ekf_full", "onehopfull": "diffusion_ekf_onehop_receiver"}
SOLO = {**MEAN_ONLY, **FULL_SHARING}
GROUP_A = list(MEAN_ONLY.values())  # the mean-only filters, for the report's rows
#: Where full sharing runs without --full-sharing: the size known to fit.
FULL_SHARING_BY_DEFAULT = {"n10"}

#: The memory smoke: two evaluations in, so the evaluation path is inside the peak.
PROBE_HORIZON = 2 * EVAL_EVERY + 1
#: Mid-grid, so no baseline diverges before the peak is reached.
PROBE_RATE = 0.01
#: A cell "fits" when its projected peak is within this share of free memory.
FITS_FRACTION = 0.95

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "nsz_status.json"


def lr_run_name(size: str, condition: str, rate: float, suffix: str = "") -> str:
    return f"nsz_lr_{size}_{condition}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(size: str, condition: str, group: str, suffix: str = "") -> str:
    return f"nsz_{size}_{condition}_{group}{suffix}"


def groups_for(size: str, full_sharing: bool) -> list[str]:
    groups = ["a", *MEAN_ONLY]
    if full_sharing or size in FULL_SHARING_BY_DEFAULT:
        groups += list(FULL_SHARING)
    return groups


def tuned_baselines() -> dict[str, dict]:
    """The three X25 baselines plus `atc_plain`, all re-tuned per (N, condition)."""
    return {**BASELINES, PLAIN["name"]: {k: v for k, v in PLAIN.items() if k != "name"}}


def selected_rates(size: str, condition: str, suffix: str = "") -> dict[str, float] | None:
    """Each baseline's own argmin in this cell, or None when it is unswept."""
    rates: dict[str, float] = {}
    for name in tuned_baselines():
        scored = [(settled(lr_run_name(size, condition, r, suffix), name), r)
                  for r in LEARNING_RATES]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def config_for(args, name: str, n: int, p: float, condition: str, entries: list[dict],
               seeds: list[int] | None = None, horizon: int | None = None):
    """One cell. N, p and the drift block move together; nothing else does."""
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": name, "horizon": horizon or args.horizon,
                    "eval_every": EVAL_EVERY, "seeds": seeds or args.seeds,
                    "device": args.device, "dtype": args.dtype},
            "graph": {"topology": "erdos_renyi", "n_nodes": n,
                      "params": {"p": p, "mixing_gap_band": list(GAP_BAND),
                                 "band_draws": BAND_DRAWS}},
            "env": {"dataset": args.dataset,
                    "samples_per_node_per_step": SAMPLES_PER_AGENT,
                    "drift": dict(CONDITIONS[condition])},
            "learners": entries,
            "eval": {"evalsets": ["prequential", "current"]},
        },
    )


def entries_for(group: str, rates: dict[str, float]) -> list[dict]:
    if group in FULL_SHARING:
        return [{"name": FULL_SHARING[group], **FILTER, "combine_exponent": 1.0}]
    if group in MEAN_ONLY:
        return [{"name": MEAN_ONLY[group], **FILTER}]
    learners = [{"name": "centralized_ekf_gamma", **CENTRALIZED}]
    learners += [{"name": n, "lr": rates[n], **o} for n, o in tuned_baselines().items()]
    return learners


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def preflight(args, suffix: str = "") -> bool:
    """The data budget, and each seed's realised mixing gap. False refuses the run."""
    from dekf_bench.env.graph import build_graphs
    from dekf_bench.runner.seeding import Seeds

    rates = {name: PROBE_RATE for name in tuned_baselines()}
    print("  pre-flight")
    print(f"    n = {SAMPLES_PER_AGENT}, T = {args.horizon}, eval_every {EVAL_EVERY}; "
          f"mixing gap target {TARGET_MIXING_GAP}")
    ok, graphs = True, {}
    for size, n, p in SIZES:
        used = n * SAMPLES_PER_AGENT * args.horizon
        config = config_for(args, "nsz_preflight", n, p, "stationary", entries_for("a", rates))
        gaps = [build_graphs(config, Seeds.from_master(s).torch_generator("graph")).comm.mixing_gap
                for s in args.seeds]
        graphs[size] = gaps
        within = used <= MNIST_TRAIN
        ok &= within
        # `!` marks a seed whose three draws all missed the band: its third is used.
        marked = " ".join(f"{g:.3f}" + (" " if GAP_BAND[0] <= g <= GAP_BAND[1] else "!")
                          for g in gaps)
        print(f"    {size}: N={n:>2} p={p:<5}  data {used:>6}/{MNIST_TRAIN} "
              f"{'ok' if within else 'OVER BUDGET'}   mixing gap {marked}  "
              f"mean {sum(gaps) / len(gaps):.3f}")
    print(f"    band {GAP_BAND[0]:.3f}-{GAP_BAND[1]:.3f}, at most {BAND_DRAWS} draws; "
          "! = outside it, the last draw kept by rule")
    if not suffix:
        status = load_status()
        status["graphs"] = graphs
        save_status(status)
    if not ok:
        print("\n  REFUSED: N*n*T exceeds the 60 000 training images. Epochs would feed the")
        print("  filter the same sample twice; lower --horizon instead (D5).")
    return ok


def _cache_allowance(args, test, condition: str, n: int, p: float) -> int:
    """Evaluation-set cache the full horizon can grow that the probe did not (D112)."""
    from dekf_bench.env.drift import build_drift
    from dekf_bench.evaluation.evalsets import MAX_CACHE_BYTES

    def cached(horizon: int) -> int:
        config = config_for(args, "nsz_preflight", n, p, condition,
                            entries_for("a", {k: PROBE_RATE for k in tuned_baselines()}),
                            horizon=horizon)
        side = config.model.input_size
        per_set = len(test) * side * side * (8 if config.run.dtype == "float64" else 4)
        drift = build_drift(config)
        steps = list(range(0, horizon, EVAL_EVERY)) + [horizon - 1]
        rotations = {round(drift.rotation_at(step), 9) for step in steps}
        return min(MAX_CACHE_BYTES, len(rotations) * per_set)

    return max(0, cached(args.horizon) - cached(PROBE_HORIZON))


def probe_memory(args, train, test, plan: list[tuple]) -> bool:
    """The memory smoke. True when every planned cell fits."""
    import gc
    import shutil

    import torch

    if args.device == "cpu" or not torch.cuda.is_available():
        print("  memory smoke: skipped -- host memory only, no device ceiling to breach\n")
        return True
    free, total = torch.cuda.mem_get_info()

    def gib(b: float) -> float:
        return b / 2**30

    print(f"  memory smoke on {torch.cuda.get_device_name(0)}: "
          f"{gib(free):.2f} GiB free of {gib(total):.2f}")
    if free < 0.85 * total:
        print(f"    note: {gib(total - free):.2f} GiB is already held elsewhere -- another "
              "process on the GPU\n    makes every cell look worse than it is. Probe with "
              "the GPU otherwise idle.")
    print(f"\n    {'cell':<24}{'learners':>9}{'peak':>9}{'+cache':>9}{'needs':>9}"
          f"{'budget':>9}   verdict")

    rates = {name: PROBE_RATE for name in tuned_baselines()}
    # One probe per (size, group): the condition changes the drift, not what is
    # resident, and the abrupt drift is the one that grows the evaluation cache.
    pairs = sorted({(size, n, p, group) for size, n, p, _c, group in plan},
                   key=lambda row: (row[1], row[3]))
    budget = FITS_FRACTION * free
    verdicts, all_fit = {}, True

    def shown(b: float | None) -> str:
        return f"{gib(b):>8.2f}G" if b is not None else f"{'-':>9}"

    for size, n, p, group in pairs:
        name = cell_name(size, "probe", group, "_memprobe")
        entries = entries_for(group, rates)
        config = config_for(args, name, n, p, "abrupt", entries,
                            seeds=[args.seeds[0]], horizon=PROBE_HORIZON)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            note = run_one(config, train, test, True)
            peak = torch.cuda.max_memory_reserved()
        except torch.cuda.OutOfMemoryError:
            note, peak = "out of memory", None
        finally:
            shutil.rmtree(ROOT / "results" / name, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
        allowance = _cache_allowance(args, test, "abrupt", n, p)
        if peak is None:
            verdict, needs = "DOES NOT FIT (out of memory in the probe)", None
        elif note != "ok":
            verdict, needs = f"INCONCLUSIVE ({note[:40]})", peak + allowance
        else:
            needs = peak + allowance
            verdict = "fits" if needs <= budget else "DOES NOT FIT"
        fits = verdict == "fits"
        all_fit &= fits
        verdicts[f"{size}_{group}"] = {"peak_bytes": peak, "needs_bytes": needs,
                                       "budget_bytes": budget, "verdict": verdict}
        print(f"    {size + '_' + group:<24}{len(entries):>9}{shown(peak)}{shown(allowance)}"
              f"{shown(needs)}{shown(budget)}   {verdict}", flush=True)

    status = load_status()
    status["memprobe"] = {"device": torch.cuda.get_device_name(0), "free_bytes": free,
                          "total_bytes": total, "at": time.strftime("%Y-%m-%d %H:%M"),
                          "cells": verdicts}
    save_status(status)
    if not all_fit:
        print("\n  Not every planned cell fits. Options, in order of cost: run without")
        print("  --full-sharing; run the main pass with --device cpu (P5.7 measured CPU at")
        print("  about 1.7x slower); or split a cell's learners across processes.")
    print()
    return all_fit


def tune(args, train, test, suffix: str = "") -> int:
    """The gradient baselines only, per (N, condition); the filter carries X20's."""
    status = load_status()
    seeds = args.seeds if suffix else LR_SEEDS
    cells = [(size, n, p, condition, rate) for size, n, p in SIZES
             for condition in CONDITIONS for rate in LEARNING_RATES]
    print(f"N>10 lr{' SMOKE' if suffix else ''}: {len(SIZES)} sizes x {len(CONDITIONS)} "
          f"conditions x {len(LEARNING_RATES)} rates at {len(seeds)} seed(s)\n", flush=True)
    started = time.time()
    for index, (size, n, p, condition, rate) in enumerate(cells, start=1):
        name = lr_run_name(size, condition, rate, suffix)
        entries = [{"name": b, "lr": rate, **o} for b, o in tuned_baselines().items()]
        note = run_one(config_for(args, name, n, p, condition, entries, seeds=seeds),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<36} {note:<28} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nlr complete in {(time.time() - started) / 60:.1f} min\n")
    names = list(tuned_baselines())
    print(f"  {'cell':>18}" + "".join(f"{b:>26}" for b in names))
    for size, _n, _p in SIZES:
        for condition in CONDITIONS:
            rates = selected_rates(size, condition, suffix)
            if rates:
                edge = [b for b in names if rates[b] in (LEARNING_RATES[0], LEARNING_RATES[-1])]
                print(f"  {size + '/' + condition:>18}"
                      + "".join(f"{rates[b]:>26g}" for b in names)
                      + (f"   <- grid edge: {', '.join(edge)}" if edge else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--lr", action="store_true",
                        help="re-tune the gradient baselines instead of running the sweep")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="20 steps at one seed, into _smoke-suffixed runs")
    parser.add_argument("--full-sharing", action="store_true",
                        help="also run full covariance sharing at N = 20 and 30")
    parser.add_argument("--probe-only", action="store_true",
                        help="run the memory smoke for the planned cells and stop")
    parser.add_argument("--skip-probe", action="store_true",
                        help="launch without re-probing (after a successful --probe-only)")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.horizon, args.seeds = 20, [0]
    if args.report_only:
        report(suffix)
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)
    if not preflight(args, suffix):
        return 1

    plan = [(size, n, p, condition, group) for size, n, p in SIZES
            for condition in CONDITIONS for group in groups_for(size, args.full_sharing)]
    if args.probe_only:
        return 0 if probe_memory(args, train, test, plan) else 1
    if args.lr:
        return tune(args, train, test, suffix)

    missing = [f"{s}/{c}" for s, _n, _p in SIZES for c in CONDITIONS
               if selected_rates(s, c, suffix) is None]
    if missing:
        print(f"no selected rates for {missing}: run --lr{' --smoke' if suffix else ''} "
              "first.\nA rate carried across conditions is the mistake D77 exists to "
              "record --\nit put a baseline at chance and inverted a damage ordering.")
        return 1
    if not args.skip_probe and not probe_memory(args, train, test, plan):
        print("  REFUSED: the memory smoke says a planned cell does not fit.")
        return 1

    print(f"\nN>10{' SMOKE' if suffix else ''}: {len(plan)} cells at {len(args.seeds)} "
          f"seeds, T={args.horizon}")
    for size, n, p, condition, group in plan:
        print(f"  {cell_name(size, condition, group, suffix):<34} N={n:<3} p={p}")
    print(flush=True)

    status = load_status()
    started, ran = time.time(), 0
    for index, (size, n, p, condition, group) in enumerate(plan, start=1):
        name = cell_name(size, condition, group, suffix)
        entries = entries_for(group, selected_rates(size, condition, suffix))
        note = run_one(config_for(args, name, n, p, condition, entries), train, test, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(plan) - index) if ran else 0.0
        print(f"[{index}/{len(plan)}] {name:<34} {len(entries):>2} learners  {note:<26}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nN>10{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


def _cell_of(learner: str, size: str, condition: str, suffix: str = "") -> str:
    """Each diffusion filter has its own cell; everything else is in group A."""
    for group, name in SOLO.items():
        if learner == name:
            return cell_name(size, condition, group, suffix)
    return cell_name(size, condition, "a", suffix)


def _table(title: str, lines: list[str], rows: list[tuple[str, object]],
           left_header: str = "") -> None:
    """One family: Holm across its rows, signed t and a 95% interval on each.

    A row whose cells are missing prints as such and spends no alpha. The label
    column is as wide as its longest label, which in the change tables carries the
    learner and its three per-N gaps.
    """
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
    width = max((len(label) for label, _r in rows), default=0) + 2
    print(f"\n    {left_header:<{width}}{STAT_HEADER}")
    live = [(label, result) for label, result in rows if result is not None]
    adjusted = dict(zip([label for label, _r in live], holm([r.p for _l, r in live]),
                        strict=True))
    for label, result in rows:
        if result is None:
            print(f"    {label:<{width}}{'-':>9}   (cells missing)")
        else:
            print(f"    {label:<{width}}{stat_columns(result, adjusted[label])}")


def report(suffix: str = "") -> None:
    """Settled error by N, then the four contrasts the sweep exists for."""
    seeds_of = lambda learner, size, condition: per_seed(  # noqa: E731
        _cell_of(learner, size, condition, suffix), learner)
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    sizes = [size for size, _n, _p in SIZES]
    central = "centralized_ekf_gamma"
    variants = [*GROUP_A, *FULL_SHARING.values()]
    every = [central, *variants, *tuned_baselines()]
    if suffix:
        print("  SMOKE: 20 rounds at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")
    graphs = load_status().get("graphs", {})
    if graphs:
        print("  realised mixing gap per seed (target "
              f"{TARGET_MIXING_GAP}): "
              + "; ".join(f"{s} {sum(g) / len(g):.3f}" for s, g in graphs.items() if g))

    for condition in CONDITIONS:
        print(f"\n  ===== {condition} =====\n")
        print(f"  settled error by N (n={SAMPLES_PER_AGENT}, T={HORIZON if not suffix else 20})\n")
        print(f"    {'learner':<36}" + "".join(f"{s:>12}" for s in sizes))
        for learner in every:
            row = f"    {learner:<36}"
            for size in sizes:
                values = seeds_of(learner, size, condition)
                row += f"{mean(values):>12.4f}" if values else f"{'-':>12}"
            print(row)

        def gap(learner: str, size: str, reference: str,
                condition: str = condition) -> dict[int, float]:
            return differences(seeds_of(learner, size, condition),
                               seeds_of(reference, size, condition))

        # Change tables: the learner, its gap at each N, then the test on the change.
        change_header = f"{'':<36}" + "".join(f"{'N=' + s[1:]:>9}" for s in sizes)

        def change_rows(reference: str, learners: list[str], reference_first: bool = False):
            rows = []
            for learner in learners:
                gaps = {s: gap(reference, s, learner) if reference_first
                        else gap(learner, s, reference) for s in sizes}
                label = f"{learner:<36}" + "".join(
                    f"{mean(gaps[s]):>+9.4f}" if gaps[s] else f"{'-':>9}" for s in sizes)
                both = gaps[sizes[0]] and gaps[sizes[-1]]
                rows.append((label, compare(gaps[sizes[-1]], gaps[sizes[0]]) if both else None))
            return rows

        _table("does decentralisation cost more as N grows?",
               ["gap = diffusion minus centralised filter, per seed; positive = diffusion worse.",
                "Tested: the change in the gap from N=10 to N=30, per seed (D54) -- the",
                "stat columns are that change, not any one gap."],
               change_rows(central, variants), change_header)
        _table("does cooperation pay more with more agents?",
               ["local_only minus each learner, per seed; positive = cooperating helps.",
                "Tested: the change from N=10 to N=30."],
               change_rows("local_only", [*variants, "diffusion_sgd_atc", PLAIN["name"]],
                           reference_first=True),
               change_header)
        _table("one-hop minus local adapt, per N (does one-hop's value scale with N?)",
               ["negative = one-hop better."],
               [(s, compare(seeds_of("diffusion_ekf_onehop_mean_receiver", s, condition),
                            seeds_of("diffusion_ekf", s, condition)))
                for s in sizes])
        _table("one-hop minus atc_plain, per N (near-matched bandwidth, 3 696 vs 2 908 scalars)",
               ["negative = the filter wins."],
               [(s, compare(seeds_of("diffusion_ekf_onehop_mean_receiver", s, condition),
                            seeds_of(PLAIN["name"], s, condition)))
                for s in sizes])
        pairs = (("diffusion_ekf_full", "diffusion_ekf"),
                 ("diffusion_ekf_onehop_receiver", "diffusion_ekf_onehop_mean_receiver"))
        rows = [(f"{s} {full.replace('diffusion_ekf_', '')}",
                 compare(seeds_of(full, s, condition), seeds_of(base, s, condition)))
                for s in sizes for full, base in pairs if seeds_of(full, s, condition)]
        if rows:
            _table("full minus mean-only sharing, per N (only where full sharing ran)",
                   ["negative = sharing the covariance helps."], rows)

    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
