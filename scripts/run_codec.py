r"""The offline-trained codec, end to end: calibrate, validate, report (C2, D137, D138).

    python scripts/run_codec.py --calibrate --device cuda      # tune per point, scales, tables
    python scripts/run_codec.py --validate --device cuda       # the frontier; C* per learner
    python scripts/run_codec.py --report-runs --device cuda    # report seeds, transfer
    python scripts/run_codec.py --report-only
    python scripts/run_codec.py --calibrate --smoke            # then --validate / --report-runs
                                                               #   --smoke: the path, on CPU

## The protocol (D137, as decided in D138)

The codec sends each vector as a quantised difference against a public copy, per module
$\ell$ with $\Delta_\ell=c\,s_\ell$, run-length and canonical-Huffman coded with tables
trained offline. Every implementation choice was asked and answered first (D138).

**Seeds split by role.** Calibration 100--104, validation 200--204, report 0--4, all
at the full horizon; no cell outside this runner has used a seed >= 10.

**Calibrate** (seeds 100--104). The source is the stationary IID cell, P5.3's ER 0.3
(one-hop at the receiver point), its filters carried exactly as recorded.
1. *The uncompressed point.* Every gradient baseline is re-tuned uncompressed, in the
   codec's ``scale`` mode -- exact arithmetic that also measures each optimiser moment's
   per-layer rms. The selected rates are the twin's rates; the selected runs' moment
   rms, pooled over seeds, agents and steps, are the moments' scales $s_\ell$
   (``scales.json``). psi's scale is rms(theta_0) per layer.
2. *Every c* in ``C_GRID``. Every gradient baseline is re-tuned again, in ``count`` mode
   (the tables never change accuracy, so the selected rate's run *is* the calibration
   run), and the filters run once at their recorded settings. The selected runs'
   counts, pooled over the five seeds, become each learner's tables
   (``tables_<c>.json``: per kind and layer, canonical Huffman with Good--Turing ESC).

**Validate** (seeds 200--204). Each c with its frozen tables, every learner at that c's
rates, against the uncompressed twin at the uncompressed point's rates. The frontier is
the result: per learner and c, the whole-run rate $R$ -- coded bits per transmitted
scalar, i.e. per parameter per link-message -- against settled error. The operating
point $\mathcal C^\star$ is per learner: the cheapest c whose mean paired settled-error
cost against the twin is within $\varepsilon=0.002$ (``cstar.json``).

**Report** (seeds 0--4). The whole frontier and its twin in the stationary condition;
then each learner's $\mathcal C^\star$ carried unchanged to linear and abrupt drift (X20's
schedules) and label skew 0.1 (X25's partition), with an uncompressed twin in each.
Learners are grouped into cells by their $\mathcal C^\star$, since c is a run setting.

⚠ Bits are the tables' code lengths summed per message, not bitstreams produced each
step; `codec.encode` and `tests/test_codec*.py` tie the two together bit for bit.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_atc_plain import PLAIN  # noqa: E402
from run_diffusion_skew import BASELINES, LEARNING_RATES, settled  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import per_seed  # noqa: E402
from run_network_size import ADAMW_RATES  # noqa: E402

from dekf_bench.codec import LayerCode  # noqa: E402
from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import paired  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

TASK = "mnist"
SOURCE = "p53_er030_a"
#: condition -> (the env section that changes, the recorded cell it is read from).
TRANSFER = {"linear": ("drift", "x20_linear_a0p03_erdos_renyi"),
            "abrupt": ("drift", "x20_every25_jump15_erdos_renyi"),
            "skew": ("partition", "x25p_onehop_b0p1_beta1")}

LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
FILTERS = [LOCAL, ONEHOP]
ATC, ATC_PLAIN, ATC_ADAMW = "diffusion_sgd_atc", PLAIN["name"], "diffusion_atc_adamw"
#: The gradient baselines, each with its options and its own grid (D138's roster).
BASELINE_OPTIONS = {ATC: BASELINES[ATC],
                    ATC_PLAIN: {k: v for k, v in PLAIN.items() if k != "name"},
                    ATC_ADAMW: {}}
GRIDS = {ATC: sorted(LEARNING_RATES, reverse=True),
         ATC_PLAIN: sorted({*LEARNING_RATES, 1.0, 0.5}, reverse=True),
         ATC_ADAMW: sorted(ADAMW_RATES, reverse=True)}
FAMILY = {ATC: "sgd", ATC_PLAIN: "sgd", ATC_ADAMW: "adamw"}
ROSTER = [*FILTERS, *BASELINE_OPTIONS]

C_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
SMOKE_C_GRID = [1e-2, 1e-4]
NONE = "none"
CALIBRATION, VALIDATION, REPORT = [100, 101, 102, 103, 104], [200, 201, 202, 203, 204], \
    [0, 1, 2, 3, 4]
EPSILON = 0.002
SMOKE_HORIZON = 20
DATA_ROOT = ROOT / "data"


# --------------------------------------------------------------------------- #
# names and files
# --------------------------------------------------------------------------- #

def tag(point) -> str:
    return NONE if point == NONE else f"c{point:g}".replace(".", "p").replace("-", "m")


def lr_name(point, family: str, rate: float, suffix: str) -> str:
    return f"cdc_lr_{tag(point)}_{family}_lr{rate:g}".replace(".", "p") + suffix


def filter_cell(point, suffix: str) -> str:
    return f"cdc_cal_{tag(point)}_filters{suffix}"


def val_cell(point, suffix: str) -> str:
    return f"cdc_val_{tag(point)}{suffix}"


def rep_cell(condition: str, point, suffix: str) -> str:
    return f"cdc_rep_{condition}_{tag(point)}{suffix}"


def out_dir(suffix: str) -> Path:
    return ROOT / "results" / "codec" / f"{TASK}{suffix}"


def scales_path(suffix: str) -> Path:
    return out_dir(suffix) / "scales.json"


def tables_path(point, suffix: str) -> Path:
    return out_dir(suffix) / f"tables_{tag(point)}.json"


def cstar_path(suffix: str) -> Path:
    return out_dir(suffix) / "cstar.json"


def grid_points(suffix: str) -> list:
    return SMOKE_C_GRID if suffix else C_GRID


def complete(run: str) -> bool:
    return (ROOT / "results" / run / "_complete").exists()


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #

def recorded(run: str) -> dict:
    return yaml.safe_load((ROOT / "results" / run / "config.yaml").read_text(encoding="utf-8"))


def filter_entries() -> list[dict]:
    """The filters exactly as the source cell recorded them: they carry their selections."""
    return [dict(e) for e in recorded(SOURCE)["learners"] if e["name"] in FILTERS]


def baseline_entries(rates: dict[str, float], names=None) -> list[dict]:
    return [{"name": n, "lr": rates[n], **BASELINE_OPTIONS[n]}
            for n in (names or BASELINE_OPTIONS) if n in rates]


def comm_for(point, mode: str, suffix: str) -> dict:
    if point == NONE:
        return ({"compressor": "codec", "codec_mode": "scale"} if mode == "scale"
                else {"compressor": "none"})
    comm = {"compressor": "codec", "codec_c": point, "codec_mode": mode,
            "codec_scales": str(scales_path(suffix))}
    if mode == "code":
        comm["codec_tables"] = str(tables_path(point, suffix))
    return comm


def condition_env(condition: str) -> dict:
    if condition == "stationary":
        return {}
    section, source = TRANSFER[condition]
    return {section: recorded(source)["env"][section]}


def config_for(args, name: str, learners: list[dict], seeds: list[int], comm: dict,
               condition: str = "stationary"):
    overrides = {"run": {"name": name, "horizon": args.horizon, "seeds": seeds,
                         "device": args.device, "dtype": args.dtype},
                 "learners": learners, "comm": comm}
    env = condition_env(condition)
    if env:
        overrides["env"] = env
    return load_config(ROOT / "results" / SOURCE / "config.yaml", overrides=overrides)


def run(args, train, test, name: str, learners: list[dict], seeds: list[int], comm: dict,
        condition: str = "stationary") -> str:
    started = time.time()
    note = run_one(config_for(args, name, learners, seeds, comm, condition), train, test,
                   args.fresh)
    print(f"    {name:<44} {len(learners)} learner(s)  {note:<14} "
          f"{(time.time() - started) / 60:.1f} min", flush=True)
    return note


# --------------------------------------------------------------------------- #
# selection and the per-seed summaries
# --------------------------------------------------------------------------- #

def summaries(run_name: str, learner: str) -> dict[int, dict]:
    """The codec's per-seed summaries for one learner in one cell."""
    out = {}
    for path in sorted((ROOT / "results" / run_name).glob(f"codec_{learner}_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[int(data["seed"])] = data
    return out


def selected(point, suffix: str) -> dict[str, float] | None:
    """Each baseline's argmin of settled error over its grid at this point (D77)."""
    rates = {}
    for name, grid in GRIDS.items():
        scored = [(settled(lr_name(point, FAMILY[name], r, suffix), name), r) for r in grid]
        scored = [(v, r) for v, r in scored if math.isfinite(v)]
        if not scored:
            return None
        rates[name] = min(scored)[1]
    return rates


def selection_cell(point, learner: str, rates: dict, suffix: str) -> str:
    """Where a learner's calibration run at this point lives."""
    if learner in FILTERS:
        return filter_cell(point, suffix)
    return lr_name(point, FAMILY[learner], rates[learner], suffix)


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #

def tune_point(args, train, test, point, suffix: str, seeds: list[int]) -> None:
    mode = "scale" if point == NONE else "count"
    comm = comm_for(point, mode, suffix)
    for family in ("sgd", "adamw"):
        rates = sorted({r for n, g in GRIDS.items() if FAMILY[n] == family for r in g},
                       reverse=True)
        for rate in rates:
            names = [n for n, g in GRIDS.items() if FAMILY[n] == family and rate in g]
            run(args, train, test, lr_name(point, family, rate, suffix),
                baseline_entries({n: rate for n in names}), seeds, comm)
    if point != NONE:
        run(args, train, test, filter_cell(point, suffix), filter_entries(), seeds, comm)


def write_scales(suffix: str) -> dict:
    rates = selected(NONE, suffix)
    if rates is None:
        raise SystemExit("  the uncompressed point is not tuned: no scales")
    scales: dict = {"rates": rates, "learners": {}}
    for learner in BASELINE_OPTIONS:
        pooled: dict = {}
        layers = None
        for summary in summaries(selection_cell(NONE, learner, rates, suffix), learner).values():
            layers = summary["layers"]
            for kind, sums in summary.get("moment_sums", {}).items():
                acc = pooled.setdefault(kind, [[0.0, 0] for _ in sums])
                for slot, (s, n) in zip(acc, sums, strict=True):
                    slot[0] += s
                    slot[1] += n
        if pooled:
            scales["learners"][learner] = {
                kind: {layer: (s / n) ** 0.5 if n else 0.0
                       for layer, (s, n) in zip(layers, acc, strict=True)}
                for kind, acc in pooled.items()}
    scales_path(suffix).parent.mkdir(parents=True, exist_ok=True)
    scales_path(suffix).write_text(json.dumps(scales, indent=1), encoding="utf-8")
    return scales


def write_tables(point, suffix: str) -> dict:
    rates = selected(point, suffix)
    if rates is None:
        raise SystemExit(f"  {tag(point)} is not tuned: no tables")
    tables: dict = {"c": point, "rates": rates, "learners": {}}
    for learner in ROSTER:
        pooled: dict = {}
        for summary in summaries(selection_cell(point, learner, rates, suffix), learner).values():
            for kind, layers in summary.get("counts", {}).items():
                for layer, entry in layers.items():
                    slot = pooled.setdefault(kind, {}).setdefault(layer, [{}, {}])
                    for side, key in ((0, "runs"), (1, "amps")):
                        for symbol, count in entry[key].items():
                            slot[side][int(symbol)] = slot[side].get(int(symbol), 0) + count
        tables["learners"][learner] = {
            kind: {layer: LayerCode(runs, amps).to_json() for layer, (runs, amps) in layers.items()}
            for kind, layers in pooled.items()}
    tables_path(point, suffix).write_text(json.dumps(tables), encoding="utf-8")
    return tables


def edges(point, rates: dict) -> list[str]:
    return [f"{n} at {r:g}" for n, r in rates.items() if r in (GRIDS[n][0], GRIDS[n][-1])]


def calibrate(args, train, test, suffix: str) -> int:
    seeds = args.seeds if suffix else CALIBRATION
    print(f"codec calibrate{' SMOKE' if suffix else ''}: seeds {seeds}, T={args.horizon}\n")
    print("  the uncompressed point (scale mode: exact, measuring the moments)")
    tune_point(args, train, test, NONE, suffix, seeds)
    scales = write_scales(suffix)
    print(f"  uncompressed rates {scales['rates']}"
          + (f"   ⚠ grid edge: {', '.join(edges(NONE, scales['rates']))}"
             if edges(NONE, scales["rates"]) else ""))
    for point in grid_points(suffix):
        print(f"\n  c = {point:g} (count mode)")
        tune_point(args, train, test, point, suffix, seeds)
        tables = write_tables(point, suffix)
        print(f"  rates {tables['rates']}"
              + (f"   ⚠ grid edge: {', '.join(edges(point, tables['rates']))}"
                 if edges(point, tables["rates"]) else ""))
    return 0


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #

def frontier_rows(cell: str, twin: str, learner: str) -> dict | None:
    """Rate, error and the paired cost against the twin, for one learner in one cell."""
    error, base = per_seed(cell, learner), per_seed(twin, learner)
    shared = sorted(set(error) & set(base))
    if not shared:
        return None
    stats = summaries(cell, learner)
    rate = {s: d["totals"].get("coded_bits", 0) / d["scalars"] for s, d in stats.items()
            if d["scalars"]}
    ideal = {s: d["totals"].get("ideal_bits", 0) / d["scalars"] for s, d in stats.items()
             if d["scalars"]}
    entropy = {s: d["totals"].get("entropy_bits", 0) / d["scalars"] for s, d in stats.items()
               if d["scalars"]}
    escapes = {s: d["totals"].get("escapes", 0) / max(d["totals"].get("symbols", 0), 1)
               for s, d in stats.items()}
    result = paired({s: error[s] for s in shared}, {s: base[s] for s in shared})
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    return {"R": mean(rate), "ideal": mean(ideal), "entropy": mean(entropy),
            "escapes": mean(escapes), "error": mean(error), "twin": mean(base),
            "cost": result.mean, "ci": result.ci(0.95), "n": result.n}


def choose(suffix: str, epsilon: float) -> dict:
    """C* per learner: the cheapest c whose mean paired cost is within epsilon."""
    twin = val_cell(NONE, suffix)
    out = {}
    for learner in ROSTER:
        accepted = []
        for point in grid_points(suffix):
            row = frontier_rows(val_cell(point, suffix), twin, learner)
            if row and row["cost"] <= epsilon:
                accepted.append((row["R"], point))
        out[learner] = min(accepted)[1] if accepted else None
    cstar_path(suffix).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def validate(args, train, test, suffix: str) -> int:
    seeds = args.seeds if suffix else VALIDATION
    if not scales_path(suffix).exists():
        print("  REFUSED: no scales -- run --calibrate first")
        return 1
    rates_none = selected(NONE, suffix)
    print(f"codec validate{' SMOKE' if suffix else ''}: seeds {seeds}\n")
    run(args, train, test, val_cell(NONE, suffix),
        filter_entries() + baseline_entries(rates_none), seeds, comm_for(NONE, "none", suffix))
    for point in grid_points(suffix):
        if not tables_path(point, suffix).exists():
            print(f"  REFUSED: no tables at c = {point:g} -- run --calibrate first")
            return 1
        rates = json.loads(tables_path(point, suffix).read_text(encoding="utf-8"))["rates"]
        run(args, train, test, val_cell(point, suffix),
            filter_entries() + baseline_entries(rates), seeds, comm_for(point, "code", suffix))
    cstar = choose(suffix, float("inf") if suffix else EPSILON)
    print(f"\n  C* per learner: {cstar}")
    return 0


# --------------------------------------------------------------------------- #
# report runs
# --------------------------------------------------------------------------- #

def report_runs(args, train, test, suffix: str) -> int:
    seeds = args.seeds if suffix else REPORT
    if not cstar_path(suffix).exists():
        print("  REFUSED: no C* -- run --validate first")
        return 1
    cstar = json.loads(cstar_path(suffix).read_text(encoding="utf-8"))
    rates_none = selected(NONE, suffix)
    rates_at = {point: json.loads(tables_path(point, suffix).read_text(encoding="utf-8"))["rates"]
                for point in grid_points(suffix)}
    print(f"codec report runs{' SMOKE' if suffix else ''}: seeds {seeds}\n")
    run(args, train, test, rep_cell("stationary", NONE, suffix),
        filter_entries() + baseline_entries(rates_none), seeds, comm_for(NONE, "none", suffix))
    for point in grid_points(suffix):
        run(args, train, test, rep_cell("stationary", point, suffix),
            filter_entries() + baseline_entries(rates_at[point]), seeds,
            comm_for(point, "code", suffix))
    for condition in TRANSFER:
        run(args, train, test, rep_cell(condition, NONE, suffix),
            filter_entries() + baseline_entries(rates_none), seeds,
            comm_for(NONE, "none", suffix), condition)
        for point in grid_points(suffix):
            members = [n for n in ROSTER if cstar.get(n) == point]
            if not members:
                continue
            learners = ([e for e in filter_entries() if e["name"] in members]
                        + baseline_entries(rates_at[point], members))
            run(args, train, test, rep_cell(condition, point, suffix), learners, seeds,
                comm_for(point, "code", suffix), condition)
    skipped = [n for n in ROSTER if cstar.get(n) is None]
    if skipped:
        print(f"\n  not transferred (no c within epsilon): {skipped}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def print_frontier(title: str, cell_of, twin: str, suffix: str, learners=ROSTER) -> None:
    print(f"\n  {title}")
    print(f"    {'learner':<36}{'c':>8}{'R':>8}{'ideal':>8}{'H':>7}{'esc':>8}{'error':>8}"
          f"{'cost':>9}  95% CI")
    for learner in learners:
        for point in grid_points(suffix):
            row = frontier_rows(cell_of(point), twin, learner) if complete(cell_of(point)) \
                else None
            if not row:
                continue
            lo, hi = row["ci"]
            print(f"    {learner:<36}{point:>8g}{row['R']:>8.2f}{row['ideal']:>8.2f}"
                  f"{row['entropy']:>7.2f}{row['escapes']:>8.4f}{row['error']:>8.4f}"
                  f"{row['cost']:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]")


def report(suffix: str) -> None:
    if suffix:
        print("  SMOKE: 20 steps, one seed per role. These numbers mean nothing.\n")
    print("  R = coded bits per transmitted scalar (per parameter per link-message), whole run;")
    print("  ideal = the bound under the trained probabilities; H = D136's empirical entropy;")
    print("  esc = escaped symbols / symbols; cost = settled error minus the uncompressed")
    print("  twin's, paired per seed. ⚠ Counted code lengths, not per-step bitstreams.")
    if scales_path(suffix).exists():
        scales = json.loads(scales_path(suffix).read_text(encoding="utf-8"))
        print(f"\n  uncompressed rates (seeds 100-104): {scales['rates']}")
    for point in grid_points(suffix):
        if tables_path(point, suffix).exists():
            rates = json.loads(tables_path(point, suffix).read_text(encoding="utf-8"))["rates"]
            print(f"  rates at c = {point:g}: {rates}"
                  + (f"   ⚠ edge: {', '.join(edges(point, rates))}" if edges(point, rates)
                     else ""))
    if complete(val_cell(NONE, suffix)):
        print_frontier("VALIDATION (seeds 200-204): the frontier",
                       lambda p: val_cell(p, suffix), val_cell(NONE, suffix), suffix)
    if cstar_path(suffix).exists():
        cstar = json.loads(cstar_path(suffix).read_text(encoding="utf-8"))
        print(f"\n  C* per learner (cheapest c within {EPSILON} of the twin): {cstar}")
    if complete(rep_cell("stationary", NONE, suffix)):
        print_frontier("REPORT (seeds 0-4), stationary: the frontier",
                       lambda p: rep_cell("stationary", p, suffix),
                       rep_cell("stationary", NONE, suffix), suffix)
        for condition in TRANSFER:
            twin = rep_cell(condition, NONE, suffix)
            if complete(twin):
                print_frontier(f"REPORT, {condition}: C* carried unchanged (esc = transfer check)",
                               lambda p, c=condition: rep_cell(c, p, suffix), twin, suffix)


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=REPORT)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--calibrate", action="store_true")
    stage.add_argument("--validate", action="store_true")
    stage.add_argument("--report-runs", action="store_true")
    stage.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="20 steps, one seed, on CPU")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.horizon, args.device = SMOKE_HORIZON, "cpu"
    if args.report_only:
        report(suffix)
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)
    out_dir(suffix).mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        args.seeds = [CALIBRATION[0]] if suffix else CALIBRATION
        code = calibrate(args, train, test, suffix)
    elif args.validate:
        args.seeds = [VALIDATION[0]] if suffix else VALIDATION
        code = validate(args, train, test, suffix)
    else:
        args.seeds = [REPORT[0]] if suffix else REPORT
        code = report_runs(args, train, test, suffix)
    report(suffix)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
