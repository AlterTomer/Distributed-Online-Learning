r"""The AdamW pass, backfilled: AdamW arms for the finished experiments the paper cites.

    python scripts/run_adamw_pass.py --lr                  # tune AdamW per condition, first
    python scripts/run_adamw_pass.py                       # the AdamW cells
    python scripts/run_adamw_pass.py --experiment p57 x20  # a subset
    python scripts/run_adamw_pass.py --report-only
    python scripts/run_adamw_pass.py --lr --smoke          # then --smoke

`--lr` must run first and the main pass refuses without it (D77).

## Why a separate runner, and why it cannot drift from the original cells

Every figure the paper carries must include the AdamW baselines (decided
2026-09-26; schedule.md, "The AdamW pass"). N>10, P5.4, P5.5, P5.8, P5.23 and the
calibration run carry them from the start; the experiments finished before that
decision do not. This backfills them **horizontally**, one experiment at a time.

**Each AdamW cell is built from the finished cell's own recorded config**
(`results/<cell>/config.yaml`, fully resolved), with the learners replaced and the
run renamed -- nothing else touched. Graph, partition, drift, horizon, cadence,
dtype and seeds are therefore identical by construction rather than by restating
them in code, which is how a backfill quietly stops matching the thing it extends.

**The merge gate checks it anyway** (D119): each AdamW cell also re-runs one SGD
learner the source cell recorded, at the rate it was recorded with -- `centralized_sgd`
where the cell has it, else ATC, local-only or `atc_plain`, in that order (X20's
cells carry no `centralized_sgd`) -- and it must reproduce per seed. X25's seeds live
in two runs (0--2 in `x25_*`, 3--4 in `x25p_*`, D101); its AdamW cells run all five
from one config and the gate compares against the pooled pair.

**Nothing is merged into the finished runs' files.** The cells stand beside them and
the figure builders read both; merging parquet would rewrite results a published
number rests on.

## The registry

`EXPERIMENTS` names each experiment's conditions, their source cells, and the one-hop
variant its figures use (X20 and X25 ran the sender point, P5.3 and P5.7 the
receiver). X27 is absent on purpose: its cells hold filters only, and the skew
baselines its figures use are X25's.

## Named before the run (confirmatory under D118)

Per experiment: **does one-hop beat ATC AdamW in every condition?** `one-hop - ATC
AdamW`, per seed, Holm across that experiment's conditions, **predicted negative**,
and tested only where the merge gate holds (the comparison crosses cells).
Exploratory: the AdamW levels, centralised AdamW against the centralised filter, and
`local_adamw` against `local_only`.
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
from run_diffusion_skew import settled  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import per_seed  # noqa: E402
from run_network_size import ADAMW, ADAMW_RATES, REPRODUCTION_TOLERANCE  # noqa: E402
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

#: experiment -> (the one-hop learner its figures use, {condition: source cells}).
#: The first source cell supplies the config; all of them supply the gate's seeds.
EXPERIMENTS: dict[str, tuple[str, dict[str, list[str]]]] = {
    "x20": ("diffusion_ekf_onehop_mean", {
        "still": ["x20_still_erdos_renyi"],
        "linear": ["x20_linear_a0p03_erdos_renyi"],
        "abrupt": ["x20_every25_jump15_erdos_renyi"],
    }),
    "x25": ("diffusion_ekf_onehop_mean", {
        "b0.1": ["x25_still_b0.1", "x25p_still_b0.1"],
        "b1": ["x25_still_b1", "x25p_still_b1"],
        "b100": ["x25_still_b100", "x25p_still_b100"],
    }),
    "p53": ("diffusion_ekf_onehop_mean_receiver", {
        "complete": ["p53_complete_a"],
        "er030": ["p53_er030_a"],
        "path": ["p53_path_a"],
        "ring": ["p53_ring_a"],
    }),
    "p57": ("diffusion_ekf_onehop_mean_receiver", {
        "per_node": ["p57_per_node_a"],
        "global": ["p57_global_a"],
    }),
}
#: The reproduction arm: the first of these the source cell recorded.
REPRODUCTION_ORDER = ["centralized_sgd", "diffusion_sgd_atc", "local_only",
                      "diffusion_sgd_atc_plain"]
ATC_ADAMW = "diffusion_atc_adamw"
LR_SEEDS = [0, 1]
SMOKE_HORIZON = 20

STATUS = ROOT / "results" / "adw_status.json"
DATA_ROOT = ROOT / "data"


def lr_run_name(experiment: str, condition: str, rate: float, suffix: str = "") -> str:
    return f"adw_lr_{experiment}_{condition}_lr{rate:g}".replace(".", "p") + suffix


def cell_name(experiment: str, condition: str, suffix: str = "") -> str:
    return f"adw_{experiment}_{condition}".replace(".", "p") + suffix


def source_config(sources: list[str]) -> dict:
    import yaml  # noqa: PLC0415

    return yaml.safe_load((ROOT / "results" / sources[0] / "config.yaml")
                          .read_text(encoding="utf-8"))


def reproduction_entry(recorded: dict) -> dict:
    """The recorded entry of the first SGD learner in REPRODUCTION_ORDER."""
    by_name = {entry["name"]: entry for entry in recorded["learners"]}
    for name in REPRODUCTION_ORDER:
        if name in by_name:
            return dict(by_name[name])
    raise SystemExit(f"{recorded['run']['name']} records no SGD learner to reproduce")


def all_seeds(sources: list[str]) -> list[int]:
    import yaml  # noqa: PLC0415

    seeds: list[int] = []
    for cell in sources:
        seeds += yaml.safe_load((ROOT / "results" / cell / "config.yaml")
                                .read_text(encoding="utf-8"))["run"]["seeds"]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"{sources}: the source cells share a seed; pooling would double it")
    return sorted(seeds)


def config_for(sources: list[str], name: str, entries: list[dict], args, suffix: str,
               seeds: list[int] | None = None):
    """The source cell's recorded config with the learners and the name replaced.

    The smoke also shortens the horizon and takes one seed; nothing else changes.
    """
    recorded = source_config(sources)
    path = ROOT / "results" / sources[0] / "config.yaml"
    run = {"name": name, "seeds": seeds or all_seeds(sources)}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "seeds": [0], "device": args.device})
    config = load_config(path, overrides={"run": run, "learners": entries})
    assert config.env.dataset == recorded["env"]["dataset"]
    return config


def selected_rates(experiment: str, condition: str, suffix: str = "") -> dict[str, float] | None:
    rates = {}
    for learner in ADAMW:
        scored = [(settled(lr_run_name(experiment, condition, r, suffix), learner), r)
                  for r in ADAMW_RATES]
        scored = [(v, r) for v, r in scored if v != float("inf")]
        if not scored:
            return None
        rates[learner] = min(scored)[1]
    return rates


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _missing_sources(experiments: list[str]) -> list[str]:
    return [cell for e in experiments for sources in EXPERIMENTS[e][1].values()
            for cell in sources if not (ROOT / "results" / cell / "_complete").exists()]


def tune(args, train, test, experiments: list[str], suffix: str) -> int:
    status = load_status()
    cells = [(e, c, r) for e in experiments for c in EXPERIMENTS[e][1] for r in ADAMW_RATES]
    print(f"AdamW pass lr{' SMOKE' if suffix else ''}: {len(cells)} cells "
          f"({', '.join(experiments)}), {len(LR_SEEDS)} seeds\n", flush=True)
    started = time.time()
    for index, (experiment, condition, rate) in enumerate(cells, start=1):
        sources = EXPERIMENTS[experiment][1][condition]
        name = lr_run_name(experiment, condition, rate, suffix)
        entries = [{"name": n, "lr": rate} for n in ADAMW]
        note = run_one(config_for(sources, name, entries, args, suffix, seeds=LR_SEEDS),
                       train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<40} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print()
    for experiment in experiments:
        for condition in EXPERIMENTS[experiment][1]:
            rates = selected_rates(experiment, condition, suffix)
            if rates:
                edge = [n for n, r in rates.items() if r in (ADAMW_RATES[0], ADAMW_RATES[-1])]
                print(f"  {experiment}/{condition:<9} " + "  ".join(
                    f"{n} {r:g}" for n, r in rates.items())
                    + (f"   <- grid edge: {', '.join(edge)}" if edge else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2, 3, 4])
    parser.add_argument("--experiment", nargs="+", choices=list(EXPERIMENTS),
                        default=list(EXPERIMENTS), help="which experiments to backfill")
    parser.add_argument("--lr", action="store_true", help="tune AdamW per condition")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="20 steps at one seed, into _smoke-suffixed runs")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(args.experiment, suffix)
        return 0
    missing = _missing_sources(args.experiment)
    if missing:
        print(f"  REFUSED: source cell(s) not finished: {missing}")
        return 1
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)
    if args.lr:
        return tune(args, train, test, args.experiment, suffix)

    unswept = [lr_run_name(e, c, r, suffix) for e in args.experiment
               for c in EXPERIMENTS[e][1] for r in ADAMW_RATES
               if not any((ROOT / "results" / lr_run_name(e, c, r, suffix) / m).exists()
                          for m in ("_complete", "_diverged"))]
    if unswept:
        print(f"  REFUSED: {len(unswept)} lr cell(s) never ran, e.g. {unswept[0]}. Run --lr first.")
        return 1

    cells = [(e, c) for e in args.experiment for c in EXPERIMENTS[e][1]]
    print(f"AdamW pass{' SMOKE' if suffix else ''}: {len(cells)} cells\n", flush=True)
    status = load_status()
    started = time.time()
    for index, (experiment, condition) in enumerate(cells, start=1):
        sources = EXPERIMENTS[experiment][1][condition]
        rates = selected_rates(experiment, condition, suffix)
        entries = [{"name": n, "lr": rates[n]} for n in ADAMW]
        entries.append(reproduction_entry(source_config(sources)))
        name = cell_name(experiment, condition, suffix)
        note = run_one(config_for(sources, name, entries, args, suffix), train, test, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<28} {note:<26} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nAdamW pass complete in {(time.time() - started) / 60:.1f} min\n")
    report(args.experiment, suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _pooled(sources: list[str], learner: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for cell in sources:
        out.update(per_seed(cell, learner))
    return out


def report(experiments: list[str], suffix: str = "") -> None:
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 steps at one seed. The merge gate cannot hold here -- the source")
        print("  cells ran the full horizon -- so it is reported, not trusted.\n")
    for experiment in experiments:
        onehop, conditions = EXPERIMENTS[experiment]
        print(f"\n  ===== {experiment} =====")
        print("\n  merge gate: the recorded SGD learner, re-run in the AdamW cell, per seed")
        gates = {}
        for condition, sources in conditions.items():
            arm = reproduction_entry(source_config(sources))["name"]
            recorded = _pooled(sources, arm)
            rerun = per_seed(cell_name(experiment, condition, suffix), arm)
            shared = sorted(set(recorded) & set(rerun))
            worst = max((abs(recorded[s] - rerun[s]) for s in shared), default=None)
            gates[condition] = worst is not None and worst <= REPRODUCTION_TOLERANCE
            shown = f"{worst:.1e}" if worst is not None else "-"
            verdict = ("reproduces" if gates[condition] else "not run" if worst is None
                       else "DOES NOT REPRODUCE")
            print(f"    {condition:<10} {arm:<26} {len(shared)} seed(s)  max |diff| "
                  f"{shown:>8}   {verdict}")

        print("\n  settled error, current set")
        print(f"    {'condition':<10}{'one-hop':>9}{'central EKF':>12}{'cent AdamW':>11}"
              f"{'ATC AdamW':>10}{'local AdamW':>12}{'local only':>11}")
        for condition, sources in conditions.items():
            cell = cell_name(experiment, condition, suffix)
            values = [mean(_pooled(sources, onehop)),
                      mean(_pooled(sources, "centralized_ekf_gamma")),
                      mean(per_seed(cell, "centralized_adamw")),
                      mean(per_seed(cell, ATC_ADAMW)),
                      mean(per_seed(cell, "local_adamw")),
                      mean(_pooled(sources, "local_only"))]
            print(f"    {condition:<10}" + "".join(f"{v:>11.4f}" for v in values))

        rows = [(condition, compare(_pooled(sources, onehop),
                                    per_seed(cell_name(experiment, condition, suffix), ATC_ADAMW)))
                for condition, sources in conditions.items() if gates[condition]]
        withheld = [c for c in conditions if not gates[c]]
        print(f"\n  CONFIRMATORY (D127): {onehop} minus ATC AdamW, per seed; negative = the")
        print("  filter wins. Predicted negative. Holm across the gated conditions.")
        if withheld:
            print(f"  withheld (merge gate): {', '.join(withheld)}")
        print(f"\n    {'':<12}{STAT_HEADER}")
        live = [(c, r) for c, r in rows if r.n]
        for (condition, result), p_adj in zip(live, holm([r.p for _c, r in live]), strict=True):
            print(f"    {condition:<12}{stat_columns(result, p_adj)}")
    print(f"\n  * = p_holm < {ALPHA} (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
