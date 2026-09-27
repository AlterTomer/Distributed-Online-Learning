r"""M7 -- the bracket's prediction on Mackey--Glass: beta_c = 2 against 1, full sharing.

    python scripts/run_m7_combine_exponent.py --device cuda   # three cells, ~3.5 GPU-h
    python scripts/run_m7_combine_exponent.py --report-only
    python scripts/run_m7_combine_exponent.py --smoke         # the path, not the numbers

`docs/mackey_glass_plan.md` M7, the X24 analogue; `schedule.md` B7.

## The question

$\beta_c$ (`combine_exponent`) scales the covariance combine of the two full-sharing
variants: $\beta_c=1$ is `lem:conservative`, which assumes the neighbours' errors
coincide; $\beta_c=2$ assumes them independent and divides the combined covariance by
about $|\mathcal M_v|$. On MNIST $\beta_c=2$ lost heavily -- $+0.134$ for full sharing,
$+0.040$ for one-hop full -- and by the same amount on IID and severely skewed shards
([[D89]]). The conclusion drawn there was structural: **the agents' errors coincide
because they mix at every step, not because their data are alike**, so $\beta_c=1$ is
right for any diffusion algorithm. That claim is about diffusion, not about images, and
this is its test on a second task: if $\beta_c>1$ won here, the MNIST result would be a
property of the task rather than of mixing.

## The design

$\beta_c=1$ is M6's group-B cells (`m6_{condition}_b`), already run. M7 adds their
$\beta_c=2$ twins -- each M6 recorded config with `combine_exponent` 2 on both learners
and the name replaced, nothing else -- in all three of M6's conditions. No tuning: the
filters carry M5's selection, as the $\beta_c=1$ cells did, so the contrast is $\beta_c$
alone (the X14 discipline; $\alpha$ is not swept, having been rejected monotonically on
both adapt scopes, X22, and carried over without re-running).

**Pairing gate.** Each twin may differ from its M6 cell in the learners'
`combine_exponent` and the run name alone, compared on resolved configs.

## Named before the run (confirmatory under D118)

$\beta_c=2$ minus $\beta_c=1$, settled RMSE per seed, for `diffusion_ekf_full` and
`diffusion_ekf_onehop_receiver` in each of the three conditions: six rows, Holm across
six, **predicted positive** in every row -- $\beta_c=2$ loses, as on MNIST, and by more
for the local adapt than for one-hop (D89's ordering: one-hop already gathers what the
covariance would carry, so shrinking it matters less).

Exploratory: the calibration of the shrunken belief (`variance_ratio`, `coverage_90`),
which $\beta_c=2$ should make over-confident; divergences, counted per seed.
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
from run_m8_tau_heterogeneity import ALPHA, STAT_HEADER, _by_seed, _stat_columns  # noqa: E402
from run_m13_data_rate import _flatten, _metric_by_seed  # noqa: E402

from dekf_bench.metrics.paired import holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

CONDITIONS = ["stationary", "linear", "abrupt"]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
EXPONENT = 2.0
SMOKE_HORIZON, SMOKE_EVAL_EVERY = 20, 5
STATUS = ROOT / "results" / "m7_status.json"


def source(condition: str) -> str:
    return f"m6_{condition}_b"


def cell_name(condition: str, suffix: str = "") -> str:
    return f"m7_{condition}_b2{suffix}"


def recorded(condition: str) -> dict:
    return yaml.safe_load((ROOT / "results" / source(condition) / "config.yaml")
                          .read_text(encoding="utf-8"))


def entries_for(condition: str) -> list[dict]:
    out = [dict(e) for e in recorded(condition)["learners"]]
    names = sorted(e["name"] for e in out)
    if names != sorted(FULL_SHARING):
        raise SystemExit(f"{source(condition)} holds {names}, not the two full-sharing variants")
    for entry in out:
        entry["combine_exponent"] = EXPONENT
    return out


def config_for(condition: str, args, suffix: str):
    run = {"name": cell_name(condition, suffix), "device": args.device}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "eval_every": SMOKE_EVAL_EVERY, "seeds": [0]})
    return load_config(ROOT / "results" / source(condition) / "config.yaml",
                       overrides={"run": run, "learners": entries_for(condition)})


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true", help="20 rounds, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0
    missing = [source(c) for c in CONDITIONS
               if not (ROOT / "results" / source(c) / "_complete").exists()]
    if missing:
        print(f"  REFUSED: M7's beta_c = 1 arm is M6's group B, and {missing} is not finished.")
        return 1

    print(f"M7{' SMOKE' if suffix else ''}: {len(CONDITIONS)} cells at beta_c = {EXPONENT:g}, "
          f"device {args.device}\n", flush=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started = time.time()
    for index, condition in enumerate(CONDITIONS, start=1):
        name = cell_name(condition, suffix)
        cell_started = time.time()
        note = run_one(config_for(condition, args, suffix), None, None, args.fresh)
        status[name] = note
        if note != "cached":
            status[f"{name}_minutes"] = round((time.time() - cell_started) / 60, 1)
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(CONDITIONS)}] {name:<22} {note:<14} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nM7{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


def report(suffix: str = "") -> None:
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 rounds at one seed, against M6's full-horizon cells, so the")
        print("  gate and every contrast below are expected to fail. The path is the check.\n")

    # run.device is exempt: M6 recorded `cuda`, and a launch under `auto` on the same
    # GPU would otherwise withhold the confirmatory table over a label.
    print("  pairing gate: each beta_c = 2 cell against its M6 cell (resolved configs;")
    print("  only the learners' combine_exponent, the run name and device may differ)")
    gated = []
    for condition in CONDITIONS:
        path = ROOT / "results" / cell_name(condition, suffix) / "config.yaml"
        if not path.exists():
            print(f"    {condition:<11} not run")
            continue
        ours = _flatten(asdict(load_config(path)))
        theirs = _flatten(asdict(load_config(ROOT / "results" / source(condition) / "config.yaml")))
        extra = sorted(k for k in set(ours) | set(theirs)
                       if ours.get(k) != theirs.get(k) and k not in ("run.name", "run.device", "learners"))
        ours_learners = [{k: v for k, v in e.items() if k != "combine_exponent"}
                         for e in ours.get("learners", [])]
        theirs_learners = [{k: v for k, v in e.items() if k != "combine_exponent"}
                           for e in theirs.get("learners", [])]
        if ours_learners != theirs_learners:
            extra.append("learners (beyond combine_exponent)")
        if not extra:
            gated.append(condition)
        print(f"    {condition:<11} " + ("paired" if not extra else f"DIFFERS in {extra}"))

    print("\n  settled RMSE on the held-out current set, and seeds that diverged")
    print(f"    {'learner':<34}{'condition':<12}{'beta_c=1':>10}{'beta_c=2':>10}"
          f"{'diverged':>10}")
    for learner in FULL_SHARING:
        for condition in CONDITIONS:
            one = _by_seed(source(condition), learner)
            two = _by_seed(cell_name(condition, suffix), learner)
            finite = {s: v for s, v in two.items() if v == v and v != float("inf")}
            print(f"    {learner:<34}{condition:<12}{mean(one):>10.4f}{mean(finite):>10.4f}"
                  f"{len(two) - len(finite):>10}")
    if suffix:
        return

    rows = [(f"{learner.replace('diffusion_ekf_', '')}, {condition}",
             compare(_by_seed(cell_name(condition), learner), _by_seed(source(condition), learner)))
            for learner in FULL_SHARING for condition in gated]
    live = [(label, r) for label, r in rows if r.n >= 2]
    print("\n  CONFIRMATORY (D132): beta_c = 2 minus beta_c = 1, per seed; positive = the")
    print("  independence assumption costs. Predicted positive in every row, and larger for")
    print("  local adapt than for one-hop. Holm across the gated rows.")
    withheld = [c for c in CONDITIONS if c not in gated]
    if withheld:
        print(f"  withheld (pairing gate): {withheld}")
    print(f"\n    {'':<30}{STAT_HEADER}")
    for (label, result), p_adj in zip(live, holm([r.p for _l, r in live]), strict=True):
        print(f"    {label:<30}{_stat_columns(result, p_adj)}")

    print("\n  EXPLORATORY: calibration of the combined belief, settled, on current")
    print(f"    {'learner':<34}{'condition':<12}{'metric':<14}{'beta_c=1':>10}{'beta_c=2':>10}")
    for learner in FULL_SHARING:
        for condition in gated:
            for metric in ("variance_ratio", "coverage_90"):
                one = _metric_by_seed(source(condition), learner, metric)
                two = _metric_by_seed(cell_name(condition), learner, metric)
                print(f"    {learner:<34}{condition:<12}{metric:<14}{mean(one):>10.3f}"
                      f"{mean(two):>10.3f}")
    print(f"\n  * = p_holm < {ALPHA} (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
