r"""P5.12 -- do the diffusion filters' agents converge to the centralised filter, or only to each other?

    python scripts/run_disagreement.py                 # six cells, ~8 GPU-h
    python scripts/run_disagreement.py --report-only
    python scripts/run_disagreement.py --smoke         # the path, on CPU

No `--lr`: every learner carries its source cell's recorded entry.

## Why this is a run and not "analysis only"

`phase5_plan.md` P5.12 asks whether the filters' agents converge, and whether "to the
centralised one or merely to each other". $E_{\text{agree}}$ is recorded everywhere and
answers the first half. The second half is where the recorded data fails:
$E_{\text{cent}}$ measures every learner against **centralised SGD**
(`REFERENCE_LEARNER` in `simulate.py`), so for a diffusion filter it is the distance
to a different method's trajectory. Parameters are not saved per step, so the right
distance cannot be recovered from disk. `simulate.py` now also records
`e_cent_filter` -- the same distance, to the pooled filter -- for every learner that
holds a covariance ([[D134]]). This runner re-runs the filters where they need it.

**The split that makes it readable.** Exactly,
$E_{\text{cent}}^{\text{filter}} = E_{\text{agree}} + \lVert\bar{\boldsymbol\theta}-\boldsymbol\theta^{\text{C}}\rVert^2$:
distance to the centralised filter is disagreement among the agents plus the
**consensus offset**, how far the network's mean sits from the centralised belief.
P5.12's "merely to each other" is an offset that dominates the disagreement.

## Cells

Three conditions, each rebuilt from finished cells' recorded configs with only the
learners and the name changed:

* **stationary** -- P5.3's ER 0.3 cells (`p53_er030_a/b`), $T=1500$, evaluations every 5;
* **global drift** and **per-node drift** -- P5.7's twin pair (`p57_global_a/b`,
  `p57_per_node_a/b`), linear, evaluations every 25. [[D115]] found diffusion agents
  "never leave consensus" under per-node drift; this measures how far from it they sit.

Cell **a** holds the centralised filter and both mean-only filters, recorded entries
verbatim. Cell **b** holds both full-sharing filters **plus the centralised filter**,
which the source b cells lacked, so their `e_cent_filter` has its reference.

**Gate.** Every learner here ran in its source cell with the same config, and a
learner's trajectory does not depend on its companions (`test_simulate.py`), so each
settled error must reproduce its source per seed within $10^{-9}$. The confirmatory
tables are withheld for any condition that fails.

## Named before the run (confirmatory under D118)

All on settled values (last 20%), per seed, for the two mean-only filters:

1. **Consensus dominates.** Offset minus $E_{\text{agree}}$ -- that is,
   $E_{\text{cent}}^{\text{filter}} - 2E_{\text{agree}}$ -- for each filter in each
   condition: six rows, Holm, **predicted positive**. The agents sit closer to one
   another than their consensus sits to the centralised filter.
2. **One-hop's consensus is nearer.** One-hop's offset minus local adapt's, in each
   condition: three rows, Holm, **predicted negative**. One-hop assimilates its
   neighbours' batches, which local adapt only reaches through the mean.
3. **Heterogeneous drift pulls agents apart.** $E_{\text{agree}}$ under per-node minus
   global drift, per filter: two rows, Holm, **predicted positive** -- and small,
   given D115.

Exploratory: the full-sharing rows, the time course (quintiles, normalised by the
centralised filter's $\lVert\boldsymbol\theta^{\text{C}}\rVert^2$), `max_pairwise_distance`,
and the recorded $E_{\text{cent}}$ against SGD beside the new one.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import per_seed  # noqa: E402
from run_p57_heterogeneous_drift import ALPHA, STAT_HEADER, stat_columns  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.metrics.paired import differences, holm, summarise  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

SOURCES = {
    "stationary": {"a": "p53_er030_a", "b": "p53_er030_b"},
    "global": {"a": "p57_global_a", "b": "p57_global_b"},
    "per_node": {"a": "p57_per_node_a", "b": "p57_per_node_b"},
}
CENTRAL = "centralized_ekf_gamma"
LOCAL, ONEHOP = "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"
MEAN_ONLY = [LOCAL, ONEHOP]
FULL_SHARING = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]
TOLERANCE = 1e-9
SMOKE_HORIZON, SMOKE_EVAL_EVERY = 20, 5
DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "dis_status.json"


def cell_name(condition: str, group: str, suffix: str = "") -> str:
    return f"dis_{condition}_{group}{suffix}"


def recorded(cell: str) -> dict:
    return yaml.safe_load((ROOT / "results" / cell / "config.yaml").read_text(encoding="utf-8"))


def entries_for(condition: str, group: str) -> list[dict]:
    """The source's recorded filter entries; b gains the centralised filter as its reference."""
    a = {e["name"]: dict(e) for e in recorded(SOURCES[condition]["a"])["learners"]}
    if group == "a":
        return [a[CENTRAL], *(a[n] for n in MEAN_ONLY)]
    b = {e["name"]: dict(e) for e in recorded(SOURCES[condition]["b"])["learners"]}
    return [a[CENTRAL], *(b[n] for n in FULL_SHARING)]


def config_for(condition: str, group: str, args, suffix: str):
    run = {"name": cell_name(condition, group, suffix), "device": args.device}
    if suffix:
        run.update({"horizon": SMOKE_HORIZON, "eval_every": SMOKE_EVAL_EVERY, "seeds": [0]})
    source = ROOT / "results" / SOURCES[condition][group] / "config.yaml"
    return load_config(source, overrides={"run": run,
                                          "learners": entries_for(condition, group)})


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=1500, seeds=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true", help="20 steps at one seed, on CPU")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.device = "cpu"
    if args.report_only:
        report(suffix)
        return 0
    missing = [c for groups in SOURCES.values() for c in groups.values()
               if not (ROOT / "results" / c / "_complete").exists()]
    if missing:
        print(f"  REFUSED: P5.12 is rebuilt from {missing}, which is not finished.")
        return 1
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)

    cells = [(c, g) for c in SOURCES for g in ("a", "b")]
    print(f"P5.12{' SMOKE' if suffix else ''}: {len(cells)} cells, device {args.device}\n",
          flush=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started = time.time()
    for index, (condition, group) in enumerate(cells, start=1):
        config = config_for(condition, group, args, suffix)
        cell_started = time.time()
        note = run_one(config, train, test, args.fresh)
        status[config.run.name] = note
        if note != "cached":
            status[f"{config.run.name}_minutes"] = round((time.time() - cell_started) / 60, 1)
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(cells)}] {config.run.name:<24} {len(config.learners)} learners  "
              f"{note:<14} {(time.time() - started) / 60:.0f} min", flush=True)
    print(f"\nP5.12{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _group(learner: str) -> str:
    return "b" if learner in FULL_SHARING else "a"


def _metric(condition: str, learner: str, metric: str, suffix: str) -> dict[int, float]:
    return per_seed(cell_name(condition, _group(learner), suffix), learner, metric)


def _offset(condition: str, learner: str, suffix: str) -> dict[int, float]:
    """||mean - centralised filter||^2 = E_cent_filter - E_agree, per seed."""
    return differences(_metric(condition, learner, "e_cent_filter", suffix),
                       _metric(condition, learner, "e_agree", suffix))


def _family(rows: list[tuple[str, dict[int, float]]]) -> None:
    results = [(label, summarise(values)) for label, values in rows]
    live = [(label, r) for label, r in results if r.n >= 2]
    print(f"\n    {'':<34}{STAT_HEADER}")
    for (label, result), p_adj in zip(live, holm([r.p for _l, r in live]), strict=True):
        print(f"    {label:<34}{stat_columns(result, p_adj)}")
    for label, result in results:
        if result.n < 2:
            print(f"    {label:<34}{'-':>9}")


def _pick(window, learner: str, metric: str) -> float:
    return window[(window.learner == learner) & (window.metric == metric)].value.mean()


def _time_course(condition: str, learner: str, suffix: str) -> list[tuple[float, float, float]]:
    """(E_agree, offset) over quintiles of t, normalised by ||theta_C||^2, seed-averaged."""
    import pandas as pd  # noqa: PLC0415

    def rows(cell: str) -> pd.DataFrame:
        files = sorted((ROOT / "results" / cell).glob("seed_*.parquet"))
        if not files:
            return pd.DataFrame()
        frame = pd.concat([pd.read_parquet(f, columns=["learner", "metric", "t", "value"])
                           for f in files], ignore_index=True)
        return frame

    ours = rows(cell_name(condition, _group(learner), suffix))
    if ours.empty:
        return []
    horizon = ours.t.max() + 1
    out = []
    for q in range(5):
        lo, hi = q * horizon / 5, (q + 1) * horizon / 5
        window = ours[(ours.t >= lo) & (ours.t < hi)]
        norm = _pick(window, CENTRAL, "theta_mean_norm_sq")
        agree = _pick(window, learner, "e_agree")
        cent = _pick(window, learner, "e_cent_filter")
        out.append((hi, agree / norm, (cent - agree) / norm))
    return out


def report(suffix: str = "") -> None:
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    if suffix:
        print("  SMOKE: 20 steps at one seed, against full-horizon sources, so the gate is")
        print("  expected to fail. The path is the check.\n")

    print("  gate: each learner's settled error against its source cell, per seed")
    gated = []
    for condition in SOURCES:
        ok = True
        for group in ("a", "b"):
            for entry in entries_for(condition, group):
                name = entry["name"]
                source = SOURCES[condition]["a" if name == CENTRAL else group]
                ours = per_seed(cell_name(condition, group, suffix), name)
                theirs = per_seed(source, name)
                shared = sorted(set(ours) & set(theirs))
                worst = max((abs(ours[s] - theirs[s]) for s in shared), default=float("nan"))
                good = bool(shared) and worst <= TOLERANCE
                ok &= good
                print(f"    {condition:<11}{group}  {name:<36} {worst:.1e} over {len(shared)} "
                      f"seed(s)  {'ok' if good else 'FAILS'}")
        if ok:
            gated.append(condition)

    print("\n  settled, per learner: E_agree, the consensus offset, and E_cent against SGD")
    print("  (recorded before, for comparison), all divided by the centralised filter's")
    print("  ||theta_C||^2")
    print(f"    {'condition':<11}{'learner':<36}{'E_agree':>10}{'offset':>10}{'vs SGD':>10}")
    for condition in SOURCES:
        norm = mean(_metric(condition, CENTRAL, "theta_mean_norm_sq", suffix))
        for learner in [*MEAN_ONLY, *FULL_SHARING]:
            agree = mean(_metric(condition, learner, "e_agree", suffix))
            offset = mean(_offset(condition, learner, suffix))
            # Recorded in the *source* cell, which carried centralized_sgd; the rebuilt
            # cells hold filters only. P5.3's and P5.7's b cells had no SGD reference.
            vs_sgd = (mean(per_seed(SOURCES[condition]["a"], learner, "e_cent"))
                      if learner in MEAN_ONLY else float("nan"))
            shown = f"{vs_sgd / norm:>10.2e}" if vs_sgd == vs_sgd else f"{'-':>10}"
            print(f"    {condition:<11}{learner:<36}{agree / norm:>10.2e}{offset / norm:>10.2e}"
                  f"{shown}")
    if suffix:
        return

    withheld = [c for c in SOURCES if c not in gated]
    if withheld:
        print(f"\n  withheld (gate): {withheld}")
    print("\n  CONFIRMATORY (D134). Q1: offset minus E_agree, per seed; positive = the")
    print("  agents agree with one another more than their consensus agrees with the")
    print("  centralised filter. Predicted positive. Holm across the rows.")
    _family([(f"{c}, {name}", differences(_offset(c, learner, ""),
                                         _metric(c, learner, "e_agree", "")))
             for c in gated for name, learner in (("local", LOCAL), ("one-hop", ONEHOP))])
    print("\n  Q2: one-hop's offset minus local adapt's; predicted negative. Holm.")
    _family([(c, differences(_offset(c, ONEHOP, ""), _offset(c, LOCAL, ""))) for c in gated])
    if {"global", "per_node"} <= set(gated):
        print("\n  Q3: E_agree under per-node minus global drift; predicted positive. Holm.")
        _family([(name, differences(_metric("per_node", learner, "e_agree", ""),
                                    _metric("global", learner, "e_agree", "")))
                 for name, learner in (("local", LOCAL), ("one-hop", ONEHOP))])

    print("\n  EXPLORATORY: the time course, quintiles of t, normalised by ||theta_C||^2")
    for condition in gated:
        for learner in [*MEAN_ONLY, *FULL_SHARING]:
            course = _time_course(condition, learner, "")
            if course:
                print(f"    {condition:<11}{learner:<36}E_agree " + " ".join(
                    f"{a:.1e}" for _t, a, _o in course))
                print(f"    {'':<11}{'':<36}offset  " + " ".join(
                    f"{o:.1e}" for _t, _a, o in course))
    print(f"\n  * = p_holm < {ALPHA} (D113).")


if __name__ == "__main__":
    raise SystemExit(main())
