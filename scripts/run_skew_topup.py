r"""X25+ -- X25's skew cells topped up to five seeds, with `atc_plain` beside them.

    python scripts/run_skew_topup.py                # seeds 3 and 4, ~4.5 h GPU
    python scripts/run_skew_topup.py --report-only  # the five-seed tables, from disk

Tier 1 of `docs/schedule.md`. X25 ran three seeds, and its sharpest claim sat on
them: under severe skew the local-adapt filter loses to `atc_plain` at equal
bandwidth by 0.0073 at $t=1.66$ -- suggestive, not established (X26). This adds
seeds 3 and 4 so every skew comparison is five-seed, like the IID ones.

## Why new cells rather than re-running X25's

A finished cell is cached, so asking X25's cells for five seeds would run nothing;
re-running all five would repeat hours of filters already on disk. The data stream
depends only on the configuration and the seed -- never on which learners are
attached -- so seeds 3 and 4 run in new cells with X25's exact settings, and the
report pools them with X25's seeds 0--2. X26 established the same property for
`atc_plain` (twelve decimals).

## What runs

* **The still cells** ($\beta_{\mathrm{dir}}\in\{0.1,1,100\}$): X25's group A at
  X25's selected rates -- the centralised filter, the local-adapt filter, the
  one-hop filter (sender point), centralised SGD, momentum ATC, local only -- plus
  `atc_plain` at X26's selected rate for that skew.
* **Full sharing at $\beta_c=1$** at the ends of the skew axis, as X25's group B;
  $\beta_c=2$ is not topped up, having lost by 0.13 at every skew.

The drifting pair at $\beta_{\mathrm{dir}}=0.1$ is not topped up: the schedule names
the still cells, where the equal-bandwidth claim lives.

## A reproduction check that costs nothing

X27 ran the sender-point one-hop filter on these same still cells at seeds 0--4.
The top-up carries it too, so on seeds 3 and 4 the two must agree **exactly** --
the same guarantee X27 gave against X25, extended to the new cells.
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
from run_atc_plain import cell_name as x26_cell  # noqa: E402
from run_atc_plain import selected_rate as x26_rate  # noqa: E402
from run_diffusion_skew import (  # noqa: E402
    BASELINES,
    CENTRALIZED,
    FILTER,
    GROUP_B,
    GROUP_B_SKEWS,
    SKEWS,
    config_for,
    group_b_name,
    selected_rates,
)
from run_diffusion_skew import cell_name as x25_cell  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import paired, per_seed  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402

SEEDS = [3, 4]
HORIZON = 1500
GROUP_A = ["diffusion_ekf", "diffusion_ekf_onehop_mean"]
DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x25p_status.json"


def condition(skew: float) -> str:
    return f"still_b{skew:g}"


def x26_condition(skew: float) -> str:
    return f"skew_b{skew:g}".replace(".", "p")


def topup_cell(skew: float) -> str:
    return f"x25p_{condition(skew)}"


def topup_group_b(learner: str, skew: float) -> str:
    return "x25p_" + group_b_name(learner, skew, 1.0).removeprefix("x25_")


def still_entries(skew: float) -> list[dict]:
    rates = selected_rates(condition(skew))
    plain = x26_rate(x26_condition(skew))
    if rates is None or plain is None:
        raise SystemExit(f"no selected rate on disk for {condition(skew)}: run X25 --lr and X26 --lr")
    return (
        [{"name": "centralized_ekf_gamma", **CENTRALIZED}]
        + [{"name": n, **FILTER} for n in GROUP_A]
        + [{"name": n, "lr": rates[n], **o} for n, o in BASELINES.items()]
        + [{**PLAIN, "lr": plain}]
    )


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.report_only:
        report()
        return 0
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)

    cells = [(topup_cell(s), s, still_entries(s)) for s in SKEWS]
    cells += [
        (topup_group_b(learner, s), s, [{"name": learner, **FILTER, "combine_exponent": 1.0}])
        for learner in GROUP_B
        for s in GROUP_B_SKEWS
    ]
    print(f"\nX25+: {len(cells)} cells at seeds {args.seeds}, T={args.horizon}")
    for name, skew, entries in cells:
        print(f"  {name:<34} skew {skew:<6g} {len(entries)} learners")
    print(flush=True)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, (name, skew, entries) in enumerate(cells, start=1):
        note = run_one(config_for(args, name, skew, None, entries, seeds=args.seeds),
                       train, test, args.fresh)
        status[name] = note
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<34} {note:<28} {elapsed:.0f} min, "
              f"~{remaining:.0f} left", flush=True)

    print(f"\nX25+ complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def pooled(*sources: tuple[str, str]) -> dict[int, float]:
    """Per-seed settled error, merged across runs; a seed in two runs must agree."""
    merged: dict[int, float] = {}
    for run, learner in sources:
        merged.update(per_seed(run, learner))
    return merged


def report() -> None:
    """Five-seed skew tables, the equal-bandwidth claim, and the reproduction check."""
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    print("  settled error by skew, seeds pooled (X25 0-2 + X25+ 3-4)")
    learners = ["centralized_ekf_gamma", *GROUP_A, *BASELINES]
    print(f"    {'skew':>6}" + "".join(f"{n.replace('diffusion_', 'd_'):>24}" for n in learners)
          + f"{'atc_plain':>12}")
    for skew in SKEWS:
        line = f"    {skew:>6g}"
        for learner in learners:
            seeds = pooled((x25_cell(condition(skew)), learner), (topup_cell(skew), learner))
            line += f"{mean(seeds):>20.4f} ({len(seeds)})"
        plain = pooled((x26_cell(x26_condition(skew)), PLAIN["name"]),
                       (topup_cell(skew), PLAIN["name"]))
        line += f"{mean(plain):>8.4f} ({len(plain)})"
        print(line)

    print("\n  equal bandwidth: diffusion_ekf minus atc_plain, paired over pooled seeds")
    for skew in SKEWS:
        ekf = pooled((x25_cell(condition(skew)), "diffusion_ekf"),
                     (topup_cell(skew), "diffusion_ekf"))
        plain = pooled((x26_cell(x26_condition(skew)), PLAIN["name"]),
                       (topup_cell(skew), PLAIN["name"]))
        diff, t, n = paired(ekf, plain)
        print(f"    skew {skew:<6g} {diff:+.4f}  t={t:.2f}  n={n}")

    print("\n  full sharing at beta_c = 1, pooled; minus mean-only local adapt")
    for skew in GROUP_B_SKEWS:
        for learner in GROUP_B:
            full = pooled((group_b_name(learner, skew, 1.0), learner),
                          (topup_group_b(learner, skew), learner))
            base = pooled((x25_cell(condition(skew)), "diffusion_ekf"),
                          (topup_cell(skew), "diffusion_ekf"))
            diff, t, n = paired(full, base)
            print(f"    {learner:<22} skew {skew:<6g} {mean(full):.4f}  vs local {diff:+.4f}"
                  f"  t={t:.2f}  n={n}")

    print("\n  reproduction: one-hop sender, X25+ against X27 on seeds 3-4 (must be 0)")
    for skew in SKEWS:
        top = per_seed(topup_cell(skew), "diffusion_ekf_onehop_mean")
        x27 = per_seed(f"x27_{condition(skew)}", "diffusion_ekf_onehop_mean")
        shared = sorted(set(top) & set(x27))
        worst = max((abs(top[s] - x27[s]) for s in shared), default=float("nan"))
        print(f"    skew {skew:<6g} max |diff| {worst:.1e} over {len(shared)} seeds")


if __name__ == "__main__":
    raise SystemExit(main())
