r"""X28 -- does covariance sharing still buy nothing at the receiver point?

    python scripts/run_receiver_full.py              # 2 cells x 5 seeds, ~2.5 h
    python scripts/run_receiver_full.py --report-only

## The question

X25 found the two repairs to be **substitutes**: with a one-hop adapt step, full
covariance sharing bought $+0.0009$ (not significant) under severe skew while
costing $1\,145\times$ the bandwidth -- where for the *local*-adapt filter the same
sharing was worth $-0.0314$ ($t=-4.61$). The conclusion was that once fresh
neighbour evidence enters the adapt step, the neighbours' accumulated uncertainty
has nothing left to add.

**That was measured at the sender point.** X27 has since shown the receiver point
is a different update -- better in all five cells, and the one the paper now
carries (D99) -- so a result about what one-hop *leaves for* covariance sharing
cannot simply be inherited. If it holds, mean-only stays the deployable variant on
the strength of a measurement rather than an analogy; if it does not, full sharing
becomes live again under skew, and that would change the headline.

## What runs, and why only two cells

`diffusion_ekf_onehop_receiver` -- one-hop, **full** sharing, receiver point -- at
$\beta_c=1$, on the ends of X25's skew axis. $\beta_c=2$ is not run: D88 and X25
both priced it as ruinous, and skew did not make it live.

Its mean-only counterpart is **already on disk from X27** at these exact conditions
and seeds. The data stream depends on the configuration and the seed, never on
which learners are attached (X26 established this to twelve decimals), so pairing
X28's cell against X27's by seed is exact and halves the work.

One full-sharing filter per cell, in its own process: X24 measured that variant at
3.43 GiB of 8.
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
from run_diffusion_skew import (  # noqa: E402
    FILTER,
    config_for,
    group_b_name,  # noqa: E402
)
from run_diffusion_skew import cell_name as x25_cell  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402
from run_linearization_point import paired, per_seed  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402

SKEWS = [0.1, 100.0]
SEEDS = [0, 1, 2, 3, 4]
HORIZON = 1500

FULL_RECEIVER = "diffusion_ekf_onehop_receiver"
MEAN_RECEIVER = "diffusion_ekf_onehop_mean_receiver"
FULL_SENDER = "diffusion_ekf_onehop"
MEAN_SENDER = "diffusion_ekf_onehop_mean"

DATA_ROOT = ROOT / "data"
STATUS = ROOT / "results" / "x28_status.json"


def cell_name(skew: float) -> str:
    return f"x28_full_receiver_b{skew:g}".replace(".", "p")


def x27_cell(skew: float) -> str:
    return f"x27_still_b{skew:g}"


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

    print(f"\nX28: {len(SKEWS)} cells at {len(args.seeds)} seeds, T={args.horizon}")
    for skew in SKEWS:
        print(f"  {cell_name(skew):<30} skew {skew:<6g} {FULL_RECEIVER}  (paired against "
              f"{x27_cell(skew)})")
    print(flush=True)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, skew in enumerate(SKEWS, start=1):
        name = cell_name(skew)
        entries = [{"name": FULL_RECEIVER, **FILTER, "combine_exponent": 1.0}]
        note = run_one(config_for(args, name, skew, None, entries, seeds=args.seeds),
                       train, test, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(SKEWS) - index) if ran else 0.0
        print(f"[{index}/{len(SKEWS)}] {name:<30} {note:<28} {elapsed:.0f} min, "
              f"~{remaining:.0f} left", flush=True)

    print(f"\nX28 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    """Full minus mean-only, at each point, paired by seed."""
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    print("  does full sharing add anything on top of a one-hop adapt?")
    print(f"    {'skew':>6}{'point':>10}{'full':>10}{'mean-only':>11}{'diff':>10}{'t':>7}{'n':>4}")
    for skew in SKEWS:
        for label, full_run, full_learner, mean_run, mean_learner in (
            ("receiver", cell_name(skew), FULL_RECEIVER, x27_cell(skew), MEAN_RECEIVER),
            ("sender", group_b_name(FULL_SENDER, skew, 1.0), FULL_SENDER,
             x25_cell(f"still_b{skew:g}"), MEAN_SENDER),
        ):
            full, mean_only = per_seed(full_run, full_learner), per_seed(mean_run, mean_learner)
            diff, t, n = paired(full, mean_only)
            if not full:
                print(f"    {skew:>6g}{label:>10}{'-':>10}")
                continue
            print(f"    {skew:>6g}{label:>10}{mean(full):>10.4f}{mean(mean_only):>11.4f}"
                  f"{diff:>+10.4f}{t:>7.2f}{n:>4}")
    print("\n  Positive diff = full sharing is WORSE. X25 measured +0.0009 (ns) at the")
    print("  sender point under severe skew, against -0.0314 for the local-adapt filter;")
    print("  if the receiver row agrees, mean-only stays the deployable variant.")


if __name__ == "__main__":
    raise SystemExit(main())
