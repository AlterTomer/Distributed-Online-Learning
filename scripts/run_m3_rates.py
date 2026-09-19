r"""M3 -- learning rates for every gradient baseline on Mackey--Glass.

    python scripts/run_m3_rates.py                  # 15 cells x 2 seeds, CPU by default
    python scripts/run_m3_rates.py --report-only    # the selections, from disk

`docs/mackey_glass_plan.md`, M3 (the X17/X26 analogue). Each baseline gets its own
rate **per condition**, never one borrowed from another -- D77 and D90 both record
what a borrowed rate costs.

## The two families, and why their grids differ by ~700x

The SGD family minimises the Gaussian NLL through the shared gradient: the score
summed over 31 positions and divided by $\bm R$. That is $31/(2R)\approx700\times$ a
per-value squared-error gradient at this task's $\bm R$ (centre $2.2\times10^{-2}$,
D96), so its rates sit ~700x below the pilot's MSE-scale ones (0.01--0.03): the grid
here spans them with a decade of margin each way. An MSE-scale rate diverged in the
first version of the M1 test. **AdamW** is close to scale-invariant, so its grid is
the pilot's, extended upward because M0 chose $10^{-2}$, the top of its grid, at
every noise level.

A cell carries all seven learners, each at the $k$-th rate of its own family, so a
condition costs five runs rather than 35.

Defaults to the **CPU** so it never contends with a GPU sweep; ``--device cuda``
when the GPU is free.
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

from _args import sweep_parser  # noqa: E402
from run_ekf_generalization import run_one  # noqa: E402

from dekf_bench.utils.config import load_config  # noqa: E402

CONDITIONS = {"stationary": "m_stationary", "linear": "m_linear", "abrupt": "m_abrupt"}

SGD_FAMILY = {
    "centralized_sgd": {"optimizer": "sgd_momentum", "momentum": 0.9},
    "diffusion_sgd_atc": {"optimizer": "sgd_momentum", "momentum": 0.9,
                          "mix_optimizer_state": "momentum"},
    "diffusion_sgd_atc_plain": {"optimizer": "sgd", "momentum": 0.0,
                                "mix_optimizer_state": "none"},
    "local_only": {"optimizer": "sgd", "momentum": 0.0, "mix_optimizer_state": "none"},
}
ADAMW_FAMILY = {"centralized_adamw": {}, "diffusion_atc_adamw": {}, "local_adamw": {}}

SGD_RATES = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
ADAMW_RATES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1], 25
STATUS = ROOT / "results" / "m3_status.json"


def cell_name(condition: str, index: int) -> str:
    return f"m3_{condition}_r{index}"


def entries(index: int) -> list[dict]:
    return [{"name": n, "lr": SGD_RATES[index], **o} for n, o in SGD_FAMILY.items()] + [
        {"name": n, "lr": ADAMW_RATES[index], **o} for n, o in ADAMW_FAMILY.items()
    ]


def rate_of(learner: str, index: int) -> float:
    return (ADAMW_RATES if learner in ADAMW_FAMILY else SGD_RATES)[index]


def settled(run: str, learner: str, metric: str = "rmse", evalset: str = "current") -> float:
    """Mean over agents of the metric on the held-out current set, last 20% of the run."""
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    files = sorted(directory.glob("seed_*.parquet"))
    if not (directory / "_complete").exists() or not files:
        return float("inf")
    frame = pd.concat(
        [pd.read_parquet(f, columns=["learner", "metric", "evalset", "t", "value"]) for f in files],
        ignore_index=True,
    )
    rows = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                 & (frame["evalset"] == evalset) & (frame["t"] >= int(0.8 * frame["t"].max()))]
    value = float(rows["value"].mean()) if len(rows) else float("inf")
    # A diverged regression learner records NaN rather than raising -- which keeps
    # the other learners in its cell, unlike MNIST, where one divergence discards
    # the cell. But NaN compares False both ways, so min() could select it.
    # Non-finite is unusable, and inf is how unusable is spelled here.
    return value if math.isfinite(value) else float("inf")


def selections() -> dict[str, dict[str, dict]]:
    """Each learner's argmin rate per condition, flagged when it sits on a grid edge."""
    chosen: dict[str, dict[str, dict]] = {}
    for condition in CONDITIONS:
        chosen[condition] = {}
        for learner in [*SGD_FAMILY, *ADAMW_FAMILY]:
            scored = [(settled(cell_name(condition, k), learner), k) for k in range(len(SGD_RATES))]
            finite = [(v, k) for v, k in scored if v != float("inf")]
            if not finite:
                continue
            value, index = min(finite)
            chosen[condition][learner] = {
                "lr": rate_of(learner, index),
                "settled_rmse": value,
                "edge": index in (0, len(SGD_RATES) - 1),
                "curve": {rate_of(learner, k): v for v, k in scored},
            }
    return chosen


def report() -> None:
    chosen = selections()
    for condition, learners in chosen.items():
        print(f"\n  {condition}")
        for learner, pick in learners.items():
            flag = "  <- EDGE: extend the grid" if pick["edge"] else ""
            print(f"    {learner:<26} lr {pick['lr']:<8g} rmse {pick['settled_rmse']:.4f}{flag}")
    (ROOT / "results" / "m3_rates.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS, device="cpu")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.report_only:
        report()
        return 0

    import torch  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    cells = [(c, k) for c in CONDITIONS for k in range(len(SGD_RATES))]
    print(f"M3: {len(cells)} cells x {len(args.seeds)} seeds, T={args.horizon}, "
          f"device {args.device}\n", flush=True)
    started, ran = time.time(), 0
    for index, (condition, k) in enumerate(cells, start=1):
        name = cell_name(condition, k)
        config = load_config(
            CONDITIONS[condition],
            overrides={
                "run": {"name": name, "horizon": args.horizon, "seeds": args.seeds,
                        "eval_every": EVAL_EVERY, "device": args.device, "dtype": args.dtype},
                "learners": entries(k),
            },
        )
        note = run_one(config, None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<22} sgd {SGD_RATES[k]:<7g} adamw {ADAMW_RATES[k]:<6g}"
              f" {note:<28} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM3 complete in {(time.time() - started) / 60:.1f} min")
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
