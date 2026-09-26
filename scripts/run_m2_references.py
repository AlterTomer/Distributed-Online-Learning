r"""M2 -- the Mackey--Glass reference lines, at every beta level the drift visits.

    python scripts/run_m2_references.py              # all 13 levels, ~13 min CPU
    python scripts/run_m2_references.py --report-only
    python scripts/run_m2_references.py --levels 0.22 --seeds 0 1 2 3 4   # e* with an error bar

**Seeds** (added 2026-09-26). M2 trained one reference per level, so $e^\star$ carried no
error bar, and D97 measured its run-to-run noise only indirectly -- 0.0015, the residual
about the fitted line. `--seeds` retrains at independent seeds: `ReferenceSettings.seed`
drives the initialisation, the minibatch order and the train, validation and test
draws, and is part of the cache key, so seed 0 is M2's own run and the others are new.
Written to `results/m2_reference_seeds.json`; the default path is unchanged.

`docs/mackey_glass_plan.md`, M2 and decision 17. For each beta level: the offline
reference $e^\star(\beta)$ (the Transformer trained to convergence on one run's
whole data budget at that fixed law, epoch chosen on validation), persistence, and
the noise floor. Cached under ``data/reference/mackey_glass/`` by everything that
determines the answer, so re-running is free and a changed law retrains.

**The levels.** The drift runs inside the chaotic window $[0.20, 0.24]$ around
$\beta_0=0.22$ (D96). The abrupt schedule's jumps are a third of the span, 1/150
in beta, so every value it can reach is a multiple of 1/150 from 0.22; the linear
schedule sweeps the upper half continuously. A grid of 1/300 covers the first
exactly (every second level) and the second at twice that resolution: 13 levels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dekf_bench.evaluation.series_reference import (  # noqa: E402
    REFERENCE_DIR,
    ReferenceSettings,
    train_reference,
)
from dekf_bench.utils.config import load_config  # noqa: E402

LEVELS = [round(0.20 + k / 300, 6) for k in range(13)]
OUT = ROOT / "results" / "m2_references.json"
SEEDS_OUT = ROOT / "results" / "m2_reference_seeds.json"


def series_config():
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": "m2_references", "seeds": [0]},
            "env": {"dataset": "mackey_glass"},
            "model": {"name": "causal_transformer", "likelihood": "gaussian", "output_dim": 31},
        },
    ).env.series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--levels", type=float, nargs="+", default=None,
                        help="beta levels to train at (default: all 13)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="reference seeds; written to m2_reference_seeds.json")
    args = parser.parse_args(argv)

    import torch  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    series = series_config()
    if args.seeds is not None:
        return seeded(series, args.levels or LEVELS, args.seeds)
    results = {}
    started = time.time()
    for index, beta in enumerate(LEVELS, start=1):
        if args.report_only and not any(REFERENCE_DIR.glob("*.json")):
            break
        score = train_reference(series, beta)
        results[str(beta)] = score.as_dict()
        print(
            f"[{index}/{len(LEVELS)}] beta {beta:.4f}  e* {score.rmse:.4f}  "
            f"full-context {score.rmse_full_context:.4f}  persistence {score.persistence_rmse:.4f}"
            f"  floor {score.noise_floor}  epoch {score.best_epoch}  "
            f"{(time.time() - started) / 60:.1f} min",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"written {OUT}")
    return 0


def seeded(series, levels: list[float], seeds: list[int]) -> int:
    """e* at each level and seed, with the spread across seeds -- its error bar."""
    import statistics  # noqa: PLC0415

    results: dict[str, dict] = {}
    started = time.time()
    for beta in levels:
        scores = {}
        for seed in seeds:
            score = train_reference(series, beta, ReferenceSettings(seed=seed))
            scores[str(seed)] = score.as_dict()
            print(f"  beta {beta:.4f} seed {seed}  e* {score.rmse:.4f}  epoch "
                  f"{score.best_epoch}  {(time.time() - started) / 60:.1f} min", flush=True)
        values = [s["rmse"] for s in scores.values()]
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        se = sd / len(values) ** 0.5 if len(values) > 1 else float("nan")
        line = 0.0816 + 0.274 * beta  # D97's fitted line, for comparison
        results[str(beta)] = {"seeds": scores, "mean": mean, "sd": sd, "se": se,
                              "d97_line": line}
        print(f"  beta {beta:.4f}: e* mean {mean:.4f}  sd {sd:.4f}  se {se:.4f}  "
              f"(D97 line {line:.4f}; seed 0 is M2's own run)\n", flush=True)
    SEEDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    SEEDS_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"written {SEEDS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
