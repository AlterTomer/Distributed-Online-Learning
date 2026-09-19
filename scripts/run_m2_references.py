r"""M2 -- the Mackey--Glass reference lines, at every beta level the drift visits.

    python scripts/run_m2_references.py              # all 13 levels, ~13 min CPU
    python scripts/run_m2_references.py --report-only

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

from dekf_bench.evaluation.series_reference import REFERENCE_DIR, train_reference  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

LEVELS = [round(0.20 + k / 300, 6) for k in range(13)]
OUT = ROOT / "results" / "m2_references.json"


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
    args = parser.parse_args(argv)

    import torch  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    series = series_config()
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


if __name__ == "__main__":
    raise SystemExit(main())
