r"""M6 -- the main comparison on Mackey--Glass: every method, three conditions.

    python scripts/run_m6_comparison.py --smoke      # ~30 min CPU: the path, not the numbers
    python scripts/run_m6_comparison.py              # the battery (GPU)
    python scripts/run_m6_comparison.py --report-only

The headline cells. Stationary is also the twin every drifting cell's damage is
measured against; linear and abrupt move the law inside the chaotic window (D96).

**Filters carry tuned settings, baselines carry tuned rates.** The centralised
filter's four knobs come from M4, the diffusion filter's from M5, the gradient
learners' rates from M3 -- per condition, as M3 selected them. The run refuses to
start if any of those are missing, because a filter carrying a guessed setting is
exactly the confound X20 spent ten hours discovering (D77, D90).

**Two groups, as the image task does.** The full-sharing variants run in their own
cells: not for memory here -- $\bm P$ is 41 MB per agent at $p=2273$, an order
below the image task -- but so that a cell's learners share one process's fate, and
a divergence in the expensive variant cannot cost the cheap ones their run.

``--smoke`` runs a short horizon at one seed with placeholder settings. It proves
the path end to end -- every learner builds, the regression scoring and the
predictive calibration produce rows, nothing runs out of memory -- and its numbers
mean nothing.
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
from run_ekf_generalization import run_one  # noqa: E402
from run_m3_rates import ADAMW_FAMILY, SGD_FAMILY, settled  # noqa: E402

from dekf_bench.utils.config import load_config  # noqa: E402

CONDITIONS = {"stationary": "m_stationary", "linear": "m_linear", "abrupt": "m_abrupt"}

#: Mean-only filters and every gradient baseline share a cell.
GROUP_A_FILTERS = ["centralized_ekf_gamma", "diffusion_ekf",
                   "diffusion_ekf_onehop_mean_receiver"]
#: Full sharing gets its own cell (see the module docstring).
GROUP_B_FILTERS = ["diffusion_ekf_full", "diffusion_ekf_onehop_receiver"]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 25
STATUS = ROOT / "results" / "m6_status.json"
M3_RATES = ROOT / "results" / "m3_rates.json"
M4_SELECTION = ROOT / "results" / "m4_selection.json"
M5_SELECTION = ROOT / "results" / "m5_selection.json"

SMOKE_FILTER = {"transition": "scalar", "gamma": 1.0, "forgetting": "process_noise",
                "process_noise_q": 1e-5, "lambda_forget": 1.0, "prior_scale": 0.01}
SMOKE_RATES = {**{n: 3e-5 for n in SGD_FAMILY}, **{n: 3e-3 for n in ADAMW_FAMILY}}


def cell_name(condition: str, group: str) -> str:
    return f"m6_{condition}_{group}"


def load_settings(smoke: bool):
    """(centralised filter, diffusion filter, rates-by-condition), or exit saying what is missing."""
    if smoke:
        return SMOKE_FILTER, SMOKE_FILTER, {c: SMOKE_RATES for c in CONDITIONS}
    missing = [p.name for p in (M3_RATES, M4_SELECTION, M5_SELECTION) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing {missing}: M6 carries tuned settings only.\n"
            "  python scripts/run_m3_rates.py        (baseline rates)\n"
            "  python scripts/run_m4_centralised.py  (centralised filter)\n"
            "  python scripts/run_m5_diffusion.py    (diffusion filter)"
        )
    m4 = json.loads(M4_SELECTION.read_text(encoding="utf-8"))
    m5 = json.loads(M5_SELECTION.read_text(encoding="utf-8"))
    rates = json.loads(M3_RATES.read_text(encoding="utf-8"))
    as_filter = lambda s: {  # noqa: E731
        "transition": "scalar", "gamma": s["gamma"], "forgetting": "process_noise",
        "process_noise_q": s["process_noise_q"], "lambda_forget": 1.0,
        "prior_scale": s["prior_scale"],
    }
    by_condition = {c: {learner: pick["lr"] for learner, pick in rates.get(c, {}).items()}
                    for c in CONDITIONS}
    return as_filter(m4), as_filter(m5), by_condition


def entries(group: str, condition: str, centralised, diffusion, rates) -> list[dict]:
    if group == "b":
        return [{"name": n, **diffusion, "combine_exponent": 1.0} for n in GROUP_B_FILTERS]
    learners = [{"name": "centralized_ekf_gamma", **centralised}]
    learners += [{"name": n, **diffusion} for n in GROUP_A_FILTERS[1:]]
    learners += [{"name": n, "lr": rates[condition][n], **o} for n, o in SGD_FAMILY.items()
                 if n in rates[condition]]
    learners += [{"name": n, "lr": rates[condition][n], **o} for n, o in ADAMW_FAMILY.items()
                 if n in rates[condition]]
    return learners


def r_scale_override(smoke: bool) -> dict:
    """M4 tunes the level of R; M6 carries it."""
    if smoke or not M4_SELECTION.exists():
        return {}
    scale = json.loads(M4_SELECTION.read_text(encoding="utf-8"))["r_scale"]
    base = list(load_config(CONDITIONS["stationary"]).model.observation_variances)
    return {"observation_variances": [v * scale for v in base]}


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS, device="cpu")
    parser.add_argument("--smoke", action="store_true", help="short horizon, one seed")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.report_only:
        report()
        return 0

    import torch  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    centralised, diffusion, rates = load_settings(args.smoke)
    horizon = 60 if args.smoke else args.horizon
    seeds = [0] if args.smoke else args.seeds
    model_override = r_scale_override(args.smoke)

    cells = [(condition, group) for condition in CONDITIONS for group in ("a", "b")]
    print(f"M6{' SMOKE' if args.smoke else ''}: {len(cells)} cells x {len(seeds)} seeds, "
          f"T={horizon}, device {args.device}\n", flush=True)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, (condition, group) in enumerate(cells, start=1):
        name = cell_name(condition, group) + ("_smoke" if args.smoke else "")
        learners = entries(group, condition, centralised, diffusion, rates)
        config = load_config(
            CONDITIONS[condition],
            overrides={
                "run": {"name": name, "horizon": horizon, "seeds": seeds,
                        "eval_every": 20 if args.smoke else EVAL_EVERY,
                        "device": args.device, "dtype": args.dtype},
                **({"model": model_override} if model_override else {}),
                "learners": learners,
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
        print(f"[{index}/{len(cells)}] {name:<28} {len(learners):>2} learners  {note:<14}"
              f" {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM6{' smoke' if args.smoke else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(smoke=args.smoke)
    return 0


def report(smoke: bool = False) -> None:
    suffix = "_smoke" if smoke else ""
    every = [*GROUP_A_FILTERS, *GROUP_B_FILTERS, *SGD_FAMILY, *ADAMW_FAMILY]
    print("  settled RMSE on the held-out current set")
    print(f"    {'learner':<34}" + "".join(f"{c:>13}" for c in CONDITIONS))
    for learner in every:
        row = f"    {learner:<34}"
        for condition in CONDITIONS:
            values = [settled(cell_name(condition, g) + suffix, learner) for g in ("a", "b")]
            values = [v for v in values if v != float("inf")]
            row += f"{values[0]:>13.4f}" if values else f"{'-':>13}"
        print(row)
    if smoke:
        print("\n  Smoke numbers mean nothing: 60 rounds, one seed, placeholder settings.")
        print("  What it checks is that every learner builds, scores and records.")


if __name__ == "__main__":
    raise SystemExit(main())
