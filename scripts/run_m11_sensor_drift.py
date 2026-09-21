r"""M11 -- the other two drift channels: the sensor moves, not the law.

    python scripts/run_m11_sensor_drift.py --smoke        # the path, not the numbers
    python scripts/run_m11_sensor_drift.py --device cuda  # the battery
    python scripts/run_m11_sensor_drift.py --report-only

Every Mackey--Glass cell so far drifts $\beta$, which moves the **law**: the dynamics
themselves change, and a filter that tracks it is tracking a moving system.
`env/series.py` implements two further channels, `gain` and `bias`, and both move the
**sensor** instead -- the law is untouched and only the observation is rescaled or
offset. That is a different kind of non-stationarity, nearer the image task's prior
drift than its rotation.

## Why it earns a run

In the filter's own language a sensor drift is a mis-specified $\boldsymbol R$ rather
than a mis-specified state: the innovation acquires a systematic component the
observation model does not explain, and no amount of process noise is the right
answer to it. So M11 stresses precisely the part of the design M4 tuned and M6 never
varied. It closes D106's gap 5 -- *"every claim here is a claim about a drifting law,
not about drift in general"*.

It also makes a prediction worth recording before the run: the one-hop advantage
should **shrink** here. One-hop wins on $\beta$ drift because neighbours' raw batches
carry information about the moving law; under a sensor drift every agent's sensor
moves identically, so a neighbour's batch is not more informative about it.

## The span, and why it is 0.1 on both channels

`channel_value` gives ``bias = span * f`` and ``gain = 1 + span * f``, where ``f`` is
the schedule's displacement as a fraction of the 45-degree cap. The series is carried
in **standardised units**, so $x$ has unit spread and a gain excess of $g$ perturbs
the observation by about $g$. ``span = 0.1`` therefore moves the sensor by **one
observation-noise standard deviation** ($\sigma = 0.1$) at full displacement on both
channels, which makes them comparable to each other by construction.

⚠ That rests on the standardisation giving unit spread. M0 measured it, but its report
is no longer in `results/`, so the assumption is **printed, not trusted**: the runner
reports the realised channel value at the cap and the perturbation it implies. If that
lands far from $\sigma$, re-run with ``--span``.

## What it deliberately does not re-tune

Filters carry M4's and M5's selections and gradient learners carry M3's rates, exactly
as M6 does, so these are the same learners M6 ran. **Rates are carried by schedule**:
the linear cells take M3's `linear` rates, the abrupt cells its `abrupt` rates.
Re-tuning per channel would answer a different question, and would break the pairing.

**The stationary twin is M6's own.** `m6_stationary_a` shares the learners, seeds,
horizon, R scale and graph and differs in the drift alone, so damage is paired against
it per seed and no fifth cell is needed.
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

#: cell suffix -> experiment config. The schedule half of the name also picks which
#: of M3's per-condition rate sets the gradient learners carry.
CONDITIONS = {
    "gain_linear": "m_gain_linear",
    "gain_abrupt": "m_gain_abrupt",
    "bias_linear": "m_bias_linear",
    "bias_abrupt": "m_bias_abrupt",
}
RATES_FOR = {"gain_linear": "linear", "gain_abrupt": "abrupt",
             "bias_linear": "linear", "bias_abrupt": "abrupt"}

#: Three filters and three gradient baselines, not fourteen arms. The question is the
#: channel, not covariance sharing, so the full-sharing variants and the gamma
#: reference arms stay out: they would double the cost for curves that answer a
#: question M6 already answered on this task.
FILTERS = ["centralized_ekf_walk", "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
BASELINES = ["centralized_adamw", "diffusion_atc_adamw", "local_only"]

#: The twin every cell's damage is paired against -- M6's own stationary cell.
TWIN = "m6_stationary_a"

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 25
STATUS = ROOT / "results" / "m11_status.json"
M3_RATES = ROOT / "results" / "m3_rates.json"
M4_SELECTION = ROOT / "results" / "m4_selection.json"
M5_SELECTION = ROOT / "results" / "m5_selection.json"

SMOKE_FILTER = {"transition": "scalar", "gamma": 1.0, "forgetting": "process_noise",
                "process_noise_q": 1e-5, "lambda_forget": 1.0, "prior_scale": 0.01}
SMOKE_RATES = {**{n: 3e-5 for n in SGD_FAMILY}, **{n: 3e-3 for n in ADAMW_FAMILY}}


def cell_name(condition: str) -> str:
    return f"m11_{condition}"


def as_filter(selection: dict) -> dict:
    return {
        "transition": "scalar", "gamma": selection["gamma"],
        "forgetting": "process_noise", "process_noise_q": selection["process_noise_q"],
        "lambda_forget": 1.0, "prior_scale": selection["prior_scale"],
    }


def load_settings(smoke: bool):
    """(centralised filter, diffusion filter, rates-by-schedule), or exit saying what is missing."""
    if smoke:
        return SMOKE_FILTER, SMOKE_FILTER, {"linear": SMOKE_RATES, "abrupt": SMOKE_RATES}
    missing = [p.name for p in (M3_RATES, M4_SELECTION, M5_SELECTION) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing {missing}: M11 carries M6's tuned settings so its cells are the\n"
            "same learners M6 ran. Run the tuning chain first."
        )
    m4 = json.loads(M4_SELECTION.read_text(encoding="utf-8"))
    m5 = json.loads(M5_SELECTION.read_text(encoding="utf-8"))
    rates = json.loads(M3_RATES.read_text(encoding="utf-8"))
    by_schedule = {s: {name: pick["lr"] for name, pick in rates.get(s, {}).items()}
                   for s in ("linear", "abrupt")}
    return as_filter(m4), as_filter(m5), by_schedule


def entries(condition: str, centralised: dict, diffusion: dict, rates: dict) -> list[dict]:
    """Assigned by name, never by position (the mis-assignment D106 caught in M6)."""
    learners = [{"name": "centralized_ekf_walk", **centralised}]
    learners += [{"name": n, **diffusion} for n in FILTERS[1:]]
    available = rates[RATES_FOR[condition]]
    for name in BASELINES:
        family = SGD_FAMILY if name in SGD_FAMILY else ADAMW_FAMILY
        if name in available and name in family:
            learners.append({"name": name, "lr": available[name], **family[name]})
    return learners


def r_scale_override(smoke: bool) -> dict:
    """M4 tuned the level of R; M11 carries it, exactly as M6 does."""
    if smoke or not M4_SELECTION.exists():
        return {}
    scale = json.loads(M4_SELECTION.read_text(encoding="utf-8"))["r_scale"]
    base = list(load_config("m_stationary").model.observation_variances)
    return {"observation_variances": [v * scale for v in base]}


def announce_span(condition: str) -> None:
    """Print the realised drift at the cap, so the span is checked rather than assumed."""
    series = load_config(CONDITIONS[condition]).env.series
    from dekf_bench.env.series import channel_value  # noqa: PLC0415

    at_cap = float(channel_value(series, 45.0))
    start = float(channel_value(series, 0.0))
    perturbation = abs(at_cap - start) if series.channel == "bias" else abs(at_cap - 1.0)
    print(f"  {condition}: {series.channel} {start:.4f} -> {at_cap:.4f} at the cap, "
          f"a perturbation of {perturbation:.4f} = {perturbation / series.sigma:.2f} sigma")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--smoke", action="store_true", help="short horizon, one seed")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--span", type=float, default=None,
                        help="override the channel span on every cell (default: the "
                             "config's 0.1, one observation-noise sd)")
    args = parser.parse_args(argv)
    if args.report_only:
        # smoke passed through: `--report-only --smoke` must read the smoke cells, not
        # silently report the real ones (and print dashes when they have not been run).
        report(smoke=args.smoke)
        return 0

    centralised, diffusion, rates = load_settings(args.smoke)
    horizon = 60 if args.smoke else args.horizon
    seeds = [0] if args.smoke else args.seeds
    model_override = r_scale_override(args.smoke)

    print(f"M11{' SMOKE' if args.smoke else ''}: {len(CONDITIONS)} cells x {len(seeds)} "
          f"seeds, T={horizon}, device {args.device}\n", flush=True)
    for condition in CONDITIONS:
        announce_span(condition)
    print(flush=True)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, condition in enumerate(CONDITIONS, start=1):
        name = cell_name(condition) + ("_smoke" if args.smoke else "")
        learners = entries(condition, centralised, diffusion, rates)
        overrides = {
            "run": {"name": name, "horizon": horizon, "seeds": seeds,
                    "eval_every": 20 if args.smoke else EVAL_EVERY,
                    "device": args.device, "dtype": args.dtype},
            **({"model": model_override} if model_override else {}),
            "learners": learners,
        }
        if args.span is not None:
            overrides["env"] = {"series": {"span": args.span}}
        note = run_one(load_config(CONDITIONS[condition], overrides=overrides),
                       None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(CONDITIONS) - index) if ran else 0.0
        print(f"[{index}/{len(CONDITIONS)}] {name:<22} {len(learners):>2} learners  "
              f"{note:<14} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM11{' smoke' if args.smoke else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(smoke=args.smoke)
    return 0


def _by_seed(cell: str, learner: str) -> dict[int, float]:
    """Settled RMSE per seed. Pooling first would bias the paired difference."""
    import pandas as pd  # noqa: PLC0415

    out: dict[int, float] = {}
    directory = ROOT / "results" / cell
    for path in sorted(directory.glob("seed_*.parquet")):
        frame = pd.read_parquet(path)
        rows = frame[(frame["learner"] == learner) & (frame["metric"] == "rmse")
                     & (frame["evalset"] == "current")]
        if rows.empty:
            continue
        rows = rows[rows["t"] >= int(0.8 * rows["t"].max())]
        if len(rows):
            out[int(path.stem.split("_")[1])] = float(rows["value"].mean())
    return out


def report(smoke: bool = False) -> None:
    suffix = "_smoke" if smoke else ""
    arms = [*FILTERS, *BASELINES]

    print("  settled RMSE on the held-out current set\n")
    print(f"    {'learner':<40}" + "".join(f"{c:>14}" for c in CONDITIONS))
    for learner in arms:
        row = f"    {learner:<40}"
        for condition in CONDITIONS:
            value = settled(cell_name(condition) + suffix, learner)
            row += f"{value:>14.4f}" if value != float("inf") else f"{'-':>14}"
        print(row)

    if smoke:
        print("\n  Smoke numbers mean nothing: 60 rounds, one seed, placeholder settings.")
        return

    print(f"\n  damage, paired per seed against {TWIN} (positive = the drift hurt)\n")
    print(f"    {'learner':<40}" + "".join(f"{c:>14}" for c in CONDITIONS))
    for learner in arms:
        base = _by_seed(TWIN, learner)
        row = f"    {learner:<40}"
        for condition in CONDITIONS:
            now = _by_seed(cell_name(condition), learner)
            shared = sorted(set(base) & set(now))
            if not shared:
                row += f"{'-':>14}"
                continue
            mean = sum(now[s] - base[s] for s in shared) / len(shared)
            row += f"{mean:>+14.4f}"
        print(row)
    # ASCII only in anything printed: this console is cp1252 and a bare "warning"
    # glyph raises UnicodeEncodeError mid-report, after two tables have already been
    # written. The docstrings above keep their symbols -- those are never encoded to
    # the terminal.
    print("\n  NOTE: absolute damage above carries the TWIN's held-out draw, which is")
    print("  drawn per (seed, channel value). One seed's twin draw being easy inflates")
    print("  damage in every cell at once, so the SIGN of a small damage is not")
    print("  trustworthy at five seeds (D108). The comparison below cancels it exactly.\n")

    print("  channel against channel, paired per seed (the twin cancels)\n")
    pairs = [("bias_linear", "gain_linear"), ("gain_linear", None),
             ("bias_abrupt", "gain_abrupt"), ("gain_abrupt", None)]
    print(f"    {'comparison':<34}{'learner':<38}{'diff':>10}{'SE':>7}")
    for left, right in pairs:
        sched = "linear" if left.endswith("linear") else "abrupt"
        right_cell = cell_name(right) if right else f"m6_{sched}_a"
        label = f"{left} - {right or 'beta ' + sched}"
        for learner in arms:
            a, b = _by_seed(cell_name(left), learner), _by_seed(right_cell, learner)
            shared = sorted(set(a) & set(b))
            if not shared:
                continue
            diffs = [a[s] - b[s] for s in shared]
            mean = sum(diffs) / len(diffs)
            sd = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
            se = abs(mean) / (sd / len(diffs) ** 0.5) if sd else float("inf")
            print(f"    {label:<34}{learner:<38}{mean:>+10.4f}{se:>7.1f}")
        print()
    print("  Negative means the left channel costs less. D108: bias < gain < beta on")
    print("  the linear schedule, unanimous across learners and seeds; under abrupt")
    print("  gain and beta are indistinguishable, because both reflect at the cap.")


if __name__ == "__main__":
    raise SystemExit(main())
