r"""M12 -- two channels drifting at once: is the damage additive?

    python scripts/run_m12_combined_drift.py --smoke        # the path, not the numbers
    python scripts/run_m12_combined_drift.py --device cuda  # the battery, ~13 h
    python scripts/run_m12_combined_drift.py --report-only

M6 drifted $\beta$ alone and M11 drifted `gain` and `bias` alone. Each is a single
channel, and D108 found they are not equally damaging: **bias < gain < β**, with a
scalar offset absorbed outright. This run moves **two at once** and asks the question
those three make precise.

## The hypothesis, and why it is a real null

`D(A+B) = D(A) + D(B)`. Additivity is not a formality here: D106 found the
communication and adapt-scope terms decomposed *exactly* additively
($0.0038+0.0035=0.0073$, in all three conditions), so the same shape has held once
already on this task.

**The prediction on record is that it fails, for one specific pair.** β and gain both
move the *observed amplitude* -- measured in D108, β at 0.24 raises the signal's
spread by $1.1386$ and gain at 1.1 by $1.0989$ -- so from the learner's side they are
partially **unidentifiable**: it cannot tell "the law sped up" from "the sensor
gained". So:

* **β+gain correlated** should be **super-additive** (the amplitude errors compound);
* **β+gain anti** should be **sub-additive**, possibly cheaper than either channel
  alone, because the net amplitude change is only ~1.035 -- the two nearly cancel;
* **β+bias** and **gain+bias** should **add cleanly**, because bias shifts the mean and
  leaves the spread at exactly $1.0000$. They are the controls.

If bias pairs add and gain+β does not, the cause is identifiability rather than
"more drift is worse", which is what makes this worth nine cells.

## The three couplings, and why `independent` needs a stochastic schedule

* `correlated` -- one displacement drives both channels.
* `anti` -- the secondary takes the negated displacement. **On a deterministic
  schedule this is the only meaningful contrast to `correlated`**: a linear ramp
  follows one path however it is seeded, so "independent" there would silently
  duplicate `correlated`. The environment refuses that pairing outright.
* `independent` -- the secondary draws its own jumps, **per run seed**, for the reason
  D98 records: a pinned jump seed makes a run's realised mean law a fixed offset from
  its twin's (measured 0.2111 against 0.22, a 12-degree spread across draws), so the
  contrast would measure a draw rather than the coupling.

## ⚠ The additivity residual cannot cancel the twin

D108's channel ladder was safe because cell-minus-cell cancels the stationary twin
exactly. **This one cannot.** Expanding the residual,

    D(A+B) − D(A) − D(B)  =  AB − A − B + twin

leaves one irreducible twin term. That term carries the held-out draw noise D108
measured (sd ≈ 0.0033 on a single draw, ≈ 0.0013 after five seeds), so **only
interactions well above ~0.0013 are readable**, and a residual near zero is "no
detectable interaction" rather than "exactly additive". The report prints the residual
with its own per-seed spread so this is visible rather than assumed.

## What it carries unchanged

Filters take M4's and M5's selections, gradient learners M3's rates by schedule, and
the twin is `m6_stationary_a` -- the same six learners, seeds, horizon, R scale and
graph as M6 and M11, so `D(A)` and `D(B)` are the numbers those runs already
measured rather than anything re-estimated here.
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
from run_m11_sensor_drift import BASELINES, FILTERS, TWIN, _by_seed, as_filter  # noqa: E402

from dekf_bench.utils.config import load_config  # noqa: E402

#: (primary, secondary, coupling) -> the experiment config. Nine cells: three
#: channel pairs against three couplings, with `independent` on the abrupt schedule
#: because a deterministic one cannot express it.
PAIRS = [("beta", "gain"), ("beta", "bias"), ("gain", "bias")]
COUPLINGS = ("correlated", "anti", "independent")
CONDITIONS = {
    f"{a}_{b}_{c}": f"m_comb_{a}_{b}_{c}"
    for a, b in PAIRS
    for c in COUPLINGS
}
#: Which of M3's per-condition rate sets each cell carries, by its schedule.
RATES_FOR = {name: ("abrupt" if name.endswith("independent") else "linear")
             for name in CONDITIONS}

#: The single-channel damages this run's additivity test is built on, by channel and
#: schedule. Filled from the cells M6 and M11 already produced -- never re-estimated.
SINGLE = {"beta": {"linear": "m6_linear_a", "abrupt": "m6_abrupt_a"},
          "gain": {"linear": "m11_gain_linear", "abrupt": "m11_gain_abrupt"},
          "bias": {"linear": "m11_bias_linear", "abrupt": "m11_bias_abrupt"}}

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 25
STATUS = ROOT / "results" / "m12_status.json"
M3_RATES = ROOT / "results" / "m3_rates.json"
M4_SELECTION = ROOT / "results" / "m4_selection.json"
M5_SELECTION = ROOT / "results" / "m5_selection.json"

SMOKE_FILTER = {"transition": "scalar", "gamma": 1.0, "forgetting": "process_noise",
                "process_noise_q": 1e-5, "lambda_forget": 1.0, "prior_scale": 0.01}
SMOKE_RATES = {**{n: 3e-5 for n in SGD_FAMILY}, **{n: 3e-3 for n in ADAMW_FAMILY}}


def cell_name(condition: str) -> str:
    return f"m12_{condition}"


def load_settings(smoke: bool):
    """(centralised filter, diffusion filter, rates-by-schedule), or exit saying what is missing."""
    if smoke:
        return SMOKE_FILTER, SMOKE_FILTER, {"linear": SMOKE_RATES, "abrupt": SMOKE_RATES}
    missing = [p.name for p in (M3_RATES, M4_SELECTION, M5_SELECTION) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing {missing}: M12 carries M6's and M11's settings so its damages are\n"
            "the same quantity theirs are. Run the tuning chain first."
        )
    m4 = json.loads(M4_SELECTION.read_text(encoding="utf-8"))
    m5 = json.loads(M5_SELECTION.read_text(encoding="utf-8"))
    rates = json.loads(M3_RATES.read_text(encoding="utf-8"))
    by_schedule = {s: {n: pick["lr"] for n, pick in rates.get(s, {}).items()}
                   for s in ("linear", "abrupt")}
    return as_filter(m4), as_filter(m5), by_schedule


def entries(condition: str, centralised: dict, diffusion: dict, rates: dict) -> list[dict]:
    """The same six learners M11 ran, assigned by name rather than position."""
    learners = [{"name": "centralized_ekf_walk", **centralised}]
    learners += [{"name": n, **diffusion} for n in FILTERS[1:]]
    available = rates[RATES_FOR[condition]]
    for name in BASELINES:
        family = SGD_FAMILY if name in SGD_FAMILY else ADAMW_FAMILY
        if name in available and name in family:
            learners.append({"name": name, "lr": available[name], **family[name]})
    return learners


def r_scale_override(smoke: bool) -> dict:
    """M4 tuned the level of R; M12 carries it, exactly as M6 and M11 do."""
    if smoke or not M4_SELECTION.exists():
        return {}
    scale = json.loads(M4_SELECTION.read_text(encoding="utf-8"))["r_scale"]
    base = list(load_config("m_stationary").model.observation_variances)
    return {"observation_variances": [v * scale for v in base]}


def announce(condition: str) -> None:
    """Print the range each channel can reach, so the spans are checked not assumed.

    ⚠ These are the values at the cap, not a realised path. Under a reflecting
    schedule the channel explores BOTH signs, and under `independent` the secondary
    draws its own jumps -- so for those the cap is an envelope and printing a single
    endpoint would read as a trajectory it never follows.
    """
    from dekf_bench.env.series import channel_value, secondary_value  # noqa: PLC0415

    series = load_config(CONDITIONS[condition]).env.series
    reflecting = load_config(CONDITIONS[condition]).env.drift.schedule == "recurring"
    rest_p, rest_s = float(channel_value(series, 0.0)), float(secondary_value(series, 0.0))

    def span_text(value_at, rest: float, label: str) -> str:
        high = float(value_at(45.0))
        if reflecting:  # explores +/- the cap and keeps returning
            return f"{label} {float(value_at(-45.0)):.4f} <-> {high:.4f}"
        return f"{label} {rest:.4f} -> {high:.4f}"

    primary = span_text(lambda d: channel_value(series, d), rest_p, series.channel)
    sign = -1.0 if series.coupling == "anti" else 1.0
    second = span_text(lambda d: secondary_value(series, sign * d), rest_s,
                       series.secondary_channel)
    note = "own draw" if series.coupling == "independent" else series.coupling
    print(f"  {condition:<26} {primary:<26} {second:<26} ({note})")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--smoke", action="store_true", help="short horizon, one seed")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--only", default=None,
                        help="run one cell, e.g. beta_gain_anti (default: all nine)")
    args = parser.parse_args(argv)
    if args.report_only:
        report(smoke=args.smoke)
        return 0

    centralised, diffusion, rates = load_settings(args.smoke)
    horizon = 60 if args.smoke else args.horizon
    seeds = [0] if args.smoke else args.seeds
    model_override = r_scale_override(args.smoke)
    wanted = [args.only] if args.only else list(CONDITIONS)
    unknown = [c for c in wanted if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown cell {unknown}; choose from {sorted(CONDITIONS)}")

    print(f"M12{' SMOKE' if args.smoke else ''}: {len(wanted)} cells x {len(seeds)} "
          f"seeds, T={horizon}, device {args.device}\n", flush=True)
    for condition in wanted:
        announce(condition)
    print(flush=True)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, condition in enumerate(wanted, start=1):
        name = cell_name(condition) + ("_smoke" if args.smoke else "")
        learners = entries(condition, centralised, diffusion, rates)
        overrides = {
            "run": {"name": name, "horizon": horizon, "seeds": seeds,
                    "eval_every": 20 if args.smoke else EVAL_EVERY,
                    "device": args.device, "dtype": args.dtype},
            **({"model": model_override} if model_override else {}),
            "learners": learners,
        }
        note = run_one(load_config(CONDITIONS[condition], overrides=overrides),
                       None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(wanted) - index) if ran else 0.0
        print(f"[{index}/{len(wanted)}] {name:<34} {len(learners):>2} learners  "
              f"{note:<14} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM12{' smoke' if args.smoke else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(smoke=args.smoke)
    return 0


def _damage(cell: str, learner: str, twin: dict[int, float]) -> dict[int, float]:
    """Per-seed damage of one cell against the shared twin."""
    now = _by_seed(cell, learner)
    return {s: now[s] - twin[s] for s in sorted(set(now) & set(twin))}


def _stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, (abs(mean) / (sd / n ** 0.5) if sd else float("inf"))


def report(smoke: bool = False) -> None:
    suffix = "_smoke" if smoke else ""
    arms = [*FILTERS, *BASELINES]

    print("  settled RMSE on the held-out current set\n")
    print(f"    {'learner':<38}" + "".join(f"{c[:11]:>13}" for c in CONDITIONS))
    for learner in arms:
        row = f"    {learner:<38}"
        for condition in CONDITIONS:
            value = settled(cell_name(condition) + suffix, learner)
            row += f"{value:>13.4f}" if value != float("inf") else f"{'-':>13}"
        print(row)

    if smoke:
        print("\n  Smoke numbers mean nothing: 60 rounds, one seed, placeholder settings.")
        return

    print("\n\n  ADDITIVITY:  D(A+B) - D(A) - D(B),  paired per seed")
    print("  positive = the pair costs MORE than its parts (they compound)")
    print("  negative = LESS (they mask, or cancel)\n")
    print("  NOTE: this residual is AB - A - B + twin, so one twin term survives. The")
    print("  held-out draw puts about 0.0013 of noise on it after five seeds (D108),")
    print("  so a residual near zero means no DETECTABLE interaction, not exact")
    print("  additivity. Read the sigma column, not the sign alone.\n")

    print(f"    {'cell':<26}{'learner':<38}{'residual':>10}{'SE':>7}")
    for condition in CONDITIONS:
        primary, secondary, _coupling = condition.split("_")
        schedule = RATES_FOR[condition]
        for learner in arms:
            twin = _by_seed(TWIN, learner)
            if not twin:
                continue
            both = _damage(cell_name(condition), learner, twin)
            alone_a = _damage(SINGLE[primary][schedule], learner, twin)
            alone_b = _damage(SINGLE[secondary][schedule], learner, twin)
            shared = sorted(set(both) & set(alone_a) & set(alone_b))
            if not shared:
                continue
            residual = [both[s] - alone_a[s] - alone_b[s] for s in shared]
            mean, se = _stats(residual)
            print(f"    {condition:<26}{learner:<38}{mean:>+10.4f}{se:>7.1f}")
        print()

    print("  The prediction on record (D108, and this script's docstring): beta+gain")
    print("  should be super-additive when correlated and SUB-additive when anti,")
    print("  because both channels move the observed amplitude and are partly")
    print("  unidentifiable. The bias pairs are the controls and should add cleanly.")


if __name__ == "__main__":
    raise SystemExit(main())
