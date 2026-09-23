r"""M8 -- per-agent delays: the agents disagree about the law itself.

    python scripts/run_m8_tau_heterogeneity.py --smoke        # the path, not the numbers
    python scripts/run_m8_tau_heterogeneity.py --device cuda  # the battery
    python scripts/run_m8_tau_heterogeneity.py --report-only

Decision 22's third heterogeneity axis, and the only one that moves the **law each
agent follows** rather than its sensor or its share of the data. Per-agent $\beta$
offsets and $\sigma_v$ remain for `m8_beta_spread` and `m8_sigma_spread`; this run
is the $\tau_v$ axis alone.

Every M-series cell so far gave all ten agents the same dynamics. Here each agent
integrates its own delay, so a neighbour's parameters are fitted to a *different*
attractor. That is a sharper test of diffusion than drift: under drift every agent
is wrong in the same direction at the same moment, and the network agrees about
where to go. Here they disagree permanently, and the disagreement has no
resolution -- no consensus exists that is right for everyone.

## Why these delays, and why not a tighter spread

$\tau$ is not a channel like $\beta$ or the sensor. Three measured facts set the
design, none of them a preference:

1. **The chaotic region is ragged.** 18.4--18.6 and 19.3--19.6 are
   *start-dependent*: some initial histories settle on a periodic attractor and
   others stay chaotic. Agents draw histories from $[0.5, 1.5]$, so a delay there
   would leave some agents chaotic and others periodic **inside one run**. 18.7 and
   19.2 are marginal (min $\lambda \approx 0.001$). All are skipped; the weakest
   value kept is 19.8 at 0.00298.
2. **A tight spread is a null by construction.** A model converged at one delay and
   evaluated at another disagrees by 0.0009--0.0015 across $\pm 0.6$ -- below the
   0.0037 floor five seeds have ever resolved -- and by 0.0082--0.0120 across
   16.4 to 20.0, inside the 0.0037--0.0146 band M12 did resolve. The narrow cell was
   measured, predicted null, and **not run**.
3. **$\tau = 17$ cannot be the centre.** Chaos ends just below 16.4, so a set wide
   enough to be detectable must run upward. Its mean is 18.21.

## The third cell, and why two would be a confound

Point 3 means the spread cell differs from the `m6_stationary_a` twin in *two* ways:
its agents disagree, and their mean delay is 1.2 higher. `m8_tau_control` holds every
agent at 18.2 -- the set's mean, which also matches its mean own-delay difficulty to
within 0.1 -- so the two separate:

    spread  - control = heterogeneity, at a matched mean delay   <- the headline
    control - twin    = the mean-delay shift alone
    spread  - twin    = their sum

All three are cell-minus-cell and paired per seed, so the twin's held-out draw
cancels. D108 found a single seed's twin draw inflating absolute damage in every
cell at once; D109 found a withdrawn damage re-entering an arithmetic that consumed
it. A contrast that never takes a damage as an *input* cannot inherit either fault.

The spread and control cells are scored on **byte-identical** held-out sets: both
configs set `tau: 18.2`, which is what `law_blocks` builds evalsets from, while
`tau_values` governs the agents. Verified with `torch.equal`, not assumed -- left at
the 17.0 default the spread cell's evalsets would sit at a delay no agent follows.

## The predictions, recorded before the run

Heterogeneity should cost *cooperation* specifically, so the ordering is the claim,
not the magnitude:

* **`local_only` should be hurt least** -- near zero. It never mixes, so it cannot be
  pulled toward a consensus that fits nobody. If it moves as much as the others, the
  effect is not about cooperation and the run says something else entirely.
* **The pooling arms should be hurt most.** A centralised learner fits one parameter
  vector to ten different attractors.
* **One-hop should be hurt more than mean-only diffusion.** One-hop consumes a
  neighbour's raw batch, and that batch now comes from a different law; mean-only
  mixes parameters, which is a milder commitment to a neighbour's world.

A null on all six arms would say the network is indifferent to whose law it is
learning, which at this spread would itself be worth recording.

## What it deliberately does not re-tune

Filters carry M4's and M5's selections and gradient learners carry M3's
**`stationary`** rates -- every cell here is stationary, so there is one rate set,
not M11's linear/abrupt split. Re-tuning at the heterogeneous delays would answer a
different question and would break the pairing against the twin.
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

#: cell suffix -> experiment config. Both are stationary, so unlike M11 there is no
#: schedule half of the name and one rate set serves both.
CONDITIONS = {
    "tau_spread": "m_tau_spread",
    "tau_control": "m_tau_control",
}
#: Every cell here is stationary; M3 selected rates per condition and this is the one.
RATES_CONDITION = "stationary"

#: The same three filters and three gradient baselines M11 ran, for the same reason:
#: the question is the delays, not covariance sharing, so the full-sharing variants
#: and the gamma reference arms stay out.
FILTERS = ["centralized_ekf_walk", "diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
BASELINES = ["centralized_adamw", "diffusion_atc_adamw", "local_only"]

#: M6's own stationary cell: homogeneous at tau = 17, and identical to these in
#: learners, seeds, horizon, R scale and graph.
TWIN = "m6_stationary_a"

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 25
STATUS = ROOT / "results" / "m8_status.json"
M3_RATES = ROOT / "results" / "m3_rates.json"
M4_SELECTION = ROOT / "results" / "m4_selection.json"
M5_SELECTION = ROOT / "results" / "m5_selection.json"

SMOKE_FILTER = {"transition": "scalar", "gamma": 1.0, "forgetting": "process_noise",
                "process_noise_q": 1e-5, "lambda_forget": 1.0, "prior_scale": 0.01}
SMOKE_RATES = {**{n: 3e-5 for n in SGD_FAMILY}, **{n: 3e-3 for n in ADAMW_FAMILY}}


def cell_name(condition: str) -> str:
    return f"m8_{condition}"


def as_filter(selection: dict) -> dict:
    return {
        "transition": "scalar", "gamma": selection["gamma"],
        "forgetting": "process_noise", "process_noise_q": selection["process_noise_q"],
        "lambda_forget": 1.0, "prior_scale": selection["prior_scale"],
    }


def load_settings(smoke: bool):
    """(centralised filter, diffusion filter, stationary rates), or exit saying what is missing."""
    if smoke:
        return SMOKE_FILTER, SMOKE_FILTER, SMOKE_RATES
    missing = [p.name for p in (M3_RATES, M4_SELECTION, M5_SELECTION) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing {missing}: M8 carries M6's tuned settings so its cells are the\n"
            "same learners M6 ran. Run the tuning chain first."
        )
    m4 = json.loads(M4_SELECTION.read_text(encoding="utf-8"))
    m5 = json.loads(M5_SELECTION.read_text(encoding="utf-8"))
    rates = json.loads(M3_RATES.read_text(encoding="utf-8"))
    stationary = {name: pick["lr"] for name, pick in rates.get(RATES_CONDITION, {}).items()}
    return as_filter(m4), as_filter(m5), stationary


def entries(centralised: dict, diffusion: dict, rates: dict) -> list[dict]:
    """Assigned by name, never by position (the mis-assignment D106 caught in M6)."""
    learners = [{"name": "centralized_ekf_walk", **centralised}]
    learners += [{"name": n, **diffusion} for n in FILTERS[1:]]
    for name in BASELINES:
        family = SGD_FAMILY if name in SGD_FAMILY else ADAMW_FAMILY
        if name in rates and name in family:
            learners.append({"name": name, "lr": rates[name], **family[name]})
    return learners


def r_scale_override(smoke: bool) -> dict:
    """M4 tuned the level of R; M8 carries it, exactly as M6 and M11 do."""
    if smoke or not M4_SELECTION.exists():
        return {}
    scale = json.loads(M4_SELECTION.read_text(encoding="utf-8"))["r_scale"]
    base = list(load_config("m_stationary").model.observation_variances)
    return {"observation_variances": [v * scale for v in base]}


def announce_delays(condition: str) -> None:
    """Print the delays each agent actually gets, so the set is checked not assumed.

    The span of a drift channel is one number; a delay set is ten, and the ones that
    matter are the extremes (they bound the disagreement) and the mean (it must match
    the control, or the contrast carries a mean shift as well as a spread).
    """
    from dekf_bench.env.series import agent_laws  # noqa: PLC0415

    config = load_config(CONDITIONS[condition])
    laws = agent_laws(config.env.series, config.graph.n_nodes)
    taus = [float(t) for t in laws.tau]
    distinct = sorted(set(taus))
    mean = sum(taus) / len(taus)
    print(f"  {condition}: {len(distinct)} distinct delays over {len(taus)} agents, "
          f"{min(taus)} to {max(taus)}, mean {mean:.2f}")
    print(f"    evalsets built at tau = {config.env.series.tau} "
          f"(law_blocks reads the scalar, agents read tau_values)")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--smoke", action="store_true", help="short horizon, one seed")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.report_only:
        # smoke passed through: `--report-only --smoke` must read the smoke cells
        # rather than silently reporting the real ones (M11 copied this bug from M6).
        report(smoke=args.smoke)
        return 0

    centralised, diffusion, rates = load_settings(args.smoke)
    horizon = 60 if args.smoke else args.horizon
    seeds = [0] if args.smoke else args.seeds
    model_override = r_scale_override(args.smoke)

    print(f"M8{' SMOKE' if args.smoke else ''}: {len(CONDITIONS)} cells x {len(seeds)} "
          f"seeds, T={horizon}, device {args.device}\n", flush=True)
    for condition in CONDITIONS:
        announce_delays(condition)
    print(f"  twin: {TWIN}, homogeneous at tau = 17.0\n", flush=True)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, condition in enumerate(CONDITIONS, start=1):
        name = cell_name(condition) + ("_smoke" if args.smoke else "")
        learners = entries(centralised, diffusion, rates)
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
        remaining = elapsed / ran * (len(CONDITIONS) - index) if ran else 0.0
        print(f"[{index}/{len(CONDITIONS)}] {name:<22} {len(learners):>2} learners  "
              f"{note:<14} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM8{' smoke' if args.smoke else ''} complete in "
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


def _paired(left: str, right: str, learner: str):
    """(mean difference, t) over the seeds both cells have, or None."""
    a, b = _by_seed(left, learner), _by_seed(right, learner)
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    diffs = [a[s] - b[s] for s in shared]
    mean = sum(diffs) / len(diffs)
    sd = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    t = abs(mean) / (sd / len(diffs) ** 0.5) if sd else float("inf")
    return mean, t


def report(smoke: bool = False) -> None:
    suffix = "_smoke" if smoke else ""
    arms = [*FILTERS, *BASELINES]
    spread, control = cell_name("tau_spread") + suffix, cell_name("tau_control") + suffix

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

    # ASCII only in anything printed: this console is cp1252 and a bare warning glyph
    # raises UnicodeEncodeError mid-report, after tables have already been written.
    # The docstrings keep their symbols -- those are never encoded to the terminal.
    print("\n  HETEROGENEITY: spread cell - control cell, paired per seed")
    print("  THE HEADLINE. Same mean delay, same held-out sets (byte-identical),")
    print("  same learners and seeds; the only difference is that the agents")
    print("  disagree. The twin cancels, and so does the mean-delay shift.")
    print("  Positive = disagreeing agents cost more than agreeing ones.")
    print("  t is |mean|/SE on 4 df: the 5% critical value is 2.78, not 2.0.\n")
    print(f"    {'learner':<40}{'spread-control':>16}{'t':>8}   verdict")
    for learner in arms:
        got = _paired(spread, control, learner)
        if got is None:
            print(f"    {learner:<40}{'-':>16}{'-':>8}")
            continue
        mean, t = got
        verdict = "COSTS" if t > 2.78 else "null"
        print(f"    {learner:<40}{mean:>+16.4f}{t:>8.1f}   {verdict}")

    print("\n  the mean-delay shift alone: control cell - twin, paired per seed")
    print("  Not the question, but it must be reported: the spread set could not be")
    print("  centred on the twin's tau = 17, because chaos ends just below 16.4.\n")
    print(f"    {'learner':<40}{'control-twin':>16}{'t':>8}")
    for learner in arms:
        got = _paired(control, TWIN, learner)
        print(f"    {learner:<40}" + (f"{got[0]:>+16.4f}{got[1]:>8.1f}"
                                      if got else f"{'-':>16}{'-':>8}"))

    print("\n  the total: spread cell - twin, and the sum it must equal\n")
    print(f"    {'learner':<40}{'spread-twin':>14}{'sum of the two':>16}{'gap':>10}")
    for learner in arms:
        total = _paired(spread, TWIN, learner)
        het = _paired(spread, control, learner)
        shift = _paired(control, TWIN, learner)
        if not (total and het and shift):
            print(f"    {learner:<40}{'-':>14}{'-':>16}{'-':>10}")
            continue
        parts = het[0] + shift[0]
        print(f"    {learner:<40}{total[0]:>+14.4f}{parts:>+16.4f}"
              f"{total[0] - parts:>+10.4f}")
    print("\n  The gap column is exact arithmetic, not a measurement: it is zero")
    print("  whenever the three contrasts share the same seeds. A non-zero entry")
    print("  means a cell is missing a seed, and the row above it is not comparable.")

    print("\n  PREDICTED, before the run: local_only hurt least (it never mixes, so it")
    print("  cannot be pulled toward a consensus fitting nobody); the pooling arms")
    print("  hurt most; one-hop hurt more than mean-only diffusion, because it")
    print("  consumes a neighbour's raw batch and that batch now obeys another law.")
    print("  The measured pairwise disagreement across 16.4 to 20.0 is 0.0082-0.0120,")
    print("  which is inside the 0.0037-0.0146 band five seeds resolved in M12.")


if __name__ == "__main__":
    raise SystemExit(main())
