r"""P5.11 / P5.14 on Mackey--Glass -- the exact Gaussian test of lem:conservative.

    python scripts/run_mg_belief_calibration.py --smoke     # the path, not the numbers
    python scripts/run_mg_belief_calibration.py             # the cells
    python scripts/run_mg_belief_calibration.py --report-only

The image task's half is `run_belief_calibration.py` on main (D120); this is the
same design on the series task, where the predictive $\mathcal N(\hat y,\kappa\bm
H\bm P\bm H^{\mathsf T}+\bm R)$ is exact -- no probit, no softmax -- and where the
observation noise is **known**: the generator adds $\sigma=0.1$ in standardised
units, so the model's own error is directly observable as $\text{MSE}-\sigma^2$.

## Why the error-shrink ratio (decided with the user 2026-09-27)

M6's settled numbers, on the `current` set, show R alone already over-covering the
residuals: the belief's NLL is *worse* than the plug-in's (R alone) by 0.004 to
0.009 for every filter in every condition, the variance ratio at $\kappa=1$ is
0.81--0.90, and 90% coverage is 0.92--0.93. So the raw $\kappa^\star$ will likely
sit at 0 here too, and a $\kappa$-based tightness test would be undecidable on both
tasks. The lemma's question does not need a covariance, though:

$$\text{shrink}=\frac{\text{MSE}_{\text{post}}-\sigma^2}{\text{MSE}_{\text{pre}}-\sigma^2}.$$

The combine replaces each $\bm\psi_u$ by $\sum_u a_{vu}\bm\psi_u$, and the mixing
weights average to one over agents, so this is the model error after averaging
against the model error the averaging started from. **1 means the errors
coincide** -- averaging moved nothing, the bound is tight; **about $1/|\mathcal
M_v|\approx0.26$ means they were independent**. MSEs are averaged over agents and
settled steps first, per seed, then the ratio is taken: a ratio of means, not a
mean of ratios.

## Named before the run (confirmatory under D118)

1. **Is the bound tight, measured without the covariance?** $\log_{10}\text{shrink}$
   per diffusion filter per condition, TOST at $\pm$`TIGHTNESS_MARGIN` decades (a
   factor of 2; independence would give about $-0.6$), Holm across the four.
   Predicted tight (D88). A seed whose $\text{MSE}_{\text{pre}}\le\sigma^2$ has no
   model error left to shrink and is dropped, and counted.
2. **The same question through the covariance**: $\log_{10}\kappa^\star_{\text{pre}}
   -\log_{10}\kappa^\star_{\text{post}}$, TOST at the same margin, Holm across four,
   under D120's degeneracy rule -- if $\kappa^\star=0$ in more than half of a
   filter's evaluations in either stage, its row is undecidable and spends no alpha.
3. **Is the pre-combine belief consistent?** $\log_{10}\kappa^\star_{\text{pre}}$
   against 0 per diffusion filter, Holm across four, under the same rule. No
   prediction.

Exploratory: belief against plug-in NLL (M6 already shows it, so it cannot be
named in advance), the R-scaled $\kappa^\star$, the moment estimate, and the
centralised filter's own $\kappa^\star$ (inferable from M6's variance ratio).

## The design

M6's three conditions, its tuned filters (M4, M5) and its R scale, five seeds,
$T=1500$, evaluations every 10 steps -- M6's 25 coincides with the abrupt schedule's
jumps -- with the beliefs scored over the settled window on every held-out block.
Cells as M6's: **a**, the centralised filter and both mean-only diffusion
filters; **b**, both full-sharing filters. The gradient baselines hold no belief;
their plug-in NLL is read from M6's own cells for context.
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
from run_m3_rates import ADAMW_FAMILY, SGD_FAMILY  # noqa: E402
from run_m6_comparison import (  # noqa: E402
    CONDITIONS,
    GROUP_B_FILTERS,
    load_settings,
    r_scale_override,
)

from dekf_bench.metrics.belief import KAPPA_GRID  # noqa: E402
from dekf_bench.metrics.paired import Paired, holm  # noqa: E402
from dekf_bench.metrics.paired import paired as compare  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402

CENTRAL = "centralized_ekf_walk"
MEAN_ONLY = ["diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]
DIFFUSION = [*MEAN_ONLY, *GROUP_B_FILTERS]

ALPHA = 0.05
#: Decades: a factor of 2, half of log10 |M_v| ~ 0.6 at ER 0.3, N = 10.
TIGHTNESS_MARGIN = 0.3
KAPPA_FLOOR = min(k for k in KAPPA_GRID if k > 0.0)
DEGENERATE_FRACTION = 0.5

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 10
STATUS = ROOT / "results" / "mcal_status.json"
STAT_HEADER = f"{'diff':>9}  {'95% CI':^20}  {'t':>7}  {'p_holm':>7}   n"


def cell_name(condition: str, group: str, suffix: str = "") -> str:
    return f"mcal_{condition}_{group}{suffix}"


def entries(group: str, centralised: dict, diffusion: dict) -> list[dict]:
    if group == "b":
        return [{"name": n, **diffusion, "combine_exponent": 1.0} for n in GROUP_B_FILTERS]
    return [{"name": CENTRAL, **centralised}, *({"name": n, **diffusion} for n in MEAN_ONLY)]


def config_for(args, condition: str, group: str, suffix: str, horizon: int, seeds: list[int],
               centralised: dict, diffusion: dict, model_override: dict):
    return load_config(
        CONDITIONS[condition],
        overrides={
            "run": {"name": cell_name(condition, group, suffix), "horizon": horizon,
                    "seeds": seeds, "eval_every": EVAL_EVERY, "device": args.device,
                    "dtype": args.dtype},
            **({"model": model_override} if model_override else {}),
            "learners": entries(group, centralised, diffusion),
            # current only: the question is the belief at the law the agents face.
            "eval": {"evalsets": ["prequential", "current"], "belief_calibration": True},
        },
    )


def noise_variance() -> float:
    """sigma^2 from the series config -- refused if it varies by agent."""
    series = load_config(CONDITIONS["stationary"]).env.series
    if series.sigma_spread:
        raise SystemExit("per-agent sigma: MSE - sigma^2 would need each agent's own sigma")
    return float(series.sigma) ** 2


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS, device="cpu")
    parser.add_argument("--smoke", action="store_true", help="short horizon, one seed")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.report_only:
        report(suffix)
        return 0

    import torch  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    centralised, diffusion, _rates = load_settings(args.smoke)
    horizon = 60 if args.smoke else args.horizon
    seeds = [0] if args.smoke else args.seeds
    model_override = r_scale_override(args.smoke)
    noise_variance()  # refuse early rather than after the compute

    cells = [(condition, group) for condition in CONDITIONS for group in ("a", "b")]
    print(f"MG P5.11/14{' SMOKE' if suffix else ''}: {len(cells)} cells x {len(seeds)} "
          f"seeds, T={horizon}, device {args.device}\n", flush=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, (condition, group) in enumerate(cells, start=1):
        config = config_for(args, condition, group, suffix, horizon, seeds,
                            centralised, diffusion, model_override)
        note = run_one(config, None, None, args.fresh)
        status[config.run.name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {config.run.name:<26} {len(config.learners):>2} learners"
              f"  {note:<14} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)
    print(f"\nMG P5.11/14{' smoke' if suffix else ''} complete in "
          f"{(time.time() - started) / 60:.1f} min\n")
    report(suffix)
    return 0


# =========================================================================== #
# report
# =========================================================================== #

def _cell_of(learner: str, condition: str, suffix: str) -> str:
    return cell_name(condition, "b" if learner in GROUP_B_FILTERS else "a", suffix)


def seed_means(run: str, learner: str, metric: str, log_kappa: bool = False
               ) -> tuple[dict[int, float], int, int]:
    """Per seed, the metric over every agent and scored step; see main's twin."""
    import pandas as pd  # noqa: PLC0415

    directory = ROOT / "results" / run
    if not (directory / "_complete").exists():
        return {}, 0, 0
    means, floored, total = {}, 0, 0
    for path in sorted(directory.glob("seed_*.parquet")):
        frame = pd.read_parquet(path, columns=["learner", "metric", "evalset", "t", "value"])
        rows = frame[(frame["learner"] == learner) & (frame["metric"] == metric)
                     & (frame["evalset"] == "current")]
        if metric in ("nll",):  # a plug-in row from M6, settled window only
            rows = rows[rows["t"] >= 0.8 * frame["t"].max()]
        values = rows["value"]
        if not len(values):
            continue
        if log_kappa:
            floored += int((values < KAPPA_FLOOR).sum())
            values = values.clip(lower=KAPPA_FLOOR).map(math.log10)
        total += len(values)
        means[int(path.stem.split("_")[1])] = float(values.mean())
    return means, floored, total


def stat_columns(result: Paired, p_adjusted: float) -> str:
    lo, hi = result.ci(0.95)
    mark = "*" if p_adjusted < ALPHA else " "
    return (f"{result.mean:>+9.4f}  {f'[{lo:+.4f}, {hi:+.4f}]':<20}  {result.t:>+7.2f}"
            f"  {p_adjusted:>7.3f}{mark}  {result.n}")


def _verdict(result: Paired, p_diff: float, p_tost: float, shrinks_when: str) -> str:
    if p_tost < ALPHA:
        return "TIGHT (shown within the margin)"
    if p_diff < ALPHA and (result.mean < 0 if shrinks_when == "negative" else result.mean > 0):
        return "LOOSE (averaging shrank the error)"
    return "neither shown"


def report(suffix: str = "") -> None:
    mean = lambda d: sum(d.values()) / len(d) if d else float("nan")  # noqa: E731
    sigma2 = noise_variance()
    if suffix:
        print("  SMOKE: a short run at one seed. These numbers mean nothing; the point")
        print("  is that every code path below ran.\n")

    for condition in CONDITIONS:
        print(f"\n  ===== {condition} =====")
        cell = lambda learner, condition=condition: _cell_of(learner, condition, suffix)  # noqa: E731

        # ---- question 1: the error-shrink ratio -------------------------------
        print("\n  Q1 [confirmatory]: is lem:conservative tight, without the covariance?")
        print(f"    log10[(MSE_post - s2) / (MSE_pre - s2)], s2 = {sigma2:g} (the generator's")
        print(f"    noise). 0 = tight; independence ~ -0.6. TOST at +/-{TIGHTNESS_MARGIN} decades,")
        print("    Holm across the four filters, on the difference and on the TOST p.\n")
        print(f"    {'':<40}{'shrink':>8}{STAT_HEADER}   {'TOST p':>7}  verdict")
        results, dropped = [], {}
        for learner in DIFFUSION:
            pre = seed_means(cell(learner), learner, "pre_mse")[0]
            post = seed_means(cell(learner), learner, "mse")[0]
            ratios = {s: math.log10((post[s] - sigma2) / (pre[s] - sigma2))
                      for s in pre if s in post and pre[s] > sigma2 and post[s] > sigma2}
            dropped[learner] = len(set(pre) & set(post)) - len(ratios)
            if ratios:
                results.append((learner, ratios, compare(ratios, {s: 0.0 for s in ratios})))
        p_diff = holm([r.p for *_x, r in results])
        p_tost = holm([r.tost_p(TIGHTNESS_MARGIN) for *_x, r in results])
        for (learner, ratios, result), pd_, pt in zip(results, p_diff, p_tost, strict=True):
            note = f"   [{dropped[learner]} seed(s) with MSE <= s2 dropped]" if dropped[learner] else ""
            print(f"    {learner:<40}{10 ** mean(ratios):>8.3f}{stat_columns(result, pd_)}   "
                  f"{pt:>7.3f}  {_verdict(result, pd_, pt, 'negative')}{note}")

        # ---- question 2: the same through kappa* ------------------------------
        print("\n  Q2 [confirmatory]: log10 kappa*_pre - log10 kappa*_post, TOST at the same")
        print("    margin; undecidable where kappa* = 0 in over half the evaluations (D120).\n")
        print(f"    {'':<40}{'pre':>8}{'post':>8}{STAT_HEADER}   {'TOST p':>7}  verdict")
        live, undecidable = [], []
        for learner in DIFFUSION:
            pre, pf, pn = seed_means(cell(learner), learner, "pre_kappa_star", log_kappa=True)
            post, qf, qn = seed_means(cell(learner), learner, "kappa_star", log_kappa=True)
            if not (pre and post):
                continue
            if max(pf / pn, qf / qn) > DEGENERATE_FRACTION:
                undecidable.append((learner, pf / pn, qf / qn))
                continue
            live.append((learner, pre, post, compare(pre, post)))
        p_diff = holm([r.p for *_x, r in live])
        p_tost = holm([r.tost_p(TIGHTNESS_MARGIN) for *_x, r in live])
        for (learner, pre, post, result), pd_, pt in zip(live, p_diff, p_tost, strict=True):
            print(f"    {learner:<40}{mean(pre):>+8.2f}{mean(post):>+8.2f}"
                  f"{stat_columns(result, pd_)}   {pt:>7.3f}  "
                  f"{_verdict(result, pd_, pt, 'positive')}")
        for learner, pre_share, post_share in undecidable:
            print(f"    {learner:<40}UNDECIDABLE: kappa* at 0 in {pre_share:.0%} (pre) and "
                  f"{post_share:.0%} (post) of evaluations")

        # ---- question 3 -------------------------------------------------------
        print("\n  Q3 [confirmatory]: is the pre-combine belief consistent? log10 kappa*_pre")
        print("    against 0; negative = conservative. Holm across four, same rule. No prediction.\n")
        print(f"    {'':<40}{STAT_HEADER}")
        rows, skipped = [], []
        for learner in DIFFUSION:
            values, floored, n = seed_means(cell(learner), learner, "pre_kappa_star",
                                            log_kappa=True)
            if n and floored / n > DEGENERATE_FRACTION:
                skipped.append((learner, floored / n))
            elif values:
                rows.append((learner, compare(values, {s: 0.0 for s in values})))
        for (learner, result), p_adj in zip(rows, holm([r.p for _l, r in rows]), strict=True):
            print(f"    {learner:<40}{stat_columns(result, p_adj)}")
        for learner, share in skipped:
            print(f"    {learner:<40}UNDECIDABLE: kappa* at 0 in {share:.0%} of evaluations")

        # ---- exploratory ------------------------------------------------------
        print("\n  exploratory: plug-in (R alone) against the belief, kappa* in plain units,")
        print("  the moment estimate, and the R-scaled reading (rho* on R, then kappa*)\n")
        print(f"    {'filter':<36}{'NLL pl.':>9}{'NLL bel.':>9}{'VR@1':>7}{'k* post':>9}"
              f"{'k* pre':>9}{'k mom':>8}{'rho*':>7}{'k~ post':>9}{'k~ pre':>9}")
        for learner in [CENTRAL, *DIFFUSION]:
            run = cell(learner)
            values = [mean(seed_means(run, learner, m)[0])
                      for m in ("plugin_nll", "belief_nll", "belief_variance_ratio")]
            kappas = [10 ** mean(seed_means(run, learner, m, log_kappa=True)[0])
                      if learner != CENTRAL or not m.startswith("pre_") else float("nan")
                      for m in ("kappa_star", "pre_kappa_star")]
            moment = mean(seed_means(run, learner, "kappa_moment")[0])
            rho = mean(seed_means(run, learner, "noise_scale_star")[0])
            tempered = [10 ** mean(seed_means(run, learner, m, log_kappa=True)[0])
                        if learner != CENTRAL or not m.startswith("pre_") else float("nan")
                        for m in ("tempered_kappa_star", "pre_tempered_kappa_star")]
            print(f"    {learner:<36}{values[0]:>9.4f}{values[1]:>9.4f}{values[2]:>7.3f}"
                  f"{kappas[0]:>9.3g}{kappas[1]:>9.3g}{moment:>8.3f}{rho:>7.3f}"
                  f"{tempered[0]:>9.3g}{tempered[1]:>9.3g}")

        print("\n  context: the gradient baselines' plug-in NLL, from M6's own cells (no belief)")
        for learner in [*SGD_FAMILY, *ADAMW_FAMILY]:
            values = seed_means(f"m6_{condition}_a", learner, "nll")[0]
            if values:
                print(f"    {learner:<36}{mean(values):>9.4f}")

    print(f"\n  * = p_holm < {ALPHA}, adjusted within each table (D113). MSEs are averaged over")
    print("  agents and settled steps before the ratio; kappa* geometrically, per seed.")


if __name__ == "__main__":
    raise SystemExit(main())
