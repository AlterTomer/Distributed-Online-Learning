r"""M4 -- the centralised filter's four knobs, tuned jointly on Mackey--Glass.

    python scripts/run_m4_centralised.py             # 24 cells x 2 seeds, ~7 h CPU
    python scripts/run_m4_centralised.py --report-only

## The four axes

$\gamma$ (the state model's contraction), $q$ (process noise), $\sigma_0^2$ (the
prior, which is a trust region as much as a prior), and the **scale of $\bm R$** --
M0 measured the per-position *shape* and decision 28 tunes only its level, so this
is one axis rather than 31.

**Jointly, not in turn.** X23 established on the image task that coordinate descent
finds the joint optimum only when the axes separate -- and there they did. That was
a measurement, not a law, and this task adds an axis, so the grid is crossed.

**Tuned on the abrupt condition and carried everywhere**, which is the X14
discipline: one setting, chosen once, so a later comparison is not confounded by
per-condition tuning. What carrying it costs is itself measurable, as the image
task's mis-tuned arm measures there (P5.24).

## Why this one runs on the CPU

The centralised filter holds **one** belief over the pooled batch, so a step is a
single Woodbury solve at $p=2273$ with $Nn_b(L-1)=310$ columns. A 60-step probe
timed that at 26 s -- 10.9 min per cell-seed -- and the grid was sized from it: a
full $2\times3\times3\times3$ would be 54 cells and 20 h, past a night, so what
runs below is 24 cells.

In the run itself the first seed took **8 min** while sharing the CPU with two
other jobs, so the grid costs roughly 7 h -- near the probe, not half it.

*(A first reading of the file timestamps said 5.5 min per cell-seed and 4.5 h.
That was wrong: this runner rewrites a seed's parquet as it goes, so a parquet
appearing means a seed has **started**. Size, not existence, is the progress
signal -- seed 1 sat at 0.64 MB against seed 0's finished 1.35 MB.)*

The cut values are *not* restored on the strength of a cheaper-looking clock:
they were dropped for the reasons below, and a grid that grows whenever a run
looks fast is a grid chosen by the budget.

**Two axes were cut, and not arbitrarily.** $\sigma_0^2=0.1$ goes because the
centralised learner's own config records that above $10^{-1}$ the filter does not
merely slow -- it diverges, reaching 1e113 by step 100 with $\bm P$ still perfectly
well conditioned (D61). That value is the documented cliff, not a candidate.
$\bm R\times0.5$ goes because the profile was measured on an *offline-converged*
model (D96): an online filter's residuals should exceed that floor, never sit
below it, so the plausible direction to stretch $\bm R$ is up. If the argmin lands
on an edge regardless, the report says so and the grid gets extended.
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
from run_m3_rates import settled  # noqa: E402

from dekf_bench.utils.config import load_config  # noqa: E402

CONDITION = "m_abrupt"
LEARNER = "centralized_ekf_gamma"

GAMMAS = [1.0, 0.9995]
QS = [6e-4, 6e-5, 6e-6]
PRIORS = [0.01, 0.001]
#: Multiplies the measured per-position profile; the shape is not tuned.
R_SCALES = [1.0, 2.0]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1], 25
STATUS = ROOT / "results" / "m4_status.json"
SELECTION = ROOT / "results" / "m4_selection.json"


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def cell_name(gamma: float, q: float, prior: float, scale: float) -> str:
    return f"m4_g{tag(gamma)}_q{tag(q)}_s{tag(prior)}_r{tag(scale)}"


def grid():
    for gamma in GAMMAS:
        for q in QS:
            for prior in PRIORS:
                for scale in R_SCALES:
                    yield gamma, q, prior, scale


def config_for(args, name, gamma, q, prior, scale, profile):
    return load_config(
        CONDITION,
        overrides={
            "run": {"name": name, "horizon": args.horizon, "seeds": args.seeds,
                    "eval_every": EVAL_EVERY, "device": args.device, "dtype": args.dtype},
            "model": {"observation_variances": [v * scale for v in profile]},
            "learners": [{
                "name": LEARNER, "transition": "scalar", "gamma": gamma,
                "forgetting": "process_noise", "process_noise_q": q,
                "lambda_forget": 1.0, "prior_scale": prior,
            }],
        },
    )


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
    profile = list(load_config(CONDITION).model.observation_variances)
    cells = list(grid())
    print(f"M4: {len(cells)} cells x {len(args.seeds)} seeds on {CONDITION}, "
          f"device {args.device}, R profile of {len(profile)} positions\n", flush=True)

    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    started, ran = time.time(), 0
    for index, (gamma, q, prior, scale) in enumerate(cells, start=1):
        name = cell_name(gamma, q, prior, scale)
        note = run_one(config_for(args, name, gamma, q, prior, scale, profile),
                       None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<30} g {gamma:<7g} q {q:<7g} s0 {prior:<6g} "
              f"R x{scale:<4g} {note:<12} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM4 complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    """The surface, its argmin, and whether the argmin sits against a grid edge."""
    scored = []
    for gamma, q, prior, scale in grid():
        value = settled(cell_name(gamma, q, prior, scale), LEARNER)
        if value != float("inf"):
            scored.append((value, gamma, q, prior, scale))
    if not scored:
        print("  no M4 cells on disk yet")
        return
    scored.sort()
    print("  best ten cells (settled RMSE on the held-out current set)")
    print(f"    {'rmse':>8}{'gamma':>9}{'q':>9}{'sigma0^2':>10}{'R scale':>9}")
    for value, gamma, q, prior, scale in scored[:10]:
        print(f"    {value:>8.4f}{gamma:>9g}{q:>9g}{prior:>10g}{scale:>9g}")

    best = scored[0]
    # An axis of two values has no interior, so "sits at an edge" is vacuous there: it
    # would fire every run and teach the reader to ignore the warning. Only an axis of
    # three or more can tell an edge from a middle -- here that is q alone.
    axes = (("gamma", best[1], GAMMAS), ("q", best[2], QS),
            ("sigma0^2", best[3], PRIORS), ("R scale", best[4], R_SCALES))
    edges = [name for name, value, axis in axes
             if len(axis) > 2 and value in (axis[0], axis[-1])]
    pinned = [name for name, _value, axis in axes if len(axis) == 2]
    selection = {"gamma": best[1], "process_noise_q": best[2], "prior_scale": best[3],
                 "r_scale": best[4], "settled_rmse": best[0], "condition": CONDITION,
                 "edges": edges, "two_valued_axes": pinned}
    SELECTION.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"\n  selected: {selection}")
    if edges:
        # ASCII only. This stream is redirected to a file and on Windows that file is
        # cp1252, where a warning-sign glyph raises UnicodeEncodeError -- which killed
        # this report once already, at the end of a seven-hour run.
        print(f"  !! the argmin sits at a grid edge on {edges} -- extend before believing it")
    if pinned:
        print(f"  note: {pinned} carry two values each, so neither end is an interior choice")


if __name__ == "__main__":
    raise SystemExit(main())
