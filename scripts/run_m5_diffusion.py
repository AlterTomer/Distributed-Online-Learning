r"""M5 -- the diffusion filter's three knobs, tuned jointly, then a five-seed tie-break.

    python scripts/run_m5_diffusion.py              # the 12-cell grid, 2 seeds
    python scripts/run_m5_diffusion.py --tie-break  # the plateau, at 5 seeds
    python scripts/run_m5_diffusion.py --report-only

The X20/X23 analogue (`docs/mackey_glass_plan.md`, M5). **Only the tie-break writes
`m5_selection.json`**, so M6 stays blocked until the plateau has been resolved at
five seeds rather than silently carrying a two-seed answer.

## Three axes, not four -- and excluding R is forced, not chosen

`observation_variances` is a **model-level** setting, shared by every learner in a
run. M5 cannot select a different R from M4 without splitting M6's cells by R, and
that would destroy the seed-pairing that makes M6's comparisons exact rather than
merely matched. M5 carries M4's R x 1 ([[D102]]). It is also right on the merits:
R describes the observation process, not how agents cooperate.

## Why the grid sits where it does

Each axis **brackets two competing predictions** instead of assuming either:

| axis | M4, centralised on this task | MNIST's *diffusion* filter | grid |
|---|---|---|---|
| $q$ | $6\times10^{-6}$ | $6\times10^{-4}$, a decade **above** its centralised twin | both, plus the midpoint |
| $\sigma_0^2$ | $0.01$ | $10^{-3}$, a decade **below** | both |
| $\gamma$ | $1.0$ | $0.9995$ | both |

The $q$ range is the crux. On the image task the diffusion filter wanted **ten times
more** process noise than the centralised one and a ten times smaller prior, because
each agent holds $1/N$ of the information and must stay adaptive -- the $1/N$ deficit
is real information, not a missing constant (D87). If that pattern transfers, this
filter wants $q\approx6\times10^{-5}$; if it does not, it wants M4's
$6\times10^{-6}$. The grid cannot be wrong in either direction.

The top of the $q$ range is deliberately **one decade below the image task's measured
divergence cliff** ($6\times10^{-3}$, where X21 found every $\gamma$ destroyed). If
the argmin lands on $6\times10^{-4}$ the grid is extended upward carefully, one step
at a time, rather than blindly.

$\gamma$ carries both values because M4 measured the axes as **interacting** on this
task: $\gamma=0.9995$ won every slice at high $q$, $\gamma=1$ every slice at low.
Pinning $\gamma$ would be the coordinate-descent mistake X23 exists to record.

## Tuned on the carried variant, carried to all four

`diffusion_ekf_onehop_mean_receiver` -- one-hop, mean-only, at the receiver point:
the variant D99 adopted and D100 confirmed, and the one the paper deploys. One
setting then serves all four diffusion variants in M6, which is the X14 discipline.
⚠ The local-adapt filter may prefer something else; carrying one setting means a
shortfall there is attributable to the adapt scope rather than confounded with
tuning, which is the trade the image task also made.

Tuned on the **abrupt** condition and carried to all three, as M4 was.

## The per-cell cost is not measured here

M4's centralised cell cost 14 minutes on the CPU; this filter holds ten beliefs
rather than one and runs on the GPU, so neither number transfers. The script prints
its own estimate after the first cell, as M4's did. The plan budgets ~8 h; if cell
one says otherwise, stop and re-cut the grid rather than trusting the budget.
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
LEARNER = "diffusion_ekf_onehop_mean_receiver"

#: The experiment itself, so constants rather than flags.
GAMMAS = [1.0, 0.9995]
PROCESS_NOISE = [6.0e-6, 6.0e-5, 6.0e-4]
PRIOR_SCALES = [0.01, 0.001]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1], 25

#: Two seeds separate a grid; they do not separate a plateau whose gaps are the size
#: of the seed spread. The tie-break settles that.
TIE_BREAK_SEEDS = [0, 1, 2, 3, 4]

#: Cells within this of the best go to the tie-break. Taken from M4's measured flat
#: region on this task -- its top four cells spanned 0.0018 and its last q decade
#: bought 0.0006 -- not from the image task, whose metric is an error rate. Reread it
#: against this grid's own spread before trusting it on a third task.
THRESHOLD = 0.002

STATUS = ROOT / "results" / "m5_status.json"
SELECTION = ROOT / "results" / "m5_selection.json"


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def run_name(gamma: float, q: float, prior: float) -> str:
    return f"m5_g{tag(gamma)}_q{tag(q)}_s{tag(prior)}"


def grid() -> list[tuple[float, float, float]]:
    return [(g, q, p) for g in GAMMAS for q in PROCESS_NOISE for p in PRIOR_SCALES]


def config_for(args, name: str, gamma: float, q: float, prior: float, seeds: list[int]):
    return load_config(
        CONDITION,
        overrides={
            "run": {"name": name, "horizon": args.horizon, "seeds": seeds,
                    "eval_every": EVAL_EVERY, "device": args.device, "dtype": args.dtype},
            "learners": [{
                "name": LEARNER, "transition": "scalar", "gamma": gamma,
                "forgetting": "process_noise", "process_noise_q": q,
                "lambda_forget": 1.0, "prior_scale": prior,
            }],
        },
    )


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def sweep(args) -> int:
    """The 12-cell grid at two seeds."""
    cells = grid()
    status = load_status()
    print(f"M5: {len(cells)} cells x {len(args.seeds)} seeds on {CONDITION}, "
          f"learner {LEARNER}, device {args.device}\n", flush=True)
    started, ran = time.time(), 0
    for index, (gamma, q, prior) in enumerate(cells, start=1):
        name = run_name(gamma, q, prior)
        note = run_one(config_for(args, name, gamma, q, prior, args.seeds),
                       None, None, args.fresh)
        status[name] = note
        save_status(status)
        if note != "cached":
            ran += 1
        elapsed = (time.time() - started) / 60
        remaining = elapsed / ran * (len(cells) - index) if ran else 0.0
        print(f"[{index}/{len(cells)}] {name:<28} g {gamma:<7g} q {q:<7g} s0 {prior:<6g} "
              f"{note:<12} {elapsed:.0f} min, ~{remaining:.0f} left", flush=True)

    print(f"\nM5 grid complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    print("\n  The grid does NOT write the selection. Run --tie-break next:")
    print("    python scripts/run_m5_diffusion.py --tie-break")
    return 0


def tied_cells() -> list[tuple[float, float, float]]:
    """Every grid cell within THRESHOLD of the best."""
    scored = sorted(
        (v, c) for v, c in
        ((settled(run_name(g, q, p), LEARNER), (g, q, p)) for g, q, p in grid())
        if v != float("inf")
    )
    return [c for v, c in scored if v - scored[0][0] <= THRESHOLD] if scored else []


def tie_break(args) -> int:
    """The plateau at five seeds, in its own directories.

    Not more seeds in the same directory: a completed cell is cached on its
    ``_complete`` marker, so asking for five where two are recorded returns "cached"
    and silently reports the two-seed answer. Re-running under separate names keeps
    both measurements, which is the point -- if the five-seed ranking differs from
    the two-seed one, that difference is the finding, and it is unreadable once the
    two-seed numbers are gone. Seeds 0-1 are recomputed rather than reused, so all
    five come from one code state. (X23's pattern, for X23's reasons.)
    """
    cells = tied_cells()
    if not cells:
        print("The grid has not run. Start with `python scripts/run_m5_diffusion.py`.")
        return 1

    status = load_status()
    print(f"M5 tie-break: {len(cells)} cells within {THRESHOLD} of the best, "
          f"at {len(args.seeds)} seeds\n", flush=True)
    started = time.time()
    for index, (gamma, q, prior) in enumerate(cells, start=1):
        name = "m5tb_" + run_name(gamma, q, prior).removeprefix("m5_")
        note = run_one(config_for(args, name, gamma, q, prior, args.seeds),
                       None, None, args.fresh)
        status[name] = note
        save_status(status)
        print(f"[{index}/{len(cells)}] {name:<30} {note:<12} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nM5 tie-break complete in {(time.time() - started) / 60:.1f} min\n")
    tie_break_report()
    return 0


def tie_break_name(gamma: float, q: float, prior: float) -> str:
    return "m5tb_" + run_name(gamma, q, prior).removeprefix("m5_")


def report() -> None:
    """The surface, one table per q, then the ranking."""
    for q in PROCESS_NOISE:
        print(f"  settled RMSE at q = {q:g}")
        print(f"    {'gamma \\ sigma0^2':>18}" + "".join(f"{p:>12g}" for p in PRIOR_SCALES))
        for gamma in GAMMAS:
            line = f"    {gamma:>18g}"
            for prior in PRIOR_SCALES:
                if (ROOT / "results" / run_name(gamma, q, prior) / "_diverged").exists():
                    line += f"{'DIV':>12}"
                    continue
                value = settled(run_name(gamma, q, prior), LEARNER)
                line += f"{value:>12.4f}" if value != float("inf") else f"{'-':>12}"
            print(line)
        print()

    scored = sorted(
        (v, c) for v, c in
        ((settled(run_name(g, q, p), LEARNER), (g, q, p)) for g, q, p in grid())
        if v != float("inf")
    )
    if not scored:
        print("  no M5 cells on disk yet")
        return
    best, (gamma, q, prior) = scored[0]
    print(f"  grid best: gamma={gamma:g}, q={q:g}, sigma_0^2={prior:g} -> {best:.4f}")
    print(f"  plateau within {THRESHOLD}: {len(tied_cells())} of {len(grid())} cells")
    if q in (PROCESS_NOISE[0], PROCESS_NOISE[-1]):
        edge = "bottom" if q == PROCESS_NOISE[0] else "top"
        print(f"  !! q sits on the {edge} of its grid. The image task put the divergence")
        print("     cliff one decade above 6e-4; extend one step at a time, not blindly.")
    print("  note: gamma and sigma_0^2 carry two values each, so neither end is interior.")
    print("  M4 chose gamma=1, q=6e-06, sigma_0^2=0.01 for the CENTRALISED filter;")
    print("  the image task's diffusion filter wanted 10x more q and 10x less prior.")


def tie_break_report() -> None:
    """Both rankings side by side, because the question is whether they agree."""
    cells = tied_cells()
    if not cells:
        return
    rows = []
    for gamma, q, prior in cells:
        few = settled(run_name(gamma, q, prior), LEARNER)
        many = settled(tie_break_name(gamma, q, prior), LEARNER)
        rows.append((many, few, (gamma, q, prior)))
    rows.sort()

    print(f"  {'gamma':>8}{'q':>10}{'sigma0^2':>10}"
          f"{f'{len(SEEDS)} seeds':>11}{f'{len(TIE_BREAK_SEEDS)} seeds':>11}{'moved':>9}")
    for many, few, (gamma, q, prior) in rows:
        moved = "-" if few == float("inf") or many == float("inf") else f"{many - few:+.4f}"
        wide = f"{many:.4f}" if many != float("inf") else "-"
        narrow = f"{few:.4f}" if few != float("inf") else "-"
        print(f"  {gamma:>8g}{q:>10g}{prior:>10g}{narrow:>11}{wide:>11}{moved:>9}")

    usable = [r for r in rows if r[0] != float("inf")]
    if not usable:
        print("\n  the tie-break has not run; no selection written")
        return
    many, few, (gamma, q, prior) = usable[0]
    narrow_best = min(cells, key=lambda c: settled(run_name(*c), LEARNER))
    agreed = narrow_best == (gamma, q, prior)
    selection = {
        "gamma": gamma, "process_noise_q": q, "prior_scale": prior,
        "settled_rmse": many, "condition": CONDITION, "learner": LEARNER,
        "seeds": TIE_BREAK_SEEDS, "plateau_size": len(cells),
        "two_seed_winner_agreed": agreed,
    }
    SELECTION.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"\n  selected: {selection}")
    if not agreed:
        print("  !! the five-seed ranking DISAGREES with the two-seed one. That is the")
        print("     finding the tie-break exists to catch; both tables are kept above.")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--tie-break", action="store_true",
                        help="re-run the plateau at five seeds and write the selection")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.report_only:
        report()
        print()
        tie_break_report()
        return 0
    if args.tie_break:
        if args.seeds == list(SEEDS):
            args.seeds = list(TIE_BREAK_SEEDS)
        return tie_break(args)
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
