r"""Does $q=6\times10^{-7}$ under-track a steadily drifting target? The probe that decides.

    python scripts/run_m5_linear_probe.py --device cuda
    python scripts/run_m5_linear_probe.py --report-only

M5's tie-break left its two plateau cells **0.0005 apart at five seeds -- exactly
the gap they showed at two**, because the extra seeds moved both by the same
$-0.0030$. That is a seed effect, not a cell effect: the data does not choose
between $q=6\times10^{-7}$ and $q=6\times10^{-6}$. Something else has to, and
guessing is what this script exists to avoid.

## Why `m_linear`, rather than yet more seeds on `m_abrupt`

$q$ is the adaptivity knob -- it is what stops the covariance collapsing -- and
$6\times10^{-7}$ is the *least* adaptive setting in M5's grid. It was selected on
`m_abrupt`, **whose schedule is `recurring`**, and recurring reflects at the
45-degree cap (D74): the target keeps coming back to where the filter already is.
A filter that adapts too slowly is barely punished by a target that returns, so
`m_abrupt` is the condition least able to expose this particular risk.

`m_linear` marches once to 45 degrees and never returns. That is where too little
process noise shows, and it is one of the three conditions M6 carries this setting
to unchanged. If the two values are indistinguishable here too, the choice is
genuinely free and the neighbourhood argument decides it. If $6\times10^{-6}$
wins, the selection M5 wrote is carrying a tuning artefact into every diffusion
cell of a 10-hour battery.

## Two learners, because the argument has two halves

`diffusion_ekf_onehop_mean_receiver` is the carried variant M5 tuned.
`diffusion_ekf` is the local-adapt one, which holds $1/N$ of the information and by
D87's logic should want *more* process noise, not less -- so if a low $q$ hurts
anywhere, it hurts there first. Both share a cell, so the second costs a fraction
of the first and tests the half of the argument the carried variant cannot.

## What it does not do

**It writes no selection.** M5's tie-break owns `m5_selection.json`; this probe
informs a human decision that is recorded in the design notes, and that file is
then edited by hand or deliberately left alone. A second script silently
overwriting the first one's output is how a tuning provenance becomes unreadable.

M4 selected `r_scale: 1.0`, so M6's R override is a no-op and this probe needs no
model override to match how M6 will actually run.
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
from run_m5_diffusion import tag  # noqa: E402

from dekf_bench.utils.config import load_config  # noqa: E402

CONDITION = "m_linear"

#: The two cells the tie-break could not separate. gamma and sigma_0^2 are the
#: values both of them share, so q is the only thing varying here.
GAMMA, PRIOR = 1.0, 0.01
QS = [6.0e-7, 6.0e-6]

#: The carried variant first, then the local-adapt one D87 says should want more q.
LEARNERS = ["diffusion_ekf_onehop_mean_receiver", "diffusion_ekf"]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1], 25

#: M5's plateau width, reused deliberately: a gap below this is not a gap, and the
#: whole point of this probe is that 0.0005 was not one.
THRESHOLD = 0.002

STATUS = ROOT / "results" / "m5lin_status.json"


def run_name(q: float) -> str:
    return f"m5lin_q{tag(q)}"


def config_for(args, q: float):
    return load_config(
        CONDITION,
        overrides={
            "run": {"name": run_name(q), "horizon": args.horizon, "seeds": args.seeds,
                    "eval_every": EVAL_EVERY, "device": args.device, "dtype": args.dtype},
            "learners": [
                {"name": name, "transition": "scalar", "gamma": GAMMA,
                 "forgetting": "process_noise", "process_noise_q": q,
                 "lambda_forget": 1.0, "prior_scale": PRIOR}
                for name in LEARNERS
            ],
        },
    )


def sweep(args) -> int:
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    print(f"m5 linear probe: {len(QS)} cells x {len(args.seeds)} seeds on {CONDITION}, "
          f"{len(LEARNERS)} learners per cell, device {args.device}\n", flush=True)
    started = time.time()
    for index, q in enumerate(QS, start=1):
        name = run_name(q)
        note = run_one(config_for(args, q), None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(QS)}] {name:<20} q {q:<8g} {note:<12} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nprobe complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    print(f"  settled RMSE on {CONDITION}, {len(SEEDS)} seeds -- lower is better\n")
    print(f"    {'learner':<36}" + "".join(f"{f'q={q:g}':>12}" for q in QS)
          + f"{'6e-6 - 6e-7':>14}")

    deltas: list[tuple[str, float]] = []
    for learner in LEARNERS:
        values = [settled(run_name(q), learner) for q in QS]
        row = f"    {learner:<36}"
        for value in values:
            row += f"{value:>12.4f}" if value != float("inf") else f"{'-':>12}"
        if all(value != float("inf") for value in values):
            delta = values[1] - values[0]
            row += f"{delta:>+14.4f}"
            deltas.append((learner, delta))
        else:
            row += f"{'-':>14}"
        print(row)

    if not deltas:
        print("\n  no cells on disk yet; run the probe first")
        return

    print(f"\n  A negative delta means q=6e-6 is better. Threshold: {THRESHOLD}.")
    decisive = [(n, d) for n, d in deltas if abs(d) >= THRESHOLD]
    if not decisive:
        print("  Neither learner separates the two values on the drifting condition")
        print("  either. The selection is then free on the evidence, and the choice")
        print("  rests on the neighbourhood argument -- recorded in the design notes.")
        return
    for learner, delta in decisive:
        better = "6e-6" if delta < 0 else "6e-7"
        print(f"  {learner}: {better} wins by {abs(delta):.4f} -- above threshold.")
    print("  Decide m5_selection.json against this, and record why.")


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=SEEDS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.report_only:
        report()
        return 0
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
