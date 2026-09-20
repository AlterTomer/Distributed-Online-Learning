r"""M6's missing arm: what does cooperating actually buy the *filter*?

    python scripts/run_m6_isolated.py --device cuda
    python scripts/run_m6_isolated.py --report-only

M6 can state what cooperation buys every gradient family -- `local_only` against
the ATC form is $+0.0111$ to $+0.0146$ across the three conditions -- and cannot
state it for the filter at all, because **every filter arm in M6 communicates**.
Even `diffusion_ekf`, the local-adapt variant, still combines its neighbours'
means. So D106 reports "distributed filtering works" with no answer to the
obvious question: *compared with not distributing it at all?*

## An edgeless graph is the non-cooperating filter, with no new learner

`disconnected` at ``n_components = N`` puts every agent in its own component, so
the graph has no edges. Metropolis weighting then gives $a_{vv} = 1/(1 + d_v) = 1$
at degree zero -- the docstring's guarantee that "no agent can discard its own
estimate", at its limit -- so the combination matrix is exactly $\boldsymbol I$ and
the combine step is the identity. `diffusion_ekf` on that graph *is* a per-agent
EKF that never communicates.

Verified before this script was written::

    build_graph(topology="disconnected", n_nodes=10, params={"n_components": 10})
    -> 0 edges, 10 components, Metropolis weight diagonal exactly 1.0000

This matters for the claim's strength: the isolated arm runs the **same learner,
the same tuned settings and the same seeds** as its connected twin, so the paired
difference is attributable to the communication and to nothing else. Had it needed
a separate learner class, a shortfall could always have been blamed on the class.

## What it pairs against

Two differences, both per seed against the completed `m6_<condition>_a` cells:

* against `diffusion_ekf` -- the filter's **cooperation gain**, the exact analogue
  of `local_only` against `diffusion_sgd_atc` for the gradient families.
* against `diffusion_ekf_onehop_mean_receiver` -- what the **whole deployable
  design** buys over running a filter per agent: cooperation and the one-hop adapt
  together, which is the number the paper actually wants.

Cross-cell pairing by seed is sound here for the reason M6 relies on it: the data
stream depends only on the configuration and the seed, so the same seed sees the
same trajectories in both cells. M6's own full-sharing contrast spans cells the
same way, and its paired spread (0.0001-0.0004) against a between-seed spread of
~0.0028 is the evidence that the pairing holds.

## What it is not

Not a competing method, and not ranked against the baselines: a filter that never
communicates is a **reference line**, as P5.24's mis-tuned arm is, and D77's floor
rule applies to it. It answers how much of the result is the cooperation.

Settings come from `m5_selection.json`, exactly as M6 takes them, so this arm is
tuned identically to the connected twin it is measured against. **It writes no
selection and no status that M6 reads.**
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

CONDITIONS = {"stationary": "m_stationary", "linear": "m_linear", "abrupt": "m_abrupt"}

#: The local-adapt, mean-combine filter. On an edgeless graph its combine is the
#: identity, so this single name gives the non-cooperating arm. `one_hop` would be
#: identical there -- with no neighbours, M_v = {v} either way -- so running both
#: would record the same filter twice under two names.
LEARNER = "diffusion_ekf"

#: The connected twins in M6 that each isolated cell is paired against.
TWINS = ["diffusion_ekf", "diffusion_ekf_onehop_mean_receiver"]

HORIZON, SEEDS, EVAL_EVERY = 1500, [0, 1, 2, 3, 4], 25

M5_SELECTION = ROOT / "results" / "m5_selection.json"
STATUS = ROOT / "results" / "m6iso_status.json"


def cell_name(condition: str) -> str:
    return f"m6iso_{condition}"


def m6_cell(condition: str) -> str:
    return f"m6_{condition}_a"


def filter_settings():
    """M5's selection, in the shape M6 carries it."""
    if not M5_SELECTION.exists():
        raise SystemExit(
            f"missing {M5_SELECTION.name}: this arm carries M5's selection so that it is\n"
            "tuned identically to the connected twin it is measured against.\n"
            "  python scripts/run_m5_diffusion.py --tie-break"
        )
    s = json.loads(M5_SELECTION.read_text(encoding="utf-8"))
    return {
        "transition": "scalar", "gamma": s["gamma"], "forgetting": "process_noise",
        "process_noise_q": s["process_noise_q"], "lambda_forget": 1.0,
        "prior_scale": s["prior_scale"],
    }


def config_for(args, condition: str, settings: dict):
    base = load_config(CONDITIONS[condition])
    return load_config(
        CONDITIONS[condition],
        overrides={
            "run": {"name": cell_name(condition), "horizon": args.horizon,
                    "seeds": args.seeds, "eval_every": EVAL_EVERY,
                    "device": args.device, "dtype": args.dtype},
            # n_components = N is the whole point: one agent per component, no edges.
            "graph": {"topology": "disconnected", "weights": "metropolis",
                      "params": {"n_components": base.graph.n_nodes}},
            "learners": [{"name": LEARNER, **settings}],
        },
    )


def sweep(args) -> int:
    settings = filter_settings()
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    print(f"M6 isolated: {len(CONDITIONS)} cells x {len(args.seeds)} seeds, "
          f"learner {LEARNER} on an edgeless graph, device {args.device}\n", flush=True)
    started = time.time()
    for index, condition in enumerate(CONDITIONS, start=1):
        name = cell_name(condition)
        note = run_one(config_for(args, condition, settings), None, None, args.fresh)
        status[name] = note
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{index}/{len(CONDITIONS)}] {name:<22} {note:<12} "
              f"{(time.time() - started) / 60:.0f} min", flush=True)

    print(f"\nisolated arm complete in {(time.time() - started) / 60:.1f} min\n")
    report()
    return 0


def report() -> None:
    print("  settled RMSE: the filter with no communication, against its connected twins\n")
    # Short labels: `diffusion_ekf_onehop_mean_receiver` is 34 characters and
    # overran its column, so the header no longer sat above the numbers it names.
    labels = {"diffusion_ekf": "vs diffusion_ekf",
              "diffusion_ekf_onehop_mean_receiver": "vs one-hop"}
    print(f"    {'condition':<14}{'isolated':>11}"
          + "".join(f"{labels.get(t, t):>27}{'gain':>11}" for t in TWINS))

    any_rows = False
    for condition in CONDITIONS:
        iso = settled(cell_name(condition), LEARNER)
        row = f"    {condition:<14}"
        row += f"{iso:>11.4f}" if iso != float("inf") else f"{'-':>11}"
        for twin in TWINS:
            connected = settled(m6_cell(condition), twin)
            if iso == float("inf") or connected == float("inf"):
                row += f"{'-':>38}"
                continue
            row += f"{connected:>27.4f}{iso - connected:>+11.4f}"
            any_rows = True
        print(row)

    if not any_rows:
        print("\n  no isolated cells on disk yet; run the sweep first")
        return
    print("\n  A positive gain is what cooperating buys: the isolated filter is worse by")
    print("  that much. Against diffusion_ekf it isolates the communication alone;")
    print("  against the one-hop variant it is the whole deployable design's value.")
    print("  For scale, M6 measured the same quantity for the gradient families at")
    print("  +0.0111 to +0.0146 (local_only against the ATC form).")
    print("\n  This is a reference line, not a ranked method (P5.24, D77).")


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
