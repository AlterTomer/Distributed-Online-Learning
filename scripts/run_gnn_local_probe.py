r"""D3c's local-adapt probe: does another q close local adapt's gap on the twin? (D148)

    python scripts/run_gnn_local_probe.py                  # q = 6e-8, 6e-7, 6e-6 at seed 0
    python scripts/run_gnn_local_probe.py --q 6e-6 --seed 1

On the twin (step 03) local adapt under the full covariance settles 0.037 behind one-hop,
against 0.003 on the Transformer's tau-spread cell, and full sharing recovers 0.031 of it.
Its settings are one-hop's, carried as on M5/M6 (D105). This runs local adapt alone on the
twin's stationary cell, full covariance, everything else as `run_gnn.py --twin`, at the q
given; 6e-7 is the carried setting, so it is the control and should reproduce
`gnt_full_a`'s local adapt at the same seed. A probe: it explains, it does not replace
the reported cells. CPU by default.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ekf_generalization import run_one  # noqa: E402
from run_gnn import HORIZON, LOCAL, config_for, read, settings_of  # noqa: E402
from run_m3_rates import settled  # noqa: E402

QS = [6e-8, 6e-7, 6e-6]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--q", type=float, nargs="+", default=QS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    args.twin, args.smoke = True, False
    reference, selection = read(args, "reference"), read(args, "filters")
    _central, diffusion, r = settings_of(selection, "full")
    for q in args.q:
        # A shorter horizon (a timing run) is its own cell, so the full run never skips it.
        horizon = "" if args.horizon == HORIZON else f"_T{args.horizon}"
        name = (f"gnt_probe_local_q{q:g}_s{args.seed}{horizon}"
                .replace(".", "p").replace("-", "m"))
        started = time.time()
        note = run_one(config_for(args, name, "stationary",
                                  [{"name": LOCAL, **diffusion, "process_noise_q": q}],
                                  [args.seed], r, reference), None, None, args.fresh)
        print(f"  {name:<34} {note:<14} settled {settled(name, LOCAL):.4f}"
              f"  ({(time.time() - started) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
