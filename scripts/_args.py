"""Shared CLI plumbing for the sweep scripts.

Every sweep carries the same run knobs -- horizon, seeds, device, dtype, dataset,
`--fresh` -- and used to carry them as module constants only, so changing one
meant editing source. As flags with those constants as defaults, ``--help``
answers "what can I change here" and zero-argument invocation still means what it
did, which keeps every command in `docs/experiments.md` valid.

Experiment *definitions* stay in code: X18's periods and X22's exponents are the
experiment, and a flag that changed them would produce a run whose name no longer
describes it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def sweep_parser(
    description: str,
    *,
    horizon: int,
    seeds: Sequence[int],
    device: str = "auto",
    dtype: str = "float64",
    dataset: str = "mnist",
) -> argparse.ArgumentParser:
    """A parser carrying the knobs every sweep shares.

    Callers add their own with ``parser.add_argument`` before parsing, and read
    the shared ones off the namespace.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--horizon", type=int, default=horizon,
                        help="steps per run")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(seeds),
                        help="seeds to run, space separated")
    parser.add_argument("--device", default=device, choices=["auto", "cpu", "cuda"],
                        help="where tensors live")
    parser.add_argument("--dtype", default=dtype, choices=["float32", "float64"],
                        help="float64 is required for the exactness checks")
    parser.add_argument("--dataset", default=dataset,
                        help="resolved through data/registry.py")
    parser.add_argument("--fresh", action="store_true",
                        help="discard completed cells and redo them")
    return parser
