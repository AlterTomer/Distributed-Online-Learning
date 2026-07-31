"""Cross-platform task runner.

``make`` is not available on the development machine (Windows), so the targets of
IMPLEMENTATION.md section 10 live here and the ``Makefile`` is a thin alias that
delegates to this file. One source of truth, two entry points.

Usage::

    python tasks.py test
    python tasks.py lint
    python tasks.py --list
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Targets that are specified but whose scripts do not exist yet. Listing them
# with the phase that introduces them beats a confusing "file not found".
PENDING: dict[str, str] = {
    "reference": "phase 2 (scripts/train_reference.py)",
    "x0": "phase 3 (scripts/run_experiment.py)",
    "x1": "phase 3 (scripts/run_experiment.py)",
    "x1b": "phase 3 (scripts/run_experiment.py)",
    "x2": "phase 3 (scripts/run_experiment.py)",
    "x3": "phase 4 (scripts/run_sweep.py)",
    "x4": "phase 4 (scripts/run_sweep.py)",
    "x5": "phase 4 (scripts/run_sweep.py)",
    "x6": "phase 4 (scripts/run_sweep.py)",
    "sweep": "phase 4 (scripts/run_sweep.py)",
    "figures": "phase 4 (scripts/make_figures.py)",
}


def run(*cmd: str) -> int:
    """Run a command from the repo root, streaming its output."""
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def task_install() -> int:
    """Install the package in editable mode with dev extras."""
    return run(PY, "-m", "pip", "install", "-e", ".[dev]")


def task_test() -> int:
    """Full test suite."""
    return run(PY, "-m", "pytest")


def task_test_fast() -> int:
    """Everything except tests marked slow."""
    return run(PY, "-m", "pytest", "-m", "not slow")


def task_lint() -> int:
    """Lint and format check. Does not modify files."""
    failed = 0
    failed |= run(PY, "-m", "ruff", "check", "src", "tests", "scripts", "tasks.py")
    failed |= run(PY, "-m", "black", "--check", "src", "tests", "scripts", "tasks.py")
    return failed


def task_format() -> int:
    """Apply formatting and auto-fixable lint rules."""
    failed = 0
    failed |= run(PY, "-m", "ruff", "check", "--fix", "src", "tests", "scripts", "tasks.py")
    failed |= run(PY, "-m", "black", "src", "tests", "scripts", "tasks.py")
    return failed


def task_typecheck() -> int:
    """Static types. Non-blocking at first (IMPLEMENTATION.md section 1)."""
    return run(PY, "-m", "mypy")


def task_clean() -> int:
    """Remove caches and build artefacts. Leaves data/ and results/ alone."""
    patterns = [
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        "src/dekf_bench.egg-info",
        ".coverage",
        "htmlcov",
    ]
    for name in patterns:
        target = ROOT / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"removed {name}/")
        elif target.is_file():
            target.unlink()
            print(f"removed {name}")
    for pycache in ROOT.rglob("__pycache__"):
        if ".venv" in pycache.parts:
            continue
        shutil.rmtree(pycache, ignore_errors=True)
    return 0


TASKS: dict[str, Callable[[], int]] = {
    "install": task_install,
    "test": task_test,
    "test-fast": task_test_fast,
    "lint": task_lint,
    "format": task_format,
    "typecheck": task_typecheck,
    "clean": task_clean,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", help="task to run")
    parser.add_argument("--list", action="store_true", help="list available tasks")
    args = parser.parse_args()

    if args.list or args.target is None:
        print("Available:")
        for name, fn in TASKS.items():
            doc = (fn.__doc__ or "").splitlines()[0]
            print(f"  {name:<12} {doc}")
        print("\nNot yet implemented:")
        for name, phase in PENDING.items():
            print(f"  {name:<12} arrives in {phase}")
        return 0

    if args.target in PENDING:
        print(f"Target '{args.target}' arrives in {PENDING[args.target]}.", file=sys.stderr)
        return 2

    if args.target not in TASKS:
        print(f"Unknown target '{args.target}'. Try --list.", file=sys.stderr)
        return 2

    return TASKS[args.target]()


if __name__ == "__main__":
    raise SystemExit(main())
