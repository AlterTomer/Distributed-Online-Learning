"""Packaging and environment smoke tests.

These are deliberately boring. Their value is that they fail loudly when the
editable install is stale or the interpreter is the wrong one -- two mistakes
that otherwise show up much later as a confusing ImportError inside a run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_package_imports() -> None:
    import dekf_bench

    assert dekf_bench.__version__ == "0.1.0"


def test_package_resolves_to_src_layout(repo_root: Path) -> None:
    """The import must come from ``src/dekf_bench``, not a stray copy at the root."""
    import dekf_bench

    assert dekf_bench.__file__ is not None
    location = Path(dekf_bench.__file__).resolve().parent
    assert location == (repo_root / "src" / "dekf_bench").resolve()


def test_python_version_meets_floor() -> None:
    assert sys.version_info >= (3, 11), "torch.func requires the 3.11+ line we target"


@pytest.mark.parametrize(
    "module",
    ["torch", "torchvision", "numpy", "pandas", "pyarrow", "networkx", "yaml", "matplotlib"],
)
def test_runtime_dependency_importable(module: str) -> None:
    __import__(module)


def test_torch_is_recent_enough() -> None:
    import torch

    major, minor = (int(part) for part in torch.__version__.split(".")[:2])
    assert (major, minor) >= (2, 2), "torch.func wrappers in models/functional.py need >= 2.2"


def test_task_runner_lists_targets(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "tasks.py", "--list"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "test-fast" in result.stdout
    assert "arrives in phase" in result.stdout


def test_task_runner_rejects_unknown_target(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "tasks.py", "does-not-exist"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Unknown target" in result.stderr
