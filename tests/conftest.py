"""Shared fixtures.

Note that nothing here inserts ``src/`` onto ``sys.path``. The package must be
installed (``python tasks.py install``) so that the tests exercise the installed
distribution and packaging mistakes surface here rather than at run time
(IMPLEMENTATION.md section 1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    """The ``configs/`` tree that experiment configs are resolved against."""
    return repo_root / "configs"


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """A throwaway output directory, so no test writes into ``results/``."""
    path = tmp_path / "out"
    path.mkdir()
    return path
