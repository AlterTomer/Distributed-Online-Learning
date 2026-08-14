r"""Where generated figures go.

**A fresh clone must be able to run every script without editing a constant.**
The figure scripts used to hardcode one machine's OneDrive folder, which meant
`make_figures.py` was unrunnable for anyone else and wrote nothing a reviewer
could find. That is the same failure as shipping a personal path in a config.

So the default is repo-relative and gitignored, and publishing elsewhere is an
environment variable rather than an edit:

    DEKF_FIGURES_DIR=/some/shared/folder python scripts/make_figures.py

Set it once in the shell profile, or in the IDE's run configuration, and every
script that writes or reads a figure agrees on the location -- including the
meeting-document builders, which are not tracked here (design note D46) but
import this to find the PNGs they embed.

A relative value is resolved against the repository root, not the working
directory, so ``DEKF_FIGURES_DIR=figures`` means the same thing from anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The environment variable that overrides the default location.
FIGURES_ENV = "DEKF_FIGURES_DIR"

#: Repo-relative default. Gitignored -- figures are generated, and a PNG in the
#: history would go stale the moment a run is re-tuned.
DEFAULT_FIGURES_DIRNAME = "figures"

#: `src/dekf_bench/utils/paths.py` -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]


def figures_dir() -> Path:
    """The directory figures are written to and read from.

    Not cached: the environment variable is read on every call so a test can
    point it at a temporary directory without reloading the module.
    """
    override = os.environ.get(FIGURES_ENV, "").strip()
    if not override:
        return REPO_ROOT / DEFAULT_FIGURES_DIRNAME
    path = Path(override).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def figure_data_dir() -> Path:
    """Where the two-stage pipeline caches the reduced tables it draws from.

    Beside the PNGs on purpose: this is the small tidy table that survives into
    a talk or a paper long after the raw ``results/`` run has been superseded,
    so the two travel together when the folder is copied.
    """
    return figures_dir() / "figure_data"
