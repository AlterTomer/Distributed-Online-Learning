r"""Where figures land.

The property under test is *runnability from a fresh clone*: the scripts must
write somewhere without anyone editing a constant, and every script must agree on
where that is. The bug being guarded against already happened once -- the figure
scripts hardcoded one machine's OneDrive folder, which made the figure builders
unrunnable for anybody else (design note D46).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dekf_bench.utils.paths import FIGURES_ENV, REPO_ROOT, figure_data_dir, figures_dir


@pytest.fixture
def unset_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer running the suite may well have the override exported."""
    monkeypatch.delenv(FIGURES_ENV, raising=False)


def test_the_default_is_inside_the_repository(unset_override: None) -> None:
    """A fresh clone runs with no configuration at all."""
    assert figures_dir() == REPO_ROOT / "figures"


def test_the_repo_root_is_the_repo_root() -> None:
    """`parents[3]` is a hop count, and a package move would silently change what
    it points at -- landing the figures somewhere plausible but wrong."""
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "src" / "dekf_bench").is_dir()


def test_an_absolute_override_is_taken_as_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "shared" / "figures"
    monkeypatch.setenv(FIGURES_ENV, str(elsewhere))
    assert figures_dir() == elsewhere


def test_a_relative_override_resolves_against_the_repo_not_the_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Otherwise a figure script would write to a different place depending on
    whether it was launched from the IDE, the repo root, or `scripts/`."""
    monkeypatch.setenv(FIGURES_ENV, "somewhere")
    monkeypatch.chdir(tmp_path)
    assert figures_dir() == REPO_ROOT / "somewhere"


def test_a_rooted_but_driveless_override_stays_rooted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Windows quirk worth pinning rather than rediscovering. `/shared/figures`
    carries no drive letter, so `is_absolute()` is False there and the value
    takes the relative branch -- but because it is rooted it still resolves to
    `<repo drive>:/shared/figures`, not to something nested under the repo. Both
    readings are defensible; what would be a bug is figures appearing at
    `<repo>/shared/figures` on Windows and `/shared/figures` on Linux."""
    monkeypatch.setenv(FIGURES_ENV, "/shared/figures")
    resolved = figures_dir()
    assert resolved.is_absolute() or resolved.root
    assert resolved.as_posix().endswith("/shared/figures")
    assert REPO_ROOT not in resolved.parents


def test_an_empty_override_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DEKF_FIGURES_DIR=` in a shell profile sets it to the empty string rather
    than unsetting it, and `Path("")` is `.` -- which would scatter figures into
    whatever directory the script happened to start in."""
    for blank in ("", "   "):
        monkeypatch.setenv(FIGURES_ENV, blank)
        assert figures_dir() == REPO_ROOT / "figures"


def test_the_override_is_read_on_every_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Not cached at import: a caller that changes the variable -- a test, or a
    script publishing two sets of figures -- must see the change."""
    monkeypatch.setenv(FIGURES_ENV, str(tmp_path / "first"))
    first = figures_dir()
    monkeypatch.setenv(FIGURES_ENV, str(tmp_path / "second"))
    assert figures_dir() != first


def test_the_cache_sits_beside_the_figures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The two travel together when the folder is copied into a talk."""
    monkeypatch.setenv(FIGURES_ENV, str(tmp_path))
    assert figure_data_dir() == figures_dir() / "figure_data"
