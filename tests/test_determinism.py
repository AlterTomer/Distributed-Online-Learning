"""Determinism flags and run provenance."""

from __future__ import annotations

import os
import warnings

import pytest

from dekf_bench.utils.determinism import (
    CUBLAS_WORKSPACE_CONFIG,
    DeterminismWarning,
    collect_metadata,
    git_revision,
    resolve_device,
    set_determinism,
    torch_dtype,
)


@pytest.fixture(autouse=True)
def restore_torch_state():
    """Undo the global flags, so these tests do not leak into the rest of the suite."""
    import torch

    was_deterministic = torch.are_deterministic_algorithms_enabled()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    yield
    torch.use_deterministic_algorithms(was_deterministic)
    if hash_seed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = hash_seed


def test_set_determinism_enables_deterministic_algorithms() -> None:
    import torch

    set_determinism(0, warn=False)
    assert torch.are_deterministic_algorithms_enabled()


def test_set_determinism_makes_global_rngs_reproducible() -> None:
    import numpy as np
    import torch

    set_determinism(1234, warn=False)
    first = (torch.randn(4).clone(), np.random.rand(4).copy())

    set_determinism(1234, warn=False)
    second = (torch.randn(4).clone(), np.random.rand(4).copy())

    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_different_seeds_give_different_draws() -> None:
    import torch

    set_determinism(1, warn=False)
    first = torch.randn(4).clone()
    set_determinism(2, warn=False)
    assert not torch.equal(first, torch.randn(4))


def test_unset_pythonhashseed_warns_rather_than_pretending() -> None:
    os.environ.pop("PYTHONHASHSEED", None)
    with pytest.warns(DeterminismWarning, match="PYTHONHASHSEED"):
        set_determinism(0)


def test_pythonhashseed_is_exported_for_child_processes() -> None:
    os.environ.pop("PYTHONHASHSEED", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeterminismWarning)
        set_determinism(0)
    assert os.environ["PYTHONHASHSEED"] == "0"


def test_no_warning_when_pythonhashseed_is_already_set() -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeterminismWarning)
        set_determinism(0)


def test_cublas_config_value_is_the_documented_one() -> None:
    """A wrong value here surfaces as a runtime error deep inside a matmul."""
    assert CUBLAS_WORKSPACE_CONFIG == ":4096:8"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def test_resolve_device_passes_concrete_names_through() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_expands_auto() -> None:
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_torch_dtype_maps_config_names() -> None:
    import torch

    assert torch_dtype("float32") is torch.float32
    assert torch_dtype("float64") is torch.float64


def test_torch_dtype_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        torch_dtype("float16")


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def test_git_revision_reports_a_sha_in_this_repo() -> None:
    sha, _dirty = git_revision()
    assert sha == "unknown" or len(sha) == 40


def test_git_revision_outside_a_repo_is_reported_not_guessed(tmp_path) -> None:
    sha, dirty = git_revision(tmp_path)
    assert sha == "unknown"
    assert dirty is True, "an untraceable result must not claim to be clean"


def test_metadata_captures_the_environment() -> None:
    import torch

    metadata = collect_metadata()
    assert metadata.torch_version == torch.__version__
    assert metadata.python_version.startswith("3.")
    assert "pandas" in metadata.packages


def test_metadata_serialises_for_the_run_record() -> None:
    recorded = collect_metadata().as_dict()
    assert set(recorded) >= {
        "git_sha",
        "git_dirty",
        "python_version",
        "platform",
        "torch_version",
        "numpy_version",
        "cuda_available",
    }
