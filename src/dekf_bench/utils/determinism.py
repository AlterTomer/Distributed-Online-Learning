"""Determinism flags and run provenance.

Two jobs. First, put the process into a state where the same seed produces the
same numbers. Second, record enough about the environment that a result which
*cannot* be reproduced can at least be explained.

Two of the required settings cannot be applied from inside a running process
and are therefore reported rather than silently ignored:

``PYTHONHASHSEED``
    Read by the interpreter at startup. Setting ``os.environ`` afterwards has no
    effect on the current process. It matters because set and dict iteration
    order feeds anything that iterates a set of node ids.

``CUBLAS_WORKSPACE_CONFIG``
    Read by cuBLAS when the CUDA context is created. Set after that, and
    deterministic matmuls raise at the first op instead of at configuration
    time. Only relevant on CUDA; phases 1-4 run on CPU.

Both are set for *child* processes and warned about for this one, so the fix
(export the variable before launching) is discoverable rather than mysterious.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: cuBLAS accepts ":4096:8" (more memory, no perf hit) or ":16:8" (less memory,
#: slower). The former is the usual recommendation.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class DeterminismWarning(UserWarning):
    """Raised when a determinism setting could not be applied in-process."""


def set_determinism(seed: int, *, device: str = "cpu", warn: bool = True) -> None:
    """Seed every RNG and switch on deterministic kernels.

    Args:
        seed: master seed. Prefer explicit generators from
            :mod:`dekf_bench.runner.seeding` over the global RNGs this seeds;
            the global seeding here is a backstop for library code that reaches
            for the global state anyway.
        device: ``"cpu"`` or ``"cuda"``. CUDA additionally needs the cuBLAS
            workspace variable.
        warn: emit :class:`DeterminismWarning` for settings that arrived too
            late to take effect in this process.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if device.startswith("cuda"):
        _configure_cublas(warn=warn)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    if warn and os.environ.get("PYTHONHASHSEED") is None:
        warnings.warn(
            "PYTHONHASHSEED is unset. It is read at interpreter startup, so it cannot be "
            "applied now; set-iteration order in this process is not pinned. Export "
            "PYTHONHASHSEED=0 before launching if you need bit-identical results across "
            "machines. Child processes started from here will inherit it.",
            DeterminismWarning,
            stacklevel=2,
        )
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _configure_cublas(*, warn: bool) -> None:
    """Set the cuBLAS workspace, warning if the CUDA context already exists."""
    import torch

    already_set = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if already_set is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        if warn and torch.cuda.is_available() and torch.cuda.is_initialized():
            warnings.warn(
                "CUBLAS_WORKSPACE_CONFIG was set after the CUDA context was created, so "
                "cuBLAS will not see it and deterministic matmuls will raise at the first "
                f"operation. Export CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG} before "
                "launching, or call set_determinism() before touching CUDA.",
                DeterminismWarning,
                stacklevel=3,
            )
    elif warn and already_set != CUBLAS_WORKSPACE_CONFIG:
        warnings.warn(
            f"CUBLAS_WORKSPACE_CONFIG is {already_set!r}, not the expected "
            f"{CUBLAS_WORKSPACE_CONFIG!r}. Leaving it alone.",
            DeterminismWarning,
            stacklevel=3,
        )


def resolve_device(device: str) -> str:
    """Turn ``"auto"`` into a concrete device string."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(name: str) -> Any:
    """Map a config dtype name onto a torch dtype."""
    import torch

    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if name not in dtypes:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(dtypes)}")
    return dtypes[name]


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunMetadata:
    """Everything needed to explain a result that will not reproduce."""

    git_sha: str
    git_dirty: bool
    python_version: str
    platform: str
    torch_version: str
    numpy_version: str
    cuda_available: bool
    cuda_version: str | None
    packages: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "packages": dict(self.packages),
        }


def git_revision(repo: str | Path | None = None) -> tuple[str, bool]:
    """The current commit and whether the working tree is dirty.

    Returns ``("unknown", True)`` outside a repository or without git on PATH.
    Dirty is reported pessimistically: a result produced from uncommitted
    changes is not traceable to a commit, and pretending otherwise is worse than
    saying so.
    """
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown", True
    return sha, bool(status)


def collect_metadata(repo: str | Path | None = None) -> RunMetadata:
    """Snapshot the environment, for the run's metadata file."""
    import numpy as np
    import torch

    sha, dirty = git_revision(repo)
    packages = {}
    for name in ("pandas", "pyarrow", "networkx", "torchvision", "matplotlib"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - all are hard dependencies
            continue
        packages[name] = getattr(module, "__version__", "unknown")

    return RunMetadata(
        git_sha=sha,
        git_dirty=dirty,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        packages=packages,
    )
