r"""Name to dataset, so a script never imports one directly.

**Why this exists.** Before it, eighteen scripts imported ``load_mnist`` by name,
``EnvConfig`` had no dataset field at all, and the only record of which dataset a
run used was the *filename* of its env config. Adding a second dataset then meant
editing eighteen scripts, and the obvious wrong fix -- a script per dataset --
would multiply the repository along an axis the scripts do not actually differ
on: a sweep script differs in what it **varies**, never in what it **loads**.

So the dataset becomes a name, resolved here, exactly as learners and models
already are (`learners/registry.py`, `models/registry.py`).

**A dataset is more than a loader, which is the part worth reading.** A
`DatasetSpec` carries the shape facts the rest of the benchmark is entitled to
ask about, because several of them are load-bearing well outside `data/`:

* ``num_classes`` sets the softmax Fisher's rank $q-1$, which is the width of
  every $\bm B$ block the filter builds and ships.
* ``image_size`` decides $p$ once a model is attached, and $p$ is what makes a
  dense covariance feasible at all -- $14\times14$ downsampling is the only
  reason $p=2908$ rather than $10^5$ (IMPLEMENTATION.md section 13.10).
* ``rotation_cap_degrees`` is a property of the **task**, not the method: past
  roughly $45^{\circ}$ a rotated 6 is a 9, so the Bayes error of the problem
  itself rises and a climbing error would measure label ambiguity rather than
  tracking failure. Every drift-rate result from X9 through X18 is stated
  relative to it. A dataset with no such collision -- or one whose drift channel
  does not act on orientation -- carries a different cap, or none, and
  ``None`` here says exactly that.

What this registry does **not** solve is that the drift channel itself is
rotation, and the schedules in `env/drift.py` describe angles. Adding a dataset
whose meaningful shift is not orientation is a task-definition change, and this
is necessary rather than sufficient for it (IMPLEMENTATION.md section 15).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dekf_bench.data.mnist import DataError, ImageSplit
from dekf_bench.data.mnist import is_cached as _mnist_is_cached
from dekf_bench.data.mnist import load_mnist as _load_mnist


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset: how to load it, and the facts other modules need about it."""

    name: str
    load: Callable[..., tuple[ImageSplit, ImageSplit]]
    cached: Callable[..., bool]
    channels: int
    image_size: int
    num_classes: int
    #: The largest rotation under which labels stay well defined, or ``None`` for
    #: a dataset whose labels rotation cannot confuse. Read by `env/drift.py`.
    rotation_cap_degrees: float | None

    @property
    def input_dim(self) -> int:
        """Pixels per sample at native resolution, before any downsampling."""
        return self.channels * self.image_size * self.image_size


#: Every dataset a config may name. One entry today; the point of the table is
#: that the second one is an entry rather than an edit to eighteen scripts.
DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        load=_load_mnist,
        cached=_mnist_is_cached,
        channels=1,
        image_size=28,
        num_classes=10,
        # 45 degrees: past it a rotated 6 is a 9 and, less sharply, a 2 is a 7.
        rotation_cap_degrees=45.0,
    ),
}


def dataset_names() -> list[str]:
    return sorted(DATASETS)


def spec(name: str) -> DatasetSpec:
    """The named dataset's spec, or a refusal listing what exists."""
    if name not in DATASETS:
        raise DataError(f"unknown dataset {name!r}; available: {dataset_names()}")
    return DATASETS[name]


def load_dataset(
    name: str, root: str | Path | None = None, *, download: bool = False
) -> tuple[ImageSplit, ImageSplit]:
    """``(train, test)`` for the named dataset.

    ``download`` defaults to **False** here, unlike the underlying loaders: a
    sweep that silently reaches for the network mid-run is a worse failure than
    one that refuses at the first cell, and `check_data.py` is the place that
    downloads on purpose.
    """
    return spec(name).load(root, download=download)


def dataset_is_cached(name: str, root: str | Path | None = None) -> bool:
    """Whether the named dataset is on disk, so a sweep can refuse early."""
    return spec(name).cached(root)
