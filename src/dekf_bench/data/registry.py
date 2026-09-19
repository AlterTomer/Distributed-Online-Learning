r"""Name to dataset, so a script never imports one directly.

A sweep script differs in what it **varies**, never in what it **loads**, so a
script per dataset would multiply the repository along an axis it does not differ
on. The dataset is a name, resolved here, as learners and models already are.

A spec carries more than a loader because three of its facts are load-bearing
outside `data/`: ``num_classes`` is the softmax Fisher's rank and so the width of
every $\bm B$ block shipped; ``image_size`` fixes $p$, which is what makes a
dense covariance feasible; and ``rotation_cap_degrees`` is a property of the
*task* -- past roughly $45^{\circ}$ a rotated 6 is a 9, so error past the cap
measures label ambiguity rather than tracking failure, and every drift-rate
result from X9 on is stated relative to it. ``None`` says a dataset has no such
collision.

This does not make a non-rotation dataset usable: the drift channel is rotation
and `env/drift.py` speaks in angles, so that is a task-definition change and this
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
    #: ``"image"`` for a dataset loaded once and rotated at serve time;
    #: ``"series"`` for one generated from a law per run (Mackey--Glass), which has
    #: no files, no classes and no image shape -- its facts live in ``env.series``,
    #: and its environment is `env/series.py` rather than `env/environment.py`.
    kind: str = "image"

    @property
    def input_dim(self) -> int:
        """Pixels per sample at native resolution, before any downsampling."""
        return self.channels * self.image_size * self.image_size


def _generated(root: str | Path | None = None, *, download: bool = False) -> tuple[None, None]:
    """A generated dataset has no splits to load: the environment integrates it."""
    return None, None


def _always_available(root: str | Path | None = None) -> bool:
    return True


#: Every dataset a config may name. The point of the table is that the second one
#: is an entry rather than an edit to eighteen scripts -- which Mackey--Glass now is.
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
    # The second task (docs/mackey_glass_plan.md). No rotation, so no cap here; the
    # drift schedules' 45-degree cap stands for the channel's usable span instead,
    # which the pilot sets (env.series.span).
    "mackey_glass": DatasetSpec(
        name="mackey_glass",
        load=_generated,
        cached=_always_available,
        channels=0,
        image_size=0,
        num_classes=0,
        rotation_cap_degrees=None,
        kind="series",
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
