"""MNIST: download, cache, tensor conversion.

Loads once into two dense tensors and keeps them there. At 60 000 x 28 x 28
float32 the training split is 188 MB, which is cheaper than a ``DataLoader`` and
avoids per-item PIL conversion on a path that runs 1500 times per run.

**This module does not normalize and does not rotate.** It returns raw
intensities in ``[0, 1]``, and the order in which later transforms are applied
matters (see :data:`MNIST_MEAN`):

    rotate at 28x28  ->  downsample to 14x14  ->  normalize

Rotation fills the corners it exposes with zero. On raw intensities zero *is*
the black background, so the fill is invisible. Normalize first and zero becomes
-0.42, so every rotated image acquires four bright corners that no unrotated
image has -- a signal correlated with the drift state, which a classifier will
happily learn instead of the digit. Keeping normalization downstream of rotation
is what prevents that, and it is why this module hands back raw values.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]

#: Expected split sizes. Checked on load, because a truncated download otherwise
#: shows up much later as a shard budget that mysteriously does not add up.
SPLIT_SIZES: dict[str, int] = {"train": 60_000, "test": 10_000}

IMAGE_SIZE = 28
NUM_CLASSES = 10

#: Standard MNIST statistics, for reference. They describe the raw 28x28 data.
#: Average-pooling to 14x14 preserves the mean but shrinks the standard
#: deviation, so a pipeline that downsamples should compute its own statistics
#: with :func:`channel_statistics` rather than reusing these.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

#: Bumping this invalidates every cached tensor file.
CACHE_VERSION = 1


class DataError(RuntimeError):
    """Raised when the dataset is missing, unreadable or the wrong shape."""


@dataclass(frozen=True)
class MnistSplit:
    """One split, fully materialised.

    Attributes:
        images: ``(n, 1, 28, 28)`` float32 in ``[0, 1]``.
        labels: ``(n,)`` int64 in ``[0, 9]``.
        split: which split this is, for error messages and provenance.
    """

    images: torch.Tensor
    labels: torch.Tensor
    split: str

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise DataError(
                f"{self.split}: expected images of shape (n, 1, {IMAGE_SIZE}, {IMAGE_SIZE}), "
                f"got {tuple(self.images.shape)}"
            )
        if self.labels.ndim != 1 or self.labels.shape[0] != self.images.shape[0]:
            raise DataError(
                f"{self.split}: {self.images.shape[0]} images but "
                f"{tuple(self.labels.shape)} labels"
            )
        # float64 is legal, not an accident: the exactness check runs the whole
        # pipeline in double precision. Anything else -- uint8 in particular --
        # means a conversion was skipped.
        if self.images.dtype not in (torch.float32, torch.float64):
            raise DataError(
                f"{self.split}: images must be float32 or float64, got {self.images.dtype}"
            )
        if self.labels.dtype != torch.int64:
            raise DataError(f"{self.split}: labels must be int64, got {self.labels.dtype}")

    def subset(self, indices: torch.Tensor) -> MnistSplit:
        """The samples at ``indices``, as a new split.

        Used by the per-agent shards. Indexing copies, which is what we want:
        a shard that aliased the full tensor would keep it alive and make an
        in-place bug in one agent visible to all of them.
        """
        return MnistSplit(
            images=self.images[indices].clone(),
            labels=self.labels[indices].clone(),
            split=f"{self.split}[{len(indices)}]",
        )

    def to(self, *, dtype: torch.dtype | None = None, device: str | None = None) -> MnistSplit:
        """A copy on another dtype or device. Labels stay int64."""
        images = self.images
        if dtype is not None:
            images = images.to(dtype)
        if device is not None:
            images = images.to(device)
        labels = self.labels.to(device) if device is not None else self.labels
        return MnistSplit(images=images, labels=labels, split=self.split)


def default_data_dir() -> Path:
    """The gitignored ``data/`` directory at the repository root."""
    return Path(__file__).resolve().parents[3] / "data"


def _cache_path(root: Path, split: str) -> Path:
    return root / "mnist" / f"{split}_v{CACHE_VERSION}.pt"


def load_split(
    split: Split = "train",
    root: str | Path | None = None,
    *,
    download: bool = True,
) -> MnistSplit:
    """Load one MNIST split, using the tensor cache when it exists.

    Args:
        split: ``"train"`` or ``"test"``.
        root: data directory. Defaults to the repository's ``data/``.
        download: fetch from the network if neither the cache nor the raw
            torchvision files are present. Set ``False`` to guarantee the
            function never touches the network.

    Raises:
        DataError: if the data is absent and ``download`` is false, if the
            download fails, or if a split has the wrong size.
    """
    if split not in SPLIT_SIZES:
        raise DataError(f"unknown split {split!r}; expected 'train' or 'test'")

    root = Path(root) if root is not None else default_data_dir()
    cache = _cache_path(root, split)

    if cache.is_file():
        data = _load_cache(cache, split)
        if data is not None:
            return data
        logger.warning("discarding unreadable cache at %s", cache)
        cache.unlink(missing_ok=True)

    data = _from_torchvision(root, split, download=download)
    _write_cache(cache, data)
    return data


def _load_cache(cache: Path, split: str) -> MnistSplit | None:
    try:
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        return MnistSplit(images=payload["images"], labels=payload["labels"], split=split)
    except (OSError, EOFError, KeyError, RuntimeError, DataError, pickle.UnpicklingError):
        # A truncated or corrupt cache is recoverable: delete and rebuild.
        # `weights_only=True` reports damage as UnpicklingError rather than
        # RuntimeError, so it has to be caught explicitly -- omitting it turns a
        # half-written file into a crashed run instead of a two-second rebuild.
        return None


def _write_cache(cache: Path, data: MnistSplit) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename, so an interrupted write cannot leave
    # a truncated cache that looks valid.
    staging = cache.with_suffix(".tmp")
    torch.save({"images": data.images, "labels": data.labels}, staging)
    staging.replace(cache)
    logger.info("cached %s split (%d samples) to %s", data.split, len(data), cache)


def _from_torchvision(root: Path, split: str, *, download: bool) -> MnistSplit:
    """Read the raw torchvision files and convert them to dense tensors."""
    from torchvision.datasets import MNIST

    raw_root = root / "torchvision"
    already_present = (raw_root / "MNIST" / "raw").is_dir()
    if not already_present and not download:
        raise DataError(
            f"MNIST is not present under {raw_root} and download=False. "
            "Run scripts/check_data.py once with downloads enabled, or point "
            "root= at a directory that already holds it."
        )

    try:
        dataset = MNIST(root=str(raw_root), train=(split == "train"), download=download)
    except Exception as error:  # noqa: BLE001 - torchvision raises a variety of types
        raise DataError(
            f"could not obtain MNIST ({type(error).__name__}: {error}). "
            "The mirrors torchvision uses are occasionally unavailable; retry, or place the "
            f"raw files under {raw_root / 'MNIST' / 'raw'} by hand."
        ) from error

    # `.data` is a uint8 (n, 28, 28) tensor already, so this is a straight cast
    # rather than 60000 PIL round trips.
    images = dataset.data.to(torch.float32).div_(255.0).unsqueeze(1).contiguous()
    labels = dataset.targets.to(torch.int64).contiguous()

    expected = SPLIT_SIZES[split]
    if images.shape[0] != expected:
        raise DataError(
            f"{split} split has {images.shape[0]} samples, expected {expected}. "
            f"The download is probably incomplete; delete {raw_root} and retry."
        )
    return MnistSplit(images=images, labels=labels, split=split)


def load_mnist(
    root: str | Path | None = None, *, download: bool = True
) -> tuple[MnistSplit, MnistSplit]:
    """Both splits, as ``(train, test)``.

    They come from separate source files, so train/test leakage is impossible by
    construction rather than by an index check.
    """
    return load_split("train", root, download=download), load_split(
        "test", root, download=download
    )


def channel_statistics(images: torch.Tensor) -> tuple[float, float]:
    """Mean and standard deviation over every pixel.

    Compute these on the *transformed* canonical training images, not on the raw
    28x28 data: downsampling changes the standard deviation, and a normalization
    constant that drifts with the rotation would confound the drift measurement
    it is supposed to leave alone.
    """
    if images.numel() == 0:
        raise DataError("cannot compute statistics of an empty tensor")
    values = images.to(torch.float64)
    return float(values.mean()), float(values.std())


def class_counts(data: MnistSplit) -> torch.Tensor:
    """Samples per class, as a length-10 int64 tensor."""
    return torch.bincount(data.labels, minlength=NUM_CLASSES)


def is_cached(root: str | Path | None = None) -> bool:
    """Whether both splits are already on disk, so no network access is needed."""
    root = Path(root) if root is not None else default_data_dir()
    return all(_cache_path(root, split).is_file() for split in SPLIT_SIZES)
