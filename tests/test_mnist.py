"""MNIST loading, caching and tensor conversion.

Structural tests build small synthetic splits, so they are fast and run
anywhere. The handful that need the real dataset are marked ``needs_data`` and
skip when it has not been downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dekf_bench.data.mnist import (
    CACHE_VERSION,
    IMAGE_SIZE,
    MNIST_MEAN,
    MNIST_STD,
    NUM_CLASSES,
    SPLIT_SIZES,
    DataError,
    MnistSplit,
    _load_cache,
    _write_cache,
    channel_statistics,
    class_counts,
    default_data_dir,
    is_cached,
    load_mnist,
    load_split,
)

needs_data = pytest.mark.needs_data


def synthetic(n: int = 8, split: str = "train") -> MnistSplit:
    """A structurally valid split, without touching the dataset."""
    generator = torch.Generator().manual_seed(0)
    return MnistSplit(
        images=torch.rand(n, 1, IMAGE_SIZE, IMAGE_SIZE, generator=generator),
        labels=torch.arange(n, dtype=torch.int64) % NUM_CLASSES,
        split=split,
    )


@pytest.fixture(scope="session")
def real_train() -> MnistSplit:
    if not is_cached():
        pytest.skip("MNIST not cached; run scripts/check_data.py once")
    return load_split("train", download=False)


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def test_split_reports_its_length() -> None:
    assert len(synthetic(5)) == 5


def test_wrong_image_shape_is_rejected() -> None:
    with pytest.raises(DataError, match="expected images of shape"):
        MnistSplit(
            images=torch.rand(4, 28, 28), labels=torch.zeros(4, dtype=torch.int64), split="x"
        )


def test_label_count_must_match_image_count() -> None:
    with pytest.raises(DataError, match="4 images but"):
        MnistSplit(
            images=torch.rand(4, 1, IMAGE_SIZE, IMAGE_SIZE),
            labels=torch.zeros(3, dtype=torch.int64),
            split="x",
        )


def test_integer_images_are_rejected() -> None:
    """uint8 here means the /255 conversion was skipped."""
    with pytest.raises(DataError, match="must be float32 or float64"):
        MnistSplit(
            images=torch.zeros(2, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8),
            labels=torch.zeros(2, dtype=torch.int64),
            split="x",
        )


def test_float64_images_are_allowed() -> None:
    """The exactness check runs the whole pipeline in double precision."""
    data = MnistSplit(
        images=torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64),
        labels=torch.zeros(2, dtype=torch.int64),
        split="x",
    )
    assert data.images.dtype == torch.float64


def test_labels_must_be_int64() -> None:
    with pytest.raises(DataError, match="must be int64"):
        MnistSplit(
            images=torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE),
            labels=torch.zeros(2, dtype=torch.int32),
            split="x",
        )


def test_split_is_frozen() -> None:
    import dataclasses

    data = synthetic()
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.split = "other"  # type: ignore[misc]


def test_unknown_split_name_is_rejected() -> None:
    with pytest.raises(DataError, match="unknown split"):
        load_split("validation")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# subsetting and conversion
# --------------------------------------------------------------------------- #


def test_subset_selects_the_requested_samples() -> None:
    data = synthetic(10)
    indices = torch.tensor([2, 5, 7])
    subset = data.subset(indices)
    assert len(subset) == 3
    assert torch.equal(subset.labels, data.labels[indices])


def test_subset_copies_rather_than_aliasing() -> None:
    """A shard must not share storage with the full tensor.

    If it did, an in-place bug in one agent would be visible to every other
    agent, which is the hardest kind of bug to attribute in this codebase.
    """
    data = synthetic(10)
    subset = data.subset(torch.tensor([0, 1]))
    subset.images.add_(1.0)
    assert not torch.allclose(subset.images[0], data.images[0])


def test_to_converts_dtype_but_leaves_labels_integral() -> None:
    converted = synthetic().to(dtype=torch.float64)
    assert converted.images.dtype == torch.float64
    assert converted.labels.dtype == torch.int64


def test_class_counts_covers_every_class() -> None:
    counts = class_counts(synthetic(30))
    assert counts.shape == (NUM_CLASSES,)
    assert int(counts.sum()) == 30


def test_channel_statistics_match_a_known_case() -> None:
    ones = torch.ones(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    mean, std = channel_statistics(ones)
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(0.0)


def test_channel_statistics_reject_an_empty_tensor() -> None:
    with pytest.raises(DataError, match="empty tensor"):
        channel_statistics(torch.empty(0))


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #


def test_cache_round_trips(tmp_path: Path) -> None:
    original = synthetic(6)
    cache = tmp_path / f"train_v{CACHE_VERSION}.pt"
    _write_cache(cache, original)

    reloaded = _load_cache(cache, "train")
    assert reloaded is not None
    assert torch.equal(reloaded.images, original.images)
    assert torch.equal(reloaded.labels, original.labels)


def test_cache_write_is_atomic(tmp_path: Path) -> None:
    """No stray .tmp file survives, so an interrupted write cannot masquerade
    as a valid cache."""
    cache = tmp_path / "train.pt"
    _write_cache(cache, synthetic(3))
    assert cache.is_file()
    assert not cache.with_suffix(".tmp").exists()


def test_corrupt_cache_is_reported_as_unreadable(tmp_path: Path) -> None:
    cache = tmp_path / "train.pt"
    cache.write_bytes(b"not a torch file")
    assert _load_cache(cache, "train") is None


def test_truncated_cache_is_reported_as_unreadable(tmp_path: Path) -> None:
    cache = tmp_path / "train.pt"
    _write_cache(cache, synthetic(6))
    payload = cache.read_bytes()
    cache.write_bytes(payload[: len(payload) // 2])
    assert _load_cache(cache, "train") is None


def test_cache_holding_the_wrong_shape_is_rejected(tmp_path: Path) -> None:
    cache = tmp_path / "train.pt"
    torch.save(
        {"images": torch.rand(3, 28, 28), "labels": torch.zeros(3, dtype=torch.int64)}, cache
    )
    assert _load_cache(cache, "train") is None


def test_missing_data_without_download_raises_rather_than_fetching(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="download=False"):
        load_split("train", root=tmp_path, download=False)


# --------------------------------------------------------------------------- #
# the real dataset
# --------------------------------------------------------------------------- #


@needs_data
def test_split_sizes_are_exact(real_train: MnistSplit) -> None:
    assert len(real_train) == SPLIT_SIZES["train"]


@needs_data
def test_intensities_are_raw_and_unnormalized(real_train: MnistSplit) -> None:
    """This module must not normalize: rotation fills with zero, and zero has to
    remain the black background (see the module docstring)."""
    assert float(real_train.images.min()) == 0.0
    assert float(real_train.images.max()) == pytest.approx(1.0)


@needs_data
def test_conversion_reproduces_the_canonical_statistics(real_train: MnistSplit) -> None:
    """A wrong uint8 scaling would show up here and nowhere else."""
    mean, std = channel_statistics(real_train.images)
    assert mean == pytest.approx(MNIST_MEAN, abs=1e-4)
    assert std == pytest.approx(MNIST_STD, abs=1e-4)


@needs_data
def test_every_class_is_present(real_train: MnistSplit) -> None:
    counts = class_counts(real_train)
    assert int(counts.min()) > 0
    assert int(counts.sum()) == SPLIT_SIZES["train"]


@needs_data
def test_train_and_test_are_different_data() -> None:
    train, test = load_mnist(download=False)
    assert len(train) == SPLIT_SIZES["train"]
    assert len(test) == SPLIT_SIZES["test"]
    assert not torch.equal(train.images[: len(test)], test.images)


@needs_data
def test_cached_load_does_not_need_the_network() -> None:
    """After one download, every later run is offline."""
    assert is_cached(default_data_dir())
    assert len(load_split("test", download=False)) == SPLIT_SIZES["test"]


@needs_data
def test_shard_sized_subset_is_what_an_agent_will_hold(real_train: MnistSplit) -> None:
    """6000 samples: one agent's shard at N=10."""
    shard = real_train.subset(torch.arange(6000))
    assert len(shard) == 6000
    assert shard.images.dtype == torch.float32
