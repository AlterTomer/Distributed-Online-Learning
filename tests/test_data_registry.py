r"""The dataset registry: a name resolves to a loader, and to the facts about it.

Before this existed, eighteen scripts imported ``load_mnist`` by name and the
only record of which dataset a run used was the *filename* of its env config.
These tests pin the properties that make a second dataset an entry in a table
rather than an edit to every script.
"""

from __future__ import annotations

import pytest

from dekf_bench.data.mnist import DataError, ImageSplit
from dekf_bench.data.registry import (
    DATASETS,
    dataset_is_cached,
    dataset_names,
    load_dataset,
    spec,
)
from dekf_bench.utils.config import ConfigError, load_config


def test_every_registered_dataset_declares_the_facts_others_depend_on() -> None:
    """Not decoration: each of these is load-bearing outside ``data/``.

    ``num_classes`` sets the softmax Fisher's rank and so the width of every
    information block the filter ships; ``image_size`` decides $p$ once a model
    is attached, and $p$ is what makes a dense covariance feasible at all.
    """
    assert dataset_names(), "the registry is empty"
    for name in dataset_names():
        entry = spec(name)
        assert entry.name == name, "a spec must know its own key"
        assert entry.channels >= 1
        assert entry.image_size >= 1
        assert entry.num_classes >= 2
        assert entry.input_dim == entry.channels * entry.image_size**2
        assert callable(entry.load) and callable(entry.cached)


def test_the_rotation_cap_is_a_property_of_the_task() -> None:
    r"""45 degrees for MNIST, because past it a rotated 6 is a 9.

    ``None`` is a legal value and means the labels survive any rotation. The
    field exists so that a dataset without the collision does not silently
    inherit MNIST's cap, under which every drift-rate result from X9 to X18 is
    stated.
    """
    assert spec("mnist").rotation_cap_degrees == pytest.approx(45.0)
    for name in dataset_names():
        cap = spec(name).rotation_cap_degrees
        assert cap is None or cap > 0.0


def test_an_unknown_dataset_is_refused_and_lists_what_exists() -> None:
    with pytest.raises(DataError, match="unknown dataset"):
        spec("cifar100")
    with pytest.raises(DataError, match="available"):
        load_dataset("cifar100")
    with pytest.raises(DataError, match="unknown dataset"):
        dataset_is_cached("cifar100")


def test_the_config_validates_the_dataset_against_the_registry() -> None:
    """A typo is a refusal at load time, not a missing-file error mid-sweep."""
    base = {"run": {"name": "probe", "horizon": 10, "seeds": [0]}}
    config = load_config("x1_stationary", overrides=base)
    assert config.env.dataset in DATASETS

    with pytest.raises(ConfigError, match="env.dataset"):
        load_config("x1_stationary", overrides={**base, "env": {"dataset": "mnsit"}})


def test_load_dataset_does_not_download_by_default() -> None:
    r"""Unlike the underlying loader, whose default is ``download=True``.

    A sweep that silently reaches for the network mid-run is a worse failure than
    one that refuses at the first cell; `check_data.py` is where downloading is
    the point.
    """
    import inspect

    assert inspect.signature(load_dataset).parameters["download"].default is False


def test_the_split_type_is_named_for_what_it_is() -> None:
    """`ImageSplit`, not `MnistSplit`.

    ``environment.py``, ``evaluation/evalsets.py`` and ``evaluation/reference.py``
    all annotate with it, so a name carrying one dataset would be a lie in three
    modules the moment a second arrives.
    """
    import torch

    data = ImageSplit(
        images=torch.rand(4, 3, 8, 8),
        labels=torch.arange(4, dtype=torch.int64),
        split="synthetic",
    )
    assert len(data) == 4
    # Three channels at 8x8 is not MNIST, and the shared type must accept it --
    # the (n, 1, 28, 28) assertion belongs to `load_mnist`, which is the only
    # thing that knows MNIST's shape.
    assert data.images.shape == (4, 3, 8, 8)


def test_the_generic_split_still_rejects_malformed_data() -> None:
    """Generalising the shape check must not have made it decorative."""
    import torch

    with pytest.raises(DataError, match="channels, height, width"):
        ImageSplit(images=torch.rand(4, 8), labels=torch.arange(4), split="flat")
    with pytest.raises(DataError, match="labels"):
        ImageSplit(
            images=torch.rand(4, 1, 8, 8),
            labels=torch.arange(3, dtype=torch.int64),
            split="mismatched",
        )
    with pytest.raises(DataError, match="int64"):
        ImageSplit(
            images=torch.rand(4, 1, 8, 8),
            labels=torch.arange(4, dtype=torch.int32),
            split="wrong-label-dtype",
        )
