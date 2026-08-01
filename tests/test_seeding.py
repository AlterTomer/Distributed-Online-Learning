"""Seed derivation.

The property these tests protect is *separability*: changing one stream must
leave the others alone, or an ablation cannot attribute a difference to the
thing that was varied.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from dekf_bench.runner.seeding import (
    STREAM_NAMES,
    Seeds,
    derive_seed,
    iter_seeds,
    seeds_for,
)


def test_derivation_is_deterministic() -> None:
    assert derive_seed(0, "init") == derive_seed(0, "init")


def test_streams_differ_from_each_other() -> None:
    seeds = seeds_for(0)
    values = [seeds[name] for name in STREAM_NAMES]
    assert len(set(values)) == len(values)


def test_streams_differ_from_the_master() -> None:
    seeds = seeds_for(7)
    assert all(seeds[name] != 7 for name in STREAM_NAMES)


def test_changing_the_master_moves_every_stream() -> None:
    a, b = seeds_for(0), seeds_for(1)
    assert all(a[name] != b[name] for name in STREAM_NAMES)


def test_stream_values_depend_only_on_name_and_master() -> None:
    """Adding a fifth stream must not shift the four that already exist.

    Sequential derivation would fail this, and would silently invalidate every
    comparison against results recorded before the new stream was added.
    """
    before = {name: derive_seed(42, name) for name in STREAM_NAMES}
    _ = derive_seed(42, "a_new_stream_inserted_alphabetically_first")
    after = {name: derive_seed(42, name) for name in STREAM_NAMES}
    assert before == after


def test_derivation_is_stable_across_processes() -> None:
    """blake2b, not hash(): the latter is randomised per interpreter."""
    code = "from dekf_bench.runner.seeding import derive_seed; print(derive_seed(0, 'init'))"
    runs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1")
    ]
    assert runs[0] == runs[1] == str(derive_seed(0, "init"))


def test_seeds_are_in_a_safe_range() -> None:
    seeds = seeds_for(123)
    for name in STREAM_NAMES:
        assert 0 <= seeds[name] < 2**63


def test_sub_seeds_separate_by_part() -> None:
    seeds = seeds_for(0)
    per_node = [seeds.sub("stream", node) for node in range(10)]
    assert len(set(per_node)) == 10


def test_sub_seeds_do_not_depend_on_request_order() -> None:
    seeds = seeds_for(0)
    forward = [seeds.sub("stream", node, 5) for node in range(4)]
    backward = [seeds.sub("stream", node, 5) for node in reversed(range(4))][::-1]
    assert forward == backward


def test_sub_seeds_of_different_streams_do_not_collide() -> None:
    seeds = seeds_for(0)
    assert seeds.sub("stream", 3) != seeds.sub("partition", 3)


def test_bare_stream_and_sub_stream_differ() -> None:
    seeds = seeds_for(0)
    assert seeds["stream"] != seeds.sub("stream", 0)


def test_unknown_stream_is_rejected() -> None:
    seeds = seeds_for(0)
    with pytest.raises(KeyError, match="unknown seed stream"):
        seeds["initialisation"]
    with pytest.raises(KeyError, match="unknown seed stream"):
        seeds.sub("initialisation", 1)


def test_negative_master_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(-1, "init")


def test_seeds_are_frozen() -> None:
    seeds = seeds_for(0)
    # setattr rather than a direct assignment: see the note in test_graph.py.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(seeds, "init", 5)  # noqa: B010


def test_iter_seeds_covers_every_master() -> None:
    assert [s.master for s in iter_seeds([0, 1, 2])] == [0, 1, 2]


def test_as_dict_records_everything_needed_to_reproduce() -> None:
    recorded = seeds_for(3).as_dict()
    assert recorded["master"] == 3
    assert set(recorded) == {"master", *STREAM_NAMES}
    assert Seeds.from_master(recorded["master"]).as_dict() == recorded


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #


def test_torch_generator_is_reproducible() -> None:
    import torch

    seeds = seeds_for(0)
    first = torch.randn(5, generator=seeds.torch_generator("init"))
    second = torch.randn(5, generator=seeds.torch_generator("init"))
    assert torch.equal(first, second)


def test_torch_generators_of_different_streams_differ() -> None:
    import torch

    seeds = seeds_for(0)
    init = torch.randn(5, generator=seeds.torch_generator("init"))
    graph = torch.randn(5, generator=seeds.torch_generator("graph"))
    assert not torch.equal(init, graph)


def test_explicit_generator_is_immune_to_global_seeding() -> None:
    """Drawing elsewhere must not perturb a stream. This is why generators are
    passed explicitly rather than relying on the global RNG."""
    import torch

    seeds = seeds_for(0)
    generator = seeds.torch_generator("init")
    before = torch.randn(3, generator=generator)

    torch.manual_seed(999)
    _ = torch.randn(100)

    generator = seeds.torch_generator("init")
    after = torch.randn(3, generator=generator)
    assert torch.equal(before, after)


def test_numpy_rng_is_reproducible() -> None:
    import numpy as np

    seeds = seeds_for(0)
    first = seeds.numpy_rng("partition").permutation(20)
    second = seeds.numpy_rng("partition").permutation(20)
    assert np.array_equal(first, second)


def test_holding_partition_fixed_while_init_varies() -> None:
    """The ablation the four streams exist to make possible."""
    import numpy as np

    shards_a = seeds_for(0).numpy_rng("partition").permutation(50)
    shards_b = Seeds(
        master=0,
        init=derive_seed(999, "init"),
        partition=derive_seed(0, "partition"),
        stream=derive_seed(0, "stream"),
        graph=derive_seed(0, "graph"),
    )
    assert np.array_equal(shards_a, shards_b.numpy_rng("partition").permutation(50))
    assert shards_b.init != seeds_for(0).init
