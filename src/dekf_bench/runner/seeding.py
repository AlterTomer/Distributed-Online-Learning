"""Separable seed streams.

One master seed derives four *independent* streams:

===============  ==========================================================
``init``         Model initialization, shared across all agents
``partition``    Shard assignment
``stream``       Sample order within a shard, label-availability draws
``graph``        Random graph realization
===============  ==========================================================

Separating them is what makes an ablation interpretable: hold the partition
fixed and vary only initialization, and any difference is attributable to
initialization. A single seed threaded through everything makes that impossible,
because changing it moves all four at once.

Independence is achieved by *deriving* each stream from a keyed hash of its own
name rather than by drawing them in sequence. Sequential derivation
(``SeedSequence.spawn``) couples the streams to their order, so inserting a
fifth stream would silently change the values of every stream after it and
invalidate comparisons against results already recorded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import numpy as np
    import torch

#: The streams, in documentation order. Adding to this tuple is safe:
#: derivation is by name, so existing streams keep their values -- which is why
#: `priors` could be appended without changing any run that predates it.
STREAM_NAMES = ("init", "partition", "stream", "graph", "priors")

#: Derived seeds are truncated to this width. 63 bits keeps them positive and
#: inside the range every backend here accepts.
_SEED_BITS = 63
_SEED_MODULUS = 1 << _SEED_BITS


def derive_seed(master: int, name: str, *parts: int | str) -> int:
    """Derive a reproducible child seed from ``master`` and a label.

    The derivation is a keyed hash, so it is deterministic across processes,
    platforms and Python versions -- unlike :func:`hash`, which is randomised
    per interpreter unless ``PYTHONHASHSEED`` is pinned.

    ``parts`` allows sub-streams, e.g. ``derive_seed(0, "stream", node, t)`` for
    a per-agent, per-step draw that does not depend on iteration order.
    """
    if master < 0:
        raise ValueError(f"master seed must be non-negative, got {master}")
    label = ":".join(["dekf_bench", name, str(master), *(str(part) for part in parts)])
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


@dataclass(frozen=True)
class Seeds:
    """The seed streams for one run, derived from ``master``.

    Frozen, because a seed that changes mid-run is not a seed.
    """

    master: int
    init: int
    partition: int
    stream: int
    graph: int
    #: The class-prior endpoints and plan. Separate from `partition` so the
    #: prior path can be held fixed while the shards are redrawn, and vice
    #: versa -- the two are otherwise easy to confound, since under prior drift
    #: the plan is what sizes the shards.
    priors: int

    @classmethod
    def from_master(cls, master: int) -> Seeds:
        return cls(
            master=master,
            **{name: derive_seed(master, name) for name in STREAM_NAMES},
        )

    def __getitem__(self, name: str) -> int:
        if name not in STREAM_NAMES:
            raise KeyError(f"unknown seed stream {name!r}; have {list(STREAM_NAMES)}")
        return int(getattr(self, name))

    def sub(self, name: str, *parts: int | str) -> int:
        """A sub-seed within one stream, e.g. per node and step.

        Derived from the *master* seed and the full label, so two sub-streams
        never collide and neither depends on the order they are requested in.
        """
        if name not in STREAM_NAMES:
            raise KeyError(f"unknown seed stream {name!r}; have {list(STREAM_NAMES)}")
        return derive_seed(self.master, name, *parts)

    def torch_generator(self, name: str, *parts: int | str, device: str = "cpu") -> torch.Generator:
        """A ``torch.Generator`` seeded from one stream.

        An explicit generator, rather than the global RNG, so that drawing in
        one part of the code cannot perturb another. Global-state RNG is the
        usual reason two runs with the same seed diverge.
        """
        import torch

        generator = torch.Generator(device=device)
        generator.manual_seed(self.sub(name, *parts) if parts else self[name])
        return generator

    def numpy_rng(self, name: str, *parts: int | str) -> np.random.Generator:
        """A ``numpy.random.Generator`` seeded from one stream."""
        import numpy as np

        return np.random.default_rng(self.sub(name, *parts) if parts else self[name])

    def as_dict(self) -> dict[str, int]:
        """For the run metadata, so a result records the seeds that produced it."""
        return {"master": self.master, **{name: self[name] for name in STREAM_NAMES}}


def seeds_for(master: int) -> Seeds:
    """Convenience alias for :meth:`Seeds.from_master`."""
    return Seeds.from_master(master)


def iter_seeds(masters: list[int]) -> Iterator[Seeds]:
    """Seeds for each master in a run's ``run.seeds`` list."""
    for master in masters:
        yield Seeds.from_master(master)


def seeds_from_config(config: Any) -> list[Seeds]:
    """The :class:`Seeds` for every master seed in a config."""
    return [Seeds.from_master(master) for master in config.run.seeds]
