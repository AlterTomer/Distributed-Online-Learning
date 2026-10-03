"""The offline codec's entropy-coding layer (D137, D138).

The counted message length (`LayerCode.message_bits`) is what every run reports, so the
central test is that it equals the real encoder's bitstream length, bit for bit, and that
the bitstream decodes back to the message -- including values the tables never saw.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter

import pytest
import torch

from dekf_bench.codec import (
    EOB,
    ESC,
    CodecError,
    LayerCode,
    canonical_codes,
    count_events,
    decode,
    elias_gamma_decode,
    elias_gamma_encode,
    elias_gamma_length,
    encode,
    events,
    huffman_lengths,
    with_escape,
)


def _sparse(rows: int, size: int, density: float, scale: float, seed: int) -> torch.Tensor:
    """Messages like quantised differences: mostly zeros, two-sided, heavy-ish tails."""
    g = torch.Generator().manual_seed(seed)
    values = torch.round(torch.randn(rows, size, generator=g, dtype=torch.float64) * scale)
    keep = torch.rand(rows, size, generator=g, dtype=torch.float64) < density
    return torch.where(keep, values, torch.zeros_like(values)).to(torch.int64)


def _trained(q: torch.Tensor) -> LayerCode:
    runs, amps = count_events(q)
    return LayerCode(dict(runs), dict(amps))


# ---- Elias-gamma ------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 9, 255, 256, 1023, 1024, 10**6])
def test_elias_gamma_round_trip_and_length(n):
    word = elias_gamma_encode(n)
    assert len(word) == elias_gamma_length(n) == 2 * int(math.floor(math.log2(n))) + 1
    value, end = elias_gamma_decode(word + "101", 0)
    assert (value, end) == (n, len(word))


def test_elias_gamma_vectorised_matches_scalar_at_powers_of_two():
    n = torch.tensor([1, 2, 3, 4, 7, 8, 15, 16, 2**20 - 1, 2**20, 2**20 + 1])
    assert elias_gamma_length(n).tolist() == [elias_gamma_length(int(v)) for v in n]


def test_elias_gamma_rejects_zero():
    with pytest.raises(CodecError):
        elias_gamma_encode(0)


# ---- Huffman ----------------------------------------------------------------------

def _reference_cost(counts: list[int]) -> int:
    """Optimal total cost: the sum of every merge's weight (independent of the module)."""
    heap = list(counts)
    heapq.heapify(heap)
    cost = 0
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        cost += a + b
        heapq.heappush(heap, a + b)
    return cost


@pytest.mark.parametrize("seed", range(5))
def test_huffman_is_complete_and_optimal(seed):
    g = torch.Generator().manual_seed(seed)
    counts = {s: int(c) + 1 for s, c in enumerate(torch.randint(0, 500, (40,), generator=g))}
    lengths = huffman_lengths(counts)
    assert sum(2.0 ** -n for n in lengths.values()) == pytest.approx(1.0)
    assert sum(counts[s] * lengths[s] for s in counts) == _reference_cost(list(counts.values()))


def test_canonical_codes_are_prefix_free_with_the_given_lengths():
    lengths = huffman_lengths({**{s: s * s + 1 for s in range(-6, 7)}, EOB: 50, ESC: 2})
    codes = canonical_codes(lengths)
    assert all(len(codes[s]) == lengths[s] for s in lengths)
    words = sorted(codes.values())
    assert all(not b.startswith(a) for a, b in zip(words, words[1:], strict=False))


def test_single_symbol_table_gets_one_bit():
    assert huffman_lengths({7: 10}) == {7: 1}


def test_good_turing_escape_weight():
    assert with_escape({1: 5, 2: 1, 3: 1, 4: 9})[ESC] == 2
    assert with_escape({1: 5, 4: 9})[ESC] == 1          # no singletons: at least one


# ---- events -----------------------------------------------------------------------

def test_events_runs_and_amplitudes():
    q = torch.tensor([[0, 0, 3, 0, -1, 0, 0], [2, 0, 0, 0, 0, 0, 0], [0] * 7])
    rows, runs, amps = events(q)
    assert rows.tolist() == [0, 0, 1]
    assert runs.tolist() == [2, 1, 0]
    assert amps.tolist() == [3, -1, 2]
    run_counts, amp_counts = count_events(q)
    assert run_counts == Counter({2: 1, 1: 1, 0: 1, EOB: 3})
    assert amp_counts == Counter({3: 1, -1: 1, 2: 1})


# ---- the counted length is the bitstream's ------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_counted_bits_equal_the_encoded_bitstream_and_it_decodes(seed):
    code = _trained(_sparse(20, 300, 0.2, 3.0, seed))
    # Different data: wider amplitudes and sparser rows force unseen runs and values.
    q = _sparse(6, 300, 0.05, 9.0, seed + 100)
    counted = code.message_bits(q)
    for row, bits in zip(q, counted, strict=True):
        stream = encode(row, code)
        assert len(stream) == int(bits)
        back, end = decode(stream, row.numel(), code)
        assert end == len(stream)
        assert torch.equal(back, row)


def test_escapes_are_exercised():
    code = _trained(torch.tensor([[0, 1, 0, -1, 0, 0]]))
    row = torch.tensor([0] * 40 + [57])                # an unseen run and an unseen value
    stream = encode(row, code)
    assert len(stream) == int(code.message_bits(row.reshape(1, -1))[0])
    assert torch.equal(decode(stream, row.numel(), code)[0], row)


def test_an_empty_message_costs_one_eob():
    code = _trained(_sparse(10, 50, 0.3, 2.0, 0))
    assert int(code.message_bits(torch.zeros(1, 50, dtype=torch.int64))[0]) == \
        code.run_lengths[EOB]


def test_huffman_sits_within_a_bit_per_symbol_of_the_ideal():
    q = _sparse(30, 400, 0.25, 4.0, 7)
    code = _trained(q)
    coded = float(code.message_bits(q).sum())
    ideal = float(code.ideal_bits(q).sum())
    _rows, runs, _amps = events(q)
    symbols = 2 * runs.numel() + q.shape[0]
    assert ideal <= coded < ideal + symbols


def test_tables_survive_json():
    code = _trained(_sparse(10, 200, 0.2, 3.0, 3))
    again = LayerCode.from_json(json.loads(json.dumps(code.to_json())))
    assert again.run_lengths == code.run_lengths
    assert again.amp_lengths == code.amp_lengths
    q = _sparse(4, 200, 0.2, 3.0, 4)
    assert torch.equal(again.message_bits(q), code.message_bits(q))


def test_run_table_needs_eob():
    with pytest.raises(CodecError):
        LayerCode({0: 3}, {1: 3})


def test_escapes_are_counted():
    code = _trained(torch.tensor([[0, 1, 0, -1, 0, 0]]))
    q = torch.tensor([[0] * 40 + [57], [0, 1, 0, 0, 0, 0] + [0] * 35])
    escaped, symbols = code.escapes(q)
    assert escaped == 2                       # row 0's run of 40 and its amplitude 57
    assert symbols == 2 * 2 + 2               # two events, two EOBs
