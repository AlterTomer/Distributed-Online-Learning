r"""The offline-trained entropy codec: run lengths, amplitudes, canonical Huffman (D137, D138).

A layer's message is its quantised difference $\boldsymbol q=\operatorname{round}(\boldsymbol e/\Delta_\ell)$,
written as events: for each nonzero entry, the **run** of zeros before it and its
**amplitude**, then an **EOB** that ends the layer. Two canonical Huffman tables per layer
-- one for runs (EOB among them), one for amplitudes -- are trained offline from pooled
counts and preloaded by every agent; nothing about them is sent.

**Unseen values** are sent as ESC followed by an Elias-gamma code of the magnitude (an
amplitude also carries a sign bit; a run $r$ is coded as $r+1$, since Elias-gamma starts
at 1). ESC's weight in each table is Good--Turing's estimate of the unseen mass: the
number of symbols seen exactly once in calibration, at least 1.

**How bits are counted (D138).** :meth:`LayerCode.message_bits` sums codeword lengths --
codewords, EOBs and ESC payloads -- per message, vectorised over agents. ⚠ That is *not*
a bitstream produced every step: :func:`encode` and :func:`decode` exist for the tests,
which check round trips and that the encoded length equals the counted one bit for bit,
so the count is the bitstream's length without packing bits during a run.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field

import torch

#: Run-alphabet token that ends a layer's message.
EOB = -1
#: The escape token, in either alphabet.
ESC = "ESC"


class CodecError(ValueError):
    """Raised for tables that cannot code what they are given."""


def module_layers(model) -> list[tuple[str, slice]]:
    """The codec's layers: (module, slice of the flat vector), weight and bias together.

    A layer is a module, not a tensor (D138): a bias starts at zero, so its own
    rms(theta_0) is no scale. MNIST's MLP gives 2 layers, the MG Transformer 8.
    """
    out: list[tuple[str, slice]] = []
    start = 0
    for name, shape in zip(model.names, model.shapes, strict=True):
        size = int(torch.Size(shape).numel())
        module = name.rsplit(".", 1)[0]
        if out and out[-1][0] == module:
            out[-1] = (module, slice(out[-1][1].start, start + size))
        else:
            out.append((module, slice(start, start + size)))
        start += size
    return out


# --------------------------------------------------------------------------- #
# Elias-gamma
# --------------------------------------------------------------------------- #

def elias_gamma_length(n: torch.Tensor | int) -> torch.Tensor | int:
    """Bits of the Elias-gamma code of ``n >= 1``: $2\\lfloor\\log_2 n\\rfloor + 1$."""
    if isinstance(n, int):
        if n < 1:
            raise CodecError(f"Elias-gamma codes integers >= 1, got {n}")
        return 2 * (n.bit_length() - 1) + 1
    n = n.to(torch.int64)
    if bool((n < 1).any()):
        raise CodecError("Elias-gamma codes integers >= 1")
    floor_log = torch.floor(torch.log2(n.double())).to(torch.int64)
    # log2 of an exact power of two can land a hair below the integer in float.
    floor_log = floor_log + ((2 ** (floor_log + 1)) <= n).to(torch.int64)
    return 2 * floor_log + 1


def elias_gamma_encode(n: int) -> str:
    if n < 1:
        raise CodecError(f"Elias-gamma codes integers >= 1, got {n}")
    binary = bin(n)[2:]
    return "0" * (len(binary) - 1) + binary


def elias_gamma_decode(bits: str, position: int) -> tuple[int, int]:
    zeros = 0
    while bits[position + zeros] == "0":
        zeros += 1
    end = position + 2 * zeros + 1
    return int(bits[position + zeros:end], 2), end


# --------------------------------------------------------------------------- #
# canonical Huffman
# --------------------------------------------------------------------------- #

def _sort_key(symbol) -> tuple:
    """A total order across ints, EOB and ESC, so the canonical code is reproducible."""
    if symbol == ESC:
        return (1, 0)
    return (0, int(symbol))


def huffman_lengths(counts: dict) -> dict:
    """Optimal prefix-code lengths for positive counts (one symbol gets length 1)."""
    items = [(c, s) for s, c in counts.items() if c > 0]
    if not items:
        raise CodecError("a table needs at least one symbol with a positive count")
    if len(items) == 1:
        return {items[0][1]: 1}
    # (weight, tiebreak, symbols in the subtree); depth grows by one per merge.
    heap = [(c, i, [s]) for i, (c, s) in
            enumerate(sorted(items, key=lambda cs: (cs[0], _sort_key(cs[1]))))]
    heapq.heapify(heap)
    depth: dict = Counter()
    tiebreak = len(heap)
    while len(heap) > 1:
        c1, _i1, s1 = heapq.heappop(heap)
        c2, _i2, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            depth[s] += 1
        heapq.heappush(heap, (c1 + c2, tiebreak, s1 + s2))
        tiebreak += 1
    return dict(depth)


def canonical_codes(lengths: dict) -> dict:
    """Canonical codewords from lengths: sorted by (length, symbol), counting up."""
    ordered = sorted(lengths, key=lambda s: (lengths[s], _sort_key(s)))
    codes, code, previous = {}, 0, lengths[ordered[0]]
    for index, symbol in enumerate(ordered):
        if index:
            code = (code + 1) << (lengths[symbol] - previous)
        previous = lengths[symbol]
        codes[symbol] = format(code, f"0{previous}b")
    return codes


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #

def events(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs and amplitudes of every row of ``q`` (rows = messages), row-major.

    Returns (rows, runs, amplitudes) for the nonzero entries; every row also ends
    with one EOB, which the caller counts as ``q.shape[0]`` EOBs.
    """
    if q.ndim != 2:
        raise CodecError(f"events take (messages, entries), got shape {tuple(q.shape)}")
    rows, cols = (q != 0).nonzero(as_tuple=True)
    if rows.numel() == 0:
        empty = torch.zeros(0, dtype=torch.int64, device=q.device)
        return empty, empty, empty
    same_row = torch.zeros_like(rows, dtype=torch.bool)
    same_row[1:] = rows[1:] == rows[:-1]
    previous = torch.where(same_row, torch.roll(cols, 1), torch.full_like(cols, -1))
    runs = cols - previous - 1
    return rows, runs.to(torch.int64), q[rows, cols].to(torch.int64)


def count_events(q: torch.Tensor) -> tuple[Counter, Counter]:
    """Run counts (EOB included) and amplitude counts for a block of messages."""
    _rows, runs, amps = events(q)
    run_values, run_counts = torch.unique(runs, return_counts=True)
    amp_values, amp_counts = torch.unique(amps, return_counts=True)
    run_counter = Counter({int(v): int(c) for v, c in zip(run_values, run_counts, strict=True)})
    run_counter[EOB] += q.shape[0]
    amp_counter = Counter({int(v): int(c) for v, c in zip(amp_values, amp_counts, strict=True)})
    return run_counter, amp_counter


# --------------------------------------------------------------------------- #
# one layer's code
# --------------------------------------------------------------------------- #

@dataclass
class LayerCode:
    """The two preloaded tables of one layer of one vector kind, and their lookups.

    Built from pooled calibration counts, ESC's Good--Turing weight included; the
    Huffman lengths follow from the counts, so the counts are what is stored.
    """

    run_counts: dict
    amp_counts: dict
    run_lengths: dict = field(init=False)
    amp_lengths: dict = field(init=False)
    _dense: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.run_counts.get(EOB, 0) < 1:
            raise CodecError("the run table needs EOB, which ends every layer's message")
        self.run_counts = with_escape(self.run_counts)
        self.amp_counts = with_escape(self.amp_counts)
        self.run_lengths = huffman_lengths(self.run_counts)
        self.amp_lengths = huffman_lengths(self.amp_counts)

    def to_json(self) -> dict:
        key = lambda s: s if s == ESC else str(s)  # noqa: E731
        return {"runs": {key(s): n for s, n in self.run_counts.items()},
                "amps": {key(s): n for s, n in self.amp_counts.items()}}

    @classmethod
    def from_json(cls, data: dict) -> LayerCode:
        key = lambda s: s if s == ESC else int(s)  # noqa: E731
        return cls({key(s): n for s, n in data["runs"].items()},
                   {key(s): n for s, n in data["amps"].items()})

    # -- per-symbol costs, vectorised ----------------------------------------- #

    def _keys_of(self, name: str, costs: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
        """The table's seen integers, sorted, and their costs -- a lookup by search.

        Not a dense array over the seen range: amplitudes can be arbitrarily large (a
        diverging learner, a jump), and a range array would have to span them.
        """
        cache = (name, str(device))
        if cache not in self._dense:
            # EOB is -1, a token of the run alphabet only: in the amplitude alphabet -1 is
            # an ordinary value and must not be filtered out with it.
            tokens = (ESC, EOB) if name.endswith("runs") else (ESC,)
            ints = sorted(s for s in costs if s not in tokens)
            keys = torch.tensor(ints, dtype=torch.int64, device=device)
            values = torch.tensor([float(costs[s]) for s in ints], dtype=torch.float64,
                                  device=device)
            self._dense[cache] = (keys, values)
        return self._dense[cache]

    def _costs(self, values: torch.Tensor, costs: dict, name: str,
               signed: bool) -> torch.Tensor:
        keys, key_costs = self._keys_of(name, costs, values.device)
        magnitude = values.abs() if signed else values + 1
        escaped = (costs[ESC] + elias_gamma_length(magnitude.clamp_min(1)).double()
                   + float(signed))
        if keys.numel() == 0:
            return escaped
        index = torch.searchsorted(keys, values).clamp(max=keys.numel() - 1)
        hit = keys[index] == values
        return torch.where(hit, key_costs[index], escaped)

    def _total(self, q: torch.Tensor, run_costs: dict, amp_costs: dict,
               tag: str) -> torch.Tensor:
        rows, runs, amps = events(q)
        per_event = (self._costs(runs, run_costs, tag + "runs", signed=False)
                     + self._costs(amps, amp_costs, tag + "amps", signed=True))
        total = torch.zeros(q.shape[0], dtype=torch.float64, device=q.device)
        total.index_add_(0, rows, per_event)
        return total + run_costs[EOB]

    def escapes(self, q: torch.Tensor) -> tuple[int, int]:
        """(escaped symbols, symbols) in a block: the transfer check (D137)."""
        _rows, runs, amps = events(q)
        unseen = 0
        for values, costs, tokens in ((runs, self.run_lengths, (ESC, EOB)),
                                      (amps, self.amp_lengths, (ESC,))):
            seen = torch.tensor([s for s in costs if s not in tokens], dtype=torch.int64,
                                device=values.device)
            unseen += int((~torch.isin(values, seen)).sum())
        return unseen, 2 * runs.numel() + q.shape[0]

    def message_bits(self, q: torch.Tensor) -> torch.Tensor:
        """Bits of each row's message (rows = agents' messages for this layer)."""
        return self._total(q, self.run_lengths, self.amp_lengths, "len").round().to(torch.int64)

    def ideal_bits(self, q: torch.Tensor) -> torch.Tensor:
        r"""$-\sum\log_2\hat p$ under the trained probabilities (counts / total).

        What an ideal coder with the same calibration statistics would spend: the gap to
        :meth:`message_bits` is Huffman's integer codeword lengths, nothing else.
        """
        return self._total(q, _information(self.run_counts), _information(self.amp_counts),
                           "info")


def with_escape(counts: dict) -> dict:
    """The seen symbols plus ESC, weighted by Good--Turing: the count of singletons, >= 1."""
    seen = {s: int(c) for s, c in counts.items() if c > 0 and s != ESC}
    return {**seen, ESC: max(1, sum(1 for c in seen.values() if c == 1))}


def _information(counts: dict) -> dict:
    total = sum(counts.values())
    return {s: -math.log2(c / total) for s, c in counts.items()}


# --------------------------------------------------------------------------- #
# the real encoder and decoder (tests)
# --------------------------------------------------------------------------- #

def encode(row: torch.Tensor, code: LayerCode) -> str:
    """One layer's message as a bit string. For tests; runs count lengths instead."""
    run_codes, amp_codes = canonical_codes(code.run_lengths), canonical_codes(code.amp_lengths)
    _rows, runs, amps = events(row.reshape(1, -1))
    out = []
    for run, amp in zip(runs.tolist(), amps.tolist(), strict=True):
        out.append(run_codes[run] if run in run_codes
                   else run_codes[ESC] + elias_gamma_encode(run + 1))
        out.append(amp_codes[amp] if amp in amp_codes
                   else amp_codes[ESC] + elias_gamma_encode(abs(amp)) + ("1" if amp < 0 else "0"))
    out.append(run_codes[EOB])
    return "".join(out)


def _read(bits: str, position: int, decoding: dict) -> tuple[object, int]:
    word = ""
    while word not in decoding:
        if position >= len(bits):
            raise CodecError("the bit string ended inside a codeword")
        word += bits[position]
        position += 1
    return decoding[word], position


def decode(bits: str, size: int, code: LayerCode, position: int = 0) -> tuple[torch.Tensor, int]:
    """The inverse of :func:`encode`: a row of ``size`` integers, and where it stopped."""
    run_words = {w: s for s, w in canonical_codes(code.run_lengths).items()}
    amp_words = {w: s for s, w in canonical_codes(code.amp_lengths).items()}
    row = torch.zeros(size, dtype=torch.int64)
    cursor = 0
    while True:
        run, position = _read(bits, position, run_words)
        if run == EOB:
            return row, position
        if run == ESC:
            run, position = elias_gamma_decode(bits, position)
            run -= 1
        amp, position = _read(bits, position, amp_words)
        if amp == ESC:
            magnitude, position = elias_gamma_decode(bits, position)
            amp = -magnitude if bits[position] == "1" else magnitude
            position += 1
        cursor += run
        if cursor >= size:
            raise CodecError(f"a run overran the layer's {size} entries")
        row[cursor] = amp
        cursor += 1


def entropy_bits(q: torch.Tensor) -> float:
    """The empirical per-entry symbol entropy of ``q``, times its entries: D136's bound."""
    _values, counts = torch.unique(q, return_counts=True)
    p = counts.double() / counts.sum()
    return float(-(p * torch.log2(p)).sum()) * q.numel()

