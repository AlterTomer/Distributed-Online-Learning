r"""The communication channel: what a message looks like after it crosses a link.

Track C, step C1 (`docs/communication_plan.md`). Every diffusing learner mixes its
neighbours' vectors through :meth:`Channel.mix`, in place of ``mixing @ stack``. The
channel decides what each *received* copy looks like and counts the bits it cost.

**The sender's own term stays exact.** An agent holds its own vector; it does not
quantise what it never sends. So with $\tilde{\boldsymbol x}_u$ the decoded message,

$$\boldsymbol x_v \leftarrow \sum_{u\ne v} a_{vu}\tilde{\boldsymbol x}_u + a_{vv}\boldsymbol x_v
  = \boldsymbol A\tilde{\boldsymbol X} + \operatorname{diag}(\boldsymbol A)\,(\boldsymbol X-\tilde{\boldsymbol X}).$$

Quantising the self term too would round every agent's state every step, and below
16 bits updates smaller than the rounding step would simply vanish: a stall that
belongs to the implementation, not to the method.

**Why here, and not between `adapt` and `combine`.** The one-hop filter recomputes
each agent's post-adapt $\boldsymbol\psi$ *inside* its combine, from its neighbours'
batches (`_one_hop_update`), and those are what cross the second link (D92). A channel
wrapped around `Intermediate` would compress the prior mean, which one-hop never
mixes. Mixing is the one place every learner's transmitted vectors pass through.

**The default is exact, bit for bit.** ``none`` returns ``mixing @ stack`` itself, the
operation the learners performed before the channel existed, so every recorded run
reproduces. The linear form above is only valid for linear combines: mean mixing,
optimiser moments. Full sharing's covariance combine is not linear in what is sent,
and it is not compressed (C-8); its mean still is.

Compressors (C1):

=============  =====================  ==================================================
name           bits per message       what the receiver gets
=============  =====================  ==================================================
``none``       $p\cdot w$             exact; $w$ the working precision (64 in float64)
``float32``    $32p$                  round-to-nearest single precision
``float16``    $16p$                  round-to-nearest half precision
``bfloat16``   $16p$                  bfloat16: float32's range, 8 mantissa bits
``stochastic`` $bp + 32$              scaled uniform, $b$ bits, stochastic rounding
=============  =====================  ==================================================

``stochastic`` scales each message by its largest magnitude $s$ (sent once as a
float32, the 32 bits of side information), maps it onto $L = 2^{b-1}-1$ levels either
side of zero, and rounds up with probability equal to the fractional part, so the
decoded message is **unbiased**: $\mathbb E\,\tilde{\boldsymbol x} = \boldsymbol x$. Its
per-coordinate error is below $s/L$ and its variance at most $(s/L)^2/4$, the number a
compensated filter (C-4) would add as process noise.

``codec`` -- the offline-trained differential codec (C2, D137, D138), :class:`CodecChannel`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from dekf_bench.codec import LayerCode, count_events, entropy_bits

COMPRESSORS = ("none", "float32", "float16", "bfloat16", "stochastic", "codec")
#: The codec's modes: an uncompressed pass measuring the moments' scales, a quantised
#: pass counting symbols for the tables, and the trained tables themselves.
CODEC_MODES = ("scale", "count", "code")
#: The kinds of vector a learner mixes. psi's scale is rms(theta_0) per layer; the
#: moments' come from the scale pass (D138).
KINDS = ("psi", "momentum", "second_moment")
_CASTS = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_CAST_BITS = {"float32": 32, "float16": 16, "bfloat16": 16}
#: One float32 per message: the stochastic quantiser's scale.
SCALE_BITS = 32
STOCHASTIC_BITS = (2, 16)

#: Bits per scalar of the *data* a one-hop agent forwards, by dataset. MNIST ships the
#: 8-bit pixels and a label per sample, the convention the scalar ledger's "788 bytes"
#: already uses; a series ships its samples at the working precision (None).
DATA_BITS: dict[str, int | None] = {"mnist": 8, "fashion_mnist": 8, "mackey_glass": None}


class ChannelError(ValueError):
    """Raised for a compressor that cannot be built or applied as configured."""


def working_bits(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits


def data_bits(dataset: str, dtype: torch.dtype) -> int:
    bits = DATA_BITS.get(dataset)
    return working_bits(dtype) if bits is None else bits


def links_of(mixing: torch.Tensor) -> int:
    """Directed links a broadcast crosses: every nonzero off-diagonal weight."""
    off_diagonal = mixing.clone()
    off_diagonal.fill_diagonal_(0)
    return int((off_diagonal != 0).sum())


@dataclass
class Channel:
    """One learner's channel. Stateless in C1; C2's public copies will live here.

    ``bits`` and ``scalars`` accumulate over the run: every :meth:`mix` adds what its
    messages cost, over every link they crossed.
    """

    compressor: str = "none"
    precision: int = 8
    generator: torch.Generator | None = None
    bits: int = 0
    scalars: int = 0
    _checked: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.compressor not in COMPRESSORS:
            raise ChannelError(f"compressor {self.compressor!r} is not one of {list(COMPRESSORS)}")
        if self.compressor == "codec" and not isinstance(self, CodecChannel):
            raise ChannelError("the codec holds public copies and tables: build a CodecChannel")
        low, high = STOCHASTIC_BITS
        if self.compressor == "stochastic":
            if not low <= self.precision <= high:
                raise ChannelError(
                    f"stochastic precision must lie in [{low}, {high}] bits, got {self.precision}")
            if self.generator is None:
                raise ChannelError("stochastic rounding needs its own generator, for pairing")

    @property
    def exact(self) -> bool:
        return self.compressor == "none"

    def message_bits(self, p: int, dtype: torch.dtype) -> int:
        if self.compressor == "none":
            return p * working_bits(dtype)
        if self.compressor in _CAST_BITS:
            return p * _CAST_BITS[self.compressor]
        return p * self.precision + SCALE_BITS

    def decode(self, stack: torch.Tensor) -> torch.Tensor:
        """What each row looks like on arrival, in the working dtype."""
        if self.compressor == "none":
            return stack
        if self.compressor in _CASTS:
            return stack.to(_CASTS[self.compressor]).to(stack.dtype)
        levels = 2 ** (self.precision - 1) - 1
        scale = stack.abs().amax(dim=1, keepdim=True)
        # The scale travels as a float32, so the receiver decodes with *that* value.
        scale = scale.to(torch.float32).to(stack.dtype)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        scaled = stack / safe * levels
        # Drawn on the CPU from the channel's own stream, then moved: the draw must
        # not depend on the device, or a CPU smoke and a GPU run would disagree.
        uniform = torch.rand(stack.shape, generator=self.generator, dtype=torch.float64)
        rounded = torch.floor(scaled + uniform.to(device=stack.device, dtype=stack.dtype))
        return torch.where(scale > 0, rounded / levels * safe, torch.zeros_like(stack))

    def mix(self, mixing: torch.Tensor, stack: torch.Tensor, kind: str = "psi") -> torch.Tensor:
        r"""$\boldsymbol A\tilde{\boldsymbol X} + \operatorname{diag}(\boldsymbol A)(\boldsymbol X - \tilde{\boldsymbol X})$,
        and the bits it cost. ``stack`` is (N, p), one sender per row; ``kind`` names the
        vector (psi, or an optimiser moment), which only the codec needs."""
        links = links_of(mixing)
        p = stack.shape[1]
        self.bits += links * self.message_bits(p, stack.dtype)
        self.scalars += links * p
        if self.exact:
            return mixing @ stack
        decoded = self.decode(stack)
        return mixing @ decoded + torch.diagonal(mixing)[:, None] * (stack - decoded)

    def finish(self, out_dir: Path, seed: int, learner: str) -> None:
        """Called once a seed's run ends. Only the codec has anything to write."""


#: The channel every learner holds until a run attaches one: exact, and counting.
def exact_channel() -> Channel:
    return Channel()


def out_degrees(mixing: torch.Tensor) -> torch.Tensor:
    """Links each sender's broadcast crosses: column v's nonzero off-diagonal weights."""
    off_diagonal = mixing.clone()
    off_diagonal.fill_diagonal_(0)
    return (off_diagonal != 0).sum(dim=0).to(torch.int64)


@dataclass
class CodecChannel(Channel):
    r"""The offline-trained differential codec: public copies, module-level steps, RLE+Huffman.

    Every agent's vector of each ``kind`` travels as $\boldsymbol q=\operatorname{round}((\boldsymbol x-\tilde{\boldsymbol x})/\boldsymbol\Delta)$
    against a public copy $\tilde{\boldsymbol x}$ that sender and receivers advance by
    $\boldsymbol q\boldsymbol\Delta$ -- the copy is the only error memory (D136). The copies start
    at $\boldsymbol\theta_0$ for psi and 0 for the moments, known to every agent, so nothing
    is charged for a first message (D138). $\Delta_\ell=c\,s_\ell$ per layer (a module).
    Receivers mix the copies; the sender's own term stays exact (C1). A received second
    moment is clamped at zero before use, never in the copy (D137).

    Modes (``CODEC_MODES``): ``scale`` mixes exactly and records each moment's per-layer
    rms; ``count`` quantises and pools run and amplitude counts per (kind, layer);
    ``code`` charges the trained tables' code lengths. Bits are charged per sender, times
    the links its broadcast crosses. ⚠ Counted code lengths, not bitstreams produced each
    step (D138); `codec.encode` and the tests tie the two together.
    """

    compressor: str = "codec"
    c: float = 0.0
    mode: str = "count"
    layers: list = field(default_factory=list)
    theta0: torch.Tensor | None = None
    #: kind -> per-layer s_l, for the moments (psi's comes from theta0).
    moment_scales: dict = field(default_factory=dict)
    #: kind -> per-layer LayerCode, in code mode.
    codes: dict = field(default_factory=dict)
    copies: dict = field(default_factory=dict, repr=False)
    deltas: dict = field(default_factory=dict, repr=False)
    counts: dict = field(default_factory=dict, repr=False)
    sums: dict = field(default_factory=dict, repr=False)
    totals: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in CODEC_MODES:
            raise ChannelError(f"codec mode {self.mode!r} is not one of {list(CODEC_MODES)}")
        if self.mode != "scale" and not self.c > 0:
            raise ChannelError(f"the codec's multiplier c must be positive, got {self.c}")
        if not self.layers:
            raise ChannelError("the codec needs the model's layers (codec.module_layers)")
        if self.theta0 is None:
            raise ChannelError("the codec's psi copies start at theta_0, which it needs")
        self.totals = Counter()

    @property
    def exact(self) -> bool:
        return self.mode == "scale"

    # -- per-kind state -------------------------------------------------------- #

    def _scales(self, kind: str) -> list[float]:
        if kind == "psi":
            theta0 = self.theta0.double()
            whole = float(theta0.pow(2).mean().sqrt()) or 1.0
            return [float(theta0[part].pow(2).mean().sqrt()) or whole for _n, part in self.layers]
        if kind not in self.moment_scales:
            raise ChannelError(
                f"no scale for {kind!r}: the moments' scales come from the scale pass (D138)")
        return list(self.moment_scales[kind])

    def _delta(self, kind: str, like: torch.Tensor) -> torch.Tensor:
        if kind not in self.deltas:
            delta = torch.empty(like.shape[1], dtype=like.dtype, device=like.device)
            for (_name, part), scale in zip(self.layers, self._scales(kind), strict=True):
                delta[part] = self.c * scale
            self.deltas[kind] = delta
        return self.deltas[kind]

    def _copy(self, kind: str, stack: torch.Tensor) -> torch.Tensor:
        if kind not in self.copies:
            start = (self.theta0.to(device=stack.device, dtype=stack.dtype).expand_as(stack)
                     if kind == "psi" else torch.zeros_like(stack))
            self.copies[kind] = start.clone()
        return self.copies[kind]

    # -- the message ------------------------------------------------------------- #

    def mix(self, mixing: torch.Tensor, stack: torch.Tensor, kind: str = "psi") -> torch.Tensor:
        if kind not in KINDS:
            raise ChannelError(f"unknown vector kind {kind!r}; have {list(KINDS)}")
        degrees = out_degrees(mixing).to(stack.device)
        p = stack.shape[1]
        self.scalars += int(degrees.sum()) * p
        if self.mode == "scale":
            # The scale pass is the uncompressed run in every column, bits included.
            self._measure(kind, stack)
            self.bits += int(degrees.sum()) * p * working_bits(stack.dtype)
            return mixing @ stack
        copy = self._copy(kind, stack)
        delta = self._delta(kind, stack)
        q = torch.round((stack - copy) / delta)
        copy.add_(q * delta)
        q = q.to(torch.int64)
        decoded = copy
        if kind == "second_moment":
            decoded = copy.clamp_min(0)
            self.totals["clamped"] += int((copy < 0).sum())
            self.totals["second_moment_entries"] += copy.numel()
        self._charge(kind, q, degrees)
        return mixing @ decoded + torch.diagonal(mixing)[:, None] * (stack - decoded)

    def _measure(self, kind: str, stack: torch.Tensor) -> None:
        if kind == "psi":
            return
        sums = self.sums.setdefault(kind, [[0.0, 0] for _ in self.layers])
        for index, (_name, part) in enumerate(self.layers):
            block = stack[:, part].double()
            sums[index][0] += float(block.pow(2).sum())
            sums[index][1] += block.numel()

    def _charge(self, kind: str, q: torch.Tensor, degrees: torch.Tensor) -> None:
        per_sender = torch.zeros(q.shape[0], dtype=torch.float64, device=q.device)
        ideal = torch.zeros_like(per_sender)
        for index, (name, part) in enumerate(self.layers):
            block = q[:, part]
            # D136's bound: the empirical per-entry symbol entropy, pooled over senders.
            self.totals["entropy_bits"] += (entropy_bits(block) / q.shape[0]
                                            * float(degrees.sum()))
            self.totals["zeros"] += int((block == 0).sum())
            self.totals["entries"] += block.numel()
            if self.mode == "count":
                runs, amps = count_events(block)
                slot = self.counts.setdefault(kind, {}).setdefault(name, [Counter(), Counter()])
                slot[0].update(runs)
                slot[1].update(amps)
                continue
            code = self._code(kind, index, name)
            per_sender += code.message_bits(block).double()
            escaped, symbols = code.escapes(block)
            self.totals["escapes"] += escaped
            self.totals["symbols"] += symbols
            ideal += code.ideal_bits(block)
        weights = degrees.double()
        if self.mode == "count":
            # Before tables exist, the ledger carries D136's entropy bound.
            self.bits += round(self.totals["entropy_bits"] - self.totals["entropy_charged"])
            self.totals["entropy_charged"] = self.totals["entropy_bits"]
            return
        self.bits += int((weights * per_sender).sum())
        self.totals["coded_bits"] += int((weights * per_sender).sum())
        self.totals["ideal_bits"] += float((weights * ideal).sum())

    def _code(self, kind: str, index: int, name: str) -> LayerCode:
        try:
            return self.codes[kind][index]
        except (KeyError, IndexError):
            raise ChannelError(
                f"no trained table for {kind!r}, layer {name!r}: build the tables from a "
                "count pass first (D137)") from None

    # -- end of a seed ------------------------------------------------------------- #

    def finish(self, out_dir: Path, seed: int, learner: str) -> None:
        names = [name for name, _part in self.layers]
        summary = {"learner": learner, "seed": seed, "mode": self.mode, "c": self.c,
                   "layers": names, "totals": dict(self.totals),
                   "scalars": self.scalars, "bits": self.bits}
        if self.mode == "scale":
            summary["moment_rms"] = {kind: [(s / n) ** 0.5 if n else 0.0 for s, n in sums]
                                     for kind, sums in self.sums.items()}
            summary["moment_sums"] = self.sums
        if self.mode == "count":
            summary["counts"] = {
                kind: {name: {"runs": {str(k): v for k, v in runs.items()},
                              "amps": {str(k): v for k, v in amps.items()}}
                       for name, (runs, amps) in layers.items()}
                for kind, layers in self.counts.items()}
        path = Path(out_dir) / f"codec_{learner}_seed{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
