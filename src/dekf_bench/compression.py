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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

COMPRESSORS = ("none", "float32", "float16", "bfloat16", "stochastic")
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

    def mix(self, mixing: torch.Tensor, stack: torch.Tensor) -> torch.Tensor:
        r"""$\boldsymbol A\tilde{\boldsymbol X} + \operatorname{diag}(\boldsymbol A)(\boldsymbol X - \tilde{\boldsymbol X})$,
        and the bits it cost. ``stack`` is (N, p), one sender per row."""
        links = links_of(mixing)
        p = stack.shape[1]
        self.bits += links * self.message_bits(p, stack.dtype)
        self.scalars += links * p
        if self.exact:
            return mixing @ stack
        decoded = self.decode(stack)
        return mixing @ decoded + torch.diagonal(mixing)[:, None] * (stack - decoded)


#: The channel every learner holds until a run attaches one: exact, and counting.
def exact_channel() -> Channel:
    return Channel()
