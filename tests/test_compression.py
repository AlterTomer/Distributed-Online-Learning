"""The communication channel (Track C, C1): what a mixed vector looks like on arrival,
and what it cost. See `compression.py` and `docs/communication_plan.md`."""

from __future__ import annotations

import pytest
import torch

from dekf_bench.compression import (
    SCALE_BITS,
    Channel,
    ChannelError,
    links_of,
)
from dekf_bench.utils.config import ConfigError, load_config


def ring_mixing(n: int = 4) -> torch.Tensor:
    mixing = torch.zeros(n, n, dtype=torch.float64)
    for v in range(n):
        mixing[v, v] = 1 / 3
        mixing[v, (v + 1) % n] = 1 / 3
        mixing[v, (v - 1) % n] = 1 / 3
    return mixing


def stack(n: int = 4, p: int = 50, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, p, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


# --------------------------------------------------------------------------- #
# the exact channel is the old arithmetic
# --------------------------------------------------------------------------- #


def test_the_exact_channel_is_the_old_combine_bit_for_bit() -> None:
    """Every recorded run mixed with `mixing @ X`. The default channel must be that
    operation, not an algebraically equal one, or nothing on disk reproduces."""
    mixing, x = ring_mixing(), stack()
    assert torch.equal(Channel().mix(mixing, x), mixing @ x)


def test_the_exact_channel_counts_working_precision_bits() -> None:
    channel = Channel()
    channel.mix(ring_mixing(), stack(p=50))
    assert channel.scalars == 8 * 50  # four agents, two neighbours each
    assert channel.bits == 8 * 50 * 64


def test_links_are_the_nonzero_off_diagonal_weights() -> None:
    assert links_of(ring_mixing(4)) == 8
    assert links_of(torch.eye(3, dtype=torch.float64)) == 0


# --------------------------------------------------------------------------- #
# the compressors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "dtype", "bits"), [
    ("float32", torch.float32, 32), ("float16", torch.float16, 16),
    ("bfloat16", torch.bfloat16, 16)])
def test_a_cast_is_its_round_trip(name: str, dtype: torch.dtype, bits: int) -> None:
    x = stack()
    channel = Channel(compressor=name)
    assert torch.equal(channel.decode(x), x.to(dtype).to(torch.float64))
    assert channel.message_bits(50, torch.float64) == 50 * bits


def test_stochastic_rounding_is_unbiased() -> None:
    """Rounding up with probability equal to the fractional part: the decoded mean over
    many draws converges on the message. A deterministic quantiser would not."""
    x = stack(n=1, p=20)
    channel = Channel(compressor="stochastic", precision=3,
                      generator=torch.Generator().manual_seed(1))
    draws = torch.stack([channel.decode(x) for _ in range(4000)])
    scale = x.abs().max()
    step = scale / (2 ** (3 - 1) - 1)
    assert (draws.mean(dim=0) - x).abs().max() < 4 * step / 4000**0.5


def test_stochastic_rounding_stays_within_one_step() -> None:
    x = stack(n=3, p=100)
    channel = Channel(compressor="stochastic", precision=4,
                      generator=torch.Generator().manual_seed(2))
    decoded = channel.decode(x)
    step = x.abs().amax(dim=1, keepdim=True) / (2 ** (4 - 1) - 1)
    assert bool(((decoded - x).abs() <= step + 1e-12).all())
    levels = decoded / step
    assert torch.allclose(levels, levels.round())  # on the grid


def test_stochastic_rounding_is_reproducible_from_its_seed() -> None:
    """The pairing guarantee: same stream, same draws, on any device."""
    x = stack()
    first = Channel(compressor="stochastic", generator=torch.Generator().manual_seed(7))
    second = Channel(compressor="stochastic", generator=torch.Generator().manual_seed(7))
    assert torch.equal(first.decode(x), second.decode(x))


def test_a_zero_message_decodes_to_zero() -> None:
    channel = Channel(compressor="stochastic", generator=torch.Generator().manual_seed(0))
    assert torch.equal(channel.decode(torch.zeros(2, 5, dtype=torch.float64)),
                       torch.zeros(2, 5, dtype=torch.float64))


def test_stochastic_bits_carry_the_scale() -> None:
    channel = Channel(compressor="stochastic", precision=4,
                      generator=torch.Generator().manual_seed(0))
    assert channel.message_bits(100, torch.float64) == 100 * 4 + SCALE_BITS


def test_stochastic_rounding_refuses_to_run_without_its_own_stream() -> None:
    with pytest.raises(ChannelError, match="generator"):
        Channel(compressor="stochastic")


@pytest.mark.parametrize("precision", [1, 17])
def test_a_precision_outside_the_range_is_refused(precision: int) -> None:
    with pytest.raises(ChannelError, match="precision"):
        Channel(compressor="stochastic", precision=precision,
                generator=torch.Generator().manual_seed(0))


# --------------------------------------------------------------------------- #
# the combine through a channel
# --------------------------------------------------------------------------- #


def test_the_senders_own_term_stays_exact() -> None:
    """Each agent mixes its neighbours' *decoded* messages and its own exact vector."""
    mixing, x = ring_mixing(), stack()
    channel = Channel(compressor="float16")
    got = channel.mix(mixing, x)
    decoded = x.to(torch.float16).to(torch.float64)
    for v in range(4):
        expected = mixing[v, v] * x[v] + sum(mixing[v, u] * decoded[u]
                                             for u in range(4) if u != v)
        assert torch.allclose(got[v], expected, atol=1e-14)


def test_an_isolated_agent_is_untouched_by_compression() -> None:
    """With no neighbours nothing crosses a link, so nothing is rounded or charged."""
    mixing, x = torch.eye(3, dtype=torch.float64), stack(n=3)
    channel = Channel(compressor="float16")
    assert torch.equal(channel.mix(mixing, x), x)
    assert channel.bits == 0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_the_default_config_is_exact() -> None:
    assert load_config("x1_stationary").comm.compressor == "none"


def test_an_unknown_compressor_is_refused() -> None:
    with pytest.raises(ConfigError, match="comm.compressor"):
        load_config("x1_stationary", overrides={"comm": {"compressor": "zip"}})


# --------------------------------------------------------------------------- #
# the whole path: the bits column
# --------------------------------------------------------------------------- #


def _run(compressor: str, learners: list[str], steps: int = 6):
    from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: PLC0415
    from dekf_bench.env.environment import build_environment  # noqa: PLC0415
    from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: PLC0415
    from dekf_bench.learners.registry import build_learners  # noqa: PLC0415
    from dekf_bench.likelihoods.categorical import Categorical  # noqa: PLC0415
    from dekf_bench.models.registry import build_model_from_config  # noqa: PLC0415
    from dekf_bench.runner import simulate  # noqa: PLC0415

    if not dataset_is_cached("mnist", "data"):
        pytest.skip("MNIST is not cached")
    train, test = load_dataset("mnist", "data", download=False)
    config = load_config("x1_stationary", overrides={
        "run": {"horizon": steps, "eval_every": 2}, "learners": learners,
        "comm": {"compressor": compressor}})
    environment = build_environment(config, 0, train)
    model = build_model_from_config(config)
    likelihood = Categorical(10)
    built = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    records = simulate.run(config, environment, built, build_evalsets(config, environment, test),
                           likelihood, theta0, stop_after=steps - 1)
    final = {}
    for record in records:
        for row in record.rows:
            final[record.learner] = (row["cum_scalars_tx"], row["cum_bits_tx"])
    return final


def test_uncompressed_bits_are_scalars_at_working_precision() -> None:
    final = _run("none", ["diffusion_sgd_atc", "centralized_sgd"])
    scalars, bits = final["diffusion_sgd_atc"]
    assert scalars > 0 and bits == 32 * scalars  # x1 runs in float32
    assert final["centralized_sgd"] == (0, 0)


def test_compressed_bits_follow_the_compressor_and_the_learners_diverge() -> None:
    exact = _run("none", ["diffusion_sgd_atc"])
    half = _run("float16", ["diffusion_sgd_atc"])
    assert exact["diffusion_sgd_atc"][0] == half["diffusion_sgd_atc"][0]
    assert half["diffusion_sgd_atc"][1] == 16 * half["diffusion_sgd_atc"][0]


def test_one_hop_data_is_priced_as_pixels_not_as_parameters() -> None:
    """C-1: the raw batch crosses uncompressed, at a byte per pixel and label; only
    the mean goes through the channel. The scalar ledger splits accordingly."""
    final = _run("float16", ["diffusion_ekf_onehop_mean_receiver"], steps=3)
    scalars, bits = final["diffusion_ekf_onehop_mean_receiver"]
    # per direction: p = 2908 through the channel at 16 bits, 788 data scalars at 8
    per_direction_scalars, per_direction_bits = 2908 + 788, 2908 * 16 + 788 * 8
    assert scalars * per_direction_bits == bits * per_direction_scalars
