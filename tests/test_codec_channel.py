"""The codec's channel (C2, D137, D138): public copies, module-level steps, three modes.

Unit tests drive `CodecChannel.mix` directly on small stacks; the last ones run the
whole chain through `simulate.run` on MNIST -- the scale pass, a count pass, tables built
from its counts, and a code pass -- and check the ledger stays consistent throughout.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
import torch

from dekf_bench.codec import EOB, LayerCode, count_events, encode, module_layers
from dekf_bench.compression import Channel, ChannelError, CodecChannel, out_degrees
from dekf_bench.utils.config import ConfigError, load_config

P = 30
LAYERS = [("a", slice(0, 20)), ("b", slice(20, 30))]


def ring(n: int = 4) -> torch.Tensor:
    mixing = torch.zeros(n, n, dtype=torch.float64)
    for v in range(n):
        mixing[v, v] = mixing[v, (v + 1) % n] = mixing[v, (v - 1) % n] = 1 / 3
    return mixing


def theta0(seed: int = 0) -> torch.Tensor:
    return torch.randn(P, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def channel(**kwargs) -> CodecChannel:
    defaults = {"c": 1e-2, "mode": "count", "layers": LAYERS, "theta0": theta0()}
    return CodecChannel(**{**defaults, **kwargs})


def walk(steps: int, n: int = 4, size: float = 0.05, seed: int = 1) -> list[torch.Tensor]:
    """Each agent's vector wandering away from theta_0."""
    g = torch.Generator().manual_seed(seed)
    x = theta0().expand(n, P).clone()
    out = []
    for _ in range(steps):
        x = x + size * torch.randn(n, P, generator=g, dtype=torch.float64)
        out.append(x.clone())
    return out


# ---- the arithmetic -----------------------------------------------------------------

def test_copies_start_at_theta0_and_track_within_half_a_step():
    ch = channel()
    delta = ch.c * torch.tensor([float(theta0()[p].pow(2).mean().sqrt()) for _n, p in LAYERS])
    for x in walk(50):
        ch.mix(ring(), x)
        error = (x - ch.copies["psi"]).abs()
        assert float(error[:, :20].max()) <= delta[0] / 2 + 1e-12
        assert float(error[:, 20:].max()) <= delta[1] / 2 + 1e-12


def test_the_senders_own_term_stays_exact():
    ch = channel(c=0.5)
    x = walk(1)[0]
    mixing = ring()
    mixed = ch.mix(mixing, x)
    decoded = ch.copies["psi"]
    assert torch.allclose(mixed - mixing @ decoded, torch.diagonal(mixing)[:, None] * (x - decoded))


def test_an_isolated_agent_is_untouched_and_pays_nothing():
    ch = channel(c=0.5)
    x = walk(1)[0]
    assert torch.equal(ch.mix(torch.eye(4, dtype=torch.float64), x), x)
    assert ch.bits == 0 and ch.scalars == 0


def test_a_fine_step_approaches_the_exact_combine():
    x = walk(1)[0]
    fine = channel(c=1e-9).mix(ring(), x)
    assert torch.allclose(fine, ring() @ x, atol=1e-8)


def test_the_received_second_moment_is_clamped_but_its_copy_is_not():
    ch = channel(c=1.0, moment_scales={"second_moment": [1e-3, 1e-3]})
    v = torch.full((4, P), 2e-4, dtype=torch.float64)         # rounds to 0: copy 0
    ch.mix(ring(), v, kind="second_moment")
    v2 = torch.full((4, P), -0.6e-3, dtype=torch.float64)     # rounds to -1: copy -1e-3
    v2[:, 0] = 5e-3
    mixed = ch.mix(ring(), v2.abs() * 0 + v2, kind="second_moment")
    assert float(ch.copies["second_moment"].min()) < 0          # the public copy keeps it
    assert ch.totals["clamped"] > 0
    received = ch.copies["second_moment"].clamp_min(0)
    expect = ring() @ received + torch.diagonal(ring())[:, None] * (v2 - received)
    assert torch.allclose(mixed, expect)


def test_a_moment_without_its_scale_is_refused():
    with pytest.raises(ChannelError, match="scale pass"):
        channel().mix(ring(), walk(1)[0], kind="momentum")


def test_an_unknown_kind_is_refused():
    with pytest.raises(ChannelError, match="kind"):
        channel().mix(ring(), walk(1)[0], kind="velocity")


def test_a_plain_channel_cannot_be_the_codec():
    with pytest.raises(ChannelError, match="CodecChannel"):
        Channel(compressor="codec")


# ---- the modes --------------------------------------------------------------------

def test_scale_mode_mixes_exactly_and_measures_the_moments():
    ch = channel(mode="scale", c=0.0)
    m = walk(1)[0] - theta0()
    assert torch.equal(ch.mix(ring(), m, kind="momentum"), ring() @ m)
    rms = [float(m[:, part].pow(2).mean().sqrt()) for _n, part in LAYERS]
    sums = ch.sums["momentum"]
    assert [(s / n) ** 0.5 for s, n in sums] == pytest.approx(rms)


def test_count_mode_pools_the_events_of_every_message():
    ch = channel()
    expected = {name: [Counter(), Counter()] for name, _p in LAYERS}
    previous = theta0().expand(4, P)
    delta = ch.c * torch.tensor([float(theta0()[p].pow(2).mean().sqrt()) for _n, p in LAYERS])
    copy = previous.clone()
    for x in walk(5):
        ch.mix(ring(), x)
        for (name, part), d in zip(LAYERS, delta, strict=True):
            q = torch.round((x[:, part] - copy[:, part]) / d)
            copy[:, part] += q * d
            runs, amps = count_events(q.to(torch.int64))
            expected[name][0].update(runs)
            expected[name][1].update(amps)
    for name, _p in LAYERS:
        assert ch.counts["psi"][name][0] == expected[name][0]
        assert ch.counts["psi"][name][1] == expected[name][1]


def test_code_mode_charges_each_senders_encoded_length_per_link():
    counter = channel()
    for x in walk(20):
        counter.mix(ring(), x)
    codes = {"psi": [LayerCode(dict(counter.counts["psi"][name][0]),
                               dict(counter.counts["psi"][name][1])) for name, _p in LAYERS]}
    coder = channel(mode="code", codes=codes)
    mixing = ring()
    mixing[0, 2] = 0.1                    # agent 2 now reaches three neighbours
    mixing[0, 0] = 1 / 3 - 0.1
    x = walk(1, seed=9)[0]
    before = coder.copies["psi"].clone() if "psi" in coder.copies else theta0().expand(4, P)
    coder.mix(mixing, x)
    q = torch.round((x - before) / coder.deltas["psi"]).to(torch.int64)
    lengths = [sum(len(encode(q[v, part], codes["psi"][i])) for i, (_n, part)
                   in enumerate(LAYERS)) for v in range(4)]
    degrees = out_degrees(mixing).tolist()
    assert coder.bits == sum(d * n for d, n in zip(degrees, lengths, strict=True))
    assert coder.totals["ideal_bits"] <= coder.bits


def test_code_mode_without_its_table_is_refused():
    with pytest.raises(ChannelError, match="trained table"):
        channel(mode="code").mix(ring(), walk(1)[0])


# ---- layers and config --------------------------------------------------------------

def test_the_mlps_layers_are_its_two_modules():
    from dekf_bench.models.registry import build_model_from_config  # noqa: PLC0415

    model = build_model_from_config(load_config("x1_stationary"))
    layers = module_layers(model)
    assert len(layers) == 2
    assert sum(part.stop - part.start for _n, part in layers) == model.num_params


@pytest.mark.parametrize("comm, message", [
    ({"compressor": "codec", "codec_c": 0.0}, "positive"),
    ({"compressor": "codec", "codec_c": 1e-3, "codec_mode": "code"}, "codec_tables"),
    ({"compressor": "codec", "codec_c": 1e-3, "codec_mode": "zip"}, "codec_mode"),
])
def test_the_config_refuses_an_incomplete_codec(comm, message):
    with pytest.raises(ConfigError, match=message):
        load_config("x1_stationary", overrides={"comm": comm})


# ---- the whole path ---------------------------------------------------------------

LEARNERS = ["diffusion_sgd_atc", "diffusion_atc_adamw", "diffusion_ekf_onehop_mean_receiver"]


def _run(comm: dict, tmp_path, steps: int = 6):
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
        "run": {"horizon": steps, "eval_every": 2}, "learners": LEARNERS, "comm": comm})
    environment = build_environment(config, 0, train)
    model = build_model_from_config(config)
    likelihood = Categorical(10)
    built = build_learners(config, model, likelihood)
    theta = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    records = simulate.run(config, environment, built, build_evalsets(config, environment, test),
                           likelihood, theta, stop_after=steps - 1)
    for name, learner in built.items():
        learner.channel.finish(tmp_path, 0, name)
    final = {}
    for record in records:
        for row in record.rows:
            final[record.learner] = (row["cum_scalars_tx"], row["cum_bits_tx"])
    summaries = {name: tmp_path / f"codec_{name}_seed0.json" for name in built}
    return final, {name: json.loads(path.read_text()) for name, path in summaries.items()
                   if path.exists()}


def test_scale_then_count_then_code_through_the_simulator(tmp_path):
    exact, _ = _run({"compressor": "none"}, tmp_path / "none")
    scale, summaries = _run({"compressor": "codec", "codec_mode": "scale"}, tmp_path / "scale")
    # The scale pass is the exact run: same scalars, same working-precision bits.
    assert scale == exact
    scales = {"learners": {name: {kind: dict(zip(s["layers"], rms, strict=True))
                                  for kind, rms in s.get("moment_rms", {}).items()}
                           for name, s in summaries.items()}}
    scales_path = tmp_path / "scales.json"
    scales_path.write_text(json.dumps(scales))
    assert set(scales["learners"]["diffusion_atc_adamw"]) == {"momentum", "second_moment"}
    assert set(scales["learners"]["diffusion_sgd_atc"]) == {"momentum"}

    comm = {"compressor": "codec", "codec_c": 1e-2, "codec_scales": str(scales_path)}
    counted, summaries = _run({**comm, "codec_mode": "count"}, tmp_path / "count")
    assert all(counted[n][0] == exact[n][0] for n in LEARNERS)   # the scalar ledger holds
    tables = {"c": 1e-2, "learners": {
        name: {kind: {layer: LayerCode({int(k): v for k, v in entry["runs"].items()},
                                       {int(k): v for k, v in entry["amps"].items()}).to_json()
                      for layer, entry in layers.items()}
               for kind, layers in s["counts"].items()}
        for name, s in summaries.items()}}
    assert str(EOB) in next(iter(next(iter(
        tables["learners"]["diffusion_sgd_atc"].values())).values()))["runs"]
    tables_path = tmp_path / "tables.json"
    tables_path.write_text(json.dumps(tables))

    coded, summaries = _run({**comm, "codec_mode": "code", "codec_tables": str(tables_path)},
                            tmp_path / "code")
    for name in LEARNERS:
        scalars, bits = coded[name]
        assert scalars == exact[name][0]
        assert 0 < bits < exact[name][1]                  # cheaper than float32
        totals = summaries[name]["totals"]
        assert totals["coded_bits"] >= totals["ideal_bits"]


def test_tables_trained_at_another_c_are_refused(tmp_path):
    tables = tmp_path / "tables.json"
    tables.write_text(json.dumps({"c": 1e-3, "learners": {}}))
    with pytest.raises(Exception, match="trained at c"):
        _run({"compressor": "codec", "codec_c": 1e-2, "codec_mode": "code",
              "codec_tables": str(tables)}, tmp_path / "x")


def test_a_diverged_message_is_mixed_exactly_and_never_counted():
    ch = channel()
    x = walk(1)[0]
    x[1, 3] = float("nan")
    mixed = ch.mix(ring(), x)
    assert torch.isnan(mixed).any()
    assert ch.totals["diverged_messages"] == 1
    assert "psi" not in ch.counts
