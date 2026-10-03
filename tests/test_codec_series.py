"""The codec on the series task (C2, D137, D138): Mackey--Glass's Transformer through it.

`test_codec_channel.py` runs the chain on MNIST; this runs it on the task this branch is
for. The Transformer's 16 tensors are 8 modules, so this also exercises the codec's
layers beyond the MLP's two.
"""

from __future__ import annotations

import json

import pytest

from dekf_bench.codec import LayerCode, module_layers
from dekf_bench.utils.config import load_config

LEARNERS = ["diffusion_sgd_atc", "diffusion_atc_adamw", "diffusion_ekf_onehop_mean_receiver"]
#: M3's stationary selections: Mackey--Glass needs rates far below the MNIST defaults,
#: which diverge here -- the first version of this test found that, and with it that a
#: diverged message must never reach the codec's counts.
ENTRIES = [{"name": "diffusion_sgd_atc", "lr": 1e-5}, {"name": "diffusion_atc_adamw", "lr": 3e-3},
           {"name": "diffusion_ekf_onehop_mean_receiver"}]


def _run(comm: dict, tmp_path, steps: int = 4):
    from dekf_bench.learners.registry import build_learners  # noqa: PLC0415
    from dekf_bench.likelihoods.registry import build_likelihood  # noqa: PLC0415
    from dekf_bench.models.registry import build_model_from_config  # noqa: PLC0415
    from dekf_bench.runner import simulate  # noqa: PLC0415
    from dekf_bench.runner.task import build_task  # noqa: PLC0415

    config = load_config("m_stationary", overrides={
        "run": {"horizon": steps, "eval_every": 2, "device": "cpu"},
        "learners": ENTRIES, "comm": comm})
    environment, evalsets = build_task(config, 0)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    built = build_learners(config, model, likelihood)
    theta = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    theta = theta.to(environment.device)
    records = simulate.run(config, environment, built, evalsets, likelihood, theta,
                           stop_after=steps - 1)
    for name, learner in built.items():
        learner.channel.finish(tmp_path, 0, name)
    final = {}
    for record in records:
        for row in record.rows:
            final[record.learner] = (row["cum_scalars_tx"], row["cum_bits_tx"])
    paths = {name: tmp_path / f"codec_{name}_seed0.json" for name in built}
    return final, {n: json.loads(p.read_text()) for n, p in paths.items() if p.exists()}


def test_the_transformers_layers_are_its_eight_modules():
    from dekf_bench.models.registry import build_model_from_config  # noqa: PLC0415

    model = build_model_from_config(load_config("m_stationary"))
    layers = module_layers(model)
    assert [name for name, _part in layers] == [
        "embed", "norm1", "qkv", "proj", "norm2", "ff1", "ff2", "head"]
    assert sum(part.stop - part.start for _n, part in layers) == model.num_params


def test_scale_then_count_then_code_on_the_series(tmp_path):
    exact, _ = _run({"compressor": "none"}, tmp_path / "none")
    scale, summaries = _run({"compressor": "codec", "codec_mode": "scale"}, tmp_path / "scale")
    assert scale == exact
    scales = {"learners": {name: {kind: dict(zip(s["layers"], rms, strict=True))
                                  for kind, rms in s.get("moment_rms", {}).items()}
                           for name, s in summaries.items()}}
    (tmp_path / "scales.json").write_text(json.dumps(scales))

    comm = {"compressor": "codec", "codec_c": 1e-2,
            "codec_scales": str(tmp_path / "scales.json")}
    counted, summaries = _run({**comm, "codec_mode": "count"}, tmp_path / "count")
    assert all(counted[n][0] == exact[n][0] for n in LEARNERS)
    tables = {"c": 1e-2, "learners": {
        name: {kind: {layer: LayerCode({int(k): v for k, v in entry["runs"].items()},
                                       {int(k): v for k, v in entry["amps"].items()}).to_json()
                      for layer, entry in layers.items()}
               for kind, layers in s["counts"].items()}
        for name, s in summaries.items()}}
    (tmp_path / "tables.json").write_text(json.dumps(tables))

    coded, summaries = _run({**comm, "codec_mode": "code",
                             "codec_tables": str(tmp_path / "tables.json")}, tmp_path / "code")
    for name in LEARNERS:
        assert coded[name][0] == exact[name][0]
        assert 0 < coded[name][1] < exact[name][1]        # cheaper than float64
        totals = summaries[name]["totals"]
        assert totals["coded_bits"] >= totals["ideal_bits"]
        assert set(summaries[name]["layers"]) == {
            "embed", "norm1", "qkv", "proj", "norm2", "ff1", "ff2", "head"}


@pytest.fixture(autouse=True)
def _cpu_threads():
    import torch  # noqa: PLC0415

    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)
