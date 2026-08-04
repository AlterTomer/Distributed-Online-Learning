# Distributed-Online-Learning

A benchmark for distributed online learning over a graph: $N$ agents on a
communication graph, each receiving a few labelled samples per time step, learning
one shared classifier online without a fusion centre. Establishes centralized and
diffusion-SGD baselines, and lays the groundwork for a diffusion extended Kalman
filter (Diff-EKF).

## Documentation

| File | What it answers |
|---|---|
| [`docs/WORKPLAN.md`](docs/WORKPLAN.md) | The research plan: questions, methods, experiments, validity checks |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | The repository specification: layout, interfaces, build order |
| [`docs/environment.md`](docs/environment.md) | What each agent observes, and the guarantees the benchmark rests on |
| [`docs/configs.md`](docs/configs.md) | Every config file and field, with legal values |
| [`docs/design_notes.md`](docs/design_notes.md) | Decisions log: what was chosen, over what, and why |

**Status:** phases 0 and 1 complete; phase 2 (models, metrics, evaluation) in progress.

## Setup

The project targets Python ≥ 3.11 and PyTorch ≥ 2.2 (for `torch.func`). The
development environment is `.venv313` (Python 3.13.5, torch 2.8.0+cu126).

```bash
# torch first, from the PyTorch index -- the CUDA builds are not on PyPI
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# then the package, in editable mode, with dev tooling
pip install -e ".[dev]"
```

On a CPU-only machine, drop the `+cu126` local version and use the
`https://download.pytorch.org/whl/cpu` index instead. Everything in phases 1–4 is
sized for a laptop CPU.

## Tasks

`make` is not available on Windows, so the targets live in `tasks.py` and the
`Makefile` is a thin alias over it. Either entry point works:

```bash
python tasks.py --list        # show all targets, including ones not yet built
python tasks.py test          # full suite
python tasks.py test-fast     # skips tests marked slow
python tasks.py lint          # ruff + black --check, read-only
python tasks.py format        # apply ruff --fix and black
python tasks.py typecheck     # mypy; non-blocking while modules stabilise
python tasks.py clean         # caches and build artefacts; leaves data/ and results/
```

Experiment targets (`x0`–`x6`, `reference`, `sweep`, `figures`) are listed but
report which phase introduces them until the corresponding scripts exist.

## Layout

```
src/dekf_bench/     the package: env, data, models, learners, metrics,
                    evaluation, runner, recording, utils
configs/            composable YAML; every swept quantity is a config field
scripts/            thin entry points; no logic
tests/              pytest; test_exactness.py is the gate
data/               gitignored MNIST cache
results/            gitignored run outputs
```

Tests import the *installed* package rather than `src/` directly, so packaging
mistakes surface immediately.
