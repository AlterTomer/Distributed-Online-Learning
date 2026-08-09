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
| [`docs/learners.md`](docs/learners.md) | The four methods, the exactness check, and what each transmits |
| [`docs/results.md`](docs/results.md) | Measured numbers, the settings behind them, and what they support |
| [`docs/figures.md`](docs/figures.md) | What each figure shows, how to read it, and the cached figure data |
| [`docs/configs.md`](docs/configs.md) | Every config file and field, with legal values |
| [`docs/design_notes.md`](docs/design_notes.md) | Decisions log: what was chosen, over what, and why |

**Status:** phases 0–4 complete; phase 5 (Diff-EKF) is next. X0 passes at
1.7e-15. Every method is tuned on a held-out grid before any comparison
(`docs/results.md` §2) — a fixed learning rate produced a headline that was an
optimizer artefact, so this is now a standing rule rather than a convenience.

### Results so far

| | |
|---|---|
| **Diffusion ≈ pooled data** | On a ring, ATC is 0.0014 behind centralized SGD against a seed s.d. of 0.0035 — *inside the noise*. Ten agents exchanging one message with two neighbours recover almost everything a fusion centre would get. |
| **Cooperation is worth 0.047 to 0.527** | Depending on sparsity and label skew. Under strong non-IID ($\beta{=}0.1$) a lone agent lands near chance while the same agent in the network reaches 0.103. |
| **Connectivity costs little** | Across seven topologies the worst *settled* penalty is 0.008 (a star); early in a run the spread is wider (0.018), so connectivity governs convergence speed more than the final answer. |
| **The spectral gap is not the best predictor of it** | Mean self-weight $\bar a_{vv}$ ranks the topologies at Spearman $+0.964$ against $-0.786$ for $1-\rho$ — but it was chosen post hoc and failed its one out-of-sample test (`docs/results.md` §7.3). |
| **ATC ≈ CTA** | 0.0768 vs 0.0774, indistinguishable at tuned settings. ATC's real advantage is robustness to step size, not accuracy. |
| **Diffusion tolerates mis-tuning** | Worst penalty for a learning rate chosen in the wrong regime: 0.028 for ATC against 0.183 for centralized. |
| **Halving the message costs little** | Dropping momentum to send $p$ scalars per link instead of $2p$ costs 0.007–0.046, worst in the sparsest corner and **flat across label skew** — the second half of the message pays under sparsity, not under heterogeneity. |
| **Nobody forgets; the lone agent lags** | X7's sinusoidal schedule revisits earlier rotations, the only regime where forgetting is well-posed. No cooperative method shows measurable forgetting (1.1–1.6 σ); `local_only` is *better* on a state it left than on the current one, at 10 σ — lag, not retention. |

**What this sets up for phase 5.** There is no headroom left on stationary
accuracy, so Diff-EKF's case has to come from tracking under drift, from
communication efficiency, or from calibrated uncertainty. Its competitor on the
communication axis is `diffusion_sgd_atc_plain` at 0.0902 — not the stronger
momentum variant — because the filter sends one $p$-vector per link.

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

## Running things

Every algorithmic entry point is a plain script with editable constants at the
top, runnable from an IDE with a debugger attached:

```bash
python scripts/run_experiment.py x1_stationary   # one of x0..x2, x5, x7; --fresh to restart
python scripts/run_topology_sweep.py             # X3
python scripts/run_sparsity_sweep.py             # X4 and X6
python scripts/sweep_hyperparameters.py          # the tuning grid
python scripts/make_figures.py                   # F1-F10
python scripts/make_figures.py --from-cache --dpi 300
python scripts/make_summary_docx.py               # the 3-page technical summary
python scripts/make_presentation.py               # the slide deck
```

Runs and sweeps are **resumable and exact**: the loop consumes no randomness, so
a resumed run reproduces an uninterrupted one bit-for-bit, and re-running a sweep
skips cells already on disk.

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

Experiment targets (`x0`–`x7`, `reference`, `sweep`, `figures`) are listed but
report which phase introduces them until the corresponding scripts exist.

`run.device` accepts only `cpu`. CUDA is *measurably slower* at this model size
-- 0.69x at batch 4, since p=2908 cannot amortise a kernel launch -- and phase 5
lifts the guard when the dense covariance makes it a 14x win (design note D43).

## Layout

```
src/dekf_bench/     the package: env, data, models, learners, metrics,
                    evaluation, runner, recording, utils
configs/            composable YAML; every swept quantity is a config field
scripts/            runnable entry points: experiments, sweeps, figures
tests/              pytest; test_exactness.py is the gate
results/            gitignored; one parquet per seed, plus a resumable checkpoint
data/               gitignored MNIST cache and the offline reference e*
```

Tests import the *installed* package rather than `src/` directly, so packaging
mistakes surface immediately.
