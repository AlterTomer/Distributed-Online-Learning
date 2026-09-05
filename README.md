# Distributed-Online-Learning

A benchmark for distributed online learning over a graph: $N$ agents on a
communication graph, each receiving a few labelled samples per time step, learning
one shared classifier online without a fusion centre. Establishes centralized and
diffusion-SGD baselines, and lays the groundwork for a diffusion extended Kalman
filter (Diff-EKF).

## Documentation

| File | What it answers |
|---|---|
| [`docs/experiments.md`](docs/experiments.md) | **X0–X16: what each asks, the command that runs it, and in what order** |
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

**Start with [`docs/experiments.md`](docs/experiments.md).** It is the index of
every experiment the benchmark has run — X0 through X16 — giving for each one the
question it answers, the exact command, the measured runtime where we have it,
and which experiments must run before which. `WORKPLAN.md` §6 states the *design*
of X0–X7 and why each exists; `docs/experiments.md` is what to type.

Every algorithmic entry point is a plain script with editable constants at the
top, runnable from an IDE with a debugger attached:

```bash
python scripts/check_environment.py              # graph stats, sample streams, smoke test
python scripts/check_data.py                     # cache MNIST, once
python scripts/train_reference.py                # the offline reference, once

python scripts/run_experiment.py x1_stationary   # x0..x2, x5, x7..x10; --fresh to restart
python scripts/run_topology_sweep.py             # X3
python scripts/run_sparsity_sweep.py             # X4 and X6
python scripts/run_recurring_sweep.py            # X11, repeated abrupt shifts
python scripts/run_linear_sweep.py               # X12, smooth drift at four rates
python scripts/sweep_hyperparameters.py          # the tuning grid
python scripts/run_ekf_sweep.py                  # X13, tuning the centralised filter
python scripts/run_ekf_generalization.py         # X14, the crossed drift sweep
python scripts/run_ekf_retune.py                 # X15, gamma against lambda
python scripts/run_ekf_ramp.py                   # X16, the filter on X9's ramp
```

Runs and sweeps are **resumable and exact**: the loop consumes no randomness, so
a resumed run reproduces an uninterrupted one bit-for-bit, and re-running a sweep
skips cells already on disk.

Each sweep has a matching reader that turns its parquet into the numbers the
design notes quote:

```bash
python scripts/report_ekf_sweep.py               # the tuning grid, ranked
python scripts/report_ekf_generalization.py      # damage per drift condition
python scripts/report_breaks.py                  # where each baseline breaks
```

Outputs land in a gitignored `results/`, and anything that writes a figure uses a
gitignored `figures/` at the repository root. To publish elsewhere -- a shared
drive, a paper's asset folder -- set `DEKF_FIGURES_DIR` rather than editing a
script; a relative value resolves against the repository root, so it means the
same thing from any working directory.

```bash
DEKF_FIGURES_DIR="/path/to/a/shared/folder" python scripts/run_experiment.py x1_stationary
```

**The figure and meeting-document builders are deliberately not in this
repository.** They draw the PNGs and write the `.docx`/`.pptx` for our own
progress reviews, so they answer "what should the slide look like" rather than
"does the benchmark run" or "is this number right". Everything needed to
reproduce a result is here: the runners above regenerate the runs, and the
readers above turn them back into numbers.

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

`run.device` was CPU-only through phases 1-4, because CUDA is *measurably
slower* at this model size -- 0.69x at batch 4, since p=2908 cannot amortise a
kernel launch. Phase 5 lifts the guard, as design note D43 said it would: the
filter is dominated by dense covariance operations, where CUDA is a 14x win
(D58). The SGD baselines stay on CPU, which is still the faster choice for them.

## Layout

```
src/dekf_bench/     the package: env, data, models, learners, metrics,
                    evaluation, runner, recording, utils
configs/            composable YAML; every swept quantity is a config field
scripts/            runnable entry points: experiments, sweeps, and the readers
                    that turn their output back into numbers
tests/              pytest; test_exactness.py is the gate
results/            gitignored; one parquet per seed, plus a resumable checkpoint
data/               gitignored MNIST cache and the offline reference e*
figures/            gitignored PNGs and their cached figure_data/ tables;
                    relocatable with DEKF_FIGURES_DIR
```

Tests import the *installed* package rather than `src/` directly, so packaging
mistakes surface immediately.
