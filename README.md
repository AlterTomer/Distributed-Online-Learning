# Distributed-Online-Learning

A benchmark for distributed online learning over a graph: $N$ agents on a
communication graph, each receiving a few labelled samples per time step, learning
one shared model online without a fusion centre, while the data drift. It holds the
centralised and diffusion-SGD/AdamW baselines and the diffusion extended Kalman
filter (diff-EKF, called D-BOLT in the paper) in its local-adapt and one-hop
variants, with mean-only or full covariance sharing, and an offline-trained codec
that prices every message in bits.

There are two tasks, on two branches until they are merged (`docs/schedule.md`,
"After A and B"):

| branch | task | model |
|---|---|---|
| `main` | rotated MNIST, $14\times14$, categorical likelihood | MLP, $p=2908$ |
| `mg-task` | Mackey–Glass prediction, Gaussian likelihood | causal Transformer, $p=2273$ |

## Documentation

| File | What it answers |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **Working with the code — data, a filter, stepping through time, scoring — and reproducing every result, runner by runner** |
| [`docs/schedule.md`](docs/schedule.md) | **Where everything stands: the four tracks, each item's status and the design note behind it** |
| [`docs/experiments.md`](docs/experiments.md) | **X0–X28: what each asks, the command that runs it, and in what order** |
| [`docs/phase5_plan.md`](docs/phase5_plan.md) | What the diffusion filter had to be tested on (the P5.x items), as a checklist |
| [`docs/howto.md`](docs/howto.md) | **"How do I…?" — change a parameter, add a learner, compare methods fairly** |
| [`docs/communication_plan.md`](docs/communication_plan.md) | Track C: the bits ledger, public copies and difference coding, the codec |
| [`docs/mackey_glass_plan.md`](docs/mackey_glass_plan.md) | Track B: the Mackey–Glass task, its 25 decisions and the M0–M14 battery |
| [`docs/WORKPLAN.md`](docs/WORKPLAN.md) | The research plan: questions, methods, experiments, validity checks |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | The repository specification: layout, interfaces, build order |
| [`docs/environment.md`](docs/environment.md) | What each agent observes, and the guarantees the benchmark rests on |
| [`docs/learners.md`](docs/learners.md) | Every method, the exactness checks, and what each transmits |
| [`docs/results.md`](docs/results.md) | Measured numbers for X0–X28 and P5.3, the settings behind them, and what they support |
| [`docs/figures.md`](docs/figures.md) | What each figure shows, how to read it, and the cached figure data |
| [`docs/configs.md`](docs/configs.md) | Every config file and field, with legal values |
| [`docs/design_notes.md`](docs/design_notes.md) | Decisions log: what was chosen, over what, and why — and the write-up of every run since X26 |
| [`docs/filter.md`](docs/filter.md) | The filter derivation, both centralised and diffusion, and what each choice costs |

**Status.** Phases 0–4 are complete, and so is phase 5's measurement of the
diffusion filter on MNIST (Track A): X19–X28, the P5.x items, network size, and an
AdamW arm in every cell the paper draws on. The Mackey–Glass battery M0–M14 is done
on `mg-task` (Track B). The communication codec is built and measured on both tasks
(Track C). Four extensions decided before the writing (Track D: $K$ inner steps and
the iterated EKF, subtractive dither, a block-diagonal covariance, a GNN on a coupled
Mackey–Glass network) are built and smoked, and their GPU runs are queued.
`docs/schedule.md` holds each item's status.

**Where results are written up.** X0–X28 and P5.3 have sections in
[`docs/results.md`](docs/results.md); everything after them is written up in its
design note (D99 onwards), which `docs/schedule.md` links item by item. Each claim
carries a tier: **confirmatory** if it was named before its run, **exploratory**
otherwise, and only confirmatory claims rest on $p_\text{holm}$ (D118).

Two standing rules came out of mistakes. Every method is tuned on a held-out grid
before any comparison (`docs/results.md` §2): a fixed learning rate once produced a
headline that was an optimiser artefact. And damage is compared only across learners
whose stationary levels are comparable, which a mis-tuned baseline taught us the hard
way (D77). X0 passes at 1.7e-15, and the diffusion filter's complete-graph identity
holds to 1e-10.

### Results so far: the baselines (X0–X12)

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

### Results so far: the diffusion filter (X19 onwards)

| | |
|---|---|
| **One-hop is the decentralised method to beat** | Under abrupt rotation (15° every 25 steps) one-hop settles at 0.107 against the centralised filter's 0.093, local adapt's 0.119, and ATC with momentum and with AdamW at 0.131 and 0.151. One-hop beats ATC AdamW in nine of twelve backfilled conditions (D127) and at every $N$ (D119). |
| **It beats ATC while sending less** | One-hop linearises a neighbour's batch at the **receiver's** own estimate (X27, D99: never worse, better where agents disagree) and sends $\psi$ plus the raw batch, 3 696 scalars per link, against momentum ATC's 5 816. At exactly matched bandwidth, mean-only local adapt beats `atc_plain` by 0.012–0.025 ($t$ = 10–24; X26). |
| **Covariance sharing buys nothing once one-hop is used** | Full minus mean-only is null at both ends of the skew axis at the receiver point (X28, D100) and on every topology (P5.3, D103), so the deployable variant sends $p$ scalars, not $p^2$. |
| **Size: local adapt falls behind, one-hop does not** | At a fixed mixing gap, local adapt's gap to the centralised filter grows with $N\in\{10,20,30\}$; one-hop's growth is not detected (bounded below 0.005). The same holds on Mackey–Glass (D117, D129). |
| **Label skew** | At $\beta_{\mathrm{dir}}=0.1$ local adapt drops to 0.111, behind the gradient baselines; one-hop, which adapts on its neighbours' labels too, reaches 0.078 and stays the most accurate decentralised method at every skew (X25, X27). |
| **Heterogeneity does not let beliefs personalise** | Per-agent drift (P5.7, D115) costs every shared model about the same, and label shift (P5.8, D124) is absorbed by cooperation in every family, with nothing extra for the filter. Per-agent delays on Mackey–Glass (M8, D111) cost the filters 0.002–0.005 and AdamW nothing, yet both D-BOLT variants stay about 0.02 below ATC AdamW. The agents converge to each other, not to the centralised filter's weights, and weight distance does not track accuracy (P5.12, D134). |
| **The bound is tight, the covariance is not a predictive** | `lem:conservative` is tight (shown covariance-free on Mackey–Glass, shrink 0.985–0.995), but the filter's belief scores worse than the plug-in prediction on MNIST and is far too wide on Mackey–Glass (P5.11/P5.14, D120). |
| **Compression is nearly free for the filter** | The offline-trained codec (quantised differences on public copies, run lengths, canonical Huffman) sends local adapt at 4.3 bits per parameter on MNIST and 1.5 on Mackey–Glass, against 64, at a cost within ±0.0002 error and +0.0013 RMSE of the uncompressed twins; on Mackey–Glass every baseline needs a step ten times finer (D138). |
| **float32 is safe, float64 stays** | Single precision is numerically safe for every filter on both tasks and about halves the MNIST GPU time, but every experiment already run is float64, so the benchmark stays there rather than mix precisions across the paper (D133). |

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
`https://download.pytorch.org/whl/cpu` index instead.

The Mackey–Glass task lives on `mg-task`; a second worktree keeps both checked out
at once (`git worktree add ../DistributedOnlineLearning-mg mg-task`). Its tests must
run with `PYTHONPATH=src`, or pytest imports `main`'s installed package instead.

### What you need to reproduce which results

| | hardware | notes |
|---|---|---|
| The SGD baselines (X0–X12) | laptop CPU | sized for it deliberately |
| Every EKF result (X13 onwards), and every AdamW arm beside one | **CUDA GPU** | a dense $p\times p$ covariance per agent |
| The test suite | laptop CPU | 1505 tests on `main` |

⚠ **The filter results were produced on CUDA and cannot be reproduced on CPU in
reasonable time.** Each agent carries a dense $2908\times2908$ covariance in
float64 — 64.5 MiB, ten of them, with a Cholesky solve per agent per step. The
reference machine is an RTX 4070 Laptop (8 GiB), where a 1500-step run at five
seeds takes about 20 minutes; a sweep is hours, and the larger ones run as
unattended GPU queues over several days. `--device cpu` works and is what the tests
and the runners' `--smoke` modes use, but it is for debugging, not for reproducing a
published cell. On the same hardware and software a re-run is bitwise identical
(X25+ reproduced X27 to 0.0e+00, D101); across GPU models and library versions it
is not.

⚠ **There is no CI, and a green suite is not a reproduction.** The repository has no
workflow (design note Q4 is still open), so the tests run only when someone runs them.
A CPU run of the suite checks that the code *runs* and that its identities hold; it
does not exercise the CUDA path every published number came from. A green suite is
not evidence that a result still reproduces — re-run the cell for that.

⚠ **The stated floor of Python 3.11 is not enforced anywhere.** mypy is pinned to
3.13 locally because `scipy-stubs` fails to parse under 3.11, so nothing checks
the claim. If you are below 3.13 and something breaks, that is why.

## Running things

**Start with [`docs/experiments.md`](docs/experiments.md)** for X0–X28 — for each,
the question it answers, the exact command, the measured runtime where we have it,
and which experiments must run before which — and with
[`docs/schedule.md`](docs/schedule.md) for everything after (P5.x, N>10, the AdamW
pass, the codec, Track D). Each runner's docstring carries its design and the command
lines; they are worth reading before running anything expensive.

Every algorithmic entry point is a plain script. The run knobs are flags —
`--horizon`, `--seeds`, `--device`, `--dtype`, `--dataset`, `--fresh`, and
`--help` lists them with their defaults — and the defaults are the values each
experiment was run at, so a bare command reproduces the published cell. What a
sweep *varies* stays in the source, since a run whose name no longer describes it
is worse than no run:

```bash
python scripts/check_environment.py              # graph stats, sample streams, smoke test
python scripts/check_data.py                     # cache MNIST, once
python scripts/train_reference.py                # the offline reference, once

# the baselines and the centralised filter
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
python scripts/run_ekf_skew.py --lr              # X17, the filter under label skew
python scripts/run_sawtooth.py --lr              # X18, sawtooth drift

# the diffusion filter
python scripts/run_diffusion_ekf.py --lr         # X19, the diffusion filter's first measurement
python scripts/run_diffusion_tuning.py --tune    # X20, re-tuning it on its own grid
python scripts/run_diffusion_gamma.py            # X21, gamma jointly with q
python scripts/run_diffusion_extrapolation.py    # X22, how far an agent may extrapolate
python scripts/run_diffusion_joint.py            # X23, all three axes at once
python scripts/run_diffusion_combine.py          # X24, alpha x beta under full sharing
python scripts/run_diffusion_skew.py --lr        # X25, Dirichlet label skew
python scripts/run_atc_plain.py --lr             # X26, the bandwidth-matched baseline
python scripts/run_linearization_point.py        # X27, sender against receiver point
python scripts/run_skew_topup.py                 # X25+, the skew cells at five seeds
python scripts/run_receiver_full.py              # X28, full sharing at the receiver point
python scripts/run_diffusion_topology.py --lr    # P5.3, topology and the spectral gap
python scripts/run_network_size.py --probe-only  # N>10: the memory smoke, then --lr, then cells
python scripts/run_p57_heterogeneous_drift.py --lr  # P5.7, heterogeneous drift
python scripts/run_belief_calibration.py --lr    # P5.11 / P5.14, the belief and the bound
python scripts/run_disagreement.py               # P5.12, agreement against the centralised filter
python scripts/run_label_sparsity.py --lr        # P5.4, sparse labels
python scripts/run_break_rate.py --lr            # P5.5, the break rate
python scripts/run_label_shift.py --lr           # P5.8, label shift
python scripts/run_midtransient_shift.py --lr    # P5.23, a second shift mid-transient
python scripts/run_adamw_pass.py --lr            # AdamW arms for the earlier experiments

# communication and precision
python scripts/run_float32_probe.py --device cuda        # can the filters run in float32?
python scripts/run_delta_entropy_probe.py --device cuda  # what would differences cost in bits?
python scripts/run_codec.py --calibrate --device cuda    # the codec: --calibrate, --validate,
                                                         # --report-runs; --dither for D2

# Track D (queued)
python scripts/run_inner_steps.py --lr           # D1, K inner steps and the iterated EKF
python scripts/run_covariance_blocks.py          # D3a, the block-diagonal covariance
```

On `mg-task` the battery is `run_m2_references.py` through `run_m14b_break_rate.py`,
plus that branch's copies of the shared runners (`run_codec.py`,
`run_inner_steps.py`, `run_covariance_blocks.py`, `run_float32_probe.py`) and
`run_gnn.py` for D3c.

Most runners take a `--lr` pass first and **refuse to start without it**, rather
than carrying a learning rate chosen for another condition. That is not caution:
doing it wrong once put a baseline at chance and inverted a damage ordering, and the
result read as a finding until the stationary column was looked at (D77). The larger
ones also run a pre-flight (memory, data budget) before any cell.

Runs and sweeps are **resumable and exact**: the loop consumes no randomness, so
a resumed run reproduces an uninterrupted one bit-for-bit, and re-running a sweep
skips cells already on disk.

Every runner from X23 on has a `--report-only` mode that turns its cells back into
the tables its design note quotes, with paired $t$, intervals and Holm correction
from `dekf_bench.metrics.paired`. The earlier sweeps have separate readers:

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
reproduce a result is here: the runners above regenerate the runs, and their
`--report-only` modes and the readers turn them back into numbers. The paper's
figures will get public builders in `scripts/figures/` as part of the reproduction
package (`docs/schedule.md`, "The reproduction package").

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
                    evaluation, runner, recording, utils; codec.py and
                    compression.py hold the communication channel
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
