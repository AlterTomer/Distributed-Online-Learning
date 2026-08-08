# Software Implementation Specification

**Companion to** `WORKPLAN.md`, which defines the research questions, the environment semantics, the learning methods, and the experiments. This document defines the repository: layout, module responsibilities, interfaces, configuration, logging, tests, figures, and tooling.

**Status.** Specification. Environment provisioned and repository initialized 2026-07-30; no package code yet.

**Where it lives.** `C:\Users\alter\PycharmProjects\DistributedOnlineLearning`, built directly in that directory rather than in a `dekf-bench/` subdirectory. It is a git repository (`main`), to be linked to a GitHub remote. The Python environment is **`.venv313`** (Python 3.13.5, torch 2.8.0+cu126, CUDA available); the sibling `.venv` is unused and should be ignored.

---

## 1. Purpose and conventions

The repository has one job: make every experiment in `WORKPLAN.md` §6 runnable from a config file, reproducibly, with results that a plotting script can turn into figures without manual intervention.

**Conventions**

- Python ≥ 3.11, PyTorch ≥ 2.2 (for `torch.func`). Provisioned: 3.13.5 and 2.8.0 (§10).
- `src/` layout, so tests exercise the installed package and import errors surface immediately.
- Everything that varies across an experiment is a config field. If it is not in a config, it cannot be swept, and it will quietly become hard-coded.
- No module imports from `scripts/`. Scripts are thin entry points; all logic lives in the package.
- **Everything on the algorithmic path must be runnable from the IDE's run/debug button.** Training, evaluation, metrics and plotting are developed interactively with breakpoints, so:
  - every `scripts/*.py` defines module-level defaults and an `if __name__ == "__main__":` block that runs with them when no CLI arguments are given. `argparse` may exist; it must never be *required*, because a script that demands arguments cannot be launched from the editor gutter without first hand-building a run configuration;
  - the main path stays **in-process**. Sweep parallelism defaults to serial (`n_workers=1`) and is opt-in, since breakpoints do not fire inside worker processes;
  - nothing on the default path obscures stack frames — no `torch.compile`, no dynamically generated functions;
  - core logic lives in importable functions taking ordinary Python arguments, so the same code can be driven from a script, a test, or a notebook.

  Terminal-only entry points remain fine for *tooling* (`tasks.py`, linting, packaging tests). This is a development-phase constraint and may be revisited once the experiments are settled.
- Type hints throughout; `mypy` in CI but non-blocking at first.
- The package is named `dekf_bench` even though phase 1 contains no filter — renaming a package later is churn for no gain.

**Naming note.** Avoid `logging` as a package name (it shadows the standard library and confuses readers); use `recording/`.

---

## 2. Repository layout

```
DistributedOnlineLearning/          # repo root; .venv313/ and .venv/ live here too
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile                          # make test / make x1 / make figures
│
├── configs/
│   ├── base.yaml                     # defaults every config inherits
│   ├── env/
│   │   ├── mnist_stationary.yaml
│   │   ├── mnist_rotating_linear.yaml
│   │   ├── mnist_rotating_piecewise.yaml
│   │   └── mnist_rotating_sinusoidal.yaml
│   ├── graph/
│   │   ├── complete.yaml
│   │   ├── ring.yaml
│   │   ├── path.yaml
│   │   ├── grid2d.yaml
│   │   ├── star.yaml
│   │   ├── erdos_renyi.yaml
│   │   └── disconnected.yaml          # negative control
│   ├── model/
│   │   ├── mlp_small.yaml             # PRIMARY: 196-14-10 on 14x14 input, p = 2908
│   │   ├── mlp.yaml                   # 784-128-10, p ~ 1e5; sanity comparison only
│   │   ├── cnn.yaml
│   │   └── linear_probe.yaml
│   ├── learner/
│   │   ├── centralized_sgd.yaml
│   │   ├── diffusion_sgd_atc.yaml     # primary
│   │   ├── diffusion_sgd_cta.yaml     # eq. (17) of [1]
│   │   ├── local_only.yaml
│   │   └── diffusion_ekf.yaml         # phase 5 placeholder
│   └── experiment/
│       ├── x0_exactness.yaml
│       ├── x1_stationary.yaml
│       ├── x1b_atc_vs_cta.yaml
│       ├── x2_rotating.yaml
│       ├── x3_topology_sweep.yaml
│       ├── x4_sparsity_sweep.yaml
│       ├── x5_abrupt_shift.yaml
│       └── x6_non_iid.yaml
│
├── src/dekf_bench/
│   ├── __init__.py
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   ├── graph.py              # topologies, weights, spectral gap; G^c and G^d kept separable
│   │   ├── partition.py          # IID and Dirichlet label-skew shard assignment
│   │   ├── drift.py              # DriftSchedule: stationary / linear / piecewise / sinusoidal
│   │   ├── stream.py             # per-agent time-indexed sampler; sparsity knobs; exactly-once
│   │   └── environment.py        # orchestrator; yields per-agent observations at each t
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mnist.py              # download, cache, tensor conversion, normalization
│   │   └── transforms.py         # rotate(28x28) -> downsample(14x14); ONE implementation,
│   │                             #   used by train, all eval sets, and the reference classifier
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # the Model protocol (§4.2)
│   │   ├── functional.py         # functional_call wrappers, flatten/unflatten, vjp/jvp
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── linear_probe.py
│   │   └── registry.py           # name -> builder
│   │
│   ├── learners/
│   │   ├── __init__.py
│   │   ├── base.py               # the Learner protocol (§4.3)
│   │   ├── centralized_sgd.py
│   │   ├── diffusion_sgd_atc.py  # primary
│   │   ├── diffusion_sgd_cta.py  # eq. (17) of [1]
│   │   ├── local_only.py
│   │   ├── optim_state.py        # how optimizer moments are mixed in combine
│   │   └── diffusion_ekf.py      # phase 5 stub; interface only
│   │
│   ├── likelihoods/              # trivial now; required by phase 5
│   │   ├── __init__.py
│   │   ├── base.py               # mu(), Lambda(), innovation(), nll()
│   │   ├── categorical.py        # softmax Fisher  diag(pi) - pi pi^T
│   │   └── gaussian.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── classification.py     # accuracy, error rate, per-agent and mean
│   │   ├── disagreement.py       # E_agree across agents; E_cent vs the centralized learner
│   │   ├── communication.py      # scalar and round counters; ledger table
│   │   └── calibration.py        # NLL, Brier, reliability curves
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── protocol.py           # prequential + periodic full-test
│   │   ├── evalsets.py           # current / backward / canonical sets under drift
│   │   └── reference.py          # offline reference classifier; train + cache
│   │
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── simulate.py           # the main loop over t; learner-agnostic, multi-learner
│   │   ├── sweep.py              # cartesian sweeps over config axes
│   │   └── seeding.py            # separable seeds
│   │
│   ├── recording/
│   │   ├── __init__.py
│   │   ├── recorder.py           # buffered writes to parquet/CSV
│   │   └── schema.py             # the column contract (§7)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # load, merge, validate, resolve inheritance
│       ├── determinism.py
│       └── linalg.py             # phase 5: Woodbury, Joseph form, PSD projection
│
├── scripts/
│   ├── run_experiment.py         # single run from a config
│   ├── run_sweep.py
│   ├── train_reference.py        # offline reference classifier
│   ├── make_figures.py
│   └── check_environment.py      # graph stats, stream visualisation, smoke test
│
├── tests/
│   ├── conftest.py
│   ├── test_graph.py
│   ├── test_partition.py
│   ├── test_stream.py
│   ├── test_drift.py
│   ├── test_transforms.py
│   ├── test_simulate.py
│   ├── test_models.py
│   ├── test_learners.py
│   ├── test_exactness.py         # §9.1 — the important one
│   ├── test_metrics.py
│   ├── test_evaluation.py
│   └── test_reproducibility.py
│
├── notebooks/
│   ├── 01_environment_visual.ipynb
│   └── 02_result_exploration.ipynb
│
├── data/                         # gitignored; MNIST cache
├── results/                      # gitignored; run outputs
├── figures/
└── docs/
    ├── WORKPLAN.md               # research plan
    ├── IMPLEMENTATION.md         # this document
    ├── configs.md                # every config file and field
    ├── learners.md               # the methods, X0, the communication ledger
    ├── environment.md            # the environment's guarantees and what enforces them
    ├── design_notes.md           # decisions log, updated as they are made
    └── diffekf_integration.md    # §13, kept live
```

---

## 3. Module responsibilities

### `env/`

| Module | Responsibility | Must not |
|---|---|---|
| `graph.py` | Build topologies; compute combination weights (Metropolis, relative-degree); expose degree, diameter, connectivity, spectral gap. Keep `G_comm` and `G_data` as separate attributes even though `G_data` is empty in phase 1. | Know anything about data or models |
| `partition.py` | Assign disjoint index shards of the training set to agents, IID or Dirichlet label-skew | Touch image tensors |
| `drift.py` | Map $t \mapsto$ transform parameters. One class per schedule, common interface | Apply the transform itself |
| `stream.py` | For each $(v,t)$, produce the sample indices this agent receives; enforce exactly-once; apply label availability | Apply drift |
| `environment.py` | Compose the above: at step $t$, gather indices, fetch tensors, apply the drift transform, return per-agent observations | Contain learner logic |

### `models/`

| Module | Responsibility |
|---|---|
| `base.py` | The `Model` protocol (§4.2) |
| `functional.py` | `functional_call` wrappers; flatten/unflatten between a parameter dict and a flat vector; `vjp`/`jvp` helpers. The one place `torch.func` is used |
| `registry.py` | Map config name → builder, so models are selectable without imports in the runner |

### `learners/`

| Module | Responsibility |
|---|---|
| `base.py` | The `Learner` protocol (§4.3); the per-agent state container |
| `optim_state.py` | Combine-step mixing of optimizer moments (`none`, `momentum`, `all`). Shared by all diffusion learners so the policy is defined in exactly one place |
| `diffusion_ekf.py` | Phase 5. In phase 1 this is the protocol implementation raising `NotImplementedError`, so the interface is exercised by the type checker |

### `evaluation/`

| Module | Responsibility |
|---|---|
| `evalsets.py` | Build and cache current / backward / canonical evaluation sets for a given drift state. **The single place that guarantees train and eval see the same rotation** |
| `reference.py` | Train the offline classifier per rotation level; cache to `data/reference/`; return $e^\star$ |
| `protocol.py` | Prequential evaluation each step; full evaluation every $K$ steps |

### `runner/`

`simulate.py` owns the loop, is learner-agnostic, and drives **several learners against one environment** (`WORKPLAN.md` §6.1):

```
for t in range(T):
    obs = env.step(t)                                    # per-agent observations, drawn ONCE
    for learner in learners:                             # e.g. centralized, atc, local_only
        preq = protocol.prequential(learner, obs)        # test-then-train: evaluate first
        intermediates = {v: learner.adapt(v, obs[v]) for v in agents}
        learner.combine(intermediates, graph.weights)
        if t % K == 0:
            full = protocol.full_eval(learner, evalsets.at(t))
        recorder.log(...)                                # rows carry the `learner` column
    if t % K == 0:
        recorder.log(disagreement.e_cent(learners, ref="centralized_sgd"))
```

Three properties of this shape are load-bearing:

- `env.step(t)` is called **once per step**, before the learner loop, and its result is shared. That is what makes X0 exact by construction rather than by seed-matching, and what makes $E_{\text{cent}}$ meaningful.
- Observations are treated as **read-only** by learners. A learner that normalizes or augments in place would corrupt every other learner's view; `Observation` is a frozen dataclass for this reason.
- The centralized learner receives the same `obs` dict and pools it internally. It does not get a privileged path through the environment.

The per-learner body — adapt, then combine — must not change when Diff-EKF arrives.

---

## 4. Core interfaces

Described at signature level. No implementation here.

### 4.1 Environment

- `Environment.reset(seed) -> Environment` — **returns a new environment** rather than mutating in place. The gym-style `-> None` sketched originally fits environments that carry episode state; this one is a pure function of (config, seed, data), and a mutating reset would be the single place where a captured component (an evaluation set, a learner's cached `graph.weights`) could silently belong to the previous seed. See design note D25.
- `Environment.step(t) -> dict[int, Observation]`
- `Environment.observe(node, t) -> Observation` — one agent, for tests and evaluation
- `Environment.assert_unmodified(observations, t) -> None` — the positive check that no learner mutated the shared observations in place
- `Observation` fields: `x` (tensor), `y` (tensor or `None`), `has_label` (bool), `n_samples` (int). **Frozen dataclass, tensors never mutated in place** — several learners share one instance per step (§3).
- `Environment.graph -> Graph`, `Environment.drift_state(t) -> DriftState`
- `Environment.horizon -> int`

### 4.2 Model

| Member | Purpose | Needed by |
|---|---|---|
| `init_params(seed) -> ParamDict` | Deterministic init, so all agents can share $\bm\theta_0$ | all |
| `forward(params, x) -> logits` | Functional; params passed explicitly | all |
| `num_params -> int` (`p`) | Sizing and cost accounting | all |
| `output_dim -> int` (`q`) | Covariance sizing | Diff-EKF |
| `flatten(params) -> Tensor` / `unflatten(vec) -> ParamDict` | Filter state is a flat vector | Diff-EKF |
| `vjp(params, x, v) -> Tensor` / `jvp(params, x, v) -> Tensor` | Jacobian products without materializing $\bm H$ | Diff-EKF |
| `param_groups() -> list[slice]` | Layer blocks for block-diagonal / Kronecker covariance and last-layer filtering | Diff-EKF |

Phase-1 SGD uses only the first three. Building the rest now costs about half a day and avoids refactoring every model later.

### 4.3 Learner

- `Learner.init(model, graph, seed) -> None`
- `Learner.adapt(node_id, observation) -> Intermediate` — local computation, no communication
- `Learner.combine(intermediates, weights) -> None` — the only communication
- `Learner.predict(node_id, x) -> logits`
- `Learner.state(node_id) -> LearnerState` — a **dict-like** container, not a bare tensor, so a covariance can be added later without touching the runner
- `Learner.flat_params(node_id) -> Tensor` — the flat $\bm\theta^v$, needed by $E_{\text{agree}}$ and $E_{\text{cent}}$
- `Learner.comm_scalars_per_step() -> int` — for the ledger

`centralized_sgd` and `local_only` implement `combine` as a no-op. This keeps the loop uniform.

**`adapt_scope`.** Every diffusion learner carries a config field `adapt_scope: local | one_hop`, defaulting to `local`. In phase 1 only `local` is implemented and `one_hop` raises. It exists now because the research note's Prop. 1 — complete-graph exactness for Diff-EKF — holds for $\mathcal M_{v,t}=\mathcal V$, the **one-hop** adapt step, not the local one, and that variant exchanges $(\bm B_{u,t},\bm H_{u,t}^{\mathsf T}\bm\nu_{u,t})$ at $O(pq')$ per link instead of $O(p)$. Retrofitting it in phase 5 would touch the learner protocol, the communication ledger, and every ledger row already recorded. The SGD exactness check needs no such distinction — its identity holds with purely local gradients — so the field is inert in phases 1–4 and merely has to exist.

### 4.4 Likelihood

- `mu(logits) -> Tensor` — mean parameter (softmax for categorical)
- `Lambda(logits) -> Tensor` — Fisher information; for categorical, $\operatorname{diag}(\pi)-\pi\pi^{\mathsf T}$
- `innovation(y, logits) -> Tensor` — $y - \mu$
- `nll(y, logits) -> Tensor`

Only `nll` is used in phase 1, but writing the rest now lets SGD runs log calibration for free and removes a phase-5 dependency.

### 4.5 Recorder

- `Recorder.log(record: dict) -> None` — validated against `schema.py`
- `Recorder.flush() -> None` — buffered; do not write per step
- `Recorder.finalize() -> Path`

---

## 5. Configuration

Composable YAML with inheritance from `base.yaml`. An experiment config selects one entry from each of `env/`, `graph/`, `model/`, `learner/` and adds run parameters.

**Top-level keys**

| Key | Fields | Defaults |
|---|---|---|
| `run` | `name`, `seeds` (list), `horizon` $T$, `eval_every` $K$, `dtype`, `device`, `out_dir` | $T=1500$, $K=25$, `float32` (`float64` for X0), `cpu` |
| `graph` | `topology`, `n_nodes` $N$, `weights` (`metropolis` \| `relative_degree`), topology params | $N=10$, `ring`, `metropolis` |
| `env` | `samples_per_node_per_step` $n$, `label_availability`, `partition` (`iid` \| `dirichlet`, `beta`), `drift`, `drift_scope` (`global` \| `per_node`), `allow_epochs` | $n=2$, $\pi_{\text{lab}}=1.0$, `iid`, `global`, `allow_epochs: false` |
| `env.drift` | `schedule` (`stationary` \| `linear` \| `piecewise` \| `sinusoidal`), **`total_degrees`**, `change_points`, `jump_degrees`, `amplitude_degrees`, `period` | `total_degrees: 45`, `jump_degrees: 15`, `amplitude_degrees: 30` |
| `model` | `name`, `input_size` (`14` \| `28`), architecture params | `mlp_small`, `input_size: 14`, hidden `[14]` |
| `learners` | **a list**; each entry `name`, `optimizer` (`sgd` \| `sgd_momentum` \| `adamw`), `lr`, `momentum`, `mix_optimizer_state` (`none` \| `momentum` \| `all`), `adapt_scope` (`local` \| `one_hop`) | `[centralized_sgd, diffusion_sgd_atc, local_only]`, `sgd_momentum`, `mix_optimizer_state: momentum`, `adapt_scope: local` |
| `eval` | `evalsets` (list), `backward_offset`, `batch_size` | `[prequential, current, canonical]`, `backward_offset: 500` (15° of separation at the capped drift rate; 200 gave only 6°) |

**Rules**

- Config validation happens once, at load, against a dataclass schema. A typo in a key is an error, not a silently ignored field.
- **`learners` is a list, not a scalar.** All entries share one environment and are stepped together (§3). A config naming a single learner is legal and simply degenerates to the old behaviour.
- **$\alpha$ is derived, never configured.** The drift schedule takes `total_degrees` and computes $\alpha=\text{total\_degrees}/T$, so changing $T$ cannot silently change how far the distribution travels (`WORKPLAN.md` §4.3). A config supplying a bare `alpha` is rejected.
- **The shard budget is validated at load**, not discovered at step 1400: if `allow_epochs` is false, reject unless $N\,n\,T\le60000$.
- The fully resolved config is written into the run's output directory, so a result is always traceable to the exact settings that produced it.
- Sweeps are expressed as lists in the experiment config; `run_sweep.py` takes the cartesian product. Note the collision: `learners` is a list *within* a run, not a sweep axis.

---

## 6. Determinism and seeding

Four **independent** seed streams, derived from one master seed:

| Stream | Controls |
|---|---|
| `seed_init` | Model initialization (shared across all agents) |
| `seed_partition` | Shard assignment |
| `seed_stream` | Sample order within a shard, label-availability draws |
| `seed_graph` | Random graph realization |

Separating them means you can hold the partition fixed and vary only initialization, which is what makes an ablation interpretable. Combining them into one seed makes that impossible.

Also: set `torch.use_deterministic_algorithms(True)`, seed Python/NumPy/Torch, and pin `PYTHONHASHSEED`. Record library versions and the git commit hash in the run metadata.

---

## 7. Logging schema

One long-format table. One row per (run, seed, step, learner, node, metric). Long format costs disk but makes every plot a groupby, and avoids schema churn when a metric is added.

| Column | Type | Notes |
|---|---|---|
| `run_id` | str | uuid or slug |
| `git_sha` | str | |
| `seed` | int | master seed |
| `experiment` | str | `x1`, `x2`, … |
| `learner` | str | |
| `topology` | str | |
| `n_nodes` | int | |
| `spectral_gap` | float | constant per run; denormalized for easy plotting |
| `samples_per_node` | int | |
| `label_availability` | float | |
| `drift_schedule` | str | |
| `drift_param` | float | $\alpha$ |
| `drift_state` | float | current rotation at step $t$ |
| `t` | int | |
| `node_id` | int \| str | integer, or `mean` / `reference` for aggregates |
| `evalset` | str | `prequential` \| `current` \| `backward` \| `canonical` |
| `metric` | str | `accuracy`, `error_rate`, `nll`, `brier`, `e_agree`, `e_cent`, … |
| `value` | float | |
| `cum_scalars_tx` | int | cumulative transmitted scalars |
| `cum_rounds` | int | cumulative exchange rounds |
| `wallclock_s` | float | |

Written as parquet (CSV fallback). Buffered — flush every few hundred steps, never per step.

---

## 8. Metrics and figures

### 8.1 Where metrics are computed

| Metric | Module | When |
|---|---|---|
| Prequential accuracy / NLL | `evaluation/protocol.py` | every step |
| Per-agent full-test accuracy | `evaluation/protocol.py` | every $K$ steps |
| Mean error rate, spread | `metrics/classification.py` | derived at plot time from per-agent rows |
| Parameter disagreement $E_{\text{agree}}$ | `metrics/disagreement.py` | every $K$ steps |
| Deviation from centralized $E_{\text{cent}}$ | `metrics/disagreement.py` | every $K$ steps; needs the co-run centralized learner (§3) |
| Communication counters | `metrics/communication.py` | every step, from `Learner.comm_scalars_per_step()` |
| Calibration (NLL, Brier, reliability) | `metrics/calibration.py` | every $K$ steps |

Aggregates (`mean`, spread, gap to reference) are **derived at plot time**, not stored. Storing only per-agent rows means a new aggregate never requires a re-run.

### 8.2 Communication ledger

A small table emitted once per experiment, in the style of Table I of [1]: one row per method, giving rounds per step, message size, and total scalars over the run. For this phase every distributed method sends one $p$-vector per link per step, so the table is nearly trivial — which is the point, and is what makes the phase-5 comparison fair by construction.

### 8.3 Figure specifications

All produced by `scripts/make_figures.py` from logged results, with no manual steps. It caches the plotted series to `figure_data/` so a redraw at a different resolution or with different labels never re-reads `results/` — see `docs/figures.md`. Bands are ±1 s.d. over seeds unless stated.

| ID | Content | Axes | Source |
|---|---|---|---|
| **F1** | Headline. Error rate vs time, three methods, reference $e^\star$ as a horizontal line. Two panels: stationary \| rotating | x: $t$, y: error rate | X1, X2 |
| **F2** | F1 replotted against communication | x: cumulative scalars transmitted, y: error rate | X1, X2 |
| **F3** | Price of connectivity. Final gap to reference vs spectral gap, one point per topology | x: $1-\rho$, y: $\bar e_T - e^\star$ | X3 |
| **F4** | Per-agent spread for the distributed method: mean line with min–max band | x: $t$, y: error rate | X1, X2 |
| **F5** | Parameter disagreement and deviation from centralized, over time, log y, two series | x: $t$, y: $E_{\text{agree}}$, $E_{\text{cent}}$ | X1, X2, X1b |
| **F6** | Sparsity heatmap: final gap as a function of $(n, \pi_{\text{lab}})$ | x: $n$, y: $\pi_{\text{lab}}$, colour: gap | X4 |
| **F7** | Adaptation transient around the abrupt shift, zoomed on $[t^\ast-50, t^\ast+300]$ | x: $t$, y: error rate | X5 |
| **F8** | ATC vs CTA: error rate and disagreement, two panels | x: $t$ | X1b |
| **F9** | Non-IID: gap vs Dirichlet $\beta$ | x: $\beta$ (log), y: gap | X6 |
| **F10** | *(phase 5)* Diff-EKF added to F1 and F2 | — | — |

Follow one style across all figures — same palette per method (centralized / distributed / local-only / reference), same axis conventions, colour-blind-safe, legible in greyscale.

---

## 9. Test suite

`pytest`. Fast tests run on every commit; the exactness test is the gate.

### 9.1 `test_exactness.py` — build this first

On a complete graph, uniform weights, plain SGD, `dtype=float64`, run centralized SGD and ATC diffusion **in one `simulate.py` call sharing one environment** (§3) for ~50 steps and assert agreement to $10^{-12}$ (`WORKPLAN.md` §7.1).

The config must pin, and the test must assert, all four preconditions of the identity:

| Precondition | Config | Why it matters |
|---|---|---|
| Equal batch sizes across agents | `label_availability: 1.0`, uniform $n$ | The average of per-agent means equals the pooled mean only for equal $|\mathcal D^v_t|$ |
| `mean` loss reduction everywhere | learner config | Centralized reduces over all $Nn$; a `sum` anywhere rescales the step by $N$ |
| Plain SGD | `optimizer: sgd`, no momentum, no weight decay | Any optimizer state makes the trajectories diverge legitimately |
| Common $\bm\theta_0$ | shared `seed_init` | Holds at $t=0$; the identity then preserves it inductively |

Violating the first produces a small, plausible, non-zero residual rather than an obvious failure — which is exactly the failure mode this test exists to catch, so the assertion is on $10^{-12}$ and not on "close enough".

This catches weight normalization, initialization mismatch, batch partition, and mean-vs-sum reduction bugs — all of which otherwise produce plausible but wrong curves. It is also the harness reused for the phase-5 filter, where the analogous check additionally requires `adapt_scope: one_hop` (§4.3).

### 9.2 The rest

| File | Asserts |
|---|---|
| `test_graph.py` | Weights row-stochastic, non-negative, respect adjacency, positive diagonal; connectivity detection; complete graph has spectral gap 1; disconnected graph is flagged |
| `test_partition.py` | Shards are disjoint and cover the training set; Dirichlet skew produces the requested concentration |
| `test_stream.py` | Exactly-once consumption; no train/test leakage; `label_availability` gives the right empirical rate; **$NnT\le60000$ rejected at config load when `allow_epochs` is false** |
| `test_drift.py` | $\alpha=0$ is identical to stationary; the same transform is applied to train and eval; piecewise change points land where specified; **$\alpha$ is derived as `total_degrees`/$T$ and a bare `alpha` key is rejected**; **total rotation never exceeds `total_degrees`** |
| `test_transforms.py` | Rotation is applied at $28\times28$ **before** downsampling to $14\times14$, and the composed pipeline is byte-identical between the training path, all three evaluation sets, and the reference classifier |
| `test_simulate.py` | `env.step(t)` is called exactly once per step regardless of learner count; learners cannot observe each other's mutations (frozen `Observation`); a one-learner config reproduces the same trajectory as that learner run alone |
| `test_models.py` | flatten/unflatten round-trip is exact; `num_params` matches; `vjp` matches autograd on random vectors; `init_params` is deterministic from a seed |
| `test_learners.py` | $N=1$ makes all learners coincide; `local_only` transmits zero scalars; combine preserves the parameter convex hull; optimizer-state mixing policies behave as configured |
| `test_metrics.py` | Accuracy on a hand-made case; disagreement is zero when agents agree; communication counters match analytic expectations |
| `test_evaluation.py` | Prequential evaluation happens strictly before the update; evaluation sets carry the correct rotation for the step |
| `test_reproducibility.py` | Same master seed gives identical logged metrics; different `seed_init` with the same `seed_partition` changes results in the expected way |

**CI.** Run the fast suite plus `ruff` and `black --check` on every push. The exactness test runs on every push too — it is worth the seconds.

---

## 10. Tooling and dependencies

**Provisioned in `.venv313` on 2026-07-30** (Python 3.13.5; the spec's floor is 3.11):

| Runtime | Version | | Dev | Version |
|---|---|---|---|---|
| `torch` | 2.8.0+cu126 | | `pytest` | 9.1.1 |
| `torchvision` | 0.23.0+cu126 | | `pytest-cov` | 7.1.0 |
| `numpy` | 2.1.2 | | `ruff` | 0.16.1 |
| `pandas` | 2.3.2 | | `black` | 26.5.1 |
| `pyarrow` | 25.0.0 | | `mypy` | 2.3.0 |
| `networkx` | 3.3 | | `pre-commit` | 4.6.1 |
| `pyyaml` | 6.0.3 | | | |
| `matplotlib` | 3.10.6 | | | |

CUDA is available but phases 1–4 target CPU (§11); `device` stays a config field. `jupyter` is not installed and is only needed for `notebooks/` — add it when those are written. Config loading uses plain `pyyaml` rather than `hydra-core`: composition here is a few dozen lines of dict merge, and Hydra's CLI ownership fights the sweep runner.

Pin exact versions in `requirements.txt`; keep the loose ranges in `pyproject.toml`. The pre-existing `old_environment_requirements.txt` and `requirements_without_torch.txt` at the repo root are from an unrelated project and should be deleted once `requirements.txt` is generated.

**Makefile targets:** `install`, `test`, `test-fast`, `lint`, `reference` (train the offline classifier), `x0` … `x6`, `sweep`, `figures`, `clean`.

---

## 11. Runtime and resource budget

Rough expectations, to catch it early if something is accidentally quadratic:

| Run | Scale | Expected |
|---|---|---|
| Single X1 run, one seed | $N=10$, $n=2$, $T=1500$, $196$–$14$–$10$ MLP, all 3 learners co-run | seconds to a couple of minutes, CPU |
| X1 full | 5 seeds (learners share each run, so 5 runs, not 15) | a few minutes |
| X3 topology sweep | 4 topologies × 5 seeds | tens of minutes |
| X4 sparsity sweep | 12 cells × 5 seeds, $T=750$ | tens of minutes |
| Offline reference | 10 rotation levels ($0°$–$45°$, step $5°$), full MNIST, ~20 epochs each | a few minutes total, GPU optional |

Everything fits on a laptop CPU. Parallelize sweeps across seeds with a process pool; do not bother with GPU until the CNN or Transformer arrives. If a single X1 run takes more than a few minutes, something is wrong — most likely the full evaluation is running every step instead of every $K$.

---

## 12. Build order

| Phase | Modules to write | Gate | Status |
|---|---|---|---|
| **0** | `utils/config.py`, `utils/determinism.py`, `runner/seeding.py`, `data/mnist.py`, repo scaffolding, CI | `make test` green; MNIST loads | ✅ done |
| **1** | `env/graph.py`, `env/partition.py`, `env/drift.py`, `env/stream.py`, `env/environment.py`, `data/transforms.py`, `scripts/check_environment.py` | `test_graph`, `test_partition`, `test_stream`, `test_drift`, `test_transforms` pass; stream visualisation renders | ✅ done |
| **2** | `models/*`, `likelihoods/*`, `metrics/*`, `evaluation/*`, `scripts/train_reference.py` | `test_models` passes; all 16 rotation-level $e^\star$ cached, each at expected MNIST accuracy for a $196$–$14$–$10$ MLP | ✅ done |
| **3** ✅ | `learners/*`, `learners/optim_state.py`, `runner/simulate.py`, `recording/*`, `scripts/run_experiment.py`, `scripts/sweep_hyperparameters.py`, `scripts/make_figures.py` | **`test_exactness` passes** (1.7e-15); `test_simulate` passes; X0, X1, X1b, X2, X5 produce F1, F2, F5, F8 |
| **4** | `runner/sweep.py`, `scripts/run_sweep.py` | X3–X6 produce F3, F4, F6, F7, F9. X3 re-tunes lr per topology (WORKPLAN §10.2 item 5) |
| **5** | `learners/diffusion_ekf.py`, `utils/linalg.py`, structured covariance | Filter reproduces the centralized EKF on a complete graph |

---

## 13. Diff-EKF forward-compatibility checklist

Keep `docs/diffekf_integration.md` open and tick these off as phases 1–4 proceed. Each is cheap now and expensive later.

1. **Flat parameter vector.** `flatten` / `unflatten` exist and are tested. The filter's state is $\bm\theta\in\mathbb R^p$, not a `state_dict`.
2. **Functional forward.** Parameters passed explicitly, so the model can be evaluated at an arbitrary $\bm\theta$ (the predictive mean) without mutating a module.
3. **Jacobian products.** `vjp` / `jvp` exposed and tested against autograd.
4. **Likelihood objects.** `mu()`, `Lambda()`, `innovation()` implemented — the softmax Fisher is the one that matters. Writing them in phase 2 also lets SGD runs log calibration for free.
5. **Learner interface is adapt/combine.** Diff-EKF then differs from diffusion SGD *only* in `adapt`. If the interface were a single `step()`, the filter would not fit.
6. **Per-agent state is a dict.** So a covariance can be added without changing `simulate.py`.
7. **Communication accounting is per-learner.** Diff-EKF sends the same $O(p)$ as diffusion SGD, and that equality is the headline of the comparison — the counter must be able to demonstrate it.
8. **float64 path.** The filter is far more numerically delicate than SGD.
9. **`linear_probe` model exists.** In the last-layer-only regime the EKF is an *exact* KF, giving a setting where the theory holds with no linearization error — the analogue of the research note's linear sanity check, and the right first Diff-EKF experiment.
10. **Size reality check — resolved.** A 784–128–10 MLP has $p\approx10^5$, so a dense covariance is $10^{10}$ entries: impossible. Shrinking the hidden width does not fix it — the budget $p\lesssim3\times10^3$ forces $h\approx3$ at 784 inputs, which is not a classifier. The **input** is what comes down: $14\times14$ downsampling gives a $196$–$14$–$10$ MLP at $p=2908$, whose dense covariance is $\approx8.5\times10^6$ entries, a few tens of MB in float64. `mlp_small.yaml` is therefore the primary model for *all* phases, not a phase-5 variant, so the comparison is like-for-like by construction (`WORKPLAN.md` §4.6).
11. **Two graphs stay separable.** `env/graph.py` keeps `G_comm` and `G_data` distinct from day one, even though phase 1 only ever populates `G_comm`. When the project reaches GNNs (Class C), the learner gains $L$ forward and $L-1$ backward message-passing rounds per step; [1] §III-A derives exactly those recursions for a GCNN and releases code, so it is an integration task rather than a derivation. Retrofitting a second graph into an environment that assumed one touches every file.
12. **`adapt_scope` exists from day one.** The research note's Prop. 1 gives complete-graph exactness only for the **one-hop** adapt step $\mathcal M_{v,t}=\mathcal V$, which exchanges $(\bm B_{u,t},\bm H_{u,t}^{\mathsf T}\bm\nu_{u,t})$ at $O(pq')$ per link rather than the $O(p)$ of the local step. Phase 1 implements `local` only and raises on `one_hop`, but the field, the ledger column, and the config plumbing exist now — adding a second communication mode later invalidates every ledger row already written (§4.3).
13. **The information form is not optional for classification.** The gain form inverts $\bm S=\bm H\bm P\bm H^{\mathsf T}+\bm R$, which does not exist for the softmax likelihood: $\bm\Lambda=\diag(\bm\pi)-\bm\pi\bm\pi^{\mathsf T}$ is singular by construction ($\bm\Lambda\one=\zero$), so $\bm R=\bm\Lambda^{-1}$ is undefined rather than merely ill-conditioned (research note, Rem. 2). `diffusion_ekf.py` carries **both** paths from the start — gain form for Gaussian regression, information form for the exponential family — and MNIST classification uses the latter. Writing only the gain form and "generalizing later" means rewriting the filter.
14. **Structure-preserving prediction.** The research note §6.4 prefers multiplicative inflation $\bm P_{t|t-1}=\lambda^{-1}\bm P_{t-1|t-1}$ to additive $\bm Q_t$, because with $\bm F=\I$ inflation is exactly $\bm\Omega_{t|t-1}=\lambda\bm\Omega_{t-1|t-1}$ and therefore closed under diagonal, block-diagonal and Kronecker structure, whereas $(\bm\Omega^{-1}+\bm Q)^{-1}$ is dense. `diffusion_ekf.yaml` exposes `lambda` and `Q` with exactly one active at a time — they are two parameterisations of the same forgetting and are jointly unidentifiable.

---

## 14. References

[1] R. Olshevskyi, Z. Zhao, K. Chan, G. Verma, A. Swami, S. Segarra, "Fully Distributed Online Training of Graph Neural Networks in Networked Systems," arXiv:2412.06105, Dec 2024. Code: `github.com/RostyslavUA/fdTrainGNN`.

[2] `WORKPLAN.md` — research plan, environment semantics, experiments, validity checks.

[3] Research note: *Distributed Online Bayesian Learning of Deep Neural Networks — A Diffusion Extended Kalman Filtering Formulation.*
