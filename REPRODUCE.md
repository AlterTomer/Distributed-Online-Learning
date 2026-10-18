# Reproducing the work

This document has two parts.

- **Part I, working with the code**, is about using the package directly: setting up
  the environment, loading data, configuring a filter, stepping it through time and
  scoring it. Read it first, even if all you want is to re-run a table. Every runner
  in `scripts/` is built from these same few calls, so once you know them you can
  check a result, change one knob and see what moves, or debug a cell that looks
  wrong.
- **Part II, reproducing the results**, gives the stages that regenerate every number
  and figure the paper reports, in dependency order. It has not been written yet;
  its outline is at the end.

Every code block in Part I was run on a laptop CPU against the current `main` (the
Mackey–Glass block against `mg-task`). Each is short enough to run there in about a
minute.

---

# Part I — working with the code

## 1. Setting up

### 1.1 Install

The project needs Python ≥ 3.11 and PyTorch ≥ 2.2. The reference environment is
Python 3.13.5 with torch 2.8.0+cu126 on an RTX 4070 Laptop GPU (8 GiB).

```bash
git clone https://github.com/AlterTomer/Distributed-Online-Learning.git
cd Distributed-Online-Learning
python -m venv .venv313 && source .venv313/bin/activate   # .venv313\Scripts\activate on Windows

# torch first, from the PyTorch index: the CUDA builds are not on PyPI
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
# then the package, editable, with the dev tools
pip install -e ".[dev]"
```

On a machine without a GPU, drop `+cu126` and use the
`https://download.pytorch.org/whl/cpu` index.

The two tasks currently live on two branches, which will be merged before Part II is
final. To have both checked out at once:

```bash
git worktree add ../Distributed-Online-Learning-mg mg-task
```

| branch | task | model | likelihood |
|---|---|---|---|
| `main` | rotated MNIST, downsampled to $14\times14$ | MLP, one hidden layer of 14, $p=2908$ | categorical |
| `mg-task` | Mackey–Glass one-step prediction | causal Transformer, $p=2273$ | Gaussian |

### 1.2 Check the install, and cache what is needed once

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/check_data.py          # downloads MNIST into data/mnist, once
python scripts/check_environment.py   # topologies, partitions, drift, the image pipeline, streams
python scripts/train_reference.py     # the offline reference e*, once (data/reference)
python tasks.py test-fast             # the suite without the slow tests
```

`check_environment.py` prints what the agents will face: every topology's mixing,
how the training set divides among the agents, each drift schedule, and who receives
which sample. It is the quickest way to see the environment before any learner is
involved. Its constants at the top (`TOPOLOGY`, `SHOW_PLOT`, …) narrow or widen it.

Nothing downloads after `check_data.py`. A runner that finds MNIST missing refuses
to start rather than going to the network in the middle of a sweep.

### 1.3 Device and precision

Two settings in every config decide where a run happens and at what precision.

- **`run.dtype`.** Every filter result is `float64`. `float32` is numerically safe
  for the filters (D133), but the benchmark stays in one precision. Experiment
  configs that only carry gradient methods default to `float32`. The sweep runners
  default to `float64`.
- **`run.device`.** This is `cpu`, `cuda` or `auto`. Each filter agent holds a dense
  $p\times p$ covariance (64.5 MiB at $p=2908$ in `float64`) and runs a Cholesky
  solve every step. A real run therefore needs CUDA, where a 1500-step, five-seed
  cell takes about 20 minutes. On CPU, short horizons like the ones below are fine
  for exploring and debugging, but they cannot reproduce a published cell.

## 2. How the pieces fit

```
config ──► data ──► environment ──► observations at step t (one per agent)
   │                                    │
   ├──► model + likelihood ──► learners ┤  adapt (local) → combine (neighbours)
   │                                    │
   └──► evaluation sets ◄───────────────┘  scored every eval_every steps
                     └──► recorder ──► results/<run>/seed_<s>.parquet
```

Five rules hold everywhere:

- **One config describes a run completely.** It is validated when loaded, and an
  unknown or contradictory field is an error, not a silent default. The run's
  resolved config is written next to its results.
- **The environment is a pure function of $t$.** `env.step(t)` returns the same
  tensors whenever it is called, in any order. That is why runs resume exactly and
  why any step can be inspected.
- **All learners see the same observations.** The runner calls `env.step(t)` once
  and hands the result to every learner, so methods in one run are compared on
  identical data. The observations are read-only.
- **Every method is adapt, then combine.** `adapt` uses only an agent's own data;
  `combine` is the only communication. The filter differs from diffusion SGD in
  `adapt` alone. Pooled ("centralised") learners take the union of all agents'
  batches through `adapt_pooled` instead.
- **Every learner starts from the same $\boldsymbol\theta_0$.** The filter needs a
  common prior, and it removes a confound from the comparison.

## 3. Configs

Configs are composable YAML under `configs/`:

- `base.yaml` holds the defaults;
- `env/`, `graph/` and `model/` hold reusable blocks;
- `learner/` holds one file per method;
- `experiment/` holds the experiment files, which pick one of each and a list of
  learners.

`docs/configs.md` lists every field and its legal values.

```python
from dekf_bench.utils.config import load_config

config = load_config("x1_stationary")          # configs/experiment/x1_stationary.yaml
print(config.run.horizon, config.graph.topology, config.env.samples_per_node_per_step)
print([entry.name for entry in config.learners])
```

`overrides` merges a mapping into the experiment file before it is resolved. It is
how every sweep runner builds its cells, and how you change anything from a script
or a notebook without editing YAML. A learner given by name picks up its
`configs/learner/<name>.yaml`, and the keys you pass go on top:

```python
config = load_config("x1_stationary", overrides={
    "run": {"name": "play", "horizon": 60, "eval_every": 20, "seeds": [0],
            "dtype": "float64", "device": "cpu"},
    "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
    "env": {"samples_per_node_per_step": 4,
            "drift": {"schedule": "recurring", "jump_every": 25,
                      "jump_degrees": 15.0, "jump_seed": 0}},
    "learners": [{"name": "diffusion_sgd_atc", "lr": 0.05}],
})
```

The filter experiments have no YAML of their own. Their runners start from
`x1_stationary` and override what they need, exactly as above. To see what a
published cell actually ran, read the `config.yaml` written in its results folder.

## 4. Data and the environment

### 4.1 Loading data

```python
from dekf_bench.data.registry import load_dataset

train, test = load_dataset(config.env.dataset, "data")   # MNIST, from the cache
print(train.images.shape, test.images.shape)             # (60000, 1, 28, 28), (10000, 1, 28, 28)
```

The splits come back as `ImageSplit`s holding `images` (in $[0,1]$) and `labels`.
Downsampling to $14\times14$, normalisation and rotation are applied when an
observation is built, not here.

### 4.2 Building the environment

```python
from dekf_bench.env.environment import build_environment

env = build_environment(config, 0, train)   # master seed 0
print(env.n_nodes, env.horizon)             # 10 60
print(env.graph.n_edges, env.graph.mixing_gap)
print(env.graph.weights)                    # the N x N combination matrix A
```

The environment combines the graph, the partition of the training set among the
agents, the sample stream and the drift schedule. The master seed is split into
separate streams for each of these, so the graph can be held fixed while the data
order varies, and vice versa.

`env.step(t)` returns a dict from agent to `Observation`:

```python
observations = env.step(0)
o = observations[0]
print(o.x.shape, o.y, o.has_label, o.rotation_degrees)  # (4, 1, 14, 14) tensor([...]) True 0.0
print(env.drift_state(30))                              # the rotation in force at t = 30
```

`Observation` is frozen. When an agent has no label this step (under
`label_availability < 1`), `x` is empty and `y` is `None`.

These are the knobs that change what the agents see:

| field | values | what it controls |
|---|---|---|
| `graph.topology`, `graph.n_nodes`, `graph.params` | `complete`, `ring`, `path`, `grid2d`, `star`, `erdos_renyi` (`p`) | who talks to whom |
| `graph.weights` | `metropolis`, `relative_degree`, `uniform` | the combination matrix $A$ |
| `env.samples_per_node_per_step` | $n$ | the batch per agent per step |
| `env.label_availability` | $\pi_\text{lab}\in(0,1]$ | the fraction of steps on which an agent gets labels |
| `env.partition` | `iid`, or `dirichlet` with `beta` | label skew across agents |
| `env.drift.schedule` | `stationary`, `linear`, `ramp`, `recurring`, `piecewise`, `sinusoidal`, `sawtooth`, `triangle` | how the rotation evolves (capped at 45°) |
| `env.drift_scope` | `global`, `per_node` | one rotation for all agents, or one per agent |
| `env.prior_drift` | `enabled`, `beta`, `total_shift` | label shift: each agent's class prior travels |

The training set is finite: $N\cdot n\cdot T$ must not exceed 60 000 unless
`env.allow_epochs` is set. That is why the network-size runs use $T=500$.

## 5. Model, likelihood and the shared starting point

```python
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.likelihoods.registry import build_likelihood

model = build_model_from_config(config)     # functional: parameters live outside the model
likelihood = build_likelihood(config)       # Categorical on main, Gaussian on mg-task
theta0 = model.flatten(model.init_params(env.seeds.torch_generator("init")))
theta0 = theta0.to(env.train.images.device)
print(model.num_params)                     # 2908
```

The model is functional. `forward(params, x)`, `vjp` and `per_sample_jacobian` take
the parameters as an argument, and `flatten`/`unflatten` convert between a parameter
dict and the flat vector the filter keeps a belief over. $\boldsymbol\theta_0$ is
drawn from its own seed stream on the CPU and then moved to the device, so a CPU run
and a CUDA run of the same config start from the same point.

## 6. Setting up a filter

A learner is chosen by **name**. The name fixes the variant, and the config fields
tune it. A config that contradicts its name, such as a one-hop name with
`adapt_scope: local`, is rejected.

| name | adapt | combine | in the paper |
|---|---|---|---|
| `centralized_ekf_gamma` | one belief over the pooled batch | — | centralised EKF |
| `diffusion_ekf` | own batch | mean only | D-BOLT, local adapt |
| `diffusion_ekf_onehop_mean_receiver` | own and neighbours' batches, linearised at the receiver | mean only | **D-BOLT, one-hop** |
| `diffusion_ekf_full` | own batch | mean and covariance | local adapt, full sharing |
| `diffusion_ekf_onehop_receiver` | as one-hop | mean and covariance | one-hop, full sharing |
| `diffusion_sgd_atc`, `diffusion_sgd_atc_plain`, `diffusion_atc_adamw` | gradient step | parameters (and moments) | ATC + momentum, ATC plain, ATC AdamW |
| `centralized_sgd`, `centralized_adamw`, `local_only`, `local_adamw` | pooled or own gradient | — | centralised and local baselines |

`docs/learners.md` describes all of them, including the older sender-point one-hop
variants that X27 retired.

The filter's state model is $\boldsymbol\theta_t=\gamma\,\boldsymbol\theta_{t-1}+\boldsymbol w_t$,
$\boldsymbol w_t\sim\mathcal N(0,q\boldsymbol I)$, with prior
$\mathcal N(\boldsymbol\theta_0,\sigma_0^2\boldsymbol I)$. These are its fields:

| field | meaning |
|---|---|
| `transition`, `gamma` | `scalar` with $\gamma\le1$, or `identity` (a random walk) |
| `forgetting`, `process_noise_q`, `lambda_forget` | `process_noise` adds $q\boldsymbol I$ every step; `lambda` inflates $\boldsymbol P$ by $1/\lambda$. Use one, not both. |
| `prior_scale` | $\sigma_0^2$. It acts as a trust region: values above about 0.1 diverge on the first steps (D61). |
| `combine_exponent` | full sharing only: $\boldsymbol P_v\leftarrow\sum_u a_{vu}^{\beta}\boldsymbol P_u$, where $\beta=1$ is the conservative bound |
| `inner_steps` | $K$ relinearisations per step, the iterated EKF (Track D, D1) |
| `covariance_blocks` | `full`, or block-diagonal by `neuron`, `layer` or `tensor` (Track D, D3a) |
| `trust_region_ratio` | the run is declared diverged when $\lVert\boldsymbol m\rVert$ passes this multiple of $\lVert\boldsymbol\theta_0\rVert$ |

Every MNIST result uses one of two tuned settings. Never carry one into the other's
place: running the diffusion filters at the centralised setting costs as much as
decentralising does (+0.016).

```python
FILTER = {"transition": "scalar", "gamma": 0.9995, "lambda_forget": 1.0,
          "process_noise_q": 6.0e-4, "prior_scale": 1.0e-3}    # diffusion filters (X20, X23)
CENTRAL = {"transition": "scalar", "gamma": 0.9995, "lambda_forget": 1.0,
           "process_noise_q": 6.0e-5, "prior_scale": 0.01}     # centralised filter (X13)

config = load_config("x1_stationary", overrides={
    "run": {"name": "play", "horizon": 60, "eval_every": 20, "seeds": [0],
            "dtype": "float64", "device": "cpu"},
    "graph": {"topology": "erdos_renyi", "params": {"p": 0.3}},
    "learners": [{"name": "centralized_ekf_gamma", **CENTRAL},
                 {"name": "diffusion_ekf_onehop_mean_receiver", **FILTER},
                 {"name": "diffusion_sgd_atc", "lr": 0.05}],
    "eval": {"evalsets": ["prequential", "current"]},
})
env = build_environment(config, 0, train)
model, likelihood = build_model_from_config(config), build_likelihood(config)
theta0 = model.flatten(model.init_params(env.seeds.torch_generator("init")))

from dekf_bench.learners.registry import build_learners, POOLING

learners = build_learners(config, model, likelihood)    # {name: learner}
for learner in learners.values():
    learner.init(theta0)
```

The gradient baselines' learning rates are tuned per condition by each runner's
`--lr` pass. 0.05 is only a placeholder here.

## 7. Stepping through time

This is the runner's loop with nothing else around it:

```python
from dekf_bench.env.environment import pool

for t in range(env.horizon):
    observations = env.step(t)
    pooled_x, pooled_y = pool(observations)
    for name, learner in learners.items():
        # test, then train: score on the batch before learning from it
        o = observations[0]
        before = (learner.predict(0, o.x).argmax(1) != o.y).double().mean()
        if name in POOLING:
            learner.adapt_pooled(pooled_x, pooled_y)
        else:
            messages = {v: learner.adapt(v, observations[v]) for v in observations}
            learner.combine(messages, env.graph.weights)
```

Between any two steps you can stop and look at the state.

```python
onehop = learners["diffusion_ekf_onehop_mean_receiver"]
m, P = onehop.belief(3)                      # agent 3's mean and covariance
print(m.shape, P.shape, P.diagonal().mean())
print(onehop.flat_params(3).shape)           # theta_v, what predict() uses
print(learners["centralized_ekf_gamma"].covariance.shape)   # one shared P (a property)
x = env.step(env.horizon - 1)[3].x
print(onehop.logit_covariance(3, x).shape)   # H P H^T per sample: (4, 10, 10)
print(onehop.comm_scalars_per_step(env.graph.n_edges))      # what it sends, network-wide
```

`adapt` returns an `Intermediate`, the message that crosses the link. It is frozen,
and `combine` receives a dict of them together with the combination matrix. Under
full sharing the message also carries $\boldsymbol P$. To write a new method,
implement this interface (`learners/base.py`) and register a name
(`docs/howto.md`, "Adding things").

## 8. Scoring

There are two error measures, and the paper reports the second.

- **Prequential.** The error on the batch an agent is about to learn from, computed
  before it learns from it. The runner records it every step.
- **Held-out ("current").** The full test set at the rotation in force at step $t$.
  It is recorded every `eval_every` steps.

```python
from dekf_bench.evaluation.evalsets import build_evalsets

evalsets = build_evalsets(config, env, test)
current = evalsets.at("current", env.horizon - 1)
print(current.images.shape, current.rotation_degrees)
for name, learner in learners.items():
    errors = [(learner.predict(v, current.images).argmax(1) != current.labels).double().mean()
              for v in range(env.n_nodes)]
    print(f"{name:<38} {sum(errors) / len(errors):.4f}")    # network-mean test error
```

The **settled error** quoted throughout the paper and the design notes is this
network-mean `current` error averaged over the last 20% of the horizon, then over
seeds. After 60 steps on CPU the filters are still far from settled, so the numbers
here say nothing about the published ones.

## 9. The whole run, recorded

`simulate.run` does sections 7 and 8 together, along with bookkeeping: test-then-
train ordering, evaluation, disagreement metrics, bits sent, checkpoints. A
`Recorder` writes the rows to parquet.

```python
from pathlib import Path
from dekf_bench.runner import simulate
from dekf_bench.recording import recorder as rec
from dekf_bench.recording.schema import RunContext

out = Path("results") / config.run.name
env = build_environment(config, 0, train)
learners = build_learners(config, model, likelihood)
evalsets = build_evalsets(config, env, test)
context = RunContext.from_config(config, 0, env.graph, run_id=rec.new_run_id(), git_sha="local")
recorder = rec.Recorder(out, context, config)
simulate.run(config, env, learners, evalsets, likelihood, theta0, recorder=recorder)
recorder.finalize()
```

`simulate.run` calls `init(theta0)` itself. Each seed writes one
`seed_<s>.parquet`, plus a `seed_<s>.checkpoint.pt` that lets an interrupted run
resume exactly. Read the parquet files, not the folder: pandas fails on the
checkpoint.

```python
import pandas as pd

frame = pd.concat(pd.read_parquet(p) for p in sorted(out.glob("seed_*.parquet")))
print(frame.metric.unique())     # error_rate, nll, brier, ece, e_agree, e_cent_filter, ...
error = frame[(frame.metric == "error_rate") & (frame.evalset == "current")]
print(error.groupby(["learner", "t"]).value.mean().unstack().round(3))

last = error[error.t >= 0.8 * error.t.max()]
print(last.groupby("learner").value.mean())      # the settled error, one seed
```

One row is one (learner, step, agent or network mean, evaluation set, metric). The
columns also carry the run's identity (seed, topology, mixing gap, drift) and the
communication so far (`cum_scalars_tx`, `cum_bits_tx`), so a frame never needs a
join to be interpreted.

Every sweep runner wraps this in its own `run_one`. That wrapper adds caching by
cell name, a `_complete` marker, a `_diverged` marker when a filter breaks its trust
region, and the `--lr` passes that tune the baselines first.

## 10. Communication

Every learner reports what it sends: `comm_scalars_per_step`, and for the diffusing
ones a `comm_payload` broken down by kind (vectors, covariance, raw data). The
recorded `cum_bits_tx` prices each kind at its own width.

To put a compressor on every diffusing learner's messages, set `comm` in the config.
It applies to the filters and the baselines alike.

| `comm.compressor` | effect |
|---|---|
| `none` | exact, the default and every result outside Track C |
| `float32`, `float16`, `bfloat16` | cast the message |
| `stochastic` | stochastic rounding at `comm.precision` bits |
| `codec` | the offline-trained codec (D137, D138): quantised differences against public copies, run lengths, canonical Huffman |

The codec needs per-learner scales and tables, which `scripts/run_codec.py
--calibrate` trains. Part II covers that order.

## 11. On `mg-task`: the same loop, a different task

The Mackey–Glass branch keeps every interface above. These are the differences:

- **The data are integrated per seed, not loaded.** `build_task(config, seed)`
  returns the environment and its evaluation sets together, for either task, and
  needs no splits for a series.
- **An observation is a block of one agent's trajectory.** `x` and `y` are the
  block's inputs and its one-step-ahead targets.
- **The likelihood is Gaussian and the metric is RMSE.** A held-out set holds fresh
  trajectories of the current law, as `inputs` and `targets`.
- **The tuned settings are on disk.** They are in `results/m4_selection.json`
  (centralised), `results/m5_selection.json` (diffusion) and `results/m3_rates.json`
  (baselines). Those files are written by `run_m3_rates.py`, `run_m4_centralised.py`
  and `run_m5_diffusion.py`.
- **Tests and imports need `PYTHONPATH=src`** in that worktree. Otherwise Python
  imports `main`'s installed package.

```python
from dekf_bench.utils.config import load_config
from dekf_bench.runner.task import build_task
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.likelihoods.registry import build_likelihood
from dekf_bench.learners.registry import build_learners

FILTER = {"transition": "scalar", "gamma": 1.0, "forgetting": "process_noise",
          "process_noise_q": 6e-7, "lambda_forget": 1.0, "prior_scale": 0.01}   # M5's selection
config = load_config("m_abrupt", overrides={
    "run": {"name": "play_mg", "horizon": 30, "device": "cpu", "seeds": [0]},
    "learners": [{"name": "diffusion_ekf_onehop_mean_receiver", **FILTER}]})
env, evalsets = build_task(config, 0)
model, likelihood = build_model_from_config(config), build_likelihood(config)
onehop = build_learners(config, model, likelihood)["diffusion_ekf_onehop_mean_receiver"]
onehop.init(model.flatten(model.init_params(env.seeds.torch_generator("init"))))

for t in range(20):
    observations = env.step(t)
    onehop.combine({v: onehop.adapt(v, observations[v]) for v in observations},
                   env.graph.weights)

current = evalsets.at("current", 19)
rmse = (onehop.predict(0, current.inputs) - current.targets).pow(2).mean().sqrt()
print(model.num_params, current.inputs.shape, float(rmse))
```

## 12. Where to go next

| to… | read |
|---|---|
| change a parameter, add a learner, compare two methods fairly | [`docs/howto.md`](docs/howto.md) |
| see every config field and its legal values | [`docs/configs.md`](docs/configs.md) |
| follow the filter's derivation and what each choice costs | [`docs/filter.md`](docs/filter.md) |
| see every method and what it transmits | [`docs/learners.md`](docs/learners.md) |
| find which experiment answered what, and its design note | [`docs/experiments.md`](docs/experiments.md), [`docs/schedule.md`](docs/schedule.md) |

---

# Part II — reproducing the results

*To be written.* The plan (`docs/schedule.md`, "The reproduction package") is a
driver, `scripts/reproduce.py`, that walks these stages in dependency order. Each
stage states its command, its inputs and outputs, its measured GPU-hours, and the
numbers it must reproduce.

1. **Environment.** Section 1 above, with the versions recorded.
2. **Tests.** The full suite.
3. **Smoke.** Every runner's `--smoke` on CPU: every code path proven before GPU
   time is spent.
4. **Calibration.** The offline references, every `--lr` pass, and every filter
   selection (the X-series re-tunes on MNIST, M3–M5 on Mackey–Glass).
5. **Experiments.** Every claim in the paper, traced to its runner, its cells and
   its tier (D118).
6. **Reports.** Each runner's `--report-only`, checked against the expected values
   within a stated tolerance.
7. **Figures.** The paper's figures only.

There are three tiers, so that checking one table does not cost the full budget:
**smoke** (CPU, about an hour), **one claim** (only that claim's dependencies), and
**full**.
