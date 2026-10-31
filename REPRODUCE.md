# Reproducing the work

This document has two parts.

- **Part I, working with the code**, is about using the package directly: setting up
  the environment, loading data, configuring a filter, stepping it through time and
  scoring it. Read it first, even if all you want is to re-run a table. Every runner
  in `scripts/` is built from these same few calls, so once you know them you can
  check a result, change one knob and see what moves, or debug a cell that looks
  wrong.
- **Part II, reproducing the results**, gives the stages that regenerate every number
  and figure the paper reports, in dependency order: every runner on both branches,
  with its commands, what it needs on disk, its recorded hours and the headline it
  must reproduce.

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

This part lists every experiment the repository has run, runner by runner, in the
order the runners depend on each other. For each one it gives the commands, what must
already be on disk, the hours it took on the reference machine, and the headline
result it must reproduce. The full tables are in the runner's `--report-only` output
and in the write-up the entry names: `docs/results.md` for X0–X28 and P5.3, and the
design note (D-number) for everything after.

The driver, `scripts/reproduce.py`, walks stages 1–3 on `main` (D141). From stage 4
on this is still a recipe to follow by hand: the rest of the driver, and the
expected-value files it will check against (`reproduce/expected/*.json`), come
after the two branches merge (`docs/schedule.md`, "The reproduction package").
Until then, the Mackey–Glass entries run from the `mg-task` worktree.

## 13. Read this first

**Hours are as recorded.** They are wall-clock hours on the reference machine (RTX
4070 Laptop, 8 GiB; Python 3.13.5; torch 2.8.0+cu126), with the GPU otherwise idle.
Each figure is the sum over an experiment's runs, taken from the result folders'
timestamps or from the GPU queue logs. Where both exist they agree to within 0.1 h.
X0–X12 ran on the CPU, as did M2–M4 on `mg-task`; everything else ran on the GPU. The
first filter sweeps (X13, X14) also paid two overheads that have since been fixed, a
checkpoint per agent (D68) and a flush that grew quadratically with the horizon
(D69). Their hours are therefore an upper bound.

**Devices.** The commands below omit `--device`. Pass `--device cuda` to every
filter runner, as the GPU queues did. M2–M4 ran on the CPU, which is their default.

**Order is enforced, but follow it anyway.** A runner with a `--lr` pass refuses its
main pass until the rates are on disk (D77). A runner that reads another
experiment's cells refuses until those cells exist. The "needs" column below gives
that order. Follow it instead of discovering it through refusals.

**Every run is cached by name.** A finished run has a `_complete` marker and is
skipped. An interrupted run resumes exactly from its checkpoint. `--fresh` discards
a run and redoes it. `--smoke` adds `_smoke` to every run name, so a smoke run never
collides with a real cell.

**What "reproduce" means.** On the same GPU model, driver and library versions, a
re-run is bitwise identical: the merge gates print `0.0e+00`, and X25+ reproduced
X27 exactly (D101). On other hardware or versions it is not. There, check each claim
through its statistics:

- the sign of each confirmatory contrast;
- whether its interval excludes zero;
- its $p_\text{holm}$ (D118);
- that each level lies within a few standard errors of the recorded one.

The expected-value files will state a tolerance per number.

**Disk.** `results/` holds 12 GB on `main` and 15 GB on `mg-task`.

## 14. The tiers

| tier | what runs | cost |
|---|---|---|
| **smoke** | stages 1–3: the environment, the tests, and every runner's `--smoke` | about 3.4 h on a laptop CPU (i7-13620H): 4 min of tests, then 3.3 h of smoke, the longest step 36 min (D141). A GPU is faster; that run is still to be timed |
| **one claim** | one row of stage 5, the rows in its "needs" column, then its report | from the tables: P5.4, for instance, needs only the reference classifier and costs 27 GPU-h |
| **full** | everything | as recorded: about 126 CPU-h and 550 GPU-h on `main`, and 11 CPU-h and 280 GPU-h on `mg-task`, plus about 250 GPU-h still running or queued (§20) |

### The driver

On `main`, one command runs stages 1–3:

```bash
python scripts/reproduce.py                  # on the CPU, so it runs on any machine
python scripts/reproduce.py --device cuda    # if you have a CUDA GPU: much faster
```

> **If you have a CUDA GPU, pass `--device cuda`.** The CPU is the default only so
> that the tier runs anywhere. Every runner honours the device it is given.

`--stage 3` runs one stage; `--stage 3 --from run_adamw_pass` resumes the smoke tier
at a step. A failure does not stop the driver: every step runs, and the summary says
which passed, which refused and which failed. The exit code is nonzero if any did not
pass. Everything it records goes to `reproduce/out/<timestamp>/`:
`environment.json`, `pip_freeze.txt`, one log per step under `logs/`, and
`summary.txt`. The sections below give the commands it runs, for running a stage by
hand.

## 15. Stage 1 — the environment

Install as in section 1, then record what you are running on:

```bash
python scripts/check_environment.py
python -c "import sys, torch; print(sys.version, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
pip freeze > environment.txt
```

`requirements.txt` pins the exact runtime versions. Every run also writes its git
SHA, its Python, torch and CUDA versions, and its package versions into
`results/<run>/metadata.json`. To compare against a recorded run, start from that
file.

MNIST is downloaded once, by `check_data.py`. Mackey–Glass needs no download: each run
integrates its series from its seed.

## 16. Stage 2 — the tests

```bash
python tasks.py test                          # main: 1 505 tests
cd ../Distributed-Online-Learning-mg
PYTHONPATH=src python -m pytest               # mg-task: 1 667 tests
```

`python tasks.py test-fast` skips the tests marked slow. The suite includes the
identities the results rest on:

- ATC reproduces centralised SGD on a complete graph (X0's identity);
- the diffusion filter's complete-graph identity holds;
- the codec's encoder reproduces its counted bits exactly.

There is no CI: the repository has no workflow, and the suite runs only when someone
runs it (design note Q4 is still open). Wherever it runs, a green suite shows that the
code runs and its identities hold on that machine, not that a published cell
reproduces on CUDA.

## 17. Stage 3 — smoke

The runners below have a `--smoke` mode: a short horizon at one seed, on the CPU,
through every code path the real pass takes. Its numbers mean nothing. Where a
runner has a `--lr` pass, smoke that pass first, exactly as for a real run. Pass
`--fresh` as well when repeating a smoke, or its cached cells make it a no-op.

A smoke writes `_smoke` runs, and a runner that reads another experiment's cells
reads that experiment's `_smoke` runs under `--smoke`. So the producers smoke first,
and the tier runs on a fresh clone with no real cells on disk (D141). The AdamW pass
and P5.12 then rebuild cells with the same horizon, cadence and seed as their smoke
sources, so their merge gates must hold in the smoke too, and the driver fails a
smoke whose gate does not.

On `main`, in this order:

- **The producers:** `run_diffusion_ekf.py --lr --smoke` (X19; its main pass has no
  smoke, since it reads X13's cells); `run_diffusion_tuning.py --tune --smoke`, then
  `--smoke` (X20); `run_diffusion_skew.py` and `run_atc_plain.py` (X25, X26),
  `--lr --smoke` then `--smoke`; `run_skew_topup.py --smoke` (X25+, at seed 3, its
  first top-up seed); `run_diffusion_topology.py --lr --smoke`, then `--smoke` (P5.3).
- **`--lr --smoke`, then `--smoke`:** `run_network_size.py`,
  `run_p57_heterogeneous_drift.py`, `run_belief_calibration.py`, `run_break_rate.py`,
  `run_label_shift.py`, `run_label_sparsity.py`, `run_midtransient_shift.py`,
  `run_adamw_pass.py`, `run_inner_steps.py`.
- **`--smoke` alone:** `run_disagreement.py`, `run_float32_probe.py`,
  `run_delta_entropy_probe.py`, `run_covariance_blocks.py`.
- **`run_codec.py`:** `--calibrate --smoke`, then `--validate --smoke`, then
  `--report-runs --smoke`.

On `mg-task`:

- **`--lr --smoke`, then `--smoke`:** `run_m9_topology.py`,
  `run_m10_network_size.py`, `run_m13_data_rate.py`, `run_m14_break_rate.py`,
  `run_m2o_readout.py`, `run_inner_steps.py`.
- **`--smoke` alone:** `run_m6_comparison.py` (about 30 minutes),
  `run_m7_combine_exponent.py`, `run_m8_tau_heterogeneity.py`,
  `run_m11_sensor_drift.py`, `run_m12_combined_drift.py`,
  `run_m12b_abrupt_coupling.py`, `run_mg_belief_calibration.py`,
  `run_covariance_blocks.py`, `run_float32_probe.py`, `run_delta_entropy_probe.py`.
- **`run_m14b_break_rate.py`:** `--pilot --smoke`.
- **`run_codec.py`:** as on `main`.
- **`run_gnn.py`:** `--smoke --reference`, `--smoke --tune-filters`, `--smoke --lr`,
  then `--smoke`; and the same four with `--twin`.

On `mg-task` these still read the real source cells; they join the chain, and the
driver, at the merge (D141).

The other runners (X0–X28 apart from the producers above; on `mg-task`, M2–M5 and
M6's isolated and shift-cycle follow-ups) have no smoke mode. Do not imitate one by
shortening a
horizon. Runs are cached by name, so a short run under a real name would be found
complete by the real pass. Part I's loop proves the code those runners use.

## 18. Stage 4 — calibration

Calibration means the runs whose output other experiments read:

- the offline references;
- the learning rates the phase-4 baselines carry;
- the filter settings every later filter cell carries.

Each experiment's own `--lr` pass is calibration too, but it belongs to that
experiment's row in stage 5.

**How the filter settings travel differs by branch.**

- **`main`.** X13's selection is read from X13's cells on disk by X14–X19
  (`tuned_settings()`), so X13 has to be on disk before any of them run. From X20 on,
  the settings are constants: `FILTER` and `CENTRALIZED` in `run_diffusion_skew.py`,
  imported by every later runner. X20–X23 chose those values. Re-running X20–X23
  therefore checks the constants; it does not feed them.
- **`mg-task`.** The selections are files: `results/m3_rates.json`,
  `results/m4_selection.json` and `results/m5_selection.json`. M3, M4 and M5's
  tie-break write them, and every later runner reads them. Re-running M3–M5
  rewrites them.

On `main`:

| | runner and passes | writes | hours | must reproduce | write-up |
|---|---|---|---|---|---|
| reference classifier | `train_reference.py`, once | `data/reference/`: $e^\star$ per rotation | CPU | $e^\star$ at every rotation the drift visits; experiments read the cache and never rebuild it | D33 |
| phase-4 tuning | `sweep_hyperparameters.py`; `--report` | `results/sweep/cells.jsonl` | CPU | each method's rate, $n$ and optimiser; the selections are carried as constants in the configs and runners | results.md §2 |
| X13, the centralised filter | `run_ekf_sweep.py`, then `--baselines`, then `--full`; read with `report_ekf_sweep.py` with the same flag | `x13_*` | 89.5 | best cell $\gamma=0.9995$, $q=6\times10^{-5}$, $\sigma_0^2=0.01$: 0.0630 ± 0.0006, against re-tuned centralised SGD 0.0918 and ATC 0.0952 | D64, D70–D72 |

On `mg-task`:

| | runner and passes | writes | hours | must reproduce | write-up |
|---|---|---|---|---|---|
| M2, the references | `run_m2_references.py`; `--levels 0.22 --seeds 0 1 2 3 4` for $e^\star$'s error bar | `m2_references.json`, `m2_reference_seeds.json` | 0.2 CPU | $e^\star(0.22)=0.1432\pm0.0013$ over five seeds | D97, D116 |
| M3, the baselines' rates | `run_m3_rates.py` | `m3_rates.json` | 5.2 CPU | one rate per learner per condition, stable across conditions | D98 |
| M4, the centralised filter | `run_m4_centralised.py` | `m4_selection.json` | 5.6 CPU | $\gamma=1$, $q=6\times10^{-6}$, $\sigma_0^2=0.01$, $\boldsymbol R\times1$, settled RMSE 0.1429 | D102 |
| M5, the diffusion filter | `run_m5_diffusion.py`, then `--tie-break` (only the tie-break writes the selection) | `m5_selection.json` | 5.1 | $\gamma=1$, $q=6\times10^{-7}$, $\sigma_0^2=0.01$, settled RMSE 0.1458 | D105 |
| M5's linear probe | `run_m5_linear_probe.py` | nothing: it informs a decision | 0.8 | on `m_linear`, $q=6\times10^{-7}$ against $6\times10^{-6}$: one-hop tied (+0.0008), local adapt prefers the smaller by 0.0088; the selection stands | D105 |

## 19. Stage 5 — the experiments

Every MNIST experiment also needs the reference classifier from stage 4. The tables
give each runner's passes in order. A *bare* pass is the command with no flag.

### 19.1 `main`: the gradient baselines (phase 4, CPU)

| | runner and passes | needs | hours (CPU) | must reproduce | write-up |
|---|---|---|---|---|---|
| X0 | `run_experiment.py x0_exactness` | — | minutes | ATC equals centralised SGD on a complete graph, to 1.7e-15 | results.md §1 |
| X1 | `run_experiment.py x1_stationary` | — | 0.3 | on a ring, ATC is 0.0014 behind centralised SGD, against a seed s.d. of 0.0035 | §3.1 |
| X1b | `run_experiment.py x1b_atc_vs_cta` | — | 0.3 | ATC 0.0768 against CTA 0.0774: indistinguishable | §3.2 |
| X2 | `run_experiment.py x2_rotating` | — | 0.3 | drift cost +0.016 for centralised and ATC, +0.022 payload-matched, +0.032 local | §3.3 |
| X3 | `run_topology_sweep.py` | — | 12.5 | worst settled penalty 0.008 (star); the graph-blind learners identical across topologies | §7 |
| X4, X6 | `run_sparsity_sweep.py` (`x4` or `x6` for one) | — | 12.0, 8.7 | cooperation worth 0.047–0.527; under $\beta_{\mathrm{dir}}=0.1$ local only 0.630 against ATC 0.102 | §9, §10 |
| X5 | `run_experiment.py x5_abrupt_shift` | — | 2.1 | one 15° shift costs every method 0.073–0.081 | §3.4 |
| X7 | `run_experiment.py x7_sinusoidal` | — | 0.3 | no cooperative method forgets measurably (1.1–1.6σ) | §11 |
| X8 | `run_experiment.py x8_global`, then `x8_per_node_drift`; `report_cooperation.py` | — | 3.2 | ⚠ confounded: its control drifted more than the treatment's mean (D53). P5.7 is the corrected measurement | §14.1 |
| X9 | `run_experiment.py x9_rate_ramp`, then `x9_control`; `report_breaks.py` | — | about 4 | the gradient methods break at 0.038–0.044 °/step (ring) | §12 |
| X10 | `run_experiment.py x10_prior_drift`, then `x10_control`; `report_cooperation.py` | — | 1.2 | under label shift the cooperation gap widens by +0.0445 | §14.2 |
| X11 | `run_recurring_sweep.py`; `run_experiment.py x11_control`; `report_recurring.py` | — | 65.3 | longer intervals between shifts produce bigger wounds | §13 |
| X12 | `run_linear_sweep.py`; `report_abrupt_vs_smooth.py` | — (its report reads X11) | 14.3 | abrupt drift is mostly cheaper than smooth drift at the same average speed | §13.2 |

### 19.2 `main`: the centralised filter (X14–X18)

| | runner and passes | needs | GPU-h | must reproduce | write-up |
|---|---|---|---|---|---|
| X14 | `run_ekf_generalization.py --lr`, then bare (`--only <condition>` runs it in instalments); `report_ekf_generalization.py` | X13 | 69.8 | one setting, tuned at linear 0.025 °/step, beats a tuned ATC in every drift condition | D67, D73, D74 |
| X15 | `run_ekf_retune.py` | X13, X14 | 4.6 | $\gamma<1$ still wins with both families re-tuned at `every2_jump5`: multiplicative forgetting is rejected on mechanism | D76 |
| X16 | `run_ekf_ramp.py` | X9, X13 | about 1 | on a shared noise bar the filter breaks at 0.0639 °/step against 0.0381–0.0399; at 0.10 °/step its damage is 0.021 against ATC's 0.036 | D123 (reproduced there) |
| X17 | `run_ekf_skew.py --lr`, then bare | X13 | 30.3 | label skew barely reaches the pooled filter | D77 |
| X18 | `run_sawtooth.py --lr`, then bare (ten seeds; raising a cached cell's seed count needs `--fresh`) | X13 | 9.4 | `atc_plain` does not gain on momentum ATC as the reset period shortens: the mechanism is dead | D78 |

### 19.3 `main`: the diffusion filter (X19–X28, P5.3)

X20–X23 select the setting every later filter cell carries (§18).

| | runner and passes | needs | GPU-h | must reproduce | write-up |
|---|---|---|---|---|---|
| X19 | `run_diffusion_ekf.py --lr`, then bare | X13 | 14.4 | full sharing buys +0.0002 to +0.0007 over mean-only, for 2 909× the bandwidth | D79, D80 |
| X20 | `run_diffusion_tuning.py --tune`, then bare | X19's `--lr` | 24.5 | re-tuning moves every cell: −0.0066, −0.0122, −0.0178 from stationary to abrupt; the filter beats tuned ATC at half its bandwidth | D81 |
| X21 | `run_diffusion_gamma.py` | — | 3.1 | the $q=6\times10^{-3}$ column is destroyed at every $\gamma$ (error 0.43–0.89) | D82 |
| X22 | `run_diffusion_extrapolation.py` | — | 3.2 | $\alpha>0$ is rejected monotonically on both adapt scopes | D87 |
| X23 | `run_diffusion_joint.py`, then `--tie-break` | — | 11.1 | a flat plateau; the carried setting $\gamma=0.9995$, $q=6\times10^{-4}$, $\sigma_0^2=10^{-3}$ | D82, D83 |
| X24 | `run_diffusion_combine.py` | — | 6.3 | $\beta=1$ beats $\beta=2$ everywhere: the conservative bound is approximately tight | D88 |
| X25 | `run_diffusion_skew.py --lr`, then bare | — | 11.7 | under severe skew, full sharing is worth −0.0314 ($t=-4.61$) to the local-adapt filter | D89 |
| X26 | `run_atc_plain.py --lr`, then bare | X20, X25 | 2.0 | `atc_plain` selects $\eta=0.05$ in all eight conditions; mean-only local adapt still beats it at equal bandwidth by 0.012–0.025 | D90, D91 |
| X27 | `run_linearization_point.py` | X25 | 10.5 | the receiver point is never worse (−0.0012 to −0.0091); the sender arm reproduces X25 exactly | D99 |
| X25+ | `run_skew_topup.py` | X25, X26, X27 | 3.9 | the sender arm reproduces X27 to 0.0e+00; at five seeds, local adapt's loss to `atc_plain` under severe skew is +0.0068 ($p=0.07$), not established | D101 |
| X28 | `run_receiver_full.py` | X27 | 2.0 | at the receiver point full sharing is null at both ends of the skew axis | D100 |
| P5.3 | `run_diffusion_topology.py --lr`, then bare | — | 17.3 | the sharing gap does not widen as connectivity falls; one-hop − local is −0.0028 at ER 0.3 and +0.0031 (ns) on the complete graph; graph-blind spread 0.0e+00 | D103 |

### 19.4 `main`: phase 5, network size and the AdamW pass

From here on, every runner has `--report-only` and a pre-registered confirmatory
family (D118).

| | runner and passes | needs | GPU-h | must reproduce | write-up |
|---|---|---|---|---|---|
| P5.7 | `run_p57_heterogeneous_drift.py --lr`, then bare | — | 6.4 | heterogeneous drift costs every shared model the same; diffusion does not personalise | D115 |
| N>10 | `run_network_size.py --probe-only` (GPU otherwise idle), `--lr`, then bare; `--full-sharing` for $N=20,30$ | — | 11.7 | from $N=10$ to 30, local adapt − centralised filter grows by +0.0069 and +0.0134 in the two conditions; one-hop's growth is not detected | D114, D117, D119 |
| AdamW backfill | `run_adamw_pass.py --lr`, then bare | X20, X25, X25+, P5.3, P5.7 | 10.7 | all twelve merge gates reproduce to 0.0e+00; one-hop beats ATC AdamW in nine of twelve conditions | D127 |
| P5.11 / P5.14 | `run_belief_calibration.py --lr`, then bare | — | 14.7 | merge gate 0.0e+00; scoring the belief is worse than the plug-in prediction for every filter (Q1 refuted) | D120 |
| P5.23 | `run_midtransient_shift.py --lr`, then bare | — | 7.2 | a mid-transient second shift costs more for all three filters: +4.07, +5.31 and +3.18 error-steps ($p_\text{holm}\le0.001$); nothing diverges | D125 |
| P5.5 | `run_break_rate.py --lr`, then bare | — | 8.1 | one-hop tolerates fast drift as well as the centralised filter; Adam pays a diffusion cost SGD does not | D123 |
| P5.8 | `run_label_shift.py --lr`, then bare | — | 3.8 | diffusion does not close its gap to the centralised filter (Q1); cooperation pays more under shift (Q2) | D124 |
| P5.4 | `run_label_sparsity.py --lr`, then bare | — | 27.4 | merge gate 0.0e+00 in all nine cells; one-hop's lead holds at $n=1,4$ and fails at $n=2$ (+0.009, $p_\text{holm}$ 0.044) | D122 |
| P5.12 | `run_disagreement.py` | P5.3, P5.7 | 8.7 | the gate reproduces every source cell; the agents converge to each other, not to the centralised filter (Q1, all six rows) | D134 |

### 19.5 `main`: communication and the Track D extensions

| | runner and passes | needs | GPU-h | must reproduce | write-up |
|---|---|---|---|---|---|
| float32 probe | `run_float32_probe.py` (lockstep, GPU otherwise idle), then `--cells` | P5.3 | 1.7 | single precision is numerically safe for every filter, and about halves the GPU time | D133 |
| delta probe | `run_delta_entropy_probe.py` | P5.3, X20 | 0.8 | the filter's differences are not cheaper to code than the baselines' (H1 refuted); no rate rise after a jump (H2) | D136 |
| codec | `run_codec.py --calibrate`, `--validate`, then `--report-runs` | P5.3, X20, X25+ | 91.6 | local adapt at 4.3 bits per parameter against 64, within ±0.0002 error of its uncompressed twin | D137, D138 |
| D1, $K$ inner steps | `run_inner_steps.py --lr`, then bare; bare again for the $K=8$ `central` re-run (D1c) | P5.4 ($K=1$, stationary) | 45.4 + ~5 | extra passes help centralised AdamW (abrupt −0.0376) and widen ATC AdamW's gap under skew (+0.0325); every filter is worse at $K>1$ | D139, D142 |
| D2, dither | `run_codec.py --dither --calibrate`, `--validate`, `--report-runs` | codec | 36.2 | no change for D-BOLT at C\* = 0.01 (local adapt 4.33 bits); ATC AdamW rescued, +0.075 → −0.003 at c = 0.01 | D139, D144 |
| D3a, block covariance | `run_covariance_blocks.py` | P5.4, D1 | 3.7 | neuron blocks: +0.005 to +0.010 error for a covariance 15× smaller; layer blocks 1.1× smaller | D139, D146 |

### 19.6 `mg-task`: the main comparison and what is built on it

Every row here needs M3–M5 (§18).

| | runner and passes | needs | GPU-h | must reproduce | write-up |
|---|---|---|---|---|---|
| M6 | `run_m6_comparison.py` | M3–M5 | 11.9 | stationary settled RMSE: centralised filter 0.1391, one-hop 0.1408. Against $e^\star=0.1432\pm0.0013$ the filters match the offline reference rather than beat it | D106, D116 |
| M6, isolated | `run_m6_isolated.py` | M6 | 1.1 | the filter's own cooperation gain, +0.0038 / +0.0031 / +0.0028, and the whole deployable design +0.0073 / +0.0083 / +0.0085, exactly additive with the adapt-scope gap | D106 (amended) |
| M6, shift cycles | `run_m6_shift_cycles.py` (`--eval-every 1` for every step) | M3–M5 | 2.3 | the shift transient, phase by phase | D107 |
| M7 | `run_m7_combine_exponent.py` | M6 | 3.7 | $\beta_c=2$ loses: +0.0864 to +0.0951 for local full sharing, +0.0534 to +0.0606 for one-hop full sharing | D132 |
| M8 | `run_m8_tau_heterogeneity.py` | M6 | 3.1 | per-agent delays cost the filters (+0.0045 centralised, +0.0040 one-hop, +0.0023 local) and the gradient methods nothing | D111 |
| M9 | `run_m9_topology.py --lr`, then bare | M6 | 13.8 | one-hop's leads hold on every graph; its gap to the centralised filter closes as the graph densifies | D128 |
| M10 | `run_m10_network_size.py --lr`, then bare; then `--lr --abrupt` and `--abrupt` | M6 | 49.8 | local adapt falls behind as $N$ grows and one-hop does not, stationary and abrupt | D129 |
| M11 | `run_m11_sensor_drift.py` | M6 | 6.2 | drift depends on what changed, not by how much: bias < gain < $\beta$ | D108 |
| M12 | `run_m12_combined_drift.py` | M6, M11 | 18.5 | drifts interact only when they contend for the same observable: $\beta$+gain anti − correlated is −0.0037 to −0.0146 | D109 |
| M12b | `run_m12b_abrupt_coupling.py`, then `--topup` (seeds 5–9) | M12 | 19.2 | M12's ordering holds on the abrupt schedule; nothing clears the pre-registered level at ten seeds | D121 |
| M13 | `run_m13_data_rate.py --lr`, then bare | M6 | 25.4 | one-hop keeps pace with the centralised filter; local adapt is the filter that pays (Q1 −0.0051, $p_\text{holm}$ 0.004, at $n_b=1$) | D131 |
| M14 | `run_m14_break_rate.py --lr`, then bare | M6 | 10.9 | nothing detected at the matched rate: the ramp is too gentle | D130 |
| M14b | `run_m14b_break_rate.py --pilot` | M14 | 9.9 | no break at any speed; closed as a null | D140 |
| M2O | `run_m2o_readout.py --lr`, then bare, then `--extend-sgd` | M3–M5 | 15.4 | no difference detected at full context; the filter's lead shrinks but holds under many-to-one | D126 |
| P5.11 / P5.14 | `run_mg_belief_calibration.py` | M3–M5 | 9.5 | `lem:conservative` is tight without the covariance: shrink 0.985–0.995 | D120 |

### 19.7 `mg-task`: communication and the Track D extensions

| | runner and passes | needs | GPU-h | must reproduce | write-up |
|---|---|---|---|---|---|
| float32 probe | `run_float32_probe.py`, then `--cells` | M6 | 2.2 | settled RMSE within 3×10⁻⁵ of float64 | D133 |
| delta probe | `run_delta_entropy_probe.py` | M6 | 0.9 | as on MNIST | D136 |
| codec | `run_codec.py --calibrate`, `--validate`, then `--report-runs` | M6 | about 65 | local adapt at 1.5 bits per parameter, within +0.0013 RMSE of its twin; every baseline needs a step ten times finer | D138 |
| D1, $K$ inner steps | `run_inner_steps.py --lr`, then bare | M6 ($K=1$) | 167.0 | centralised SGD gains −0.016 to −0.019 from $K=8$, ATC's gap grows +0.010 to +0.019; every filter worse at $K>1$ | D139, D143 |
| D2, dither | `run_codec.py --dither …`, as on MNIST | codec | 49.3 | every baseline from C\* = 0.001 to 0.01; D-BOLT's plain 1.50 bits stays its operating point | D139, D145 |
| D3a, block covariance | `run_covariance_blocks.py` | M6, D1 | 6.9 | neuron blocks (106× smaller): +0.011 to +0.036 RMSE, local adapt most; layer blocks (3.9×): +0.004 to +0.009 | D139, D147 |
| D3c, the GNN | `run_coupling_smoke.py` (CPU, chooses $\kappa$); `run_gnn.py --twin --reference`, `--twin --tune-filters`, `--twin`; then `--reference`, `--tune-filters`, `--lr`, bare | — | ~65 (est.) | ⏳ queued; $\kappa=0.6$ was chosen | D139 |

## 20. Still running

Queue 5 (D1, D2, D3a) ended on 2026-10-30 after 309 GPU-h; its rows above carry
their recorded hours and get their headlines as each is analysed. Queue 6 (the GNN)
started after it; its estimate comes from CPU step timings and is ±50%. D1c, the
$K=8$ re-run on MNIST (D142), follows queue 6.

## 21. Stage 6 — the reports

Each runner turns its cells back into the numbers its write-up quotes:

- **From X23 on:** `--report-only`. It reads the cells from disk and runs no
  learner, so it costs seconds to minutes on a CPU.
- **X9–X14:** `report_breaks.py`, `report_recurring.py`,
  `report_abrupt_vs_smooth.py`, `report_cooperation.py`, `report_ekf_sweep.py` and
  `report_ekf_generalization.py`.
- **Before X23:** the main pass prints its tables as it ends. Re-running it on a
  complete set of cells runs nothing and prints them again.

A report checks its own preconditions before it prints a confirmatory table:

- the pairing gates (`assert_paired_runs`);
- the merge gates against a source cell (D119);
- the graph-blind gates across topologies.

A gate that fails withholds the tables it guards, so read the gate lines first.

## 22. Stage 7 — the figures

*To come.* The paper's figures will be rebuilt by a public `scripts/figures/`,
holding only the paper's figures and reading only `results/`. It is built once the
paper's figure set is final. `docs/figures.md` describes every figure drawn so far
and the data behind it.
