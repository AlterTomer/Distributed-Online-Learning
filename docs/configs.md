# Configuration reference

Every quantity that varies across an experiment is a config field. If it is not
in a config it cannot be swept, and it will quietly become hard-coded.

This document covers what each file in `configs/` is for, every parameter it can
hold, the legal values, and which module consumes it. The schema itself lives in
`src/dekf_bench/utils/config.py`; if the two ever disagree, the code is right and
this file is stale.

To see a resolved config without reading YAML, run `scripts/check_config.py`.

---

## 1. How composition works

A config is assembled in four layers, each overriding the one before:

```
configs/base.yaml                 defaults for every field
    ↓
include: {env, graph, model}      one named file per section
    ↓
the experiment file's own keys    e.g. run:, env:, graph:
    ↓
overrides= passed to load_config  sweeps, notebooks, one-off edits
```

Mappings merge key by key, so an experiment that sets `env.drift.total_degrees`
keeps every other `env.drift` field from the layers below. **Lists replace
wholesale** — otherwise it would be impossible to *shorten* a list in an
override.

`learners` is not an `include:` section. It is a top-level list, because several
learners share one environment per run (`WORKPLAN.md` §6.1). Each entry is either
a name, resolved against `configs/learner/<name>.yaml`, or a mapping with `name:`
plus overrides:

```yaml
learners:
  - centralized_sgd                  # the file as-is
  - name: diffusion_sgd_atc          # the file, with two fields changed
    optimizer: sgd
    momentum: 0.0
```

**Unknown keys are errors**, with a typo suggestion. This is deliberate: a
misspelled `label_availabilty` silently falling back to a default is how a run
produces a plausible curve that answers a different question than the one asked.

---

## 2. `base.yaml` — the defaults every config inherits

The values here are the phase-1 decisions from `WORKPLAN.md` §10. Nothing else
needs to restate them.

### `run`

| Field | Type | Default | Legal values | Notes |
|---|---|---|---|---|
| `name` | str | `unnamed` | any | Names the output directory and the `experiment` column in the log |
| `seeds` | list[int] | `[0,1,2,3,4]` | non-negative, no duplicates | Each is a *master* seed; the four streams are derived from it (§7) |
| `horizon` | int | `1500` | ≥ 1 | $T$. Bounded by the shard budget (§6) |
| `eval_every` | int | `25` | ≥ 1 | $K$. Full evaluation cadence; prequential runs every step regardless |
| `dtype` | str | `float32` | `float32`, `float64` | `float64` is required for X0, where the identity must hold to $10^{-12}$ |
| `device` | str | `cpu` | `cpu`, `cuda`, `auto` | `auto` resolves at runtime. Phases 1–4 are CPU-sized |
| `out_dir` | str | `results` | any path | Gitignored |

*Consumed by:* `runner/simulate.py` (phase 3), `recording/` (phase 3),
`runner/seeding.py` (now).

### `graph`

| Field | Type | Default | Legal values | Notes |
|---|---|---|---|---|
| `topology` | str | `ring` | `complete`, `ring`, `path`, `grid2d`, `star`, `erdos_renyi`, `watts_strogatz`, `disconnected` | See §3 |
| `n_nodes` | int | `10` | ≥ 1 | $N$. Raising it *shortens* the feasible horizon (§6) |
| `weights` | str | `metropolis` | `metropolis`, `relative_degree`, `uniform` | Combination rule $a_{vu}$ |
| `params` | mapping | `{}` | topology-specific | Only `grid2d` is validated today |

`weights` choices:

- **`metropolis`** (default), $a_{vu} = 1/(1+\max(d_v,d_u))$ on an edge. Symmetric and **doubly stochastic**, and $a_{vv}\ge 1/(1+d_v)>0$ so an agent never discards its own estimate. Doubly stochastic means the network average is invariant under combining — information is redistributed, never created.
- **`relative_degree`**, $a_{vu} = d_u/\sum_{j\in\mathcal N_v\cup\{v\}}d_j$. Row-stochastic but **not** symmetric: an agent leans toward better-connected neighbours. An isolated agent keeps all its own weight.
- **`uniform`**, $a_{vu} = 1/(1+d_v)$ over the closed neighbourhood. On a complete graph $d_v = N-1$, so this is exactly the $1/N$ the X0 identity requires; defining it over the neighbourhood rather than as a flat $1/N$ means it also respects a sparser adjacency instead of weighting agents that sent nothing.

On a **regular** graph — a ring, a complete graph — all three coincide.

**Two mixing measures, and the difference matters.** `spectral_gap` is the `WORKPLAN.md` §4.1 definition and is **only valid for doubly stochastic weights**; `env/graph.py` raises rather than returning the negative value it would otherwise produce for `relative_degree` or `uniform` on an irregular graph. `mixing_gap` $= 1-\mathrm{SLEM}$ is always valid and agrees with the spectral gap wherever both are defined. `summary()` reports `spectral_gap: None` in the undefined case and always populates `mixing_gap`.

*Consumed by:* `env/graph.py` (now), every diffusion learner (phase 3).

### `env`

| Field | Type | Default | Legal values | Notes |
|---|---|---|---|---|
| `samples_per_node_per_step` | int | `2` | ≥ 1 | $n$. The "not much data" knob |
| `label_availability` | float | `1.0` | $[0,1]$ | $\pi_{\text{lab}}$. Below 1, some agents idle on some steps but still gain from the combine step |
| `partition.kind` | str | `iid` | `iid`, `dirichlet` | Shard assignment |
| `partition.beta` | float | `1.0` | > 0 | Dirichlet concentration; ignored when `kind: iid`. Small $\beta$ = severe label skew |
| `drift_scope` | str | `global` | `global`, `per_node` | `global` keeps a single shared $\theta$ the correct object |
| `allow_epochs` | bool | `false` | | `true` lets samples repeat and **forfeits the exactly-once guarantee**; it also disables the shard-budget check |

**`partition.kind`** decides *which* images land in which shard. Under **`iid`**
the split is uniform, so every agent sees roughly the same class distribution;
this is deliberate for X1–X3, because it makes drift the only source of
non-stationarity and keeps "what does decentralization cost" (Q1) from being
confounded with "what does heterogeneity cost". Under **`dirichlet`**, each agent
draws a class preference $\bm q_v \sim \text{Dir}(\beta\mathbf 1_K)$ and is filled
according to it: $\beta = 0.1$ gives agents that each see only three or four
digits, and $\beta \to \infty$ approaches IID. That is the sharpest form of Q2 —
an agent that never sees a 7 can only learn 7s through the combine step — and it
is what X6 sweeps.

Measured on MNIST at $N=10$, over the shipped `beta` values:

| `beta` | Skew (mean TV from global) | Classes per agent | Shard sizes |
|---|---|---|---|
| `iid` | 0.017 | 10 | 6000 |
| 100 | 0.060 | 10 | 6000 |
| 1.0 | 0.365 | 7–10 | 6000 |
| 0.1 | 0.651 | 3–7 | 6000 |

**Shard sizes stay equal under skew.** Only the *composition* varies. The
classical construction — a Dirichlet draw over agents, per class — makes sizes
vary by 7× or more, which would both confound X6 with data volume and break the
shard budget check, since that check assumes equal shards. At $\beta = 0.1$ every
seed tested left some agent with fewer samples than a default run consumes.
`env/partition.py` takes `balance_sizes=False` if you want the classical
behaviour for comparison, and `min_shard_size=` to reject a partition that
cannot feed the run.

**`allow_epochs`** decides whether a sample may be consumed twice. It is not
merely bookkeeping. The research note's Assumption 5 requires every labelled
observation to enter the network likelihood **exactly once**: reusing a sample
double-counts its information, so the filter's covariance shrinks faster than
the evidence justifies and the posterior becomes overconfident — a failure that
surfaces in the phase-5 diagnostics as innovation whiteness breaking down. It
also keeps the online claim honest, since "no replay buffer" is one of the
method's stated properties. Setting it `true` waives both the guarantee and the
shard-budget check, and is a real trade rather than a convenience.

*Consumed by:* `env/stream.py`, `env/partition.py`, `env/environment.py` (now).

### `env.drift`

| Field | Type | Default | Legal values | Applies to |
|---|---|---|---|---|
| `schedule` | str | `stationary` | `stationary`, `linear`, `piecewise`, `sinusoidal` | all |
| `total_degrees` | float | `45.0` | $[0, 45]$ | `linear` |
| `change_points` | list[int] | `[]` | non-negative, sorted | `piecewise` |
| `jump_degrees` | float | `15.0` | any | `piecewise` |
| `amplitude_degrees` | float | `30.0` | $\lvert\cdot\rvert \le 45$ | `sinusoidal` |
| `period` | int | `500` | ≥ 1 | `sinusoidal` |
| `per_node_spread` | float | `0.5` | $[0, 1)$ | all, under `drift_scope: per_node` |

Under **`drift_scope: per_node`**, agent $v$'s rotation is scaled by a multiplier
evenly spaced over $[1-\text{spread},\,1]$. The multipliers top out at **1, not
$1+\text{spread}$**: scaling any agent above the configured rate would carry it
past the 45° cap the schedule was validated against, so the spread slows the
laggards rather than accelerating the leaders. `spread: 0` reduces exactly to
global drift.

Note that per-node drift has **no single network-wide drift state**, so
`Drift.state_at(t)` raises without a node argument — returning agent 0's state
would be a silent lie. Use `states_at(t)` for all of them.

**There is no `alpha` field, and supplying one is an error.** The per-step rate
is derived as `total_degrees / run.horizon`, so changing the horizon cannot
silently change how far the distribution travels. At the defaults that is
$45/1500 = 0.03°$/step.

The 45° ceiling is a validity constraint, not a preference. Past roughly that
much rotation MNIST accuracy degrades for reasons unrelated to decentralization,
and near 180° a 6 *is* a 9 — the reference error $e^\star$ would then be
measuring label ambiguity and the headline gap would be uninterpretable.

*Consumed by:* `env/drift.py` and `data/transforms.py` (now);
`evaluation/evalsets.py` (phase 2). The rotation at a step comes from
`build_drift(config).rotation_at(t)` -- the schedule lives in `env/drift.py`,
not on the config object (design note D19).

### `model`

| Field | Type | Default | Legal values | Notes |
|---|---|---|---|---|
| `name` | str | `mlp_small` | see §4 | Selects the builder in `models/registry.py` |
| `input_size` | int | `14` | ≥ 1 | Image side length **after** downsampling; $14 \Rightarrow 196$ inputs |
| `hidden` | list[int] | `[14]` | each ≥ 1 | Empty list means a linear probe |
| `output_dim` | int | `10` | ≥ 2 | $q$; number of classes |

`config.model.num_params` computes $p$ from these — 2908 at the defaults.

*Consumed by:* `models/` (phase 2).

### `learners`

A list. See §5 for the per-learner fields.

### `eval`

| Field | Type | Default | Legal values | Notes |
|---|---|---|---|---|
| `evalsets` | list[str] | `[prequential, current, canonical]` | `prequential`, `current`, `current_mean`, `backward`, `canonical` | No duplicates |
| `backward_separation_degrees` | float | `15.0` | > 0, and reachable by the schedule | How far back the forgetting probe looks, **in degrees** |
| `batch_size` | int | `1000` | ≥ 1 | Evaluation batching only |

The evaluation sets: **`prequential`** is test-then-train on the incoming batch,
the per-step signal. **`current`** is the held-out test set rotated by the same
amount as the training data at step $t$ — the headline metric, and the one that
is easy to get wrong. **`backward`** measures forgetting. **`canonical`** is
unrotated and stays comparable to published MNIST numbers. **`current_mean`** is
added automatically under `per_node` drift, scoring every agent at the
network-mean rotation so the per-agent spread can be separated from the rotation
spread.

**The backward probe is anchored by rotation, not by a step count.** It
evaluates at the *most recent* earlier step whose rotation differs from the
current one by at least `backward_separation_degrees`. A fixed step offset
degenerates: at an offset equal to the sinusoidal period the separation is
identically zero, so the schedule chosen to expose forgetting could not measure
it. Where no qualifying earlier step exists — early in a run, or a stationary one
— the probe is **undefined and logs nothing**, rather than logging a zero that
would read as "no forgetting".

Config validation rejects a separation the schedule cannot reach, so asking for
`backward` on a stationary run fails at load rather than silently producing
nothing.

## The reference classifier

| Field | Type | Default | Legal values | Notes |
|---|---|---|---|---|
| `epochs` | int | `100` | ≥ 1 | At 20, 9 of 16 levels stopped at the budget |
| `batch_size` | int | `128` | ≥ 1 | |
| `lr` | float | `0.003` | > 0 | AdamW |
| `init_strategy` | str | `shared_seed` | `shared_seed`, `independent_seeds`, `warm_start` | How the per-rotation models relate |
| `selection` | str | `validation` | `validation`, `fixed_budget` | Which epoch is reported |
| `validation_size` | int | `5000` | $(0, 60000)$, or 0 under `fixed_budget` | Held out of **train** |
| `rotation_min_degrees` | float | `-30.0` | | Grid start |
| `rotation_max_degrees` | float | `45.0` | | Grid end |
| `rotation_step_degrees` | float | `5.0` | > 0 | 16 levels at the defaults |
| `seed` | int | `0` | | Separate from `run.seeds` |

$e^\star$ is a **fixed asset**: trained once by `scripts/train_reference.py`,
cached per `(init_strategy, selection)` so variants never overwrite each other,
and never rebuilt by an experiment run. Its own seed means re-running an
experiment cannot silently retrain the thing it is measured against.

The grid covers every rotation the configured schedules visit — linear
$[0°,45°]$, piecewise $[0°,15°]$, sinusoidal $[-30°,+30°]$ — so nothing
extrapolates. Lookup between grid points interpolates linearly; outside the grid
it raises.

Measured at the defaults: $e^\star$ ranges over roughly 4.6–5.5% for the
$196$–$14$–$10$ MLP, and the symmetry $e^\star(-arphi) = e^\star(+arphi)$
holds to within 0.004 — checked rather than assumed.

*Consumed by:* `evaluation/` (phase 2).

---

## 3. `configs/graph/` — topologies

Each file sets `topology` and any `params` it needs. All inherit `n_nodes` from
`base.yaml`, so a graph file alone never fixes the network size.

### 3.1 At a glance

Measured at $N = 10$ with Metropolis weights, seed 0:

| Topology | Edges | Degrees | Diameter | Mixing gap | `params` |
|---|---|---|---|---|---|
| `complete` | 45 | 9–9 | 1 | **1.000** | — |
| `erdos_renyi` | 19 | 2–5 | 3 | 0.257 | `p` |
| `watts_strogatz` | 20 | 3–6 | 3 | 0.197 | `k`, `beta` |
| `ring` | 10 | 2–2 | 5 | 0.127 | — |
| `star` | 9 | 1–9 | 2 | 0.100 | — |
| `grid2d` | 13 | 2–3 | 5 | 0.096 | `rows`, `cols` |
| `path` | 9 | 1–2 | 9 | **0.033** | — |
| `disconnected` | 20 | 4–4 | ∞ | **0.000** | `n_components` |

Higher mixing gap means information spreads faster. Note that diameter and
mixing are *not* the same ordering: a star has diameter 2 but mixes worse than
a ring of diameter 5, because everything must funnel through one hub.

### 3.2 Setting parameters

**From a config.** `params` is a mapping under `graph`, and it merges key by key:

```yaml
# configs/experiment/my_experiment.yaml
include:
  graph: erdos_renyi
graph:
  n_nodes: 20
  params:
    p: 0.5           # overrides the 0.3 in configs/graph/erdos_renyi.yaml
```

**From Python**, for a sweep or a notebook — `overrides` reaches nested keys:

```python
from dekf_bench.utils.config import load_config
from dekf_bench.env.graph import build_graphs

config = load_config(
    "x1_stationary",
    overrides={"include": {"graph": "erdos_renyi"}, "graph": {"params": {"p": 0.5}}},
)
graph = build_graphs(config).comm       # 29 edges at N=10, p=0.5
```

**Bypassing configs entirely**, when you just want a graph object:

```python
import torch
from dekf_bench.env.graph import build_graph

graph = build_graph(
    topology="watts_strogatz",
    n_nodes=10,
    weights="metropolis",
    params={"k": 4, "beta": 0.5},
    generator=torch.Generator().manual_seed(3),
)
print(graph.n_edges, graph.mixing_gap)   # 20  0.2380
```

The `generator` argument is only read by the random topologies. Pass one derived
from the `graph` seed stream (`seeds.torch_generator("graph")`) so a topology can
be redrawn without disturbing the partition or the sample order.

**Sensible defaults for any size.** `default_topology_params(n_nodes)` returns a
mapping from every topology to parameters that work at that $N$ — it picks the
most square grid factorisation and an Erdős–Rényi density above the connectivity
threshold. Used by the sweep runner and by `scripts/check_environment.py`, so
those choices live in one place:

```python
from dekf_bench.env.graph import default_topology_params

default_topology_params(10)["grid2d"]        # {'rows': 2, 'cols': 5}
default_topology_params(16)["grid2d"]        # {'rows': 4, 'cols': 4}
default_topology_params(10)["erdos_renyi"]   # {'p': 0.4605...}
```

### 3.3 The topologies

**`complete`** — every pair adjacent, $\binom N2$ edges. Mixing gap exactly 1:
one combine step reaches full consensus, so the distributed method reproduces
the centralized one. That is what makes it the X0 exactness setting, and the
best-case anchor of the topology sweep. No parameters.

**`ring`** — a cycle, each agent adjacent to $i \pm 1 \bmod N$. Regular
(degree 2), $N$ edges, diameter $\lfloor N/2 \rfloor$. The default for X1/X2: a
realistic middling case where information takes several steps to cross the
network. The mixing gap decays roughly as $1/N^2$, so a large ring is genuinely
hard — doubling $N$ from 10 to 20 cuts the gap by about four. No parameters.
Below three agents it degenerates to a path, deliberately: `networkx` would
otherwise return a **self-loop** for a one-node cycle.

**`path`** — a line, $N-1$ edges, diameter $N-1$. The worst connected case and
the pessimistic end of the sweep: at $N=10$ the mixing gap is 0.033, so a sample
at one endpoint needs nine steps even to reach the other. No parameters.

**`star`** — node 0 is the hub, adjacent to all $N-1$ leaves; every leaf has
degree 1. Diameter 2, but that understates the difficulty: all traffic funnels
through the hub, so the mixing gap (0.100) is *worse* than a ring's. Maximally
irregular, which makes it the case where `relative_degree` and `uniform` weights
stop being doubly stochastic — see §2's note on the two gap measures. No
parameters.

**`grid2d`** — a `rows × cols` lattice, four-neighbour connectivity, no wrap-
around. Node $(r, c)$ has id $r \cdot \text{cols} + c$ (row-major), so a shard
assignment can be read straight off the index.

- `rows`, `cols` — **required**, and `rows * cols` must equal `n_nodes`. This is validated twice, at config load and at build, because a silent mismatch would change $N$ rather than fail.
- Edges: $\text{rows}(\text{cols}-1) + \text{cols}(\text{rows}-1)$. Diameter: $(\text{rows}-1)+(\text{cols}-1)$, the Manhattan span.
- A prime $N$ forces a $1 \times N$ grid, which **is** a path. Worth knowing before reading a sweep in which the two rows look identical.

**`erdos_renyi`** — each of the $\binom N2$ pairs is an edge independently with
probability $p$.

- `p` — required, in $[0,1]$. Below about $\ln(N)/N$ the graph is usually disconnected; at $N=10$ that threshold is 0.23.
- `ensure_connected` — default `true`. Redraws with a new seed until connected, up to 20 attempts, then raises with the threshold quoted in the message. Set `false` if a disconnected draw is the point.
- Reproducible from `generator`; the same seed always gives the same realization. Because a rejected draw increments the seed, `p` and the seed jointly determine the graph.

**`watts_strogatz`** — a small-world graph: start from a ring where each node
joins its `k` nearest neighbours, then rewire each edge with probability `beta`.

- `k` — neighbours in the starting ring, default 4, must be `< n_nodes`. Edge count is exactly $Nk/2$ and rewiring preserves it.
- `beta` — rewiring probability, default 0.2. `beta: 0.0` leaves a plain $k$-regular ring; `beta: 1.0` is essentially random. The interesting middle is where a few long-range shortcuts collapse the diameter at almost no extra cost — at $N=20$, $k=4$: `beta` 0.0 gives diameter 5 and mixing 0.096, while `beta` 0.5 gives diameter 4 and mixing 0.151.
- `ensure_connected` — default `true`, as for Erdős–Rényi.

**`disconnected`** — the **negative control**. Splits the agents into
`n_components` groups (sizes differing by at most one) with **no** edge between
them; each group is internally complete.

- `n_components` — default 2, must be $\le$ `n_nodes`.
- Mixing gap is exactly 0 and `diameter` is `None`, since a disconnected graph has no diameter.
- Each component being complete is deliberate: diffusion works *perfectly* within a component, so any shortfall against the connected case is attributable to the missing cross-component links and to nothing else. With `n_components: n_nodes` every agent is isolated and the combine step becomes the identity — which should reproduce `local_only` exactly, and is a cheap consistency check on the learners.

### 3.4 Adding a topology

`TOPOLOGY_BUILDERS` in `env/graph.py` maps a name to a
`(n_nodes, params, generator) -> nx.Graph` builder. Add an entry, add the name
to `TOPOLOGIES` in `utils/config.py` so configs may reference it, and add it to
`default_topology_params`. The builder returns a plain `networkx` graph;
adjacency conversion, self-loop rejection, weight construction and validation
are handled for you.

---

## 4. `configs/model/` — architectures

| File | Shape | $p$ | Purpose |
|---|---|---|---|
| `mlp_small.yaml` | 196–14–10 | **2908** | **Primary.** Small enough that a dense $p \times p$ covariance is affordable in phase 5, so phases 1–4 run the same architecture the filter will |
| `mlp.yaml` | 784–128–10 | 101 770 | Sanity comparison at full resolution. A dense covariance here would be $10^{10}$ entries |
| `linear_probe.yaml` | 196–10, no hidden | 1970 | $\theta \mapsto$ logits is **linear**, so the EKF becomes an exact KF and complete-graph exactness holds with no linearisation error |

The budget cannot be met by shrinking the hidden layer: at 784 inputs,
$p \le 3000$ forces a hidden width of about 3. The *input* is what comes down,
hence `input_size: 14`. Rotation is applied at 28×28 and downsampling second —
rotating a 14×14 image directly destroys far more information.

`cnn.yaml` is named in the layout but not written; it arrives with its builder.

---

## 5. `configs/learner/` — methods

Shared fields:

| Field | Type | Default | Legal values | Notes |
|---|---|---|---|---|
| `name` | str | — | must match the filename | Selects the class |
| `optimizer` | str | `sgd_momentum` | `sgd`, `sgd_momentum`, `adamw` | `sgd` is required for X0 |
| `lr` | float | `0.05` | > 0 | |
| `momentum` | float | `0.9` | $[0,1)$ | Must be 0 for X0 |
| `mix_optimizer_state` | str | `momentum` | `none`, `momentum`, `all` | Whether the combine step mixes optimizer moments as well as parameters |
| `adapt_scope` | str | `local` | `local`, `one_hop` | Phase 5; `one_hop` raises today |
| `transition` | str | `identity` | `identity`, `scalar` | Phase 5; $\bm F_t$ |
| `gamma` | float | `1.0` | $(0,1]$ | Phase 5; $\bm F_t = \gamma\bm I$ under `scalar` |
| `forgetting` | str | `lambda` | `lambda`, `process_noise` | Phase 5; how $\bm P$ is loosened |
| `lambda_forget` | float | `0.997` | $(0,1]$ | Phase 5; memory $\approx 1/(1-\lambda)$ |
| `process_noise_q` | float | `1e-6` | > 0 | Phase 5; used under `forgetting: process_noise` |
| `prior_scale` | float | `1.0` | > 0 | Phase 5; $P_0 = \text{prior\_scale}\cdot I$ |

### The state model has two independent axes

`transition` acts on the **mean**, `forgetting` on the **covariance**. All four
combinations are legal, so which state model performs better is measured rather
than assumed.

$\gamma$ is commonly called a "forgetting factor" and **is not one**. Propagating
the moments gives $\bm P_{t|t-1} = \gamma^2\bm P_{t-1|t-1} + \bm Q_t$, and
$\gamma^2 \le 1$ *contracts* the covariance — the opposite of forgetting, which
requires loosening the prior so a new sample counts for relatively more. What
$\gamma$ actually does is $\bm m_{t|t-1} = \gamma\bm m_{t-1|t-1}$: $L_2$ weight
decay written in state-space form.

Defaults are `identity` + `lambda`, because a single $\gamma$ ties two things you
would want to tune separately, and because multiplicative inflation is exactly
structure-preserving in the information domain while $(\bm\Omega^{-1}+\bm Q)^{-1}$
is dense. `process_noise` is offered anyway: while $\bm P$ is carried densely —
all of phase 5 at $p = 2908$ — both rules cost the same, and $\bm Q$ buys
anisotropy a scalar cannot express.

**`lambda_forget` should be read as a memory length.** Effective memory is
$\approx 1/(1-\lambda)$ steps, over which the data rotates $\alpha W$ degrees:

| $\lambda$ | memory | drift over it | inflation over $T$ |
|---|---|---|---|
| 0.9999 | 10 000 | 300° | 1.2 |
| **0.997** | **333** | **10°** | **91** |
| 0.994 | 167 | 5° | 8 300 |

A memory longer than the horizon means no forgetting at all, and the tracking
claim could not be tested. Shorter memory tracks better but inflates unexcited
directions harder — the value is a phase-5 pilot item, not a settled constant.

Three validation rules worth knowing before you hit them:

- **An optimizer that carries per-node state cannot have `mix_optimizer_state: none`.** Local momentum drifts apart across agents and the run diverges; that is a known failure mode, not an open question. Use plain `sgd` if you want no mixing.
- **`gamma` under `transition: identity` is rejected**, rather than silently ignored: $\bm F_t = \bm I$ means it does nothing, and a config setting it is asking for behaviour it will not get.
- **`lambda_forget` and `process_noise_q` are both always present**, but only the one `forgetting` names is used. They are two parameterisations of the same effect and are jointly unidentifiable, so the selector makes the choice explicit rather than inferred from which field is non-null.

| File | Role |
|---|---|
| `centralized_sgd.yaml` | Upper reference. Sees the pooled batch $\bigcup_v \mathcal D_t^v$; `combine` is a no-op, so `mix_optimizer_state` is inert |
| `diffusion_sgd_atc.yaml` | **Primary.** Adapt-then-combine. Diff-EKF is also ATC, so phase 5 differs in the adapt step alone |
| `diffusion_sgd_cta.yaml` | Combine-then-adapt, eq. (17) of Olshevskyi et al. Measured in X1b so the ATC choice is reported rather than assumed |
| `local_only.yaml` | Lower reference. No communication; the gap to ATC *is* the value of cooperation |
| `diffusion_ekf.yaml` | Phase-5 placeholder. Interface only; raises today |

---

## 6. `configs/env/` — data regimes

Each file sets only the `drift` block; everything else comes from `base.yaml`.

| File | Schedule | What it answers |
|---|---|---|
| `mnist_stationary.yaml` | `stationary` | Baseline. Do the methods learn at all, and what is the gap? |
| `mnist_rotating_linear.yaml` | `linear`, 45° total | Does the gap widen under drift; does local-only collapse? |
| `mnist_rotating_piecewise.yaml` | `piecewise`, 15° at $t=500$ | Adaptation transient — the cleanest tracking test |
| `mnist_rotating_sinusoidal.yaml` | `sinusoidal`, ±30°, period 500 | Forgetting, since the distribution returns to states already seen |

### The shard budget

Agents draw from **disjoint** shards, so a run needs $N \cdot n \cdot T$ samples
and MNIST supplies 60 000. This is checked at load, not discovered at step 1400:

| $N$ | $n$ | Max $T$ |
|---|---|---|
| 10 | 2 | 3000 |
| 10 | 4 | 1500 |
| 10 | 8 | 750 |
| 100 | 2 | 300 |

The default $N{=}10$, $n{=}2$, $T{=}1500$ uses half the training set. Note that
**raising $N$ shortens the feasible run** — which is why $N \in [50,100]$ is
still an open question rather than a default. `allow_epochs: true` waives the
check at the cost of the exactly-once guarantee.

---

## 7. `configs/experiment/` — the runs

| File | Setup | Question |
|---|---|---|
| `x0_exactness.yaml` | complete graph, uniform weights, plain SGD, float64, $T{=}50$ | Does ATC diffusion reproduce centralized SGD exactly? |
| `x1_stationary.yaml` | ring, $N{=}10$, $n{=}2$, no drift | Do all methods learn, and what is the gap? |
| `x1b_atc_vs_cta.yaml` | ring, stationary, both orderings | Which diffusion ordering, and how much does it matter? |
| `x2_rotating.yaml` | as X1, linear drift to 45° | Does the gap widen; does local-only collapse? |
| `x5_abrupt_shift.yaml` | piecewise, 15° jump at $t{=}500$ | How fast does each method recover? |

X3 (topology sweep), X4 (sparsity) and X6 (non-IID) are sweeps and arrive with
`runner/sweep.py` in phase 4.

**`x0_exactness.yaml` is the one to read carefully.** Every field in it is a
precondition of the algebraic identity, not a preference:

```yaml
graph:   {topology: complete, weights: uniform}   # a_vu = 1/N exactly
run:     {dtype: float64}                         # agreement to 1e-12
env:     {label_availability: 1.0}                # equal batch sizes per agent
learners:
  - {name: centralized_sgd,   optimizer: sgd, momentum: 0.0}
  - {name: diffusion_sgd_atc, optimizer: sgd, momentum: 0.0,
     mix_optimizer_state: none}
```

Equal batch sizes matter because the average of per-agent means equals the
pooled mean only when every $\lvert\mathcal D_t^v\rvert$ is the same. Relax
`label_availability` and the test fails with a small, plausible, non-zero
residual rather than an obvious error — which is exactly the failure mode it
exists to catch. `tests/test_config.py` asserts all four preconditions stay
pinned.

---

## 8. Seeds

Seeds are not in the YAML beyond `run.seeds`. Each master seed derives four
**independent** streams by keyed hash of the stream name:

| Stream | Controls |
|---|---|
| `init` | Model initialization, shared across all agents |
| `partition` | Shard assignment |
| `stream` | Sample order within a shard, label-availability draws |
| `graph` | Random graph realization |

Separation is what makes an ablation interpretable — hold `partition` fixed,
vary `init`, and any difference is attributable to initialization. Derivation is
by name rather than by draw order, so adding a fifth stream later will not shift
the four that already exist and invalidate recorded results.

---

## 9. Quick reference: every enumerated value

```
run.dtype                 float32 | float64
run.device                cpu | cuda | auto
graph.topology            complete | ring | path | grid2d | star |
                          erdos_renyi | watts_strogatz | disconnected
graph.weights             metropolis | relative_degree | uniform
env.partition.kind        iid | dirichlet
env.drift.schedule        stationary | linear | piecewise | sinusoidal
env.drift_scope           global | per_node
learner.optimizer         sgd | sgd_momentum | adamw
learner.mix_optimizer_state   none | momentum | all
learner.adapt_scope       local | one_hop
eval.evalsets             prequential | current | backward | canonical
```
