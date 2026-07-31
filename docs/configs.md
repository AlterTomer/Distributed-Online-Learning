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

`weights` choices: **`metropolis`** is the default and guarantees
$a_{vv} \ge 1/(1+d_v) > 0$, so an agent never discards its own estimate.
**`relative_degree`** weights by neighbour degree. **`uniform`** is $a_{vu}=1/N$
and is what the X0 exactness identity requires — it is not a sensible choice on
a graph that is not complete.

*Consumed by:* `env/graph.py` (phase 1), every diffusion learner (phase 3).

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
confounded with "what does heterogeneity cost". Under **`dirichlet`**, for each
class $c$ a distribution over agents is drawn from $\text{Dir}(\beta\mathbf 1_N)$
and that class's images are allocated accordingly: $\beta = 0.1$ gives agents
that each see only two or three digits, and $\beta \to \infty$ approaches IID.
That is the sharpest form of Q2 — an agent that never sees a 7 can only learn 7s
through the combine step — and it is what X6 sweeps.

**`allow_epochs`** decides whether a sample may be consumed twice. It is not
merely bookkeeping. The research note's Assumption 5 requires every labelled
observation to enter the network likelihood **exactly once**: reusing a sample
double-counts its information, so the filter's covariance shrinks faster than
the evidence justifies and the posterior becomes overconfident — a failure that
surfaces in the phase-5 diagnostics as innovation whiteness breaking down. It
also keeps the online claim honest, since "no replay buffer" is one of the
method's stated properties. Setting it `true` waives both the guarantee and the
shard-budget check, and is a real trade rather than a convenience.

*Consumed by:* `env/stream.py`, `env/partition.py`, `env/environment.py` (phase 1).

### `env.drift`

| Field | Type | Default | Legal values | Applies to |
|---|---|---|---|---|
| `schedule` | str | `stationary` | `stationary`, `linear`, `piecewise`, `sinusoidal` | all |
| `total_degrees` | float | `45.0` | $[0, 45]$ | `linear` |
| `change_points` | list[int] | `[]` | non-negative, sorted | `piecewise` |
| `jump_degrees` | float | `15.0` | any | `piecewise` |
| `amplitude_degrees` | float | `30.0` | $\lvert\cdot\rvert \le 45$ | `sinusoidal` |
| `period` | int | `500` | ≥ 1 | `sinusoidal` |

**There is no `alpha` field, and supplying one is an error.** The per-step rate
is derived as `total_degrees / run.horizon`, so changing the horizon cannot
silently change how far the distribution travels. At the defaults that is
$45/1500 = 0.03°$/step.

The 45° ceiling is a validity constraint, not a preference. Past roughly that
much rotation MNIST accuracy degrades for reasons unrelated to decentralization,
and near 180° a 6 *is* a 9 — the reference error $e^\star$ would then be
measuring label ambiguity and the headline gap would be uninterpretable.

*Consumed by:* `env/drift.py`, `data/transforms.py`, `evaluation/evalsets.py`
(phases 1–2). The rotation at a step is already computable today:
`config.rotation_at(t)`.

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
| `evalsets` | list[str] | `[prequential, current, canonical]` | `prequential`, `current`, `backward`, `canonical` | No duplicates |
| `backward_offset` | int | `500` | ≥ 0, and < `run.horizon` if `backward` is requested | How far back the forgetting probe looks |
| `batch_size` | int | `1000` | ≥ 1 | Evaluation batching only |

The four evaluation sets: **`prequential`** is test-then-train on the incoming
batch, the per-step signal. **`current`** is the held-out test set rotated by the
same amount as the training data at step $t$ — the headline metric, and the one
that is easy to get wrong. **`backward`** is rotated as of $t -$ `backward_offset`
and measures forgetting. **`canonical`** is unrotated and stays comparable to
published MNIST numbers across every run.

`backward_offset` has to be read together with the drift rate, because what
matters is the *rotation* between the two sets, not the number of steps. At the
capped rate of $45°/1500 = 0.03°$/step the separation is
`backward_offset` $\times\ 0.03°$:

| `backward_offset` | Separation | Verdict |
|---|---|---|
| 200 | 6° | Too small — indistinguishable from `current` |
| **500** | **15°** | Default. A third of the total range |
| 1000 | 30° | Large, but leaves only 500 steps where the probe exists |

`tests/test_config.py` asserts the separation stays above 10°, so lowering the
drift rate without revisiting the offset fails loudly rather than producing a
forgetting curve that is really a noise curve.

The last three sets score 10 000 images per agent every $K$ steps, which is the
expensive part of a run and the reason $K = 25$ rather than 1.

*Consumed by:* `evaluation/` (phase 2).

---

## 3. `configs/graph/` — topologies

Each file sets `topology` and any `params` it needs. All inherit `n_nodes` from
`base.yaml`, so a graph file alone never fixes the network size.

| File | `params` | Spectral gap $1-\rho$ | Why it exists |
|---|---|---|---|
| `complete.yaml` | — | 1 (best) | Reproduces centralized behaviour; required by X0 |
| `ring.yaml` | — | moderate | Default for X1/X2; a realistic middling case |
| `path.yaml` | — | near 0 (worst) | The pessimistic end of the topology sweep |
| `grid2d.yaml` | `rows`, `cols` | between | `rows × cols` **must** equal `n_nodes`, and this is validated |
| `star.yaml` | — | moderate | A hub; degree distribution is maximally uneven |
| `erdos_renyi.yaml` | `p: 0.3` | random | Edge probability; realization is drawn from `seed_graph` |
| `disconnected.yaml` | `n_components: 2` | 0 | **Negative control.** Communication cannot cross components, so the distributed method must degrade toward local-only |

`watts_strogatz` is a legal `topology` value but has no shipped file yet.

Spectral gap is the natural x-axis for "what does connectivity cost" (Q1) and is
computed by `env/graph.py` in phase 1, then denormalized into every log row.

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
| `lambda_forget` | float or null | `null` | $(0,1]$ | Phase 5 only |
| `process_noise_q` | float or null | `null` | > 0 | Phase 5 only |
| `prior_scale` | float | `1.0` | > 0 | Phase 5 only; $P_0 = \text{prior\_scale} \cdot I$ |

Two validation rules worth knowing before you hit them:

- **An optimizer that carries per-node state cannot have `mix_optimizer_state: none`.** Local momentum drifts apart across agents and the run diverges; that is a known failure mode, not an open question. Use plain `sgd` if you want no mixing.
- **`lambda_forget` and `process_noise_q` cannot both be set.** They are two parameterisations of the same forgetting effect and are jointly unidentifiable.

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
