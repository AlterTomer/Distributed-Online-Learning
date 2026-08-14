# The environment

The environment answers one question: **what does agent $v$ observe at step $t$?**
Everything else in the project — the learners, the metrics, the filter — consumes
that answer and nothing else.

This document explains how the answer is assembled, and states the guarantees the
rest of the benchmark relies on together with what enforces each one. Those
guarantees are the reason to trust a curve. If shards overlap, if a sample is
served twice, if the evaluation set carries a different rotation than the training
data, then every method degrades together and the comparison still *looks* fine —
which is why each is a checked invariant rather than a convention.

The API reference is `IMPLEMENTATION.md` §4.1; the decisions are in
`design_notes.md`; this file is the argument connecting them.

**If this document and the code disagree, the code is right and this is stale.**

---

## 1. The five components

`env/environment.py` composes five things, each of which knows nothing about the
others:

```
env/graph.py       who talks to whom, and with what weights
env/partition.py   which images belong to which agent
env/stream.py      which of an agent's images arrive at which step
env/drift.py       how far the distribution has moved by step t
data/transforms.py what a raw image looks like after rotation and pooling
```

The separation is load-bearing in two places. `drift.py` computes *how much*
rotation and `transforms.py` *applies* it, so the evaluation sets can be built at
an arbitrary drift state without duplicating the rotation code. And `graph.py`
knows nothing about data, so a topology can be swapped without touching anything
that reads a pixel.

### The chain from index to pixel

```
partition.shards[v]                  6000 indices, disjoint across agents
        ↓  stream.order[v]           shuffled once, from the `stream` seed
        ↓  stream.offsets[v, t]      prefix sum over labelled steps
    indices                          n indices for this (v, t)
        ↓  train.images[indices]     raw uint8→float, in [0, 1]
        ↓  transform.apply(·, φ)     rotate at 28×28 → pool to 14×14 → normalize
    Observation.x                    (n, 1, 14, 14), normalized
```

with `φ = drift.rotation_at(t, v)`.

At the defaults ($N{=}10$, $n{=}2$, $T{=}1500$) one step produces
$10\times2 = 20$ images of shape $(1,14,14)$, and the whole run consumes
**30 000 of the 60 000** training images.

---

## 2. The guarantees

### G1. Shards are disjoint

No image belongs to two agents. Enforced in `Partition.__post_init__`, which
compares the concatenated shard length against the number of unique indices and
raises if they differ.

*Why it matters.* Two agents holding the same sample would make their updates
correlated in a way the diffusion analysis does not model, and would break the
research note's Assumption 5 (each observation enters the network likelihood
exactly once) at the network level rather than the agent level.

*Guarded by* `test_shards_are_disjoint`, `test_shards_cover_the_training_set`.

### G2. Each sample is consumed at most once

Over a run, no index is served twice — not to the same agent, and not to two
agents. Enforced structurally: `stream.order[v]` is a permutation of shard $v$,
and the offset into it is a prefix sum that only advances on labelled steps, so
the consumed set is always a *prefix* of that permutation.

`build_stream` additionally checks the whole run up front and refuses one that
would outrun a shard, quoting the numbers:

> agent 0 would consume 3200 samples over the run but its shard holds 3000.

*Why it matters.* A sample counted twice double-counts its information, so the
filter's covariance shrinks faster than the evidence justifies and the posterior
becomes overconfident. It also falsifies the "no replay buffer" property that is
one of the method's stated advantages.

*Escape hatch.* `env.allow_epochs: true` permits wrapping and forfeits this
guarantee explicitly. It also waives the budget check, so it is the one setting
that turns G2 off.

*Guarded by* `test_no_sample_is_served_twice_to_an_agent`,
`test_no_sample_is_served_to_two_agents`,
`test_exactly_once_holds_over_a_full_length_run` (on real MNIST, 30 000 served /
30 000 distinct).

### G3. No train/test leakage

The test split is never reachable from the environment. This is structural rather
than checked: `data/mnist.py` loads the two splits from separate source files, the
partition indexes only into the training labels, and the stream only into the
partition. There is no code path from an `Observation` to a test image.

*Guarded by* `test_train_and_test_are_different_data`,
`test_no_training_index_falls_outside_the_split`.

### G4. Training and evaluation see the same rotation

At step $t$ the data carries rotation $\varphi(t)$, and any evaluation set built
for step $t$ must carry the same one. Enforced by having exactly one
`ImageTransform` instance and one `Drift` object in the environment, and one call
that both paths make:

```python
transform.at(images, drift.state_at(t))
```

The `Observation` also records `rotation_degrees`, so an evaluation set can be
*checked* against the data it is being compared with rather than trusted.

*Why it matters.* `WORKPLAN.md` §5.3 calls this "the thing that is easy to get
wrong". Evaluate on an unrotated set while training on rotated data and every
method appears to fail, for a reason that has nothing to do with any of them.

*Guarded by* `test_train_and_eval_paths_produce_identical_pixels`,
`test_rotation_recorded_on_the_observation_matches_the_schedule`.

### G5. Observations are shared and read-only

`simulate.py` calls `step(t)` **once** and hands the result to every learner, so
that X0 compares two learners on identical data by construction rather than by
seed-matching. `Observation` is frozen, and
`Environment.assert_unmodified(observations, t)` recomputes the step and compares
fingerprints.

*Why the positive check.* A frozen dataclass stops `obs.x = ...` but not
`obs.x.add_(1)`, and torch has no read-only tensor flag. The failure would be
near-invisible: the run completes, curves look like curves, and X0 fails with a
small residual that reads as a numerical issue rather than data corruption.

*Guarded by* `test_in_place_mutation_of_a_shared_observation_is_caught`,
`test_labels_are_copied_not_aliased`.

### G6. `step(t)` is positional

`step(t)` is a pure function of $t$. Calling it twice returns the same tensors;
visiting steps out of order gives the same answers as visiting them in order;
nothing advances as a side effect.

*Why it matters.* An evaluation set, a test, or a resumed sweep must be able to
ask about an arbitrary step and get the answer the run would have produced. With
a cursor, a second read at step $t$ returns step $t+1$ and shifts everything
downstream — silently, and for every learner at once.

*Guarded by* `test_the_same_step_returns_the_same_data`,
`test_steps_can_be_visited_out_of_order`,
`test_a_step_gives_the_same_answer_however_it_is_reached`.

### G7. Batches are full or empty, never partial

An agent either receives its full $n$ samples or none at all. Every agent active
at a step therefore holds the same count.

*Why it matters.* The X0 identity

$$\sum_v \tfrac1N\bigl(\bm\theta-\eta\nabla L(\bm\theta;\mathcal D^v)\bigr)
= \bm\theta - \eta\,\tfrac1N\sum_v \nabla L(\bm\theta;\mathcal D^v)$$

holds only when the per-agent means average to the pooled mean, which requires
equal batch sizes. Unequal batches produce a small, plausible, non-zero residual
rather than an obvious failure.

*Guarded by* `test_a_batch_is_either_full_or_empty`,
`test_every_active_agent_receives_the_same_count`.

### G8. Everything is reproducible from one master seed

Four independent streams are derived from the master seed by keyed hash of the
stream's *name*, so each component can be varied while the others are held fixed:

| Stream | Drives | Held fixed lets you vary |
|---|---|---|
| `init` | model initialization | everything else |
| `partition` | shard assignment | initialization alone |
| `stream` | sample order, label draws | which agent owns what |
| `graph` | random topology realization | the data, at fixed connectivity |

Derivation is by name rather than draw order, so adding a fifth stream later
cannot shift the four that exist and invalidate recorded results.

*Guarded by* `test_holding_partition_fixed_while_init_varies`,
`test_stream_seed_does_not_disturb_the_partition`,
`test_the_same_seed_gives_the_same_environment`.

---

## 3. What each component decides

### Graph — `env/graph.py`

Two matrices, and keeping them distinct is the point. **Adjacency** has a *zero*
diagonal: an agent sends itself no message, and a self-edge would bill the
communication ledger for $N$ vectors per step that nobody transmitted.
**Combination weights** have a *strictly positive* diagonal: an agent always keeps
some of the estimate it just computed from its own data.

Also two *graphs*: `graphs.comm` carries diffusion messages, `graphs.data` is the
graph a predictor's forward pass would exchange over. For every Class L
architecture — MLP, CNN, RNN, Transformer — the data graph is **empty**, and it
stays empty for this whole project. It exists now because retrofitting a second
graph into an environment that assumed one touches every file.

Full parameter reference: `configs.md` §3.

### Partition — `env/partition.py`

IID or Dirichlet label skew. Under skew, **shard sizes stay equal** and only the
composition varies, so X6 measures label skew rather than skew confounded with
data volume — and so the config-time budget check, which assumes equal shards,
remains meaningful.

#### What "label skew" means here

Every agent holds the same *number* of images. What differs is **which digits**
they are. Agent 3 might hold mostly 2s and 3s while agent 7 holds mostly 8s and
9s. Nothing about the images themselves is changed — the skew is entirely in
*who got which*, which is why `centralized_sgd`, which pools every shard, is
provably flat across $\beta$ and serves as a free correctness check (F9).

It is a statement about the **marginal $P(y)$ per agent**, not about $P(x \mid
y)$. A "3" looks the same on every agent; some agents just see very few of them.
This is the standard federated-learning notion of non-IID, and it is the one that
matters for diffusion: an agent whose local gradient only ever points toward
"separate 2 from 3" needs its neighbours to learn anything about 8s.

Reported as one number by `Partition.skew`: the mean total-variation distance
between an agent's class histogram and the pooled one. $0$ means every agent
mirrors the global data; it approaches $1 - 1/K = 0.9$ when each agent holds a
single class.

#### How $\beta$ controls it

Each agent draws a preference vector over the $K = 10$ classes,

$$\bm q_v \sim \mathrm{Dir}(\beta \bm 1_K),$$

and is then filled to its target size, taking as much of each class as $\bm q_v$
asks for and the remaining pool can supply. $\beta$ is the **concentration**: it
is the only knob, and it controls how far a typical draw strays from the uniform
vector $\bm 1/K$.

The mechanism is the variance of a Dirichlet coordinate,

$$\operatorname{Var}(q_{v,k}) = \frac{\tfrac1K\left(1 - \tfrac1K\right)}{\beta K + 1},$$

which is $O(1/\beta)$. So:

- **Small $\beta$ (0.1).** Draws are extreme — most of the mass lands on a couple
  of coordinates and the rest are near zero. Agents get sharply different
  preference vectors, hence sharply different digits.
- **Large $\beta$ (100).** Draws concentrate on the mean $\bm 1/K$. Every agent's
  preference is *nearly the same uniform vector*, so every agent's shard is
  nearly a uniform sample of the pool. As $\beta \to \infty$ this converges to
  IID.

**"More independence" is the wrong intuition, and worth being precise about.**
The $\bm q_v$ are drawn independently at *every* $\beta$ — that never changes.
What large $\beta$ does is make them **nearly identical**, because they all
concentrate on the same mean. The consequence is that an agent's shard becomes
*uninformative about which agent it is*: knowing you are looking at agent 7's
data tells you nothing about which digits you will see. That statistical
independence between **agent identity and label** is what "IID across agents"
names, and it is produced by low variance in $\bm q_v$, not by any change in how
the $\bm q_v$ are drawn.

Measured on MNIST, $N = 10$, five seeds:

| $\beta$ | skew (mean TV) | classes present | classes with ≥5 % of the shard | perplexity $e^{H(\bm q_v)}$ |
|---|---|---|---|---|
| 0.1 | 0.647 | 5.5 [3–8] | 3.1 [1–5] | 2.9 |
| 1.0 | 0.363 | 9.3 [6–10] | 6.0 [4–9] | 6.5 |
| 100 | 0.060 | 9.9 [9–10] | 9.9 [8–10] | 9.8 |

Two counts are given because they answer different questions. **Classes
present** counts anything with at least one image, so it overstates what an agent
can actually learn — a shard with 400 images of one digit and three of another
"has" two classes. **Perplexity** is the effective count, $e^{H}$ of the shard's
class histogram, and equals $K$ exactly when the shard is uniform. At $\beta =
0.1$ an agent nominally touches 5.5 digits but has meaningful data on about 3.
At $\beta = 100$ the two agree at ~9.9, which is what "IID" should look like.

The skew column never reaches its 0.9 ceiling because sizes are held equal: an
agent must be filled to its quota even after its preferred classes are exhausted,
so it takes some of everything. That is a deliberate trade — see the note on
`balance_sizes` in the module docstring — and it makes $\beta = 0.1$ a *milder*
regime than the classical unbalanced Dirichlet partition, not a harsher one.

### Stream — `env/stream.py`

Which of an agent's images arrive when. Label availability $\pi_{\text{lab}}$ is
drawn once as an $(N,T)$ block, and **an idle step consumes nothing**: the agent
receives no samples and its shard is untouched, so $\pi_{\text{lab}}$ controls
only how often an update happens, never how fast the data runs out.

### Drift — `env/drift.py`

Four schedules. The per-step rate is **derived**, never configured
($\alpha = \text{total\_degrees}/T$), and the 45° well-posedness cap is checked
against what the schedule *does* over the horizon rather than against its
individual fields.

### Transform — `data/transforms.py`

`rotate at 28×28 → downsample to 14×14 → normalize`, in that order, with the
constants fitted once on canonical data. Normalizing before rotating would give
every rotated image four bright corners that no unrotated image has — a signal
perfectly correlated with the drift state.

---

## 4. How to inspect it

```
python scripts/check_environment.py
```

Run button, no arguments. Sections for topologies, partitions, drift schedules and
the image pipeline, sample streams, and the composed environment — including a
deliberate in-place mutation so you can watch G5 fire.

`scripts/make_preliminary_figures.py` renders figure 10, which shows what each
agent receives at four steps under both regimes, with identical indices in both
halves so the only visible difference is the rotation. It writes to the
gitignored `figures/` at the repository root, or wherever `DEKF_FIGURES_DIR`
points — see `docs/figures.md` §1.

---

## 5. Test coverage

| Module | Tests |
|---|---|
| `env/graph.py` | 352 |
| `env/partition.py` | 101 |
| `env/stream.py` | 63 |
| `env/environment.py` | 43 |
| `env/drift.py` | 43 |
| `data/transforms.py` | 37 |

Tests marked `needs_data` skip when MNIST has not been downloaded; run
`scripts/check_data.py` once to enable them.

---

## 6. What the environment does *not* do

- **No learner logic.** It does not know what a gradient is. Pooling the per-agent
  batches into $\mathcal D_t = \bigcup_v \mathcal D_t^v$ lives here as `pool()`
  because it is data handling and because the X0 identity depends on the pooled
  batch being *exactly* that union — but the environment never differentiates
  anything.
- **No evaluation sets.** `evaluation/evalsets.py` (phase 2) builds those, reading
  `drift_state(t)` and the shared `transform` so it cannot disagree with the
  training path.
- **No metrics.** It records `rotation_degrees` on each observation so a metric can
  be attributed to a drift state, and nothing else.
- **No simulated network.** Messages are not sent; the combine step is a matrix
  product against the weights. `WORKPLAN.md` lists distributed *systems*
  engineering as an explicit non-goal.
