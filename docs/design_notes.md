# Design notes

A running log of decisions: what was chosen, what it was chosen over, and why.

This exists because the *why* is the part that decays fastest. Six months on, a
config value looks arbitrary and gets "cleaned up" by someone — often the author
— who no longer remembers what it was protecting against. Each entry below is
something that would be easy to undo by accident.

Entries are append-only and dated. When a decision is reversed, the original
entry stays and a new one records the reversal, so the reasoning chain survives.
If an entry and the code disagree, the code is right and the entry is stale;
say so in a new entry rather than editing history.

**Legend.** ✅ settled · 🔄 revised · ❓ open

---

## 2026-07-30 — Spec review, before any code

### ✅ D1. Input downsampled to 14×14; primary model is 196–14–10

**Decision.** MNIST images are downsampled to 14×14, and the primary model is a
196–14–10 MLP with $p = 2908$. It is used for *all* phases, not just phase 5.

**Alternative rejected.** Keeping 784 inputs and shrinking the hidden layer.

**Why.** Phase 5 needs a dense $p \times p$ covariance, which caps $p$ at about
$3\times10^3$. That budget cannot be met by narrowing the hidden layer: at 784
inputs, $784h + h + 10h + 10 \le 3000$ forces $h \approx 3$, which is not a
classifier. The input dimension is what has to come down. The alternative —
running phases 1–4 on a 784–128–10 MLP and switching models for phase 5 — would
make the Diff-EKF comparison a comparison of two different architectures.

**Consequence if undone.** A dense covariance at $p \approx 10^5$ is $10^{10}$
entries. Phase 5 stalls.

**Guarded by.** `test_small_mlp_hits_the_phase_5_parameter_budget`.

### ✅ D2. Rotation is applied at 28×28, downsampling second

**Decision.** `data/transforms.py` composes `rotate → downsample`, in that order,
and is the single implementation used by training, all evaluation sets, and the
reference classifier.

**Why.** Rotating a 14×14 image directly destroys far more information than
rotating at full resolution and then pooling. More importantly, one shared
implementation is what makes "train and eval see the same rotation" checkable
rather than hoped for.

**Guarded by.** `test_transforms.py` (phase 1, not yet written).

### ✅ D3. Total rotation capped at 45°; α is derived, never configured

**Decision.** `env.drift.total_degrees` defaults to 45 and may not exceed it. The
per-step rate is `total_degrees / run.horizon`, and supplying `alpha` directly is
a config error.

**Alternative rejected.** The spec's original $\alpha \in [0.1, 0.5]°$/step.

**Why.** Per-step drift being slow is not sufficient; *cumulative* drift is what
decides whether the task stays well-posed. $\alpha = 0.2°$ over $T = 1500$ is
300° of rotation, where a 6 is a 9 and $e^\star$ measures label ambiguity rather
than decentralization cost. Deriving α additionally means changing the horizon
cannot silently change how far the distribution travels — the two were coupled
in a way that made runs of different length incomparable.

**Consequence if undone.** Every method looks equally bad, and the headline gap
$\bar e_t - e^\star$ answers nothing.

**Guarded by.** `test_rotation_beyond_the_well_posed_cap_is_rejected`,
`test_configuring_alpha_directly_is_rejected`,
`test_shortening_the_horizon_does_not_change_the_total_rotation`.

### ✅ D4. All learners in an experiment share one environment instance

**Decision.** `learners` is a list within a run. `simulate.py` calls
`env.step(t)` **once** per step and drives every learner against that same
observation dict, stepping them in lockstep.

**Alternative rejected.** One run per learner, matched afterwards by seed.

**Why.** Three things follow. X0 becomes exact *by construction* rather than
contingent on every RNG draw lining up — seed-matching holds until someone adds
a `torch.randn` somewhere, at which point the exactness test fails for a reason
unrelated to what it tests. $E_{\text{cent}}$ (research note §7.3) becomes
computable in phase 1 instead of requiring a re-run and a trajectory join. And
every cross-method comparison is paired, so fewer seeds buy the same resolution.

**Cost.** `Observation` must be a frozen dataclass and learners must treat it as
read-only, or one learner's in-place normalization corrupts another's view.

**Guarded by.** `test_simulate.py` (phase 3, not yet written).

### ✅ D5. Defaults N=10, n=2, T=1500; shard budget checked at load

**Decision.** $N = 10$, $n = 2$, $T = 1500$, and a config is rejected at load
unless $N n T \le 60000$ or `allow_epochs` is set.

**Why.** Agents draw from disjoint shards, so the run needs $NnT$ samples and
MNIST supplies 60 000. The spec's own runtime table specified $N{=}10$,
$n{=}4$, $T{=}2000$ — which needs 80 000 and does not fit. Discovering that at
step 1400 of a five-seed sweep wastes the sweep.

**Non-obvious consequence.** Raising $N$ *shortens* the feasible horizon:
$N{=}100$ at $n{=}2$ leaves 300 steps. Scaling the network is not free, which is
why $N \in [50,100]$ stayed an open question rather than becoming a default.

**Guarded by.** `test_shard_budget_rejects_the_combination_from_the_early_drafts`,
`test_scaling_n_agents_shortens_the_feasible_horizon`.

### ✅ D6. ATC is primary for both diffusion SGD and Diff-EKF

**Decision.** Adapt-then-combine is the primary ordering. CTA is a labelled
variant, measured in X1b and reported.

**Why.** Diff-EKF is ATC. If the SGD baseline were CTA, the phase-5 comparison
would confound *EKF vs SGD* with *ATC vs CTA*, and since ATC generally has better
mean-square performance the baseline would be handicapped in a way a reviewer
will notice. With both ATC, the methods differ in the adapt step alone at
identical communication.

### ✅ D7. Exactness preconditions are pinned in the X0 config, not assumed

**Decision.** `x0_exactness.yaml` explicitly sets `dtype: float64`, complete
graph, `weights: uniform`, `label_availability: 1.0`, `optimizer: sgd`,
`momentum: 0.0`, `mix_optimizer_state: none`.

**Why.** The identity $\sum_v \frac1N(\theta - \eta\nabla L_v) = \theta - \eta
\frac1N\sum_v \nabla L_v$ needs equal batch sizes across agents, mean loss
reduction, no optimizer state, and a common $\theta_{t-1}$. Violating the first
produces a small, plausible, **non-zero** residual rather than an obvious
failure — precisely the failure mode the test exists to catch.

**Guarded by.** `test_exactness_config_pins_every_precondition`.

---

## 2026-07-31 — Phase 0 implementation

### ✅ D8. Seeds derived by keyed hash of the stream name, not sequential spawning

**Decision.** `derive_seed(master, name)` is
`blake2b("dekf_bench:<name>:<master>")`. Four streams: `init`, `partition`,
`stream`, `graph`.

**Alternative rejected.** `numpy.random.SeedSequence(master).spawn(4)`.

**Why.** Sequential spawning couples each stream to its *position*. Adding a
fifth stream later — for evaluation subsampling, say — would shift the values of
every stream after the insertion point, silently invalidating comparisons
against results already recorded. Name-keyed derivation is positionally stable.
It is also stable across processes, unlike `hash()`, which is randomised per
interpreter unless `PYTHONHASHSEED` is pinned.

**Guarded by.** `test_stream_values_depend_only_on_name_and_master`,
`test_derivation_is_stable_across_processes`.

### ✅ D9. Explicit generators, not global RNG state

**Decision.** `seeds.torch_generator("init")` returns a generator that is passed
explicitly to every draw.

**Why.** Global-state RNG is the usual reason two runs with the same seed
diverge: a draw anywhere perturbs every subsequent draw everywhere.
`set_determinism` still seeds the globals as a backstop for library code that
reaches for them.

**Guarded by.** `test_explicit_generator_is_immune_to_global_seeding`.

### ✅ D10. Determinism settings that cannot be applied warn rather than pretend

**Decision.** `set_determinism` warns when `PYTHONHASHSEED` is unset, and when
`CUBLAS_WORKSPACE_CONFIG` is set after the CUDA context exists. Both are still
exported for child processes.

**Why.** `PYTHONHASHSEED` is read at interpreter startup and
`CUBLAS_WORKSPACE_CONFIG` when the CUDA context is created; setting either later
is a no-op for this process. Silently calling `os.environ[...] = ...` and moving
on would make the code *look* deterministic while set-iteration order stayed
unpinned. On CUDA the failure is worse: deterministic matmuls raise inside the
first op rather than at configuration time.

**Related.** `git_revision()` returns `("unknown", dirty=True)` outside a repo —
a result that cannot be traced to a commit must not claim to be clean.

### ✅ D11. Unknown config keys are errors, with typo suggestions

**Decision.** Config loading validates against a dataclass schema; any unknown
key raises, and the message suggests the nearest known key.

**Why.** A misspelled `label_availabilty` silently taking the default is how a
run produces a plausible curve that answers a different question. This happened
in practice within a day of writing the loader — see D12.

### 🔄 D12. `backward_offset` raised from 200 to 500

**Decision.** Default `eval.backward_offset` is 500 steps.

**Why.** What the forgetting probe measures is the *rotation* between the
backward and current evaluation sets, not the step count. At the D3 capped rate
of $0.03°$/step, a 200-step lookback is 6° — below what a classifier visibly
forgets, so the backward curve would have tracked the current one and been read
as "no forgetting" when it actually meant "no separation". 500 steps gives 15°,
a third of the total range, while leaving 1000 steps where the probe is defined.

**Note.** This is a consequence of D3 that was missed when D3 was taken: capping
the drift rate silently shrank the separation the offset produces. Coupled
defaults need re-checking together.

**Guarded by.** `test_backward_probe_sees_a_distinguishable_rotation`, which
asserts ≥ 10° and so fails loudly if the drift rate is lowered again without
revisiting the offset.

### ✅ D13. `mnist.py` returns raw intensities; normalization happens after rotation

**Decision.** The loader hands back float images in $[0,1]$ and does **not**
normalize. The pipeline order is fixed as `rotate at 28×28 → downsample to 14×14
→ normalize`.

**Alternative rejected.** Normalizing at load, which is what almost every MNIST
example does.

**Why.** Rotation fills the corners it exposes with zero. On raw intensities
zero *is* the black background, so the fill is invisible. Normalize first and
zero becomes $-0.4242$, so every rotated image acquires four bright corners that
no unrotated image has — a signal **correlated with the drift state**. A
classifier will happily learn the corners instead of the digit, and the drift
experiments would then measure how well each method tracks an artefact we
introduced. Nothing about the resulting curves would look wrong.

**Consequence.** The normalization constants must be computed on the
*transformed* canonical training images, not taken from the standard 28×28
values, because average-pooling to 14×14 preserves the mean but shrinks the
standard deviation. `channel_statistics()` exists for that, and it must be
computed once on canonical data and then held fixed — a normalization that
drifts with the rotation would confound the very thing being measured.

**Guarded by.** `test_intensities_are_raw_and_unnormalized`,
`test_conversion_reproduces_the_canonical_statistics` (which pins mean/std to
the canonical 0.1307/0.3081 and so catches any scaling error in the uint8
conversion).

### ✅ D14. Splits are cached as dense tensors, not served by a `DataLoader`

**Decision.** Both splits are converted once to dense float tensors and cached
to `data/mnist/<split>_v1.pt`. Writes are staged to a `.tmp` file and renamed;
an unreadable cache is deleted and rebuilt rather than raising.

**Why.** The training split is 180 MB, which fits in memory comfortably, and the
run indexes it 1500 times. A `DataLoader` would add per-item PIL conversion and
collation to the hot path for no benefit. Measured: 8.4 s from source, 0.07 s
from cache.

**Two things found by the tests rather than by reasoning.**
`torch.load(weights_only=True)` reports a damaged file as
`pickle.UnpicklingError`, not `RuntimeError` — omitting it from the except
clause turned a half-written cache into a crashed run instead of a two-second
rebuild. And the `MnistSplit` dtype invariant originally demanded float32,
which rejected the float64 conversion the X0 exactness check requires; it now
accepts either and rejects integer dtypes, since uint8 there means the $/255$
conversion was skipped.

**Related.** `subset()` clones rather than views. A shard that aliased the full
tensor would make an in-place bug in one agent visible to every other agent,
which is the hardest class of bug to attribute in this codebase.

---

## 2026-07-31 — Phase 1, `env/graph.py`

### ✅ D15. Adjacency has a zero diagonal; combination weights have a positive one

**Decision.** Two matrices, never one. `Graph.adjacency` is who talks to whom
and its diagonal is **zero**; `Graph.weights` is $a_{vu}$ and its diagonal is
**strictly positive**.

**Why.** An agent does not send itself a message, but it certainly keeps its own
estimate — it just computed that estimate from its own data. Conflating the two
gives two distinct failures. A zero diagonal in $\bm A$ makes an agent discard
the update it just produced. A non-zero diagonal in the adjacency bills the
communication ledger for $N$ vectors per step that nobody transmitted, which
inflates the denominator of every accuracy-versus-communication plot — the
convention the whole comparison is reported in.

Metropolis weights guarantee $a_{vv} \ge 1/(1+d_v) > 0$, so the positive
diagonal is structural rather than a special case.

**Guarded by.** `test_adjacency_has_no_self_loops` and
`test_weights_have_a_strictly_positive_diagonal`, both run across every
topology × weight-rule combination, plus
`test_no_agent_combines_an_estimate_it_did_not_receive`.

**Found by these tests.** `nx.cycle_graph(1)` returns a single node **with a
self-loop**. A one-agent ring would therefore have had a 1 on the adjacency
diagonal. `_ring` now falls back to `path_graph` below three nodes.

### 🔄 D16. The spec's spectral gap is undefined for non-doubly-stochastic weights

**Decision.** `spectral_gap` implements `WORKPLAN.md` §4.1's
$1 - \lVert\bm A - \tfrac1N\mathbf1\mathbf1^{\mathsf T}\rVert_2$ but **raises**
unless the weights are doubly stochastic. A new `mixing_gap` $= 1 - \text{SLEM}$
is always defined and equals the spectral gap whenever both are.

**Why.** The $\tfrac1N\mathbf1\mathbf1^{\mathsf T}$ term is the projector onto
the consensus direction only when the all-ones vector is a *left* eigenvector as
well as a right one — that is, only for doubly stochastic $\bm A$. Metropolis
weights are; relative-degree and uniform weights are doubly stochastic only on a
*regular* graph. Off that case the norm exceeds 1 and the formula returns a
negative number:

| topology / rule | spec gap | doubly stochastic | $1-\text{SLEM}$ |
|---|---|---|---|
| star / metropolis | +0.100 | yes | +0.100 |
| star / relative_degree | **−1.565** | no | **+0.600** |
| star / uniform | −0.265 | no | +0.500 |

The star with relative-degree weights in fact mixes *very well* — the hub
aggregates the whole network in one hop — so the reported value is not merely
imprecise, it is **wrong in sign and in ranking**. Had this reached F3, the
price-of-connectivity figure would have placed the best-mixing configuration off
the left-hand end of the axis.

**Consequence.** X3 uses Metropolis weights, where the two measures coincide, so
no experiment changes. What changes is that a non-Metropolis sweep now fails
loudly instead of plotting nonsense. `summary()` reports `spectral_gap: None`
alongside a populated `mixing_gap` rather than emitting the bad number.

**Guarded by.** `test_spectral_gap_refuses_non_doubly_stochastic_weights`,
`test_the_two_gap_measures_agree_when_both_are_defined`,
`test_star_with_relative_degree_weights_actually_mixes_fast`.

**Spec impact.** `WORKPLAN.md` §4.1 states the formula without the double
stochasticity condition. It has been amended.

### ✅ D17. Dirichlet skew holds shard *sizes* equal and skews only composition

**Decision.** `balance_sizes=True` by default. Each agent draws a class
preference $\bm q_v \sim \mathrm{Dir}(\beta\bm 1_K)$ and is then filled to
exactly $60000/N$ samples, taking as much of each class as its preference asks
for and the pool can supply.

**Alternative rejected.** The classical construction (Hsu et al.): per class,
draw a distribution over *agents*. This is what most federated-learning papers
use, and it makes shard sizes vary enormously.

**Why.** Two independent problems with unequal sizes.

*It confounds the experiment.* X6 asks "does cooperation still work when agents
see different classes?" If shard sizes also vary by 7×, the answer mixes label
skew with data volume and X6 no longer isolates what it is named after.

*It silently starves runs.* The config-time shard budget check ($N n T \le
60000$) assumes equal shards. Measured on MNIST with the classical
construction, the smallest shard falls below the 3000 a default run consumes:

| $\beta$ | seeds (of 10) whose smallest shard < 3000 | smallest observed |
|---|---|---|
| 1.0 | 1 | 2001 |
| 0.5 | 8 | 1496 |
| 0.1 | 10 | 50 |

So at $\beta = 0.1$ **every** seed would exhaust an agent part-way through a run
that passed validation. Balancing makes the budget check meaningful again.

**Escape hatch.** `balance_sizes=False` still available for comparison against
the literature, and `build_partition(min_shard_size=...)` rejects a partition
that cannot feed the run, with a message explaining the interaction.

**Guarded by.** `test_mnist_unbalanced_dirichlet_would_starve_a_default_run`
(asserted over seeds, since the shortfall is a property of the construction
rather than of one draw), `test_mnist_balanced_dirichlet_never_starves`,
`test_balancing_removes_the_size_variation_but_keeps_the_skew`.

### ✅ D18. Dirichlet is sampled through numpy, not `torch.distributions`

**Decision.** `_dirichlet()` consumes the torch generator once to seed a
`numpy.random.Generator` and draws from that.

**Why.** `torch.distributions.Dirichlet.sample()` accepts **no generator
argument** and draws from the global RNG. The first implementation used it, and
`test_the_same_seed_gives_the_same_shards` failed for every $\beta$: the
partition depended on whatever else had consumed random numbers first, so a run
could not be reproduced from `seed_partition`.

This is D9 (explicit generators, never global state) reappearing from a
direction I did not anticipate — not our own code reaching for the global RNG,
but a *library* doing it silently behind an object that looks stateless. Worth
checking any other `torch.distributions` use the same way; `torch.randperm` and
`torch.randint` do take generators, which is why the IID path was unaffected.

**Guarded by.** `test_the_same_seed_gives_the_same_shards`, plus a cross-process
check that the same seed gives byte-identical shards in a fresh interpreter.

---

## 2026-08-01 — Phase 1, drift and transforms

### 🔄 D19. Schedule evaluation moved out of the config schema into `env/drift.py`

**Decision.** `DriftConfig` validates fields and nothing else. Turning a step
into a rotation lives in `env/drift.py`, one class per schedule.
`Config.rotation_at` and `alpha_per_step` were removed and their tests moved to
`test_drift.py`.

**Why.** The schedule maths was in `utils/config.py` because that is where it
was first needed. Once `env/drift.py` existed it had two options, both bad:
duplicate the piecewise logic, or have `env` call *up* into a config object for
behaviour. Two implementations of "where does the jump land" is precisely the
kind of pair that diverges silently — one of them gets a fix and the other does
not, and the evaluation sets end up built at a different rotation than the
training data.

**Cost.** Three call sites updated (`check_config.py`,
`make_preliminary_figures.py`, `test_config.py`). Worth doing now; it would have
been ten call sites after phase 3.

### ✅ D20. The rotation cap is checked against behaviour, not against fields

**Decision.** `build_drift` computes `total_travel` over the horizon and rejects
anything past 45°. The per-field checks in the config schema stay, but they are
no longer the last word.

**Why.** Every individual field can look reasonable while the schedule as a
whole travels too far. `jump_degrees: 15` is inside the cap; four change points
is a normal-looking list; together they are 60° of rotation and the run is
measuring label ambiguity. Field-level validation cannot see the interaction.

**Guarded by.** `test_the_cap_is_checked_against_what_the_schedule_does`.

### ✅ D21. Per-node drift multipliers top out at 1, not at $1+\text{spread}$

**Decision.** Under `drift_scope: per_node`, rates are spread over
$[1-\text{spread},\,1]$ — the spread slows the laggards rather than
accelerating the leaders.

**Why.** The symmetric choice, $[1-s, 1+s]$, has a mean rate equal to the global
rate and is the obvious construction. But it puts the fastest agent at
$1.5 \times 45° = 67.5°$, past the cap the schedule was validated against — so
enabling per-node drift would silently invalidate the well-posedness guarantee
for some agents while every configured field still read as legal. Anchoring the
top at 1 keeps the cap true for every agent by construction.

**Related.** `Drift.state_at(t)` raises under per-node scope unless given a
node: there is no single network-wide state, and returning agent 0's would be a
silent lie.

### ✅ D22. `data/transforms.py` is the single implementation, and it is fitted once

**Decision.** One `ImageTransform` instance, frozen, with its normalization
constants baked in as values. Training, all three evaluation sets, and the
offline reference classifier call the same object. Constants are computed on the
**canonical** (unrotated, downsampled) training images.

**Why.** Under drift the evaluation set must carry the *same* rotation as the
training data at step $t$. One shared object is what makes that checkable rather
than hoped-for — `transform.at(images, state)` is the only call either path
makes, so neither can pass a rotation the other did not.

Fitting on canonical data matters twice over. The published 28×28 constants no
longer apply after pooling (average pooling preserves the mean but shrinks the
standard deviation — measured 0.3081 → 0.279 on MNIST, so the normalization
offset is 0.47σ rather than 0.42σ). And statistics recomputed per step would
drift with the rotation, confounding the very thing being measured.

**Guarded by.** `test_train_and_eval_paths_produce_identical_pixels`,
`test_exposed_corners_match_the_interior_background`,
`test_normalizing_before_rotating_brands_every_rotated_image`, and
`test_mnist_corners_are_clean_under_rotation` on the real data. Figure 06 of the
preliminary work now calls the shipped transform, so it fails visibly if the
ordering ever changes.

**Also decided.** Rotation is **bilinear**, not nearest-neighbour: at a few
degrees, nearest-neighbour quantises the digit into staircase artefacts that
vary with the angle — another signal correlated with the drift state. And
`rotate(images, 0.0)` returns the input object unchanged, so an $\alpha=0$ run
is bit-identical to a stationary one rather than merely close.

### ✅ D23. The stream is a pure function of (agent, step), not a cursor

**Decision.** Label-availability draws are made once up front as an $(N, T)$
Bernoulli block, and the offset into a shard at step $t$ is a prefix sum over
them. `indices_at(v, t)` can be answered directly for any $(v, t)$.

**Alternative rejected.** A per-agent pointer advanced each step, which is the
obvious implementation.

**Why.** A cursor works until something needs step 900 without having walked
0..899 — an evaluation set, a test, a restarted runner, a sweep resuming — and
then the answer depends on how you got there. It also makes the realised label
rate something you infer *after* a run rather than check before one. The prefix
sum costs an $(N,T)$ int tensor, which at the defaults is 15 000 entries.

**Guarded by.** `test_a_step_gives_the_same_answer_however_it_is_reached`,
`test_querying_out_of_order_is_stable`,
`test_repeated_queries_do_not_advance_anything`.

### ✅ D24. An unlabelled step consumes nothing from the shard

**Decision.** With $\pi_{\text{lab}} < 1$ an agent idles: no samples served, no
shard entries consumed, and its offset does not advance.

**Alternative rejected.** Serving the samples but withholding the labels.

**Why.** Three reasons, in order of weight.

*No learner here can use an unlabelled sample.* Every method in the project is
fully supervised, and the Diff-EKF adapt step needs an innovation
$\bm\nu = \bm y - \bm\mu$; with no $\bm y$ there is no measurement, and
Algorithm 1 line 8 passes the prediction through unchanged. So the sample is
discarded on arrival — and since shards are disjoint and finite, consuming it
destroys it permanently. At $\pi_{\text{lab}} = 0.25$ that is 4 500 of each
agent's 6 000 samples thrown away to no purpose.

*It would cap the horizon at the dense-label value.* Under consumption the
shard drains at $n$ per step regardless of $\pi_{\text{lab}}$, so the maximum
horizon is the same for every label rate:

| $n$ | $\pi_{\text{lab}}$ | max $T$, consuming | max $T$, not consuming |
|---|---|---|---|
| 2 | 1.0 | 3000 | 3000 |
| 2 | 0.5 | 3000 | 6000 |
| 2 | 0.25 | 3000 | 12000 |

Q4 asks where the online signal becomes too weak to learn from. Answering it may
well require running the sparse-label regime *longer* to see whether it
eventually catches up — and under consumption that experiment is simply not
available, because the data runs out at the same step either way.

*Forward compatibility.* A semi-supervised variant would want those samples, and
a design that has already destroyed them cannot be extended to use them.

**Correction.** An earlier version of this entry said unlabelled consumption
"would silently also be a shorter run". That is wrong: consumption makes the run
length *independent* of $\pi_{\text{lab}}$, not shorter than the dense case. The
table above is the accurate statement.

Measured at the defaults: $\pi_{\text{lab}}=0.25$ consumes 7 380 samples against
30 000, leaving 5 228 of each 6 000-sample shard available.

**Guarded by.** `test_an_idle_step_consumes_nothing_from_the_shard`,
`test_sparser_labels_consume_less_of_the_shard`,
`test_sparse_labels_let_a_longer_horizon_fit`.

**Also decided.** A batch is full or empty, never partially filled, so every
agent active at a step receives the same count — which is what the X0 identity
requires. And $\pi_{\text{lab}} = 1.0$ is handled exactly rather than as
`rand() < 1.0`, because "1.0 minus a rounding accident" is not a guarantee X0
can rest on.

### ✅ D25. Shared observations are guarded by a positive check, not by convention

**Decision.** `Observation` is a frozen dataclass, and
`Environment.assert_unmodified(observations, step)` recomputes the step and
compares fingerprints. The runner calls it on evaluation steps.

**Why.** D4 has one environment feeding every learner, which is what makes X0
exact by construction. The cost is that a learner mutating its input in place —
normalising, augmenting, casting — corrupts the data every *subsequent* learner
in that iteration sees. Freezing the dataclass stops `obs.x = ...` but not
`obs.x.add_(1)`, and torch has no read-only tensor flag.

The failure would be near-invisible: the run completes, the curves look like
curves, and X0 fails with a small residual that reads as a numerical issue
rather than as data corruption. Since the environment is positional (D23), the
check is cheap — recompute and compare two floats per agent — so it is a real
guard rather than a comment asking learners to behave.

**Guarded by.** `test_in_place_mutation_of_a_shared_observation_is_caught`,
`test_label_mutation_is_caught_too`, `test_labels_are_copied_not_aliased`.

**Also decided.** `pool()` lives in the environment rather than in the
centralized learner. The exactness identity requires the pooled batch to be
*exactly* the union of the per-agent ones; putting that in one place means the
learner and the test that checks it call the same code.

**Deviation from the spec.** `IMPLEMENTATION.md` §4.1 sketches
`Environment.reset(seed) -> None`. Here `reset` returns a *new* environment
instead. Everything else in the environment is frozen and positional; a mutating
reset would be the single place where a stale reference could hand back data
from a previous seed.

---

## 2026-08-02 — Phase 2, the state model

### ✅ D26. $\bm F$ and the forgetting rule are two axes, and both are selectable

**Decision.** The phase-5 state model exposes two independent choices:

| Field | Values | Acts on |
|---|---|---|
| `transition` | `identity`, `scalar` ($\bm F_t = \gamma\bm I$) | the **mean** |
| `forgetting` | `lambda` ($\bm P \mathbin{{*}{=}} \lambda^{-1}$), `process_noise` ($\bm P \mathbin{{+}{=}} \bm Q$) | the **covariance** |

Defaults `identity` + `lambda`. All four combinations are legal, so which state
model performs better is measured rather than assumed.

**The conflation this exists to prevent.** $\gamma$ is routinely called a
"forgetting factor", but propagating the moments gives
$\bm P_{t|t-1} = \gamma^2\bm P_{t-1|t-1} + \bm Q_t$ — and $\gamma^2 \le 1$
**contracts** the covariance. Forgetting means *loosening* the prior so a new
sample counts for relatively more, so $\gamma$ works against it. What $\gamma$
actually does is $\bm m_{t|t-1} = \gamma\bm m_{t-1|t-1}$: it pulls the estimate
toward the origin, which is $L_2$ weight decay written in state space. The two
knobs are orthogonal and were treated as one in the first config draft.

**Why `identity` is the default.** A single $\gamma$ ties "how fast may the model
change" to "how hard is it pulled to zero". And the origin is not a neutral point
for a network — it is where the model computes approximately the constant-zero
map, so shrinking toward it is a *bad* prior rather than a weak one. (The note
also argues from ReLU positive-rescaling and LayerNorm scale invariance; that
argument is weak *here*, since we use GELU and no normalization layers, and it is
recorded as not load-bearing for us.) `scalar` remains available because $\gamma$
does buy something real: it makes the state a mean-reverting AR(1) with a proper
stationary prior $\tfrac{q}{1-\gamma^2}\bm I$, where a random walk's prior
variance grows without bound.

**Why `lambda` is the default.** With $\bm F=\bm I$, multiplicative inflation in
the information domain is $\bm\Omega_{t|t-1} = \lambda\bm\Omega_{t-1|t-1}$ —
*exactly* structure-preserving, where $(\bm\Omega^{-1}+\bm Q)^{-1}$ is dense even
for diagonal $\bm Q$. That is decisive once the covariance is structured.

**Why `process_noise` is nonetheless offered.** The objection above is stated in
the *information* domain. While the covariance is carried as a dense $\bm P$ —
which is the whole of phase 5 at $p = 2908$ — both rules are one line and the
same cost, since the measurement update goes through Woodbury either way. $\bm Q$
also buys anisotropy a scalar $\lambda$ cannot express ("the read-out drifts, the
feature extractor does not"). The cost only appears if the project later adopts
the (S4) diagonal-plus-low-rank structure, which is defined on $\bm P^{-1}$.

### 🔄 D27. `lambda_forget` default 0.9999 → 0.997

**Decision.** Default $\lambda = 0.997$.

**Why.** Effective memory is $\approx 1/(1-\lambda)$ steps. The original 0.9999
gives **10 000 steps against a horizon of 1 500** — the filter would average over
the whole run and then some, which is *no forgetting at all*. X2 and X5 test the
tracking claim (M5); with that default the filter would have failed to track for
a reason having nothing to do with the method.

$\lambda$ should be set from the drift timescale, not chosen for looking close to
1. The model averages over $W$ steps during which the data rotates $\alpha W$
degrees:

| $\lambda$ | memory $W$ | drift over $W$ | inflation $\lambda^{-T}$ |
|---|---|---|---|
| 0.9999 | 10 000 | 300° | 1.2 |
| 0.999 | 1 000 | 30° | 4.5 |
| **0.997** | **333** | **10°** | **91** |
| 0.994 | 167 | 5° | 8 300 |

**The tension, which is why this stays a pilot item.** Shorter memory tracks
better but inflates *unexcited* directions harder — in an unexcited direction
nothing balances $\lambda^{-t}$. There is such a direction here: adding a
constant to every output bias shifts all logits equally, which softmax cannot
see. It is the $\bm\Lambda\mathbf 1 = \bm 0$ null direction reappearing in
parameter space, and its variance grows ~90× over a run at $\lambda = 0.997$.

The right *scaling* is $W \propto T$, since $\alpha = \text{total}/T$ — the same
argument as D3. That derivation is **not** built: phase 5 should sweep $\lambda$
against tracking error in X2 first (WORKPLAN §9 calibrates $\alpha$ the same
way), and building the machinery before knowing whether it matters would be
speculative.

**Diagnostics this obliges.** Record $\operatorname{cond}(\bm P)$ and
$\lambda_{\min}(\bm\Omega)$ from the first filter run: null-direction growth is
the failure mode this choice risks, and it is invisible to the innovation-based
checks, which only probe directions the data excites.

### ✅ D29. The ledger counts real payload, and X1 runs a payload-matched baseline

**Decision.** The communication ledger charges what a learner actually
transmits, including optimizer state. X1 runs **two** ATC variants:
`diffusion_sgd_atc` (momentum mixed, $2p$ per link) and
`diffusion_sgd_atc_plain` (no state, $p$ per link).

**The problem this fixes.** `WORKPLAN.md` §3.2 says diffusion SGD exchanges "one
$p$-vector per link per step". That is true of the payload-matched variant. But §3.4 makes the
primary configuration for X1–X6 *SGD with momentum, momentum also mixed* — and a
neighbour cannot mix a momentum buffer it was never sent. The primary baseline
therefore broadcasts $(\bm\psi, \bm m) = 2p$, while Diff-EKF broadcasts
$\bm\psi$ alone.

Measured at $N{=}10$, ring, $p{=}2908$:

| learner | per link | scalars/step | relative |
|---|---|---|---|
| `diffusion_sgd_atc_plain` | $p$ | 58 160 | 1.0× |
| `diffusion_sgd_atc` (momentum) | $2p$ | 116 320 | 2.0× |
| `diffusion_sgd_atc` (AdamW, `all`) | $3p$ | 174 480 | 3.0× |
| `diffusion_ekf` local | $p$ | 58 160 | 1.0× |
| `diffusion_ekf` one_hop | $p(q'{+}1)$ | 581 600 | 10.0× |

So **"at identical communication" is a claim about a particular pairing**, not
about the methods in general. Left unnoticed, phase 5 would have compared a
filter sending $p$ against a baseline sending $2p$ and reported the result as
equal-cost. F2 plots error against cumulative scalars, so the curves would have
separated on the *x*-axis for a reason the caption did not mention.

**Cost of the fix.** One extra learner per X1 run, which is nearly free since
learners share the environment (D4).

**Guarded by.** `test_the_ekf_and_plain_atc_send_exactly_the_same`,
`test_the_primary_sgd_baseline_sends_twice_what_the_filter_does`.

### ✅ D30. Centralized is a reference line on F2, not a point on the axis

**Decision.** `centralized_sgd` is recorded with `diffuses=False` and appears on
F2 as a horizontal line, like $e^\star$. Its notional pooling cost is stored in
the ledger but not plotted.

**Why.** It is an upper reference *for the online setting*, not a deployable
competitor — nothing in the paradigm proposes running it. Placing it at $x=0$
would read as "free", and placing it on the axis would invite a comparison it
was never meant to enter.

**The number is kept anyway, and it is uncomfortable.** Shipping raw samples to
a centre costs $N n (d{+}1) = 3\,940$ scalars per step, against $58\,160$ for
ring diffusion — **15× less**. Bandwidth is not the argument for
decentralization at this scale; latency, privacy, and the absence of a reliable
fusion centre are. Better to have that in the ledger than to be asked about it.

**Guarded by.** `test_pooling_is_cheaper_than_ring_diffusion`,
`test_centralized_is_not_on_the_communication_axis`.

### ✅ D31. $E_{\text{cent}}$ shares $\bm\theta_0$ and runs its own trajectory

**Decision.** The centralized reference for $E_{\text{cent}}$ starts from the
*same* $\bm\theta_0$ as the agents and runs its own trajectory from $t=0$.

**Why.** The research note asks for an "independently initialised" centralized
run. Read literally — a different $\bm\theta_0$ — the metric acquires an
irreducible floor that never vanishes even for a perfect method, and there is no
value of $E_{\text{cent}}$ meaning "these coincide". Reading it as *runs its own
trajectory rather than being re-anchored to the agents each step* gives
$E_{\text{cent}}(0) = 0$ exactly, so the metric measures algorithmic divergence
alone. It is also what `WORKPLAN.md` §4.5 mandates for every learner in a run.

**Also decided.** $E_{\text{agree}}$ and $E_{\text{cent}}$ are logged
**unnormalised**, with $\lVert\bar{\bm\theta}_t\rVert^2$ alongside, so any
normalisation — per-parameter, or relative to the mean's own size — is derivable
at plot time without a re-run. `max_pairwise_distance` is logged too, since
$E_{\text{agree}}$ is a mean and can stay small while one agent drifts far off.

### 🔄 D32. The backward probe is anchored by rotation, not by a step offset

**Decision.** `eval.backward_separation_degrees` (default 15°) replaces
`eval.backward_offset`. The backward set is built at

$$t' = \max\{\,s < t : |\varphi(s)-\varphi(t)| \ge \Delta\varphi\,\}$$

— the *most recent* earlier step far enough away in rotation. Where no such step
exists the probe is **undefined and logged as absent**, never as zero.

**Why.** A fixed step offset degenerates, and did:

| schedule | old (500-step offset) | new (15°) |
|---|---|---|
| linear | 15.0° ✓ | 15.0° ✓ |
| piecewise | **0.0° after $t{=}1000$** | 15.0°, anchored to step 499 |
| sinusoidal | **0.0° at every step** | 15.1° throughout |
| stationary | 0.0°, silently | undefined, reported |

The sinusoidal case is the one that matters: `backward_offset` was 500 and the
period is 500, so $\varphi(t-500) \equiv \varphi(t)$. **The schedule chosen
specifically to expose forgetting had a forgetting probe that measured nothing**,
and would have reported "no forgetting" for the whole run. Piecewise collapsed
once $t$ passed the last change point plus the offset.

Anchoring by rotation also guarantees the probe evaluates a state the model
**actually visited** — a fixed *rotation* offset ($\varphi(t) - 15°$) would not,
asking for $-45°$ at the trough of a $\pm30°$ sine, and forgetting of a
distribution never seen is not forgetting.

This is D3's move applied again: state the physically meaningful quantity and
derive the mechanical one. Config validation now rejects a separation the
schedule cannot reach, so `x1_stationary` asking for the backward set fails at
load with *"the stationary schedule only travels 0.0 degrees"*.

**Guarded by.** `test_the_backward_probe_survives_a_sinusoidal_schedule`,
`test_the_backward_probe_survives_a_piecewise_schedule`,
`test_a_stationary_run_has_no_backward_probe`,
`test_the_backward_state_is_one_the_model_actually_visited`.

**Also decided.** Under `per_node` drift, `current` is built **per agent** at
that agent's own rotation, and `current_mean` is scored alongside. Scoring
everyone at the mean alone would make the per-agent spread conflate "this agent
learned worse" with "this agent is further from the mean rotation"; logging both
separates them.

### ✅ D33. The reference offers three init strategies and two selection rules

**Decision.** `reference.init_strategy` ∈ {`shared_seed` (default),
`independent_seeds`, `warm_start`} and `reference.selection` ∈ {`validation`
(default), `fixed_budget`}, with `epochs` and `validation_size` configurable.
Each combination caches to its own file.

**Why all three.** They differ in what $e^\star(\varphi)$ *means*.
`shared_seed` trains each level independently from a common $\bm\theta_0$, so
$e^\star$ is genuinely "best achievable at this rotation" while the curve stays
smooth in $\varphi$. `independent_seeds` is honest about run-to-run variance but
puts jitter into the subtrahend of the headline gap. `warm_start` is ~3× cheaper
but makes $e^\star(45°)$ depend on having passed through $40°$ — contaminated by
exactly the history the reference exists to be free of. Making the choice
measurable costs a config field.

**Selection never touches the test split.** Under `validation` a slice is held
out of *train*, the best epoch is chosen on it, and test is scored once at the
end. Test error is still recorded per epoch, for inspection only — the code says
so at the point where it would be tempting to use.

**Convergence is reported, not assumed.** At the original 20-epoch budget, **9
of 16 levels selected the final epoch** — the budget, not convergence, decided
where training stopped. The budget is now 100 and `all_converged` states the
outcome. The direction matters: an under-trained $e^\star$ is too high, so every
gap comes out too small, flattering the online methods.

**Two smaller choices.** `Reference.at()` **interpolates** between grid points
(nearest-neighbour on a 5° grid would put a sawtooth into the gap curve at the
scale of the effect being measured) and **raises** outside the grid rather than
extrapolating.

**Measured, not assumed.** The full $[-30°, +45°]$ union is trained rather than
mirroring negatives, and the symmetry $e^\star(-\varphi) = e^\star(+\varphi)$ is
then checked: the largest mismatch is **0.0037**. The cheaper ten-level grid
would have been defensible — but that is now a measurement rather than a hope.

### 🔄 D34. $e^\star$'s run-to-run noise is measured, and the rotation trend is not established

**Decision.** `reference.repeat_seeds()` retrains one rotation several times,
varying only the seed, and `seed_spread()` reports the result against the
binomial floor. Figure 11 carries a $\pm1\sigma$ band from that measurement.

**Why.** The grid has **one draw per point** and, as first built, no stated
uncertainty. A reader looking at the curve cannot tell a rotation effect from
run-to-run noise — and neither could I. `WORKPLAN.md` §5.4 already requires five
seeds and a band for every *experiment*; the quantity all of those experiments
are measured against had one run and no band.

**Measured at $0°$, five runs, identical but for the seed:**

| | |
|---|---|
| mean $e^\star$ | 0.0455 |
| std across seeds | 0.00163 |
| range | 0.0043 |
| binomial floor $\sqrt{e(1-e)/n}$ on 10k | 0.00208 |
| std across the 16 rotations | 0.00222 |

**What this shows, and what it does not.** The seed noise alone is nearly the
size of the whole grid's variation, so most of the curve's shape is not a
rotation effect. A $\chi^2$ test on whether the grid's variance exceeds the seed
variance gives $27.8$ on 15 df against a $0.05$ threshold of $25.0$ — marginally
over. But $\sigma_{\text{seed}}$ comes from five runs, and its own 95% interval
is $[0.00106, 0.00469]$, which contains the grid's std. **The test is not
conclusive either way.**

The honest statement is therefore: *most of the variation across the rotation
grid is run-to-run noise; whether a small genuine rotation effect remains cannot
be settled from one run per grid point.*

**Correction.** An earlier reading of this data — that the difference between
$e^\star(-30°)$ and $e^\star(+5°)$ "is not real" — was stated too strongly, on a
back-of-envelope range argument before the noise was measured. The measurement
is more ambiguous than that claim. Recorded here rather than quietly softened,
because the direction of the error matters: overstating "this is noise" is how a
real effect gets dismissed.

**What would settle it.** Five seeds per grid point: 80 trainings, ~100 minutes.
Worth doing before any figure asserts a rotation trend, and not needed for the
gap itself — which uses $e^\star$ pointwise, where a $\pm0.0016$ uncertainty is
small against the gaps phase 3 will measure.

---

## 2026-08-05 — Phase 3, the learners

### ✅ D35. X0 tests every *linear* update rule, not just plain SGD

**Finding.** The exactness identity survives heavy-ball momentum, mixed or not.
It breaks for AdamW. Measured, 30 steps, complete graph, float64:

| optimizer | mixing | residual | |
|---|---|---|---|
| plain SGD | — | 9.99e-16 | exact |
| momentum $\beta=0.9$ | mixed | 7.22e-16 | exact |
| momentum $\beta=0.9$ | **not** mixed | 7.77e-16 | exact |
| AdamW | all | 5.24 | **breaks** |

**Why.** Averaging commutes with *linear* maps. Heavy-ball is linear in the
gradients — $\bm m \leftarrow \beta\bm m + \bm g$, $\bm\theta \leftarrow
\bm\theta - \eta\bm m$ — so

$$\tfrac1N\textstyle\sum_v \bm m_v = \beta\,\tfrac1N\sum_v \bm m_v^{\text{old}} + \tfrac1N\sum_v \bm g_v$$

is exactly the centralized momentum recursion. On a complete graph every agent
evaluates its gradient at the same point, so the *average* trajectory matches
centralized whether or not the buffers are exchanged. Adam's second moment
carries $\bm g^2$, which is not linear, and the identity fails immediately.

**Three consequences.**

*X0 is stronger than advertised.* It certifies the diffusion algebra for the
whole class of linear update rules, not only for the configuration it runs in.

*A claim in this codebase was wrong.* The precondition check's message said
optimizer state "makes the two trajectories diverge legitimately". False for
heavy-ball. Corrected: plain SGD is required as the **canonical** configuration
so the check leans on nothing but the diffusion algebra — not because momentum
would break it.

*The check is now known not to be vacuous.* `test_adamw_does_break_the_identity`
is a positive control: without a case that fails, a test that always passes is
indistinguishable from one that checks nothing. Three other controls exist —
float32 exceeds the tolerance, a ring breaks by $>10^{-6}$, and unequal batch
sizes break it.

**The numbers in that table are snapshots; the conclusion is not.** Re-measured
at the tuned lr 0.01 the exact rows read 9.99e-16, 1.33e-15 and 9.99e-16 and
AdamW reads 0.76 — float64 accumulation moves by a factor of two, and AdamW's
breakage scales with the step size. `learners.md` §3 carries the current values
and says so; D41 records the test that broke by asserting one of them literally.
The unmixed row also stopped being constructible when D36 added the guard, and is
now reproduced by relaxing the field after validation.

**Note the limit of the result.** This holds on a *complete* graph, where every
agent linearises at the same $\bm\theta$. It says nothing about a ring, where
the agents differ and $\nabla L_v$ is evaluated at different points — and
nothing about the D-Adam divergence the plan cites, which is an
adaptive-optimizer phenomenon over many steps on a sparse graph.

### ✅ D36. Momentum mixing treats the whole learner state as one object

**Decision.** `mix_optimizer_state: momentum` averages $\bm m$ with the *same*
weights as $\bm\theta$, in one exchange of $(\bm\psi, \bm m)$ costing $2p$ per
link.

**Why.** Olshevskyi et al. (Fig. 2a): **D-Adam**, which mixes parameters and
keeps moments local, converges then *diverges*; **D-AMSGrad**, which runs
consensus on the moments too, is their best distributed method. `WORKPLAN.md`
§3.4 concludes unmixed adaptive state is a known failure mode rather than an
open question, and the config rejects it.

Structurally, mixing makes combine a single operator on the whole state:

$$\begin{pmatrix}\bm\theta_v \\ \bm m_v\end{pmatrix} \leftarrow \sum_u a_{vu}\begin{pmatrix}\bm\psi_u \\ \bm m_u\end{pmatrix}$$

so every property established for $\bm A$ covers all of it — row-stochasticity
keeps the result inside the neighbours' convex hull, double stochasticity
preserves the network average. Unmixed, $\bm\theta$ gets those guarantees and
$\bm m$, the part that diverges, gets none.

**What was not a reason.** An earlier draft justified this by analogy to the
filter, which "treats its state uniformly". That is wrong: Diff-EKF's default
(eq. 44) mixes the **mean only** and keeps the covariance local; covariance
combining is explicitly optional at $O(p^2)$ per link. The analogy was dropped.

**Cost, and why it is affordable.** $2p$ per link means the primary baseline
sends twice what Diff-EKF does. `diffusion_sgd_atc_plain` carries no state and
therefore sends $p$, so X1 runs both and the phase-5 comparison has a
payload-matched baseline as well as a stronger one (D29).

### ✅ D37. The combine step reads all messages before writing any

**Decision.** `_combine_states` stacks every $\bm\psi_u$, applies the weight
matrix once, and only then writes the results back.

**Why.** The obvious loop — update agent 0, then agent 1, … — would let agent 1
combine agent 0's *already-updated* parameters. The result would depend on node
ordering, and on a complete graph it would break the X0 identity while still
producing a plausible curve. Stacking makes the step a genuine matrix product,
which is also what the algebra says it is.

**Related.** `init()` clones $\bm\theta_0$ per agent rather than sharing one
tensor. Sharing would make the first in-place update change every agent at once,
and the run would show perfect consensus for a reason unconnected to the combine
step.

---

### ✅ D38. Resumption is exact, and guarded by the config fingerprint

**Decision.** Re-running an experiment resumes from the last completed
evaluation step rather than starting over. The checkpoint stores the learner
states, the ledger and the last completed step; it is refused if the config
fingerprint changed.

**Why it is exact and not approximate.** The run loop consumes no randomness.
The environment is *positional* — agent $v$'s samples at step $t$ are a function
of $(v, t)$ and the shard, computed by cumulative-sum offsets rather than by
advancing a cursor — and the stream, partition and graph were all drawn at
construction time. So there is no RNG state to save and restore, which is the
usual thing resumption gets wrong. `test_recording.py` asserts a resumed run
matches an uninterrupted one bit-for-bit rather than assuming it.

This is a payoff from the positional-stream decision (D18) that was not the
reason for making it.

**Why the fingerprint guard.** A resumed run with a changed config is not the
run it claims to continue, and the parquet would mix two experiments under one
name with nothing recording the seam. The failure is silent and permanent — the
result looks like one clean run.

The guard is sharper than it first appears because $\alpha$ is *derived*
(`total_degrees / horizon`, D14). Shortening the horizon to simulate an
interruption does not truncate the run — it changes the drift rate, so step 100
of the short run carries different data than step 100 of the long one. My first
version of the resumption test did exactly that and read the refusal as a false
positive; the guard was right. `simulate.run` therefore takes a `stop_after`
argument, which interrupts a run **without reconfiguring it**. It is deliberately
a function parameter and not a config field: a config field would be part of the
fingerprint and would change the very thing it is meant to hold fixed.

**Cost.** A checkpoint write per evaluation step, and the states must be
serialisable — which they are, being tensors in a dict. Writes are atomic
(temp file, then rename) so a crash *during* the checkpoint cannot leave a
truncated file that fails to load or, worse, loads with partial state.

---

### ✅ D39. Every method is tuned before any comparison is drawn

**Decision.** Learning rate, $n$, and whether momentum is used at all are chosen
per method on a held-out grid (`scripts/sweep_hyperparameters.py`) rather than
shared by assumption. X1–X6 then run at the selected values.

**What forced it.** The first full X1 run, at the planned primary (SGD momentum
0.9, lr 0.05, $n=2$):

| learner | optimizer | held-out error |
|---|---|---|
| `diffusion_sgd_atc_plain` | plain SGD | **0.095** |
| `centralized_sgd` | momentum | 0.117 |
| `diffusion_sgd_atc` | momentum | 0.146 |
| `local_only` | momentum | 0.897 |

Two things are wrong here. `local_only` sits at chance for ten classes, and the
only method *without* momentum finishes ahead of `centralized_sgd` — which pools
every agent's samples each step and therefore cannot legitimately be beaten by a
distributed method.

**The cause.** Momentum 0.9 at lr 0.05 gives an effective step
$\eta/(1-\beta) = 0.5$, which is unstable at a batch of $n=2$. One agent, 1500
steps, varying only the optimizer:

| lr | momentum | $n$ | held-out error |
|---|---|---|---|
| 0.05 | 0.9 | 2 | **0.898** |
| 0.05 | 0.0 | 2 | 0.194 |
| 0.01 | 0.9 | 2 | 0.356 |
| 0.005 | 0.9 | 2 | 0.188 |
| 0.05 | 0.9 | 20 | 0.188 |

It is not divergence — $\|\bm\theta\|^2$ stays comparable to the other methods —
but the model settles into a near-uniform output, mean confidence 0.154 against
a floor of 0.1. Averaging over $N=10$ agents cuts the gradient noise like a
tenfold batch increase, which is exactly why the diffusion methods survive the
same setting and `local_only` does not.

**Why this could not be reported as a result.** "Cooperation is essential" would
have been the headline of F1, and the mechanism is real — averaging *is* variance
reduction. But at a tuned lr the same lone agent reaches 0.188 against ATC's
0.18, so nearly the whole gap is an optimizer artefact rather than a learning
benefit. Reporting it would not survive the first reviewer who asks whether the
baseline was tuned.

**How the grid is shaped.** The shard budget $NnT \le 60000$ caps $n$ at 4 when
$T = 1500$, so the sweep runs at $T = 600$, where $n = 10$ lands at exactly
60 000 and every cell stays epoch-free. Enabling `allow_epochs` for the large-$n$
cells instead would let them train on repeated data while $n = 2$ did not,
biasing the sweep toward the axis being measured. $n$ is swept at fixed $T$
rather than fixed $nT$: $n$ is how fast an agent samples and $T$ is the horizon,
so the question is "does sampling faster help at a fixed number of rounds?".

Selection is on the **held-out** set, not the prequential stream, because
prequential error is what the tuning gets reported against and choosing on it
would select for the noise in that estimate.

**Related.** `diffusion_sgd_atc` under `optimizer: sgd` *is*
`diffusion_sgd_atc_plain`, so the optimizer axis subsumes that learner and the
sweep carries three rather than four.

---

### ✅ D40. ATC's advantage over CTA is robustness, not accuracy

**Measured.** A full grid (5 lr $\times$ 5 $n$ $\times$ 2 optimizers $\times$ 2
seeds) with ATC and CTA in the *same* cells, so every comparison is on identical
data.

ATC wins **47 of 50** cells — the sign is not chance. But at each ordering's own
optimum the difference is an order of magnitude below seed noise:

| $n$ | ATC seeds | CTA seeds | difference | ATC seed spread |
|---|---|---|---|---|
| 2 | 0.1025, 0.1146 | 0.1045, 0.1162 | +0.0018 | **0.0121** |
| 4 | 0.0952, 0.0976 | 0.0959, 0.0988 | +0.0010 | **0.0023** |
| 6 | 0.0905, 0.0946 | 0.0910, 0.0952 | +0.0006 | **0.0041** |
| 8 | 0.0876, 0.0917 | 0.0877, 0.0919 | +0.0002 | **0.0041** |

Where ATC actually separates is the *unstable* region:

| optimizer | lr | $n$ | ATC | CTA | CTA − ATC |
|---|---|---|---|---|---|
| momentum | 0.2 | 10 | 0.118 | 0.251 | **+0.133** |
| momentum | 0.2 | 8 | 0.169 | 0.295 | +0.127 |
| sgd | 0.2 | 4 | 0.113 | 0.140 | +0.027 |

**Interpretation.** ATC averages *after* stepping, so the combine step damps a
too-large update. CTA averages first and steps afterwards, so the damping
arrives before the step it would have absorbed. That makes ATC tolerant of step
size rather than better at the right one.

**What this changes.** `WORKPLAN.md` §3.2 justified ATC as primary partly on
"ATC generally has the better mean-square performance [3], so the baseline would
be handicapped". For *this* benchmark that is too strong — at tuned settings the
two are indistinguishable. ATC stays primary because **Diff-EKF is ATC**, so
matching the ordering removes a confound from the phase-5 comparison. That reason
stands on its own and does not depend on CTA being worse.

**What it was run to check, and the answer.** Whether CTA's optimum sits
somewhere ATC's does not — which would make F8 report a tuning difference as an
ordering difference. It does not: both select momentum at lr 0.01 for
$n \in \{2,4,6,8\}$, diverging only at $n=10$. So F8 runs both at matched
settings as designed, and that is now measured rather than assumed.

---

### ✅ D41. Changing $n$ broke eleven tests, and that was the tests working

**What happened.** Raising the default $n$ from 2 to 4 (D39, WORKPLAN §3.7)
failed 10 tests and errored 10 more. None was a defect in the change; every one
was a test encoding a *consequence* of $n=2$ as a literal.

| what broke | why |
|---|---|
| `obs.x.shape == (2, 1, 14, 14)` | the batch shape is $n$ |
| pooled union `== 20` | the pooled batch is $Nn$ |
| "consumes exactly half the training set", `== 30_000` | at $n=4$ it consumes **all** of it |
| `test_evaluation` fixtures | a hardcoded 4000-sample synthetic split no longer covers $NnT$ |
| AdamW positive control `residual > 1.0` | see below |

**The fix is not to bump the literals.** Each assertion now derives its expected
value from the config — `config.env.samples_per_node_per_step`,
$N \times n$, $N n T$ — so the test states the *invariant* rather than a
snapshot of one configuration. The evaluation fixture sizes its synthetic split
from $NnT$ for the same reason.

**The AdamW control deserves its own note.** It asserted `residual > 1.0`, chosen
when lr was 0.05 and the residual was 5.24. At the tuned lr 0.01 the residual is
0.76 — AdamW still breaks the identity by fourteen orders of magnitude against
the 1e-12 tolerance, but the literal bound failed. A threshold tied to a *tuned*
quantity tracks the tuning rather than the property. It now reads
`residual > 1e6 * TOLERANCE`, stated relative to what the identity is checked at.

**Worth recording because the same trap is still live.** Any test that encodes a
default rather than an invariant will break at the next tuning pass, and the
tempting fix — editing the number until it passes — destroys the test. The
$n=4$/60 000 coincidence is the sharpest case: `test_stream` now asserts the
default run consumes the training split *exactly*, with nothing spare, which is
the assertion that would catch a change silently pushing the run into reuse.

---

### ✅ D42. Atomic rename is retried, because it is not reliably atomic on Windows

**Decision.** `_replace_with_retry` wraps every `staging.replace(target)` in the
recorder — both the parquet write and the checkpoint — retrying on
`PermissionError` with a short backoff, about 3 s of total patience.

**What happened.** A re-run of X2 died at its first checkpoint with

```
PermissionError: [WinError 5] Access is denied:
  'results\x2_rotating\seed_0.checkpoint.tmp' -> '...\seed_0.checkpoint.pt'
```

`os.replace` is genuinely atomic on POSIX. On Windows it fails outright when the
target is held open by **any** process — an antivirus scanner, a search indexer,
a cloud-sync client, or a handle the OS has not finished reaping from a killed
run. The write was complete and correct; the rename simply could not land at that
instant.

**Why it mattered more than it looks.** The atomic-write pattern exists so a
crash cannot leave a truncated file (D38). Here the safety mechanism *was* the
crash: a condition that clears in milliseconds killed a 15-minute experiment, and
because the driver was a shell `for` loop, it carried on to the next experiment
and left the results out of order — X5 complete while X2 sat at one seed.

**The retry is deliberately narrow.** Only `PermissionError`, only for a few
seconds, and the original error is re-raised after the last attempt rather than
swallowed — a genuinely read-only directory raises the same class and must still
fail. Both branches are tested: `test_a_transient_lock_on_the_target_is_retried`
injects a lock that clears on the third try,
`test_a_permanent_permission_error_still_raises` injects one that never clears.

**Related.** This is a Windows-specific hazard the project will keep meeting,
since the whole benchmark runs there and `results/` sits under a path that sync
clients watch.

---

### ✅ D43. CUDA is measurably slower for phases 1–4, and essential for phase 5

**Measured**, on an RTX 4070 Laptop against 10 CPU threads:

| workload | batch | CPU | CUDA | |
|---|---|---|---|---|
| one agent's gradient | 4 | 1.34 ms | 1.93 ms | **0.69×** |
| one agent's gradient | 40 | 1.58 ms | 2.04 ms | **0.77×** |
| one agent's gradient | 400 | 2.16 ms | 1.61 ms | 1.34× |
| wider model, hidden 512 | 512 | 4.33 ms | 1.68 ms | 2.57× |
| reference trainer, one epoch | 128 | 0.57 s | 0.62 s | **0.92×** |
| **dense $p\times p$ matmul** | — | **120 ms** | **8.6 ms** | **14×** |

**Why.** The model is 2 908 parameters on a $14\times14$ input, and the runs use
$n=4$ per agent, 40 pooled. At that size kernel-launch overhead exceeds the
arithmetic, and there is nothing for 36 SMs to do. The crossover is around batch
400 — an order of magnitude above anything phases 1–4 use. Even the longest CPU
job in the project, the 20-minute reference trainer, comes out slower.

**Phase 5 inverts this completely.** A dense covariance is $2908^2 = 8.5$M
entries per agent, and the EKF update needs several $p \times p$ products per
agent per step. Across 10 agents and 1500 steps, 120 ms versus 8.6 ms is the
difference between a run measured in days and one measured in minutes. CUDA is
not an optimisation for Diff-EKF — it is the assumption the $p=2908$ budget was
chosen under (WORKPLAN §4.6).

**What was done about it.** `run.device` was *validated but never used*: no code
path moved a tensor to it, so `device: cuda` would have been accepted and
silently ignored. A config field that lies is worse than one that does not exist,
so it now raises, and the message says where CUDA does pay rather than only
saying no. **Phase 5 removes the guard when it wires the device through.**

**The general point.** "Use the GPU where it helps" is a measurement, not a
default. Here the intuition was wrong in both directions — slower where it was
expected to help, and decisive in a place that had not been benchmarked.

### ✅ D44. A derived quantity whose sign is fixed by construction is a test

Two figures compute a difference whose sign cannot vary if the computation is
right, and both produced the impossible sign before anyone noticed — because an
impossible value renders as an unremarkable cell rather than as an error.

**F6b's penalty** is $e(\text{headline lr}) - \min_{\text{lr}} e(\text{lr})$ over
a grid that *contains* the headline. A minimum over a set cannot exceed a member
of it, so the penalty is $\ge 0$ by construction. It reached $-0.042$, from two
independent causes:

1. **lr 0.2 was missing from the sweep grid**, so for a method whose headline
   sat there the "headline" term came from one place and the minimum from
   another.
2. **The two terms were different estimators** — a five-seed X4 run error minus
   a two-seed sweep minimum. Seed noise alone then makes negatives routine.

The fix for (2) is the interesting one: the five-seed numbers are *better*, and
are still what every reported table quotes. They are simply not what a
*difference* can be built from, because a difference needs both sides measured
the same way. `x4_headline`'s docstring says exactly this so the "improvement"
of switching it back to the five-seed runs is not made twice.

**F6a's payload cost** is $e(\text{payload-matched}) - e(\text{ATC})$ at matched
tuning. The payload-matched variant *is* ATC minus momentum, so at each one's own
optimum the cost is $\ge 0$: the constrained optimum cannot beat the
unconstrained one. It came out **exactly 0.000 in all twelve cells** — which
looks like a clean null result and is a definition being overridden. Both names
map to one class, the sweep sets the optimizer for every learner it runs, so
unconstrained the variant picked momentum and became numerically identical to
ATC. Carrying no optimizer state is precisely what makes its message $p$ per link
rather than $2p$, so it is now tuned within the plain-SGD arm only.

**What was done about it.** Two tests in `tests/test_figures.py` assert the
signs. The meeting-document builders — which live outside the repo, see D46 —
additionally raise rather than printing a negative penalty into a
supervisor-facing document, and now import `x4_tuned`/`x4_headline` from
`make_figures` instead of rebuilding the tables, because one of them had
independently reproduced bug (2). **The tracked guard is the test**; the
builders' check is a second line of defence on a document nobody can review
before it is sent.

**The general point.** Every derived quantity in this project should be asked
"what values can this not take?" before it is plotted. Where the answer is
non-empty it is a test, and the test is cheap. Neither of these was caught by
looking at the figure.

### ✅ D45. The second $p$ scalars pay under sparsity, not under heterogeneity

With the payload cost finally well-defined, it has a shape. Across the tuned X4
plane it runs 0.007–0.046, rising toward the sparse corner: about **2.5x** along
each axis in the marginals (0.0118 → 0.0304 as $\pi_\text{lab}$ falls 1.0 →
0.25; 0.0125 → 0.0292 as $n$ falls 8 → 1). Across X6's three decades of label
skew it is **flat** — 0.015, 0.013, 0.015.

Both X4 axes control how much signal one step carries; skew does not. Momentum
accumulates a consistent direction out of noisy gradients, so the extra $p$
scalars are worth most where each gradient is worst, and worth nothing extra
when the gradients are merely *different* from a neighbour's.

**Why this matters for phase 5.** Diff-EKF sends a mean, and if it sends no
covariance it is a $p$-per-link method. This bounds what that costs against the
$2p$ baseline at ~0.046 worst case, and locates it: the sparse corner, not the
non-IID regime. It also predicts that a Diff-EKF *would* recover the difference
if its curvature estimate does the job momentum was doing — which is a testable
claim, not a hope.

**A caveat that was tested and withdrawn.** This note originally carried one:
the payload-matched variant's tuned optimum sat at lr 0.2, the largest rate then
swept, in 7 of 12 cells, so the costs were "mildly pessimistic" and its
0.6-decade span was the grid's width rather than the method's. The mechanism
argued for it — momentum's $\eta/(1-\beta) = 10\eta$ means a plain learner needs
roughly 10x the nominal rate for the same effective step, which should put its
optimum at or past the edge.

The grid was extended to **lr 0.5 and 1.0** (192 further cells, 432 per tag).
**Neither rate wins a single cell, for any method.** Every number above is
unchanged to four decimals, so the costs are measured rather than pessimistic,
and the 0.6-decade span survives a grid 0.7 decades wider — the payload-matched
variant genuinely has the narrowest optimum range of the four, which the ceiling
argument had written off as an artefact. Centralized's 1.9-decade span is
likewise no longer a lower bound; it still picks 0.2 twice, but 0.2 is interior
now.

**What went wrong in the reasoning, since the mechanism was right.** The $10\eta$
rule predicts a *band*: ATC's optima run 0.005–0.05, so it points at 0.05–0.5,
and the plain variant's 0.05–0.2 sits inside that band at the low end. The rule
was fine; treating the top of its range as a point prediction, and then treating
"optimum at the boundary" as evidence of truncation rather than as a hypothesis
to test, was the error. **A boundary optimum is a question, not a conclusion** —
and it costs 90 minutes of compute to answer.

### ✅ D46. The meeting-document builders live outside the repo

Four scripts — `make_presentation.py`, `make_summary_docx.py`, and since
2026-08-23 `make_drift_presentation.py` and `make_drift_docx.py` — are untracked
and `.gitignore`d. They stay on disk, rebuilt before every supervisor meeting,
but they are not part of the published benchmark.

**The drift pair is separate rather than appended, and the originals are left
untouched.** The stationary documents build toward "there is no accuracy gap
left to close, so the filter must win somewhere else"; the drift documents *are*
that somewhere else. Two arguments, each better told from its own beginning than
as a seventh section of the other. The new pair imports the old pair's styling,
layout and overflow guard, so they read as a set rather than as two house
styles.

**The criterion.** A file belongs in the repo if it helps someone else *run the
models, check them, or understand the environment*. These two do neither: they
render a three-page .docx and a 25-slide .pptx aimed at one reader, and they
write to a personal OneDrive path. Someone cloning this repository to reproduce
X1 or to add a learner gains nothing from them and has to read past them.

**What went with them.** `tests/test_figures.py` had a slide-overflow test that
imported `make_presentation`; it was removed rather than skipped, because a test
that can only ever skip in CI is noise. The builder still refuses to write an
overflowing deck, which is where that check belongs — it guards a local action.

**What deliberately stayed.** `make_figures.py` (the F1–F10 pipeline, the results
of record) and `make_preliminary_figures.py` — that one renders the environment
illustrations, and `environment.md` §4 points at it as the way to *inspect* what
the agents actually see. It is a checking tool that happens to also produce
slides material.

**The wart this exposed, now fixed.** Both surviving figure scripts wrote to a
hardcoded `C:\Users\alter\OneDrive\...` path — the same criterion failing in a
smaller way, since a fresh clone could not run them without editing a constant,
and a reviewer who did run them would find the output nowhere they thought to
look. `utils/paths.py` now resolves it: the default is repo-relative `figures/`
(gitignored), and publishing elsewhere is `DEKF_FIGURES_DIR` rather than an edit.

Two details worth keeping. The override is read on **every call** rather than
cached at import, so a test can point it at `tmp_path` without reloading the
module. And a relative value resolves against the *repository root*, not the
working directory, so `DEKF_FIGURES_DIR=figures` means one place whether the
script is launched from the IDE, the repo root, or `scripts/`.

The untracked builders import the same helper, which is why the deck and the
.docx keep finding the PNGs `make_figures.py` wrote: one variable moves all four
scripts together, and nothing can half-move.

---

## 2026-08-14 — Phase 4b, benchmarks for tracking under drift

### ✅ D47. Rate is the drift axis, not displacement — so the schedule accelerates

**The question posed.** "After how many steps do the existing algorithms break,
and how much does the distribution have to change?"

**Why those are not two questions, and why the second is nearly closed.**
Rotation is capped at `MAX_WELL_POSED_DEGREES = 45`, enforced against what a
schedule *travels* rather than what it configures. Past that a 6 is a 9, the
Bayes error itself moves, and a rising curve would measure label ambiguity —
precisely the artifact the question is trying to avoid. The default
`total_degrees` is already 45, so the displacement axis is at its ceiling.

Worse, under a *constant* rate a tracking learner reaches a steady-state lag and
then stops degrading. Run it longer at the same speed and nothing new happens,
so "after how many steps" has no non-trivial answer either: an apparent break at
large $t$ is the transient not having finished, or it is the 6↔9 degeneracy.

What remains, and what actually governs tracking, is the **rate** $\alpha$ in
degrees per step. `rate_at` is now defined for every schedule as a difference
rather than a derivative — it is what a learner experiences per update, it needs
no special case for `piecewise`, and it is the quantity a break threshold is a
threshold *on*.

**The bind, and how the ramp escapes it.** The cap makes rate and duration trade
off: $\alpha T \le 45$. A constant-rate sweep therefore has to shorten the run
to go faster, and the run must stay long enough to clear the transient. Holding
$T \ge 500$ leaves $\alpha \le 0.09$ — a factor of three over the current
default, which is very unlikely to contain a break.

A `ramp` breaks the bind, because the cap constrains the *mean* rate while the
peak is `exponent` times the mean:

| $T$ | linear | ramp $p{=}2$ | ramp $p{=}4$ | ramp $p{=}6$ |
|---|---|---|---|---|
| 1500 | 0.030 | 0.060 | 0.120 | 0.180 |
| 750 | 0.060 | 0.120 | 0.240 | 0.359 |
| 500 | 0.090 | 0.180 | 0.359 | 0.537 |

At full length, with no compromise on run duration and ending at exactly 45°, a
$p = 6$ ramp reaches six times the default rate. **This revises a claim made
earlier in the same discussion** — that rotation could reach only a factor of
three and therefore probably could not break anything. That was true of
constant-rate runs and false in general, because it reasoned about mean rate as
if it were peak rate.

**Progress became the primitive.** `DriftSchedule` now defines
`progress_at(step)`, normalised so 1.0 is fully travelled, and rotation is
`degrees_scale * progress`. One schedule therefore drives every channel
coherently, and D48's channel needed no new schedule.

**Read a ramp's answer as an upper bound.** The learner lags, so it crosses
threshold slightly after the rate that would break it in steady state. The ramp
is the cheap instrument that says where to look; constant-rate `linear` runs
bracketing the located value are what confirm it.

### ✅ D48. The class-prior channel exists because rotation has a ceiling and label shift does not

**Decision.** A second drift channel: each agent's class distribution travels
from $1/K$ to a Dirichlet draw along the schedule's progress. Off by default.

**Why.** Everything in D47 is downstream of one fact — rotation is bounded by
label ambiguity at 45°. Label shift has no analogous limit. The Bayes-optimal
classifier moves the whole way, but no label ever becomes ambiguous, so the
distribution can travel as far as total variation allows and the gap to the
reference stays interpretable throughout. It also stresses a different part of
the model than rotation does, which makes the pair a stronger benchmark than
either alone.

**The implementation turned out to be a permutation.** The obvious design —
sample a class per step from $\bm q_v(t)$ — would make the stream stateful, and
`stream.py` is deliberately a *pure function of (agent, step)* so that step 900
can be answered without walking steps 0..899. Instead the whole $(N, T, n)$
table of classes is drawn up front, exactly as the label-availability mask
already is. Then:

* the **partition** is sized from the plan's `demand()`, so a shard cannot run
  dry part-way through a run;
* the **stream** orders each shard so that consuming it front-to-back realises
  the plan.

The second point is the one worth keeping. Prior drift reaches the learner as a
*serving order*, so offsets stay a prefix sum and exactly-once is untouched —
each index is still popped from its class queue exactly once. `Stream` itself
gained no new machinery.

**Feasibility is a property of the plan, and is checked before a run starts.**
The shards are disjoint, so the agents compete for one finite pool per class. A
plan that oversubscribes a class has no valid partition at all — there is
nothing to degrade into — so `check_plan_is_feasible` refuses it and names the
class, rather than letting a shard come up short at step 1400.

**A wart found by measuring rather than by reasoning.** The first version filled
each shard to its target size by taking the whole shortfall from whichever class
pool was largest. The served composition was correct, but `partition.skew()`
read 0.545 — reporting a skew the experiment did not have, entirely from filler
that is never served. Apportioning the filler across classes drops it to 0.128,
the irreducible part that comes from the demand itself. The lesson is the
recurring one: a derived diagnostic can be wrong while the thing it describes is
right, and only looking at it catches that.

**Only the labelled steps draw from the plan.** With $\pi_\text{lab} < 1$ an
agent idles and consumes nothing, so taking its plan entry anyway would slide
every later class one step out of step with the drift schedule that chose it.

### ✅ D49. Two break definitions, kept separate because they disagree

**Decision.** A break is recorded under both an absolute and a comparative
definition, and neither is reduced to the other.

*Absolute*: the tracking gap $e_v(t) - e^\star(\varphi_v(t))$ stays above a
threshold. *Comparative*: the learner stops beating `frozen_atc` — the same
algorithm, same optimizer, same data, which adapted for a warmup and then
stopped.

**Why both, rather than picking one.** They answer different questions and can
disagree in both directions, and each disagreement is informative:

| absolute | comparative | what it means |
|---|---|---|
| broke | did not | adaptation is helping and still not keeping up |
| did not | broke | drift is slow enough that standing still was fine; the online updates add more variance than they remove |

One number cannot say both. The comparative definition also needs **no
threshold**, which is its real advantage — the absolute one requires someone to
choose 0.05, and that choice is not measurable.

**The rate is the answer; the step is how it was found.** Every located step is
converted immediately to `schedule.rate_at(step)`. Under a constant rate the
step is nearly meaningless (D47), and under a ramp the instantaneous rate at the
crossing is not the run's average — reporting the step alone would invite
reading it as though it were.

**Three things the locator refuses to do.**

*Call a single excursion a break.* One evaluation over the line is sampling
noise, so the condition must hold for three consecutive evaluations. There is a
positive control asserting that with `persistence=1` the excursion **is**
located, so the guard is known to be doing work rather than passing vacuously.

*Report the end of the run when nothing broke.* `step=None` survives as an
outcome, carried alongside `max_rate_probed` so "did not break" is interpretable
— without the rate actually reached, a null result says nothing.

*Compare across runs.* The frozen baseline rides inside the same experiment as
the learners it is compared with, so it shares one environment and one
$\bm\theta_0$ (D4) and the comparison is paired by construction. A missing
baseline raises rather than falling back to a separate run.

**The frozen baseline stops transmitting, not only stepping.** Averaging
identical estimates is a numerical no-op but would still be *counted* by the
ledger, and a baseline paying bandwidth to change nothing would distort every
plot against cost. `comm_scalars_per_step` is asked once per step, so it reports
the warmup cost honestly and drops to zero only once the learner has stopped.

**`centralized_sgd` cannot be frozen**, and this is rejected rather than
silently ignored: it adapts through `adapt_pooled()`, which the runner calls
without a step, so the freeze point could not be honoured.

### ✅ D50. The absolute threshold is derived from a paired control, not chosen

**The plan was** to threshold the tracking gap $e(t) - e^\star(\varphi_t)$ at
some multiple of the seed noise. Measuring first showed that would not work.

**What the existing runs say.** On x2 the gap does not grow with drift — it
*shrinks*, because the learner is still converging:

| learner | gap early | gap late | drift cost (x2 − x1, late) |
|---|---|---|---|
| centralized_sgd | 0.072 | 0.044 | 0.015 |
| diffusion_sgd_atc | 0.076 | 0.046 | 0.015 |
| diffusion_sgd_atc_plain | 0.104 | 0.066 | 0.022 |
| local_only | 0.190 | 0.119 | 0.029 |

Two things follow. A threshold on the raw gap would fire at step 0 for every
method, for reasons that have nothing to do with tracking — the gap is mostly
the price of learning online from a stream rather than offline from the whole
split. And the *whole* drift effect at $\alpha = 0.03$ (0.015 to 0.029) is at or
below $3\sigma$ of the seed noise (0.025 to 0.049), which is a useful
calibration fact in its own right: **nothing is close to breaking at the rate
the benchmark has been running at.**

**The fix is a paired stationary control**, `x9_control`: identical seeds,
horizon, cadence and learners, with the drift switched off. The break is then
measured on $e_\text{drift}(t) - e_\text{control}(t)$, paired by seed *and* by
step, so the convergence trend and the online-versus-offline penalty cancel
exactly. $e^\star$ drops out of the subtraction, so this definition needs no
reference table at all.

**The noise is the s.e.m., not the s.d.** The quantity thresholded is the
five-seed mean, so the relevant spread is that of the mean: $3\,\text{sem} =
0.016$ against $3\sigma = 0.035$. The s.d. answers a different and more
conservative question — whether any single run would show it — and is available
via `statistic="sd"` for when that is what is wanted.

**Estimated from the quiet opening of the run.** A $p = 6$ ramp has barely moved
over its first quarter, so the excess there is zero by construction and its
spread is noise and nothing else. Using a window from the same run keeps the
estimate matched to the evaluation-set size and learner set actually in play,
and it is not circular: early-window noise does not depend on whether the run
breaks later.

**The control must match exactly, and mismatch raises.** Unpaired rows would be
dropped silently by the merge and the excess would then be computed over a
different population than it claims, so a missing seed, step or learner is an
error rather than a smaller intersection.

### ✅ D52. Erdős–Rényi(0.3) becomes the standard topology, and x9/X11 stay on ring

**Decision.** `erdos_renyi` with $p = 0.3$ is the default from 2026-08-17.
x9 and X11 keep their ring, deliberately.

**It is a choice about realism, not one that moves results.** X3 ran both with
everything else held fixed:

| topology | centralized | ATC | ATC − centralized |
|---|---|---|---|
| ring | 0.0754 | 0.0768 | 0.0014 |
| **erdos_renyi (0.3)** | 0.0754 | **0.0759** | 0.0005 |
| path | 0.0754 | 0.0789 | 0.0035 |
| star | 0.0754 | 0.0831 | 0.0077 |

ATC differs by **0.0009** between ring and ER — about a fifth of the seed-noise
floor. They are close because at $N=10$ their average degrees are similar (2.7
against 2). The topologies that do move results are star and path.

**Why the finished runs stay on ring.** x9 and X11 are a *matched pair*: the
whole point of `x11_control` is to compare repeated abrupt shifts against smooth
drift at matched average speed, and that only means something if the graph is
the same on both sides. Moving one would confound the comparison with topology;
moving both costs ~47 hours to shift numbers by less than the noise.

**Two topologies now live in the repo**, which is a real cost. It is paid down
by `tests/test_topology_provenance.py`, which pins each experiment's graph — so
a change to `base.yaml` cannot quietly move a finished experiment onto a
different one and invalidate a comparison already drawn from it. X0's complete
graph is pinned there too: the identity needs one combine step to reach full
consensus, and a default that moved it would turn the highest-value test in the
suite into a test of nothing.

**Three things measuring turned up that reasoning had not.**

*The density cannot live in `base.yaml`.* `params` merges key by key, so a
`p: 0.3` there follows every other topology around — a grid2d config reported
`{p: 0.3, rows: 5, cols: 2}` and misdescribed itself. It belongs in
`configs/graph/erdos_renyi.yaml`, which the experiments include; a config that
selects `erdos_renyi` without it fails loudly asking for `p`.

*Seeds now vary in topology as well as data.* The mixing gap over the five run
seeds spans 0.043 to 0.197, a 4.6× spread, and 8% of 200 draws fall below 0.06
— worse than half the ring's 0.127. This matters less than it looks: the graph
is derived from the seed, so in `paired_excess` the drifting run and its control
share it and the variance cancels. It would only bite in unpaired cross-seed
comparisons.

*Disconnected draws are already handled.* `_erdos_renyi` defaults
`ensure_connected=True` and retries, which is why all five seeds came out
connected rather than by luck.

**X8 needed a new twin.** Its config claimed the comparison was against X2,
which was true on ring and false the moment X8 moved to ER — the difference
would have been scope *and* topology. `x8_global` is now that twin: same
topology, seeds, horizon, cadence and learners, differing only in
`drift_scope`.

### ✅ D53. A control must match the *mean* of what the treatment does

**The bug.** X8 compares per-node drift against global drift. Under `per_node`
the multipliers run over $[0.5, 1.0]$ and average 0.75, so the agents end at
22.5°–45.0° with a mean of 33.75. The global twin was set to 45 — the treatment's
*maximum* — so it drifted 33% further on average, and X8 measured heterogeneity
and drift amount at once.

**What caught it.** `frozen_atc` came out **0.10 better** under per-node drift.
It stops adapting, so its error tracks pure displacement; no account of
heterogeneity explains that and "less drift" explains it exactly. The result had
looked plausible — the cooperation gap appeared to shrink by 0.0106, in the
predicted direction — and would have been reported as the headline.

**The lesson.** Matching the maximum felt natural and was wrong. The control has
to match whatever the treatment's own agents *average* to, since that is the
quantity the treatment realises. With the twin at 33.75° the spread is the only
difference, and `frozen_atc`'s damage falls to +0.0050.

**A non-adapting baseline earns its place in every run.** It is the only learner
whose error is a direct read-out of displacement, so it is the one that catches a
control which does not match. Nothing else in the table looked wrong.

### ✅ D54. The change in a cooperation gap is a paired difference

**Decision.** Report $\text{damage}(\texttt{local\_only}) -
\text{damage}(\texttt{diffusion\_sgd\_atc})$, estimated as a per-seed paired
difference, rather than differencing two independently-computed gaps.

**Why it matters, measured.** They are the same quantity and their standard
errors differ by a factor of seven on X8: 3 s.e.m. of **0.0013** paired against
**0.0110** unpaired. The unpaired estimate called a real −0.0070 shrinkage noise.

The paired version is correct because the two runs share a seed, hence the same
graph and the same $\bm\theta_0$; the subtraction cancels variation the unpaired
comparison leaves in. Both are printed, the weaker one labelled as such, because
the size of the discrepancy is itself worth seeing.

### ✅ D55. What the drift benchmarks found

The four axes, and what each says for phase 5.

**Smooth drift breaks everything at 0.038–0.044°/step** (X9), only 1.3–1.5× the
rate the benchmark has been running at. The methods sit far closer to each other
than any does to a frozen baseline, and none ever loses to it.

**Abrupt shifts are mostly *cheaper* than smooth drift at matched average speed**
— 0.43–0.47× across three speeds (X11 against X12). Between shifts the
distribution is exactly stationary, so the learner settles; smooth drift never
allows it. It is not the average speed that hurts, it is the continuous motion.

**Longer intervals produce bigger wounds**, monotone across the grid: the learner
specialises harder the longer it sits, so the next shift costs more. That is the
pressure a filter's covariance should relieve, and it is why `recurring` is the
phase-5 benchmark.

**Cooperation's value moves in opposite directions on the two axes.** Under
heterogeneous drift the gap narrows 10% and everything that merges information
pays about 0.007 while `local_only` pays exactly nothing (X8). Under label shift
the gap widens 59% and `local_only` is hurt 3.5× more than anyone else (X10).
The filter's cooperation story is therefore strongest under label shift and
weakest under heterogeneous drift.

**Two things remain out of reach in this design.** The X11 crossover at $J = 30$
rests on one shift and cannot be firmed up, because a longer matched window needs
a longer smooth run and the 45° cap forbids one. And the `recovered` column is
confounded along $J$, since the cap forces reflection and larger jumps reach
fewer states — 11, 6 and 3 rotations at $J = 5, 15, 30$.

## 2026-08-25 — Phase 5, the centralised filter

### ✅ D56. The centralised EKF has two variants, not four

**The request was four**: the γ and λ state models, each with "sending only the
mean" and "sending mean and covariance". The last axis does not exist here.
Equations 45–46 are the *combine* step of the **diffusion** filter; a
centralised filter has one processor and one belief and sends nothing. So the
centralised filter has only the state-model axis, and the four variants appear
when the combine step does.

**γ = 1 is not a third model.** It gives $\bm F=\bm I$ and
$\bm P\leftarrow\bm P+\bm Q$ — the random walk — so it is the boundary of the γ
grid, and `transition: identity` is already how the config expresses it. Putting
it *in* the sweep is worth doing, because it separates the γ family's two
effects and yields two clean comparisons: γ=1 against γ<1 isolates whether
shrinking the mean helps, and γ=1 against λ compares additive with multiplicative
loosening **on equal hyperparameter budgets**.

That last point dissolves a concern about fairness. γ<1 needs three knobs
(σ₀, γ, **Q**) against λ's two (σ₀, λ), but not because the comparison is
rigged — it is asking a strictly additional question. At γ=1 the budgets match.

**The γ family requires $\bm Q\succ\bm 0$**, which follows from the algebra
rather than from taste: γ² *contracts* the covariance, so with $\bm Q=\bm 0$ the
filter's confidence grows monotonically and it stops learning. Only one of
λ<1 and $\bm Q\succ\bm 0$ may be active at a time; both together are
unidentifiable.

### ✅ D57. A step's samples are stacked, not applied sequentially

**Decision.** The $n=4$ samples an agent receives at step $t$ enter as one
update, $\bar{\bm H}$ of shape $(nK)\times p$ with block-diagonal
$\bar{\bm\Lambda}$ — equation 34 applied within an agent.

**Why not four sequential rank-9 updates.** They are not the same operation:
sequential updates relinearise between samples, which would give the filter four
linearisation points per step where every SGD baseline gets one. The filter
would then look better partly because it was granted a larger budget of the very
resource under comparison. Stacking keeps the per-step budget identical and
matches what `centralized_sgd` does with its pooled batch.

### ✅ D58. float64 on CUDA, and D43's guard comes off

**Decision.** The filter runs in float64 on CUDA, and `run.device` accepts
`cuda`.

D43 measured CUDA at **0.69×** on the SGD path — slower, because $p=2908$ cannot
amortise a kernel launch — and **14×** on dense covariance operations, then
closed the device to `cpu` with the note that phase 5 would reopen it when the
dense covariance made it a win. This is that moment, and the guard coming off is
the plan working rather than a plan changing.

float64 costs 68 MB per covariance and buys a clean failure mode: the paper
reports positive definiteness lost within a few hundred steps in single
precision, against runs of 1500. In float64, with the Joseph form and per-step
symmetrisation, a PD failure means a real bug rather than accumulated rounding —
which is worth more than the memory.

### ✅ D59. Woodbury is required, not an optimisation

$\bm\Lambda=\operatorname{diag}(\bm\pi)-\bm\pi\bm\pi^\top$ has rank at most
$K-1=9$, so stacked over ten agents and four samples the information increment
has rank 360 against $p=2908$. Inverting directly is $O(p^3)\approx2.5\times
10^{10}$ flops per step — roughly an hour of inversion per seed, before any
sweep. Woodbury inverts $360\times360$ instead for the same answer.

Recorded as a design note rather than left as an implementation detail because
"the filter is too slow to sweep" would otherwise look like a property of the
method rather than of one avoidable choice.

### ✅ D60. The mean update takes the score, not the innovation

The filter's mean update is $\bm m^+ = \bm m^- + \bm P^+\sum_v\bm H_v^\top\bm s_v$
where $\bm s = \partial\log p/\partial\bm h$. Under softmax the logits *are* the
natural parameter, so $\bm s = \bm y-\bm\pi = \bm\nu$ and the two words name one
vector. Under a Gaussian the logits are the *mean* parameter and
$\bm s = \bm R^{-1}(\bm y-\bm h) = \bm\nu/\sigma^2$.

Using $\bm\nu$ throughout is therefore correct for every experiment X0–X6 and
wrong by $\sigma^{-2}$ the moment a regression likelihood appears. Gradient
training cannot detect the difference — a constant factor on the gradient is
absorbed by the learning rate — but the filter can, because $\bm P$ carries
absolute units and has no step size to absorb it into.

`score()` is now on the likelihood protocol, `innovation()` keeps its own
meaning, and `loss_gradient` takes the score too (a no-op under softmax, a
correction under a Gaussian, where it previously computed the gradient of a
different loss than `nll` reported).

**This is what the Gaussian likelihood was kept for.** `gaussian.py` argued in
its own docstring that an interface with one implementation is shaped around
that implementation whether or not anyone intended it, and that untested code
the filter will one day depend on is the kind that turns out to be wrong. The
linear-Gaussian exactness test found the mean off by a factor of $\sigma^{-2}$
on the first run, with the covariance already correct to $4\times10^{-15}$ —
a discrepancy the categorical likelihood is structurally incapable of exposing.

The two quantities are tied by $\bm\Lambda = \operatorname{Cov}(\bm s)$, which
holds in both families and is now tested by sampling.

### ✅ D61. $\sigma_0^2$ is a trust region, and too large a value diverges

The EKF mean update is a Gauss–Newton step whose trust region is $\bm P$ itself.
At $\sigma_0^2=1$ and $p=2908$, the **first** step moves $\lVert\bm m\rVert$ by
7.9 against $\lVert\bm\theta_0\rVert=6.1$ — more than doubling the parameter
norm in one jump. That lands far outside the region the linearisation describes,
so the next linearisation is taken somewhere worse, and the run reaches
$10^{113}$ by step 100 with the covariance still perfectly well conditioned
(condition number $2.2\times10^3$). Measured on the same data at smaller priors:

| $\sigma_0^2$ | first step $\lVert\Delta\bm m\rVert$ | relative to $\lVert\bm\theta_0\rVert$ | outcome |
|---|---|---|---|
| 1.0   | 7.91  | 1.29  | diverges by step 25 |
| 0.1   | 1.51  | 0.25  | stable, NLL flat |
| 0.01  | 0.26  | 0.042 | stable |
| 0.001 | 0.041 | 0.007 | stable |

This is the method behaving as derived, not a defect, and it decides two things.
The sweep runs over $\sigma_0^2\le10^{-1}$, since larger values do not fail
gracefully — they fail catastrophically. And the filter carries an $O(p)$
per-step guard that raises on a non-finite mean or a non-positive variance, so a
diverged cell is reported as diverged rather than reaching the metrics as NaN
and averaging into a seed mean.

The useful diagnostic is the **first step's relative jump**
$\lVert\Delta\bm m\rVert/\lVert\bm\theta_0\rVert$: it is computable before
committing to a run, and it separates the stable settings from the divergent one
cleanly at this model size.

### ✅ D62. The Joseph form is the expensive half of a false choice

D58 committed to "float64, Joseph form, per-step symmetrisation" as the three
numerical defences. The Joseph form does not survive contact with D59.

Written the usual way it forms $\bm I-\bm K\bar{\bm B}^\top$ at $p\times p$ and
multiplies it by $\bm P$, which is $O(p^3)$ — the cost Woodbury exists to avoid.
Expanding the product keeps every term $O(p^2q')$, but the expansion telescopes
back to $\bm P-\bm A\bm S^{-1}\bm A^\top$: the short form. The two are
algebraically the same matrix, and the Joseph form's entire value is the
numerical behaviour of the *un-expanded* product, which is the $O(p^3)$ one.

So the real choice is Joseph **or** Woodbury, not Joseph over the short form.
Woodbury wins: float64 plus symmetrisation plus the D61 guard, with positive
definiteness soaked over a full 1500-step run rather than assumed. Worth
recording because "we used the Joseph form for stability" is the kind of claim
that reads as obviously correct and would have quietly cost an order of
magnitude, or been silently untrue.

### ✅ D63. Both predictive approximations are reported, because the gap is the finding

The filter's posterior is Gaussian over logits; the metrics need a distribution
over classes. The integral $\int\operatorname{softmax}(\bm h)\,\mathcal N(\bm h;
\bm h(\bm m),\bm\Sigma)\,\mathrm d\bm h$ has no closed form, and two
approximations are standard:

* **Probit**, $\operatorname{softmax}\bigl(\bm h/\sqrt{1+\tfrac\pi8\bm\sigma^2}\bigr)$
  — free, deterministic, uses only $\operatorname{diag}\bm\Sigma$.
* **Monte Carlo** over the full $\bm\Sigma$ — keeps the correlations the probit
  form discards, costs $S$ softmaxes and a sampling seed.

Reporting both, rather than picking one, because the distance between them is
measurable and turns out to be **approximation error rather than sampling
noise**. At $S=200{,}000$ two seeds differ by $\le0.002$ while the probit form
sits this far from the sampled answer:

| $\sigma^2$ | MC noise | probit gap |
|---|---|---|
| 0.05 | 0.0003 | 0.0021 |
| 0.25 | 0.0006 | 0.0093 |
| 1.0  | 0.0011 | 0.0246 |
| 4.0  | 0.0016 | 0.0404 |
| 16.0 | 0.0019 | 0.0418 |

The error grows with the spread and then saturates, because both forms tend to
uniform. So the probit form carries the headline metric — it is free and the
distortion is bounded — and sampling runs on a subset of eval steps as the check
that keeps the bound honest. If the filter's calibration advantage over SGD ever
turns out to be the same size as the row above it, that is a result about the
approximation and not about the filter.

**Sampling uses a symmetric eigendecomposition, not a Cholesky.** $\bm H\bm P\bm
H^{\mathsf T}$ is positive semi-definite, not definite — nothing makes $\bm H$'s
rows independent — and Cholesky requires strict definiteness. Rescuing it by
jittering the diagonal injects spread the belief does not contain, which is
visible at an exactly-zero covariance as a prediction that is not quite the
plug-in one ($\sim10^{-6}$). At $q=10$ the eigendecomposition is free and exact
for singular and zero inputs alike, so there is one path and no fallback.

### ✅ D64. X13 pilot: the filter wins, and $Q$ is the knob

80 cells, 2 seeds, linear drift at $\alpha=0.025$ deg/step over $T=1500$. **No
cell diverged**, so the $\sigma_0^2\le10^{-1}$ bound from D61 was conservative.
Seed noise, taken as the median $|{\rm seed}_0-{\rm seed}_1|$ across all 80
cells, is **0.0021**, so a two-seed mean carries about $\pm0.0011$.

| | settled error |
|---|---|
| best EKF ($\sigma_0^2=0.1$, $\gamma=0.999$, $Q=10^{-4}$) | **0.0620** ±0.0007 |
| `centralized_sgd` | 0.0948 ±0.0003 |
| `diffusion_sgd_atc` | 0.0970 ±0.0007 |
| `frozen_atc` | 0.4101 ±0.0138 |

0.033 against the centralized learner, at roughly 30× the noise. **Provisional**,
because the baselines ran at the shipped `lr 0.01`, which was tuned on stationary
data in X1 — see the re-tune below.

**The axes are wildly unequal.** Best cell per level:

| axis | profile | span | verdict |
|---|---|---|---|
| $Q$ | 1e-6: 0.0825 → 1e-5: 0.0679 → **1e-4: 0.0620** → 1e-3: 0.0852 | 0.0232 | dominant |
| $\lambda$ | 0.9999: 0.0911 → 0.999: 0.0815 → **0.997: 0.0694** → 0.995: 0.0698 | 0.0217 | dominant |
| $\gamma$ | 1: 0.0650 → 0.9999: 0.0644 → 0.9995: 0.0626 → **0.999: 0.0620** | 0.0030 | marginal |
| $\sigma_0^2$ | 0.003: 0.0626 → 0.01: 0.0635 → 0.03: 0.0629 → 0.1: 0.0620 | 0.0016 | below noise |

All seven cells statistically tied with the best have $Q=10^{-4}$, spanning the
entire $\sigma_0^2$ range and two $\gamma$ values. So 64 of the 80 cells went to
the family whose two extra axes are worth 0.0030 and nothing, while the axis
worth 0.0232 got four points three decades apart.

$\gamma$ is marginal but probably real: at $Q=10^{-4}$, $\gamma=0.999$ beat
$\gamma=1$ in **all four** $\sigma_0^2$ blocks (+0.0020, +0.0008, +0.0035,
+0.0055), and the margin grows with $\sigma_0^2$ — consistent with D26's reading
of $\gamma$ as weight decay, since shrinking the mean offsets a looser prior.

**$\sigma_0^2$ is flat for the $\gamma$ family and monotone for the $\lambda$
family**, which is mechanistic rather than incidental. $\bm P\leftarrow\bm
P/\lambda$ is scale-preserving, so $\bm P_0$ never washes out and small is
better (0.0694 → 0.0751 as $\sigma_0^2$ goes 0.003 → 0.1 at $\lambda=0.997$);
additive $Q$ erases $\bm P_0$ within a few hundred steps, so it cannot matter.

**The $\lambda$ family's 0.0074 deficit is not yet a fair result.** Its optimum
was a tie between the two *lowest* $\lambda$ tested with $\sigma_0^2$ pinned at
its low edge, against a $\gamma$ family bracketed on both sides — a tuned method
against a less-tuned one, which is the D39 error in miniature. The refined grid
gives it its own downward-extended prior axis before the comparison is reported.

### ✅ D65. The baselines are re-tuned per drift condition, not once

`lr 0.01` was selected on 2026-08-05 against **stationary** data (X1). X13 then
compared a filter tuned under drift against a baseline that was not, which is
exactly what D39 exists to prevent — and a 0.033 headline deserves a baseline
that had its best step size for the condition it is being beaten in.

The project already does this elsewhere: X3 re-tunes the learning rate per
topology, X4 per cell. Doing it per drift condition is the same convention, not
a new one.

`run_ekf_sweep.py --full` therefore **refuses to start** until the re-tune has
run, rather than silently producing the unfair comparison. `frozen_atc` takes
ATC's re-tuned rate too: it is ATC stopped at step 300, so a different rate would
make it a different algorithm rather than the same one frozen, and that is the
whole basis of the comparative break (the X9 bug where the two disagreed on `lr`
is the precedent). Note that `frozen_atc`'s *own* best rate is 0.05, and it is
deliberately not given it — that would tune the baseline into a different
algorithm to make it look better at being frozen.

**Outcome: the shipped rate was already right, and the drift does not want a
faster one.** Settled error under the same drift, two seeds:

| lr | `centralized_sgd` | `diffusion_sgd_atc` | `frozen_atc` |
|---|---|---|---|
| 0.01 | 0.0948 | 0.0970 | 0.4101 |
| 0.02 | **0.0941** | **0.0968** | 0.4083 |
| 0.05 | 0.1089 | 0.1204 | 0.4041 |
| 0.1 | 0.1360 | 0.1845 | 0.4971 |
| 0.2 | 0.7034 | 0.7393 | 0.7937 |

0.02 wins by +0.0007 and +0.0002 against seed spreads of 0.0027 and 0.0056, so
the two are **indistinguishable**; everything larger is much worse, by 7.5× at
lr 0.2. That the drift optimum barely moved from the stationary one is itself a
reading of the regime: at $\alpha=0.025$ the baselines are not step-size limited,
which fits damage of 0.0126 against a base error near 0.09.

The full pass uses 0.02 anyway. It is the argmin of a noisy curve — a
winner's-curse selection rather than a measured optimum — but the bias makes the
*baseline* look better, which is the safe direction, and it costs 0.0007 to make
the comparison unarguable. The headline moves from +0.0328 to **+0.0321** against
the centralized learner and from +0.0350 to **+0.0348** against ATC.

### ✅ D66. What X13's break analysis can actually use

Checked against the pilot data *before* buying the fine evaluation cadence for
the full pass, since the 1.57× premium is paid specifically for break and
recovery resolution.

**Pooling across directories is sound.** An X13 cell holds only its EKF; the
baselines live in `x13_baselines`. The two runs share a seed set, a horizon and a
cadence, so their step grids are identical and `error_by_step` on the
concatenation yields all four learners. `assert_paired_runs` is not the right
guard here — it exists for drift/control pairs and would correctly reject a
cell against the baselines, whose learner lists differ by design.

**The comparative break is vacuous at this rate.** Every learner returns
`step=None` against `frozen_atc`, because the frozen baseline settles at 0.4101
against the filter's 0.0620 and is never competitive — nothing ever falls behind
it, so "when does the learner stop beating a frozen one" has no answer here. That
is a property of $\alpha=0.025$ over 1500 steps, not a defect: D49's two
definitions were introduced precisely because either can be uninformative in a
given regime. So X13 rests on the **absolute** definition — damage against a
paired stationary control — which makes the controls load-bearing rather than
confirmatory, and is why they run at all.

**`persistence` is meaningless without a cadence.** `DEFAULT_PERSISTENCE = 3` is
documented as 75 steps, which it is *at `eval_every` 25*. X13's tuning pass runs
at 25 and its measurement pass at 5, where the same three points span 15 steps —
a five-fold weaker bar on the same axis, invisible at the call site. Since the
thing being defended against is a noise excursion in *time*, the window is the
invariant and the point count is derived: `persistence_for(eval_every)` now does
that conversion, and the default is left untouched so no published number moves.

Worth recording as its own note because it is the same failure as D50's threshold
and D53's control: a quantity that looks like a constant but is really a ratio
against something the config can change.

### ✅ D67. X14 generalises one tuned setting; the conditions are picked by damage

X13 tunes at linear $\alpha=0.025$, which measurement shows is among the
**mildest** conditions the project has. Damage across everything already run:

| abrupt | | linear | |
|---|---|---|---|
| every25 jump30 | **0.0677** | $\alpha=0.15$ ($T{=}299$) | 0.0303 |
| every25 jump15 | 0.0518 | $\alpha=0.10$ ($T{=}449$) | 0.0257 |
| every50 jump15 | 0.0498 | $\alpha=0.05$ ($T{=}899$) | 0.0196 |
| every200 jump30 | 0.0398 | $\alpha=0.025$ ($T{=}1499$) | **0.0126** |
| every100 jump15 | 0.0183 | | |
| every200 jump5 | 0.0054 | | |

So the tuning condition sits 5× below the worst abrupt cell, and a claim resting
on it alone would be a claim about slow smooth drift.

**One setting per family, tested across regimes — not tuned per condition.**
Re-tuning each cell would answer "can the filter be tuned to win anywhere", which
is both weaker and less useful than "one setting tracks across regimes"; the
latter is what the diffusion version needs, since a Diff-EKF cannot re-tune
itself per drift regime in deployment. The settings are *read* from X13's grid at
run time rather than hard-coded, so the two experiments cannot drift apart.

**The faster linear rates are excluded, and not for being mild.** A constant rate
reaches the 45° cap at $T=45/\alpha$, so $\alpha=0.15$ runs 299 steps and its
damage mixes drift with "had less time to converge". The tell is already in the
table: at $\alpha=0.15$ the *frozen* baseline's damage equals ATC's exactly
(0.0303), because `freeze_after` is 300 and the run ends at 299 — it never froze,
so the row describes one algorithm twice. Recurring shifts reflect at the cap
(D51) and therefore sustain any rate for the full 1500 steps, which is why all
eight X14 conditions share a horizon and a single stationary control.

**The baselines are re-tuned once, on the worst condition.** D65 established
per-condition re-tuning, but doing it for all seven would cost more than the
experiment. `every25 jump30` is the strongest case for a fast step the project
has, so a rate that does not want to move there will not want to move at 0.0054
damage either — and if the X13 choice survives, one number covers the set.

---

## Open questions

### ❓ Q1. Network size $N$

Is $N = 10$ the target, or should the benchmark reach 50–100? Bears on the
topology sweep and, via D5, on the feasible horizon. Decide before phase 4.

### ❓ Q2. Non-IID in scope for the first paper?

X6 and the Dirichlet axis are built either way; this is a question about what
gets written up. Decide before phase 4.

### 🔄 Q3. Per-node drift — now a first-class axis, still an open question

Is the interesting story "all agents drift together" (a shared $\theta$ stays
correct) or "agents drift differently" (a shared $\theta$ becomes wrong,
motivating the hierarchical shared/local extension)? `drift_scope` is
configurable so this can be answered empirically.

**No longer just configurable: `x8_per_node_drift` exists** and is the paired
comparison against X2, which is the same run with `drift_scope: global` and
nothing else changed. The prediction is that diffusion's advantage over
`local_only` shrinks or reverses, because under global drift the neighbours have
adapted to the *same* state and mixing is pure variance reduction, whereas under
per-node drift mixing also drags each agent toward a state that is not its own.
The question stays open until that run says so.

### ✅ D51. Repeated abrupt shifts: the SGD baseline is measured *before* the filter

X5 measures the transient after **one** 15° jump (F7). `recurring` repeats it
every $t'$ steps, so the learner never settles and the run measures recovery
over and over. X11 sweeps $t' \in \{25, 50, 100, 200\}$ against $J \in \{5, 15,
30\}$, five seeds.

**Why measure it now rather than alongside the filter.** Gradient methods
recover only as fast as a step size allows, and that step size is tuned for the
stationary regime — so shifts arriving faster than the recovery time should
compound into a standing error. A filter can respond faster in principle,
because its covariance says how much to trust new evidence rather than applying
a fixed gain. That makes this the regime where Diff-EKF should show a real
advantage, which is precisely why the SGD numbers must exist *first*: measuring
them after the filter exists would invite tuning one against the other.

**The grid is two-dimensional because $J/t'$ is only the average speed.** The
same average can arrive as rare-large or frequent-small shifts, and those are
different problems — one asks whether recovery finishes before the next shift,
the other whether many small perturbations accumulate. Matched-speed cells are
the ones to read against each other: $(t'{=}50, J{=}15)$ and $(t'{=}100,
J{=}30)$ both average 0.300°/step, confirmed measured.

**Reflected, not clipped, at the band edge.** The rotation must stay inside
$\pm45°$, so repeated jumps necessarily revisit states. What is controlled
instead is that every jump has *exactly* magnitude $J$ — clipping at the edge
would shorten those jumps and make their transients incomparable with the rest,
destroying the one property the schedule exists to provide. The direction is
randomised so a learner cannot pre-position, and `jump_seed` is separate from
the run seeds so the pattern can be held while the data varies.

**Evaluation runs at `eval_every: 5`.** A transient is a few tens of steps wide,
so at $t' = 25$ the default cadence of 25 would sample roughly once per shift —
measuring the standing error and missing the recovery entirely.

Raised by the user, 2026-08-15, and brought forward from phase 5 at their
request.

### ❓ Q4. CI environment

GitHub Actions has no GPU, so CI must install CPU torch from a different index
than the dev machine uses. Acceptable, but it means CI is not bit-identical to
local runs and the difference should be deliberate. Also unresolved: whether to
run mypy on 3.11 in CI to genuinely enforce the stated floor, since locally it
is pinned to 3.13 to avoid a `scipy-stubs` parse error.

### ❓ Q5. Diff-EKF `adapt_scope`: `local` or `one_hop`?

**Decide before phase 5 starts.** In diffusion SGD the adapt step is
unambiguous — agent $v$ takes a gradient on its own batch, and using a
neighbour's batch would mean shipping data, which the setting forbids. The EKF
has a genuine second option, because its measurement update consumes
$(\bm H, \bm R, \bm y)$ rather than raw samples, and a neighbour can send those
without sending an image.

**`local`.** Agent $v$ updates on its own measurement only, then combines:

$$\bm\theta_v^+ = \bm\theta_v + \bm K_v(\bm y_v - h(\bm\theta_v)),
\qquad \bm\theta_v \leftarrow \sum_u a_{uv}\bm\theta_u^+$$

**`one_hop`.** Agent $v$ additionally assimilates its neighbours' measurements,
stacking $\{(\bm H_u, \bm R_u, \bm y_u)\}_{u \in \mathcal N_v}$ (or applying them
sequentially) before combining.

| | `local` | `one_hop` |
|---|---|---|
| Matches the SGD baselines | **Yes** — same information per step, so F1/F3/F6 comparisons are like-for-like | No — it sees $\deg(v)+1$ batches to ATC's one, so a win is confounded with seeing more data |
| Message size | $p$ (mean), or $p + \tfrac{p(p+1)}2$ with covariance | $+\ \deg(v) \cdot (mp + m^2 + m)$ for the Jacobian, noise and residual, $m$ = measurement dim |
| Effective batch under sparse labels | Same $\eta \cdot n_\text{active}/N$ scaling as ATC (`results.md` §9.3) | Larger, and **exactly where ATC's automatic scaling helps** — likely its strongest regime |
| Non-IID ($\beta = 0.1$) | Relies on the combine to spread class information | Gets neighbours' classes directly in one step, so consensus is not the only channel |
| Convergence theory | Standard diffusion-EKF results apply | Closer to a distributed/consensus EKF; correlated innovations across agents complicate the covariance bookkeeping |
| Implementation | Straightforward | Needs a second message type in the ledger and a Jacobian that is meaningful to a *neighbour's* parameters |
| Risk | Might underperform for a boring reason (too little information per step) | Might outperform for a boring reason (too much) |

**The confound is the crux.** Q1 of the workplan is "what does the filter buy
over backpropagation?", and `one_hop` changes two things at once — the estimator
*and* the information per step — so a positive result would not answer it.

**Provisional recommendation: `local` for the headline, `one_hop` as an
ablation.** `local` keeps every existing figure a valid comparison. Then run
`one_hop` on X4 and X6 only, where the mechanism above predicts it should help
most, and report it as "what an extra hop of measurement sharing buys" — a
separate, well-posed question rather than a contaminated headline. Cost is one
extra sweep, which §10.1c already accepted.

**Open sub-question if `one_hop` is chosen as the headline instead:** the
communication ledger needs a fair matched baseline, presumably an SGD variant
that also exchanges $\deg(v)$ gradients per step. That baseline does not exist
yet and would have to be built and tuned.
