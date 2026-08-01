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

## Open questions

### ❓ Q1. Network size $N$

Is $N = 10$ the target, or should the benchmark reach 50–100? Bears on the
topology sweep and, via D5, on the feasible horizon. Decide before phase 4.

### ❓ Q2. Non-IID in scope for the first paper?

X6 and the Dirichlet axis are built either way; this is a question about what
gets written up. Decide before phase 4.

### ❓ Q3. Per-node drift

Is the interesting story "all agents drift together" (a shared $\theta$ stays
correct) or "agents drift differently" (a shared $\theta$ becomes wrong,
motivating the hierarchical shared/local extension)? `drift_scope` is
configurable so this can be answered empirically. Decide before phase 4.

### ❓ Q4. CI environment

GitHub Actions has no GPU, so CI must install CPU torch from a different index
than the dev machine uses. Acceptable, but it means CI is not bit-identical to
local runs and the difference should be deliberate. Also unresolved: whether to
run mypy on 3.11 in CI to genuinely enforce the stated floor, since locally it
is pinned to 3.13 to avoid a `scipy-stubs` parse error.
