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
gives two distinct failures. A zero diagonal in $\boldsymbol A$ makes an agent discard
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
$1 - \lVert\boldsymbol A - \tfrac1N\mathbf1\mathbf1^{\mathsf T}\rVert_2$ but **raises**
unless the weights are doubly stochastic. A new `mixing_gap` $= 1 - \text{SLEM}$
is always defined and equals the spectral gap whenever both are.

**Why.** The $\tfrac1N\mathbf1\mathbf1^{\mathsf T}$ term is the projector onto
the consensus direction only when the all-ones vector is a *left* eigenvector as
well as a right one — that is, only for doubly stochastic $\boldsymbol A$. Metropolis
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
preference $\boldsymbol q_v \sim \mathrm{Dir}(\beta\boldsymbol 1_K)$ and is then filled to
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
$\boldsymbol\nu = \boldsymbol y - \boldsymbol\mu$; with no $\boldsymbol y$ there is no measurement, and
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

### ✅ D26. $\boldsymbol F$ and the forgetting rule are two axes, and both are selectable

**Decision.** The phase-5 state model exposes two independent choices:

| Field | Values | Acts on |
|---|---|---|
| `transition` | `identity`, `scalar` ($\boldsymbol F_t = \gamma\boldsymbol I$) | the **mean** |
| `forgetting` | `lambda` ($\boldsymbol P \mathbin{{*}{=}} \lambda^{-1}$), `process_noise` ($\boldsymbol P \mathbin{{+}{=}} \boldsymbol Q$) | the **covariance** |

Defaults `identity` + `lambda`. All four combinations are legal, so which state
model performs better is measured rather than assumed.

**The conflation this exists to prevent.** $\gamma$ is routinely called a
"forgetting factor", but propagating the moments gives
$\boldsymbol P_{t|t-1} = \gamma^2\boldsymbol P_{t-1|t-1} + \boldsymbol Q_t$ — and $\gamma^2 \le 1$
**contracts** the covariance. Forgetting means *loosening* the prior so a new
sample counts for relatively more, so $\gamma$ works against it. What $\gamma$
actually does is $\boldsymbol m_{t|t-1} = \gamma\boldsymbol m_{t-1|t-1}$: it pulls the estimate
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
stationary prior $\tfrac{q}{1-\gamma^2}\boldsymbol I$, where a random walk's prior
variance grows without bound.

**Why `lambda` is the default.** With $\boldsymbol F=\boldsymbol I$, multiplicative inflation in
the information domain is $\boldsymbol\Omega_{t|t-1} = \lambda\boldsymbol\Omega_{t-1|t-1}$ —
*exactly* structure-preserving, where $(\boldsymbol\Omega^{-1}+\boldsymbol Q)^{-1}$ is dense even
for diagonal $\boldsymbol Q$. That is decisive once the covariance is structured.

**Why `process_noise` is nonetheless offered.** The objection above is stated in
the *information* domain. While the covariance is carried as a dense $\boldsymbol P$ —
which is the whole of phase 5 at $p = 2908$ — both rules are one line and the
same cost, since the measurement update goes through Woodbury either way. $\boldsymbol Q$
also buys anisotropy a scalar $\lambda$ cannot express ("the read-out drifts, the
feature extractor does not"). The cost only appears if the project later adopts
the (S4) diagonal-plus-low-rank structure, which is defined on $\boldsymbol P^{-1}$.

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
see. It is the $\boldsymbol\Lambda\mathbf 1 = \boldsymbol 0$ null direction reappearing in
parameter space, and its variance grows ~90× over a run at $\lambda = 0.997$.

The right *scaling* is $W \propto T$, since $\alpha = \text{total}/T$ — the same
argument as D3. That derivation is **not** built: phase 5 should sweep $\lambda$
against tracking error in X2 first (WORKPLAN §9 calibrates $\alpha$ the same
way), and building the machinery before knowing whether it matters would be
speculative.

**Diagnostics this obliges.** Record $\mathrm{cond}(\boldsymbol P)$ and
$\lambda_{\min}(\boldsymbol\Omega)$ from the first filter run: null-direction growth is
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
therefore broadcasts $(\boldsymbol\psi, \boldsymbol m) = 2p$, while Diff-EKF broadcasts
$\boldsymbol\psi$ alone.

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

### ✅ D31. $E_{\text{cent}}$ shares $\boldsymbol\theta_0$ and runs its own trajectory

**Decision.** The centralized reference for $E_{\text{cent}}$ starts from the
*same* $\boldsymbol\theta_0$ as the agents and runs its own trajectory from $t=0$.

**Why.** The research note asks for an "independently initialised" centralized
run. Read literally — a different $\boldsymbol\theta_0$ — the metric acquires an
irreducible floor that never vanishes even for a perfect method, and there is no
value of $E_{\text{cent}}$ meaning "these coincide". Reading it as *runs its own
trajectory rather than being re-anchored to the agents each step* gives
$E_{\text{cent}}(0) = 0$ exactly, so the metric measures algorithmic divergence
alone. It is also what `WORKPLAN.md` §4.5 mandates for every learner in a run.

**Also decided.** $E_{\text{agree}}$ and $E_{\text{cent}}$ are logged
**unnormalised**, with $\lVert\bar{\boldsymbol\theta}_t\rVert^2$ alongside, so any
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
`shared_seed` trains each level independently from a common $\boldsymbol\theta_0$, so
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
gradients — $\boldsymbol m \leftarrow \beta\boldsymbol m + \boldsymbol g$, $\boldsymbol\theta \leftarrow
\boldsymbol\theta - \eta\boldsymbol m$ — so

$$\tfrac1N\textstyle\sum_v \boldsymbol m_v = \beta\,\tfrac1N\sum_v \boldsymbol m_v^{\text{old}} + \tfrac1N\sum_v \boldsymbol g_v$$

is exactly the centralized momentum recursion. On a complete graph every agent
evaluates its gradient at the same point, so the *average* trajectory matches
centralized whether or not the buffers are exchanged. Adam's second moment
carries $\boldsymbol g^2$, which is not linear, and the identity fails immediately.

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
agent linearises at the same $\boldsymbol\theta$. It says nothing about a ring, where
the agents differ and $\nabla L_v$ is evaluated at different points — and
nothing about the D-Adam divergence the plan cites, which is an
adaptive-optimizer phenomenon over many steps on a sparse graph.

### ✅ D36. Momentum mixing treats the whole learner state as one object

**Decision.** `mix_optimizer_state: momentum` averages $\boldsymbol m$ with the *same*
weights as $\boldsymbol\theta$, in one exchange of $(\boldsymbol\psi, \boldsymbol m)$ costing $2p$ per
link.

**Why.** Olshevskyi et al. (Fig. 2a): **D-Adam**, which mixes parameters and
keeps moments local, converges then *diverges*; **D-AMSGrad**, which runs
consensus on the moments too, is their best distributed method. `WORKPLAN.md`
§3.4 concludes unmixed adaptive state is a known failure mode rather than an
open question, and the config rejects it.

Structurally, mixing makes combine a single operator on the whole state:

$$\begin{pmatrix}\boldsymbol\theta_v \\ \boldsymbol m_v\end{pmatrix} \leftarrow \sum_u a_{vu}\begin{pmatrix}\boldsymbol\psi_u \\ \boldsymbol m_u\end{pmatrix}$$

so every property established for $\boldsymbol A$ covers all of it — row-stochasticity
keeps the result inside the neighbours' convex hull, double stochasticity
preserves the network average. Unmixed, $\boldsymbol\theta$ gets those guarantees and
$\boldsymbol m$, the part that diverges, gets none.

**What was not a reason.** An earlier draft justified this by analogy to the
filter, which "treats its state uniformly". That is wrong: Diff-EKF's default
(eq. 44) mixes the **mean only** and keeps the covariance local; covariance
combining is explicitly optional at $O(p^2)$ per link. The analogy was dropped.

**Cost, and why it is affordable.** $2p$ per link means the primary baseline
sends twice what Diff-EKF does. `diffusion_sgd_atc_plain` carries no state and
therefore sends $p$, so X1 runs both and the phase-5 comparison has a
payload-matched baseline as well as a stronger one (D29).

### ✅ D37. The combine step reads all messages before writing any

**Decision.** `_combine_states` stacks every $\boldsymbol\psi_u$, applies the weight
matrix once, and only then writes the results back.

**Why.** The obvious loop — update agent 0, then agent 1, … — would let agent 1
combine agent 0's *already-updated* parameters. The result would depend on node
ordering, and on a complete graph it would break the X0 identity while still
producing a plausible curve. Stacking makes the step a genuine matrix product,
which is also what the algebra says it is.

**Related.** `init()` clones $\boldsymbol\theta_0$ per agent rather than sharing one
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

It is not divergence — $\|\boldsymbol\theta\|^2$ stays comparable to the other methods —
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
sample a class per step from $\boldsymbol q_v(t)$ — would make the stream stateful, and
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
$\boldsymbol\theta_0$ (D4) and the comparison is paired by construction. A missing
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
graph and the same $\boldsymbol\theta_0$; the subtraction cancels variation the unpaired
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

**γ = 1 is not a third model.** It gives $\boldsymbol F=\boldsymbol I$ and
$\boldsymbol P\leftarrow\boldsymbol P+\boldsymbol Q$ — the random walk — so it is the boundary of the γ
grid, and `transition: identity` is already how the config expresses it. Putting
it *in* the sweep is worth doing, because it separates the γ family's two
effects and yields two clean comparisons: γ=1 against γ<1 isolates whether
shrinking the mean helps, and γ=1 against λ compares additive with multiplicative
loosening **on equal hyperparameter budgets**.

That last point dissolves a concern about fairness. γ<1 needs three knobs
(σ₀, γ, **Q**) against λ's two (σ₀, λ), but not because the comparison is
rigged — it is asking a strictly additional question. At γ=1 the budgets match.

**The γ family requires $\boldsymbol Q\succ\boldsymbol 0$**, which follows from the algebra
rather than from taste: γ² *contracts* the covariance, so with $\boldsymbol Q=\boldsymbol 0$ the
filter's confidence grows monotonically and it stops learning. Only one of
λ<1 and $\boldsymbol Q\succ\boldsymbol 0$ may be active at a time; both together are
unidentifiable.

### ✅ D57. A step's samples are stacked, not applied sequentially

**Decision.** The $n=4$ samples an agent receives at step $t$ enter as one
update, $\bar{\boldsymbol H}$ of shape $(nK)\times p$ with block-diagonal
$\bar{\boldsymbol\Lambda}$ — equation 34 applied within an agent.

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

$\boldsymbol\Lambda=\mathrm{diag}(\boldsymbol\pi)-\boldsymbol\pi\boldsymbol\pi^\top$ has rank at most
$K-1=9$, so stacked over ten agents and four samples the information increment
has rank 360 against $p=2908$. Inverting directly is $O(p^3)\approx2.5\times
10^{10}$ flops per step — roughly an hour of inversion per seed, before any
sweep. Woodbury inverts $360\times360$ instead for the same answer.

Recorded as a design note rather than left as an implementation detail because
"the filter is too slow to sweep" would otherwise look like a property of the
method rather than of one avoidable choice.

### ✅ D60. The mean update takes the score, not the innovation

The filter's mean update is $\boldsymbol m^+ = \boldsymbol m^- + \boldsymbol P^+\sum_v\boldsymbol H_v^\top\boldsymbol s_v$
where $\boldsymbol s = \partial\log p/\partial\boldsymbol h$. Under softmax the logits *are* the
natural parameter, so $\boldsymbol s = \boldsymbol y-\boldsymbol\pi = \boldsymbol\nu$ and the two words name one
vector. Under a Gaussian the logits are the *mean* parameter and
$\boldsymbol s = \boldsymbol R^{-1}(\boldsymbol y-\boldsymbol h) = \boldsymbol\nu/\sigma^2$.

Using $\boldsymbol\nu$ throughout is therefore correct for every experiment X0–X6 and
wrong by $\sigma^{-2}$ the moment a regression likelihood appears. Gradient
training cannot detect the difference — a constant factor on the gradient is
absorbed by the learning rate — but the filter can, because $\boldsymbol P$ carries
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

The two quantities are tied by $\boldsymbol\Lambda = \mathrm{Cov}(\boldsymbol s)$, which
holds in both families and is now tested by sampling.

### ✅ D61. $\sigma_0^2$ is a trust region, and too large a value diverges

The EKF mean update is a Gauss–Newton step whose trust region is $\boldsymbol P$ itself.
At $\sigma_0^2=1$ and $p=2908$, the **first** step moves $\lVert\boldsymbol m\rVert$ by
7.9 against $\lVert\boldsymbol\theta_0\rVert=6.1$ — more than doubling the parameter
norm in one jump. That lands far outside the region the linearisation describes,
so the next linearisation is taken somewhere worse, and the run reaches
$10^{113}$ by step 100 with the covariance still perfectly well conditioned
(condition number $2.2\times10^3$). Measured on the same data at smaller priors:

| $\sigma_0^2$ | first step $\lVert\Delta\boldsymbol m\rVert$ | relative to $\lVert\boldsymbol\theta_0\rVert$ | outcome |
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
$\lVert\Delta\boldsymbol m\rVert/\lVert\boldsymbol\theta_0\rVert$: it is computable before
committing to a run, and it separates the stable settings from the divergent one
cleanly at this model size.

### ✅ D62. The Joseph form is the expensive half of a false choice

D58 committed to "float64, Joseph form, per-step symmetrisation" as the three
numerical defences. The Joseph form does not survive contact with D59.

Written the usual way it forms $\boldsymbol I-\boldsymbol K\bar{\boldsymbol B}^\top$ at $p\times p$ and
multiplies it by $\boldsymbol P$, which is $O(p^3)$ — the cost Woodbury exists to avoid.
Expanding the product keeps every term $O(p^2q')$, but the expansion telescopes
back to $\boldsymbol P-\boldsymbol A\boldsymbol S^{-1}\boldsymbol A^\top$: the short form. The two are
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
over classes. The integral $\int\mathrm{softmax}(\boldsymbol h)\,\mathcal N(\boldsymbol h;
\boldsymbol h(\boldsymbol m),\boldsymbol\Sigma)\,\mathrm d\boldsymbol h$ has no closed form, and two
approximations are standard:

* **Probit**, $\mathrm{softmax}\bigl(\boldsymbol h/\sqrt{1+\tfrac\pi8\boldsymbol\sigma^2}\bigr)$
  — free, deterministic, uses only $\mathrm{diag}\boldsymbol\Sigma$.
* **Monte Carlo** over the full $\boldsymbol\Sigma$ — keeps the correlations the probit
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

**Sampling uses a symmetric eigendecomposition, not a Cholesky.** $\boldsymbol H\boldsymbol P\boldsymbol
H^{\mathsf T}$ is positive semi-definite, not definite — nothing makes $\boldsymbol H$'s
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
family**, which is mechanistic rather than incidental. $\boldsymbol P\leftarrow\boldsymbol
P/\lambda$ is scale-preserving, so $\boldsymbol P_0$ never washes out and small is
better (0.0694 → 0.0751 as $\sigma_0^2$ goes 0.003 → 0.1 at $\lambda=0.997$);
additive $Q$ erases $\boldsymbol P_0$ within a few hundred steps, so it cannot matter.

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

**Two ceilings decide the horizon, and which binds differs by schedule.** Linear
drift reaches the 45° cap at $T=45/\alpha$; every schedule also obeys the shard
budget $NnT\le60000$:

| $\alpha$ | $45/\alpha$ | $T$ at $n{=}4$ | binds |
|---|---|---|---|
| 0.025 | 1800 | 1500 | shard budget |
| **0.030** | **1500** | **1500** | both, exactly |
| 0.050 | 900 | 900 | 45° cap |
| 0.150 | 300 | 300 | 45° cap |

So **$\alpha=0.03$ is the fastest linear rate that still runs a full 1500-step
horizon**, and it is included. The faster rates are still excluded, and not for
being mild: their damage would mix drift with "had less time to converge". The
tell is already in the table above — at $\alpha=0.15$ the *frozen* baseline's
damage equals ATC's exactly (0.0303), because `freeze_after` is 300 and the run
ends at 299, so that row describes one algorithm twice.

**Lowering $n$ cannot rescue a fast linear run**, because above $\alpha\approx
0.04$ the 45° cap binds at every $n$ — monotone drift runs out of *room*, not out
of data. Where $n$ does buy horizon is exactly where the budget is the binding
constraint: recurring shifts, which reflect at the cap (D51) and are therefore
limited only by $NnT$. At $n=2$ the budget allows $T=3000$, which doubles the
shifts a cell delivers (120 at $t'=25$ instead of 60) and halves the data each
step supplies. Both matter: scarce data is the regime where a second-order method
should have most to offer, and X4 already established $n$ as a real axis.

X14 therefore carries a low-$n$ block of two conditions — one smooth, one abrupt,
at the same $n=2$ and $T=3000$, so the contrast between them is confounded by
neither. **Controls are keyed by $(n,T)$** rather than shared globally: a control
at a different horizon or sample count produces perfectly well-formed rows whose
subtraction means nothing, which is precisely what `assert_paired_runs` exists to
refuse.

**The baselines are re-tuned once, on the worst condition.** D65 established
per-condition re-tuning, but doing it for all seven would cost more than the
experiment. `every25 jump30` is the strongest case for a fast step the project
has, so a rate that does not want to move there will not want to move at 0.0054
damage either — and if the X13 choice survives, one number covers the set.

### ✅ D68. The checkpoint wrote one covariance per agent, and cost 3× the run

X13's full pass ran at 157 min/cell against a 35-min estimate. The arithmetic
closed exactly once measured:

| | per seed |
|---|---|
| filter arithmetic (benchmarked) | 7.1 min |
| checkpointing, 300 flushes × 646 MB | **25.9 min** |
| predicted | 33 min |
| **observed** | **31 min** |

`recorder._write_checkpoint` clones each learner's `extras` **per agent**. That
is right for the diffusion learners, whose agents hold genuinely different
parameters, and catastrophic for a centralized one, which by definition holds a
*single* belief every agent reports. At $p=2908$ the covariance is 68 MB, so ten
clones made a 677 MB checkpoint — written at every flush.

`torch.save` already stores shared storage once; the `.clone()` was what defeated
it. Cloning each **distinct** tensor once (keyed on `id`) keeps the protection
against later mutation and leaves the on-disk structure unchanged. Measured:
646 MB → 67.7 MB, 5.17 s/flush → 0.36, and the full pass 158 h → **51 h**.

**Two lessons, and the second is the one that generalises.**

*The benchmark omitted the component that dominated.* `check_cell_cost.py` timed
`simulate.run` **without a recorder**, so it measured 425 s/seed of a 1860 s
reality — it was measuring the part I had thought about. A cost model built from
a benchmark is only as good as the benchmark's fidelity to the real call, and the
cheap way to check that is to compare against one real run before extrapolating
to sixty.

*The pilot could not have caught it.* At `eval_every` 25 there are 60 flushes
rather than 300, so the same defect cost 5.2 min/seed instead of 26 — visible as
"a bit slower than expected", not as a blocker. A staged design that changes two
things between stages (seeds **and** cadence) hides anything that scales with the
one you were not watching.

**A second bug surfaced while fixing it.** `CentralizedEKF.state()` built a fresh
`LearnerState` per call, but `recorder.resume()` restores by *mutating* the object
`state(node)` hands back — so a resumed filter silently kept $\boldsymbol\theta_0$ while
the recorder skipped to the checkpoint step. Every arithmetic test passed, because
the defect only exists across an interruption. The filter now holds one stable
state object. This is the same failure as the sweep harness's poisoned-cell bug
(D66's neighbourhood): partial state plus a resume that looks successful.

Completed runs now delete their checkpoints — a finished run has nothing to
resume from, and 470 files had accumulated to 118 GB.

### ✅ D69. Flushing is quadratic in the horizon, so X14 rate-limits it

`Recorder.flush` rebuilds a frame from **every** accumulated row and rewrites the
**whole** parquet. One flush is therefore $O(\text{rows so far})$ and a run costs
$O(T^2)$ in flushing. Measured at the real row width:

| rows held | one flush |
|---|---|
| 2,000 | 39 ms |
| 10,000 | 209 ms |
| 26,000 | 798 ms |
| 52,000 | 1573 ms |

The original docstring justified whole-file rewriting as costing "less than the
machinery to avoid it", at "~2 MB per seed rewriting 60 times". That reasoning is
sound and **load-bearing on the 60**: at `eval_every` 5 over $T=1500$ it is 300
rewrites of a frame growing to 52k rows, and X14's low-$n$ conditions run
$T=3000$ — 600 rewrites past 100k rows, roughly 15 minutes a seed spent writing.

It also explains why extrapolating a $T=300$ benchmark predicted 50 min/cell
against an observed 59: linear scaling catches 47 s of flush cost where the
quadratic reality is ~236 s. **That is the third time in this phase a short
benchmark understated a superlinear cost** (D68 was the first two), and the
pattern is worth naming: extrapolating a cost model from a shortened run is only
valid for the terms that are linear in what was shortened.

`min_flush_steps` caps the write *rate* without changing what is written. Rows
keep accumulating and `finalize` always writes, so **the parquet is identical** —
tested by running the same rows through recorders at 0 and 40 and comparing every
column. Only `wallclock_s` differs, which is the quantity being changed. On a
synthetic run the recorder cost fell 13.8× (201 writes to 11).

The default stays 0, so every experiment before X14 is untouched. X14 uses 100:
at most 100 steps of work lost to a crash, against ~20× less write cost. X13 is
left alone deliberately — it is mid-run, and restarting it to save ~5 h of a
49 h remainder would risk more than it saves.

### ✅ D70. X13 refined: the filter wins, and $Q$ is a trade rather than a setting

45 cells, 5 seeds, one divergence. Median s.e.m. **0.0009**, so a difference
between two cells must clear $\sqrt2\times0.0009=\mathbf{0.0013}$ before their
order means anything.

| | settled error |
|---|---|
| best EKF ($\sigma_0^2=0.01$, $\gamma=0.9995$, $Q=6\times10^{-5}$) | **0.0630** ±0.0006 |
| `centralized_sgd` (re-tuned `lr 0.02`) | 0.0918 ±0.0015 |
| `diffusion_sgd_atc` | 0.0952 ±0.0018 |
| `frozen_atc` | 0.3976 ±0.0071 |

**+0.0288 against the centralized learner and +0.0322 against ATC**, at 22× and
25× the threshold. The pilot's provisional +0.0328 came down to +0.0288 once the
baselines were re-tuned and both sides ran five seeds — the claim shrank slightly
and got much harder to argue with.

**The optimum is genuinely flat.** Seven cells sit within 0.0013 of the best, the
same count as the pilot found within its looser 0.0021 — five seeds tightened the
floor without resolving the tie. All seven are $\gamma$-family, at
$Q\in\{3\times10^{-5},6\times10^{-5},10^{-4}\}$ and every $\gamma$.

**Best cell per level**, with the 0.0013 threshold:

| axis | profile | span | verdict |
|---|---|---|---|
| $Q$ | 3e-5: 0.0641 → **6e-5: 0.0630** → 1e-4: 0.0637 → 2e-4: 0.0655 → 3e-4: 0.0684 | 0.0054 | real |
| $\lambda$ | 0.998: 0.0729 → 0.997: 0.0694 → **0.996: 0.0683** → 0.995: 0.0690 → 0.993: 0.0772 | 0.0089 | real |
| $\gamma$ | 1: 0.0641 → 0.9995: 0.0630 → 0.999: 0.0635 | 0.0012 | **within noise** |
| $\sigma_0^2$ ($\gamma$) | 0.01: 0.0630 → 0.1: 0.0639 | 0.0009 | **within noise** |
| $\sigma_0^2$ ($\lambda$) | 0.001: 0.0690 → **0.003: 0.0683** → 0.01: 0.0700 | 0.0017 | real |

$Q$ and $\lambda$ both have interior optima, so both families are now bracketed
on their own terms. **$\gamma$ did not survive.** The pilot called it "marginal
but probably real" on a 0.0030 span with the sign consistent across all four
$\sigma_0^2$ blocks; at five seeds the span is 0.0012 against a 0.0013 threshold.
The consistency argument was suggestive and the extra seeds did not confirm it —
which is the honest outcome of having made the prediction explicitly.

The $\sigma_0^2$ asymmetry **did** survive: flat for $\gamma$, real for $\lambda$,
exactly as the mechanism predicts. $\boldsymbol P\leftarrow\boldsymbol P/\lambda$ is
scale-preserving so $\boldsymbol P_0$ never washes out; additive $Q$ erases it.

**The $\lambda$ family loses fairly now.** Given its own bracketed axes it reaches
0.0683 against the $\gamma$ family's 0.0630 — a gap of 0.0054, down from the
pilot's less-well-tuned 0.0074, and still four times the threshold.

**One divergence**: $\lambda=0.993$ at $\sigma_0^2=0.01$, seed 4, step 860. The
grid bracketed the stability edge correctly. It also exposed a misleading error
message — the singular-covariance path blamed `prior_scale`, which at 0.01 was
mid-range and blameless, when the culprit was $\lambda$ inflating $\boldsymbol P$ by 0.7%
a step for 860 steps. The message now reports the realised variance and names
both mechanisms.

**$Q$ is a trade, not a setting** — the finding that changes how the result should
be read. Damage (drift minus its own stationary twin) runs *opposite* to
stationary error across the $Q$ axis:

| $Q$ | stationary | under drift | damage |
|---|---|---|---|
| 3e-5 | 0.0548 | 0.0641 | 0.0093 |
| 6e-5 | 0.0561 | 0.0630 | 0.0069 |
| 1e-4 | 0.0583 | 0.0637 | 0.0054 |
| 2e-4 | 0.0619 | 0.0655 | 0.0035 |

More process noise loosens the covariance, which buys tracking and costs
precision when nothing moves. So "the best cell" depends on which is being
bought, and the answer is a choice rather than a measurement.

Either way the filter is **less damaged than the baselines**, whose damage is
0.0124 (ATC) and 0.0129 (centralized) against the EKF's 0.0035–0.0110. It both
tracks better in absolute terms and loses less to the drift.

### ✅ D71. Where the data does not decide, carry both branches

$\gamma$'s entire span on the refined grid is **0.0012** against a significance
threshold of **0.0013**. The measurement cannot separate the driftless random
walk from a shrinking transition, so selecting the empirical argmin would be
choosing between cells the data does not distinguish — the winner's curse with
extra steps, and it would silently commit the diffusion filter to an extra
hyperparameter on the strength of a noise draw.

Both therefore go forward into X14, and the choice is left to be made on grounds
the numbers do not supply: parsimony, and how many hyperparameters Diff-EKF
should have to carry and re-tune in deployment.

| carried | setting | settled | damage |
|---|---|---|---|
| $\gamma=1$, the random walk | $\sigma_0^2{=}0.01$, $Q{=}3\times10^{-5}$ | 0.0641 ±0.0009 | 0.0093 |
| $\gamma<1$, shrinking | $\sigma_0^2{=}0.01$, $\gamma{=}0.9995$, $Q{=}6\times10^{-5}$ | 0.0630 ±0.0006 | 0.0069 |
| $\lambda$ family | $\sigma_0^2{=}0.003$, $\lambda{=}0.996$ | 0.0683 ±0.0009 | 0.0121 |

**Implementation consequence.** Two $\gamma$-family settings in one run need two
learner names: the config refuses duplicates, loudly, which is how this was
caught rather than by one setting silently overwriting the other in a dict keyed
on name. `centralized_ekf_walk` is that second name, and it asserts $\gamma=1$
exactly as `centralized_ekf_lambda` asserts $\boldsymbol F=\boldsymbol I$ — the naming convention
of D56 applied one level down. It is **not** the same as
`centralized_ekf_lambda` at $\lambda=1$, which also forces $Q=0$ and so cannot
express a random walk that is actually driven.

### ✅ D72. The two families differ in tracking, not in fit

Running the missing stationary twin for the best $\lambda$ cell — the top-16
controls were all $\gamma$ cells, so the family comparison had damage for one
side only — turns the 0.0054 gap into something much sharper:

| | stationary | under drift | damage |
|---|---|---|---|
| $\gamma$ family, best | 0.0561 | 0.0630 | **0.0069** |
| $\lambda$ family, best | 0.0562 | 0.0683 | **0.0121** |
| `diffusion_sgd_atc` | 0.0828 | 0.0952 | 0.0124 |

The two families are **indistinguishable when nothing is moving** — 0.0561
against 0.0562, a difference of 0.0001 against a 0.0013 threshold. The entire
gap between them is drift damage, and $\lambda$'s damage is within noise of
ATC's.

So the finding is not "$\gamma$ fits slightly better". It is that multiplicative
inflation buys **essentially no tracking advantage over SGD**, while additive
process noise halves the damage — and that both reach the same place when there
is nothing to track. That is a statement about the mechanism rather than about
the tuning: $\boldsymbol P\leftarrow\boldsymbol P/\lambda$ preserves the covariance's shape,
scaling confident and unconfident directions alike, while $\boldsymbol P\leftarrow\boldsymbol
P+Q\boldsymbol I$ compresses toward isotropy and so re-opens precisely the directions
the data had pinned down. Under drift it is those directions that need to move.

Worth stating carefully because it was invisible until the control ran: on
settled error alone the two families look like near neighbours, 0.0630 against
0.0683.

**⚠ Superseded in part by D73.** The *ordering* holds, but "multiplicative
inflation buys essentially no tracking advantage over SGD" was drawn from a
condition where there was almost nothing to detect, and it does not survive a
harder one.

### ✅ D73. The tracking share triples where the drift actually hurts

D70 measured the filter's advantage and split it: at linear 0.025°/step, **83 %**
of the win over ATC is present with no drift at all — the EKF being a
second-order method — and only **17 %** is tracking. That was the honest reading,
and it made a prediction: if the tracking component is small *because there is
little tracking to win*, it should grow at a condition that does more damage.

Tested at `every 25, jump 30°`, the most damaging condition in the project (D67),
with the **same hyperparameters** chosen under smooth drift and not re-tuned:

| | stationary | under drift | damage |
|---|---|---|---|
| EKF, $\gamma<1$ | 0.0561 | 0.1007 | **0.0446** |
| EKF, $\gamma=1$ | 0.0548 | 0.1015 | 0.0467 |
| EKF, $\lambda$ | 0.0562 | 0.1034 | 0.0472 |
| Centralized SGD | 0.0785 | 0.1453 | 0.0669 |
| ATC | 0.0798 | 0.1499 | 0.0701 |

| split, vs ATC | linear 0.025 | every 25, jump 30 |
|---|---|---|
| total advantage | +0.0322 | **+0.0493** |
| present with no drift | 0.0267 (83 %) | 0.0237 (48 %) |
| from tracking | 0.0056 (17 %) | **0.0255 (52 %)** |

The tracking component grew **4.6×** in absolute terms while the fitting
component barely moved — which is what should happen if the two are measuring
what they claim to, since fitting is a property of the method and tracking a
property of the method *against a condition*. The prediction held.

Two things make this stronger than the smooth-drift result rather than merely
consistent with it. The hyperparameters were **selected at a different
condition**, so it is not tuned-here-reported-here. And the baselines were
**re-tuned again here**: ATC prefers `lr 0.01` under abrupt shifts where it
preferred 0.02 under smooth drift, worth 0.0050, so it is measured against its
own best step size.

**What did not survive.** D72 concluded that the $\lambda$ family "buys
essentially no tracking advantage over SGD", because its damage (0.0121) was
within noise of ATC's (0.0124). Here $\lambda$ is damaged 0.0472 against ATC's
0.0701 — a real advantage of 0.023 — while remaining 0.0026 behind the $\gamma$
family. The ordering of the two families holds and the mechanism argument for
*why* still stands; the strong form of the claim does not. It was drawn from a
condition where only 13 % of ATC's error was the drift's fault, so there was
almost nothing there to detect either way.

The general lesson is the one D67 was built around and this is the second
instance of: **a null result at a mild condition is not a null result.** Before
concluding that a mechanism does nothing, check whether the condition gave it
anything to do.

$\gamma=1$ and $\gamma<1$ remain tied (0.1015 against 0.1007, gap 0.0008 against
a 0.0021 floor), so carrying both forward under D71 was the right call and the
question is still open.

### ✅ D74. Rate and state count are separate axes, and $J$ confounds them

A recurring cell's rate is $J/t'$, but $J$ and $t'$ are not interchangeable ways
of reaching it. Jumps reflect inside the 90°-wide band $[-45,45]$, so a larger
jump reaches fewer distinct rotations:

| $J$ | 5 | 15 | 30 | 45 |
|---|---|---|---|---|
| distinct rotations | 12 | 6–7 | **3** | **3** |

Shortening $t'$ raises the rate and leaves the state space alone; enlarging $J$
raises it by collapsing the state space. A learner facing three rotations
revisited twenty times is doing **recurring-concept recall** as much as tracking,
which is a real and studied problem but not the one this project claims to study.
Linear drift sits at the far end of the same axis, visiting a new rotation every
step.

**This lands on D73's headline condition.** `every25_jump30` reaches three
rotations across sixty shifts. Every method faces the same three, so the
comparison and the damage figures stand — but "the filter tracks better" is
narrower than it sounded, and both meeting documents now say so.

X11's own documentation already carried the warning — *"larger J reaches fewer
states (11, 6, 3), so it partly measures recall"* — as a footnote about reading
one panel. It was never promoted to a property of the conditions themselves, so
it did not travel to X14 when those conditions were reused. **A caveat attached
to a figure does not survive being reused as an experimental condition**; it has
to be attached to the condition.

X14 therefore crosses the two axes rather than pushing the rate alone:
$t'\in\{25,10,5,2\}$ against $J\in\{5,15,30\}$, capped at 3°/step by `MAX_RATE`.
Reading down a column gives rate at roughly fixed state count, across a row gives
state count at fixed interval, so "is it rate or is it recall?" becomes a
measurement rather than a caveat.

**The ceiling is chosen, not derived.** Above ~3°/step the learner has fewer than
five steps and twenty samples per agent between shifts, so no transient completes
and every method degrades to the same floor — the comparison stops
discriminating. Finding that floor exactly would be a different experiment.

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
$(\boldsymbol H, \boldsymbol R, \boldsymbol y)$ rather than raw samples, and a neighbour can send those
without sending an image.

**`local`.** Agent $v$ updates on its own measurement only, then combines:

$$\boldsymbol\theta_v^+ = \boldsymbol\theta_v + \boldsymbol K_v(\boldsymbol y_v - h(\boldsymbol\theta_v)),
\qquad \boldsymbol\theta_v \leftarrow \sum_u a_{uv}\boldsymbol\theta_u^+$$

**`one_hop`.** Agent $v$ additionally assimilates its neighbours' measurements,
stacking $\{(\boldsymbol H_u, \boldsymbol R_u, \boldsymbol y_u)\}_{u \in \mathcal N_v}$ (or applying them
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

### ✅ D75. The figure layer follows the meeting documents out of the repo

D46 moved the four `.docx`/`.pptx` builders out on the grounds that they "help
nobody run or check the benchmark". The same criterion, applied consistently,
reaches further than D46 took it. Nine more files are now untracked and
`.gitignore`d, still on disk:

| file | lines | what it draws |
|---|---|---|
| `make_figures.py` | 1689 | F1–F10, the phase 1–4 results figures |
| `make_preliminary_figures.py` | 911 | environment illustrations 01–11, `SUMMARY.md` |
| `plot_ekf_pilot.py` | 441 | figures 27–28 |
| `plot_abrupt_vs_smooth.py` | 240 | figure 25 |
| `plot_recurring.py` | 232 | figure 24 |
| `plot_breaks.py` | 204 | figure 23 |
| `plot_ekf_advantage.py` | 201 | figure 29 |
| `plot_drift_damage.py` | 181 | figure 26 |
| `tests/test_figures.py` | 187 | axis-label claims in `make_figures` |

That is 4,286 lines, about 14 % of the tracked Python. The pattern
`/scripts/plot_*.py` also catches the three written since (`plot_ekf_generalization`,
`plot_gamma_vs_lambda`, `plot_shift_cycles`), so the rule holds without an edit
per file.

**`test_figures.py` went with them rather than being orphaned.** It imports
`F3_WINDOWS`, `x4_headline` and `x4_tuned` from `make_figures`, so on a fresh
clone it would fail at collection rather than skip. It was the only test coupled
to the script layer — everything else in `tests/` mentions scripts solely in skip
messages, and `src/` does not reference them at all. Verified: 1274 tests pass
with it absent.

**What stayed, and why the line is there.** The `run_*` and `sweep_*` scripts and
the `report_*` readers are kept. Someone who clones this to check a number needs
to regenerate the run and read it back, and those two layers are exactly that
path; their docstrings are also the best design documentation in the repository —
`run_ekf_sweep.py` explains why the grid is proportioned as it is, and
`run_ekf_generalization.py` why the axes are crossed. A figure answers "what
should the slide look like", which is our question and not a reader's.

**This reverses part of D46, deliberately.** D46 kept `make_preliminary_figures.py`
because `environment.md` §4 pointed at it as a way to *inspect* what the agents
receive — "a checking tool that happens to also produce slides material". That
reading no longer holds up: the section leads with `check_environment.py`, which
prints sample streams directly and is staying, so the inspection survives without
the renderer. `environment.md` §4 now says so rather than pointing at a file that
is not there.

**Dangling references, all fixed rather than left.** `README.md` (the run list,
the `DEKF_FIGURES_DIR` example, the `scripts/` line in the layout tree),
`tasks.py` (the `figures` target now explains the absence instead of naming a
missing script), `utils/paths.py` (its docstring used `make_figures.py` as the
worked example of the hardcoded-path wart it exists to fix),
`docs/IMPLEMENTATION.md` (the repo tree) and `docs/figures.md` (a banner: the
document is kept because *what each figure shows* is worth having, but every
command in it is now a record of how a figure was made rather than something a
clone can run).

### ✅ D76. Multiplicative forgetting is rejected on mechanism, not on tuning

D72 concluded the $\lambda$ family "buys no tracking advantage"; D73 recorded
that this did not survive the abrupt condition and called the mechanism claim too
strong. X14 then put $\gamma<1$ ahead across twenty-one conditions — the less
damaged in fifteen, $\lambda$ in five, one tie. That ordering was still open to a
fair objection, and it is the objection D71 exists to take seriously: **both
families were running settings tuned once, at 0.025°/step, and never re-tuned.**
$\lambda$'s memory of $1/(1-\lambda)=250$ steps is a poor match to a condition
that shifts every 2, so it may simply have been mistuned.

**The test.** Both families re-swept at `every2_jump5` — 2.50°/step, 19 distinct
angles — chosen because it carries the largest gap in the sweep, so the objection
had the most room to be right. Ten cells, five seeds, each family compared at its
own optimum against its own paired stationary twin.

| | selected | twin | under drift | damage |
|---|---|---|---|---|
| EKF, $\gamma<1$ | $q=2\times10^{-4}$ | 0.0619 | 0.1127 | **0.0508** |
| EKF, $\lambda$ | $\lambda=0.996$ | 0.0562 | 0.1514 | **0.0951** |
| diffusion ATC | lr 0.05 | 0.0798 | 0.1746 | **0.0947** |

**The gap does not close — it widens, 0.0376 → 0.0443**, against a 0.0013
threshold. And the decisive number is the last column: *$\lambda$'s damage is
ATC's damage.* At its own optimum the multiplicative filter delivers **no
tracking advantage over a tuned gradient method at all**; everything by which it
beats ATC is fitting. Decomposed, $\gamma$ splits 0.0179 fitting against
0.0439 tracking, and $\lambda$ 0.0236 against $-0.0004$.

**Why it could not be tuned to the condition.** The re-tune kept $\lambda=0.996$
— the *longest* memory tried, against a shift every 2 steps. Shortening it made
the filter worse (0.99 → 0.2011) and then made it diverge: 0.98 at steps 943 and
963, 0.95 at 211 and 313, at both priors tested. The useful half of the axis is
unreachable.

In information coordinates the recursion is $\boldsymbol\Omega_{t|t}=\lambda
\boldsymbol\Omega_{t-1|t-1}+\boldsymbol\Delta_{v,t}$: a fixed **fraction** $(1-\lambda)$ of
everything known is discarded each step, so the sustainable level is
$\boldsymbol\Delta/(1-\lambda)$ — bounded *only if* $\boldsymbol\Delta=\boldsymbol H^{\mathsf T}
\boldsymbol\Lambda\boldsymbol H$ is bounded below. That is exactly the noise assumption's
$c_{\min}>0$, and D-note `rem:cmin_fails` shows the softmax denies it: as the
classifier grows confident $\boldsymbol\Lambda\to\boldsymbol 0$ in *every* direction, so the
forgetting outruns the information arriving and $\boldsymbol P\to\infty$. Measured
against pure unopposed inflation $\sigma_0^2\lambda^{-t}$, the divergences land
within a factor of 2–5 — the arriving information was barely resisting at all.

The additive recursion has no such exposure. $\gamma^{2}\boldsymbol P+\boldsymbol Q$ is bounded
by $q/(1-\gamma^{2})\approx0.06$ **with no data whatsoever**, and even
$\gamma=1$ grows linearly rather than geometrically. *$\lambda$'s stability is
conditional on a likelihood property the softmax does not provide; $\gamma$'s is
not conditional on anything.*

**This is why the claim is about mechanism.** D72's version was drawn from a
condition where there was almost nothing to detect, and D73 rightly withdrew it.
This one is drawn from the condition most favourable to the opposite conclusion,
with the family given its own optimum, and the failure has a derivation rather
than a p-value.

**What it does not establish, and the note should not be read as more.** The
re-tune ran at *one* condition, deliberately the one where $\gamma$ already led
by the most, and the five conditions where $\lambda$ was ahead were **not**
re-tuned — so $\lambda$'s optimum may move there and it may genuinely track
better at moderate rates. The recommendation does not turn on it: one setting
must be committed to before the drift is known, $\lambda$'s best margin over
$\gamma$ anywhere in the sweep is $+0.0083$ against a worst of $-0.0376$, and
only one of the two families can diverge. **"$\gamma<1$ with additive $\boldsymbol Q$ is
the right commitment" is supported; "$\gamma$ dominates everywhere" is not.**

**Consequences.** The $\lambda$ convention is removed from the note's machinery —
symbol table, time update, algorithm, the `as:noise` remedies — and survives only
as a recorded rejection plus one live caveat: `sec:closure` keeps it as the
simplest way to preserve matrix structure through the prediction step *above* the
scale at which $\boldsymbol P$ must be structured, which is a regime we have not reached
and where the same shape-preserving property that loses here would win.

### ✅ D77. Label skew does not reach the filter, and a broken baseline looks robust

Asked whether the filter had ever been tested under label skew. It had not: of
242 completed runs exactly three used a Dirichlet partition, and all three were
X6, which predates the filter. Everything from X9 onward — the 21/21
generalisation of D74, the $\gamma$/$\lambda$ decision of D76, the break rate of
X16 — was measured IID. True, unstated, and the kind of gap that survives until
someone asks.

**The obvious experiment would have measured almost nothing, and X6 said so in
advance.** The centralised filter trains on $\bigcup_v\mathcal D_t^v$, and skew
is a statement about how labels are split *across* agents; pooling undoes it.
X6's settled error across $\beta\in\{0.1,1,100\}$:

| | $\beta=0.1$ | $1$ | $100$ | spread |
|---|---|---|---|---|
| centralized SGD (pooled) | 0.0788 | 0.0777 | 0.0797 | 0.0020 |
| ATC (per-agent) | 0.1021 | 0.0812 | 0.0809 | 0.0212 |
| local only (per-agent) | 0.6301 | 0.2757 | 0.1419 | 0.4882 |

So it was run for two sharper questions instead.

**(a) The curvature hypothesis, and it is refuted.** Under skew the per-step
pooled batch has higher variance in its label *composition* even though the
marginal is right. SGD averages gradients and cannot see that; the filter
estimates $\boldsymbol H^{\mathsf T}\boldsymbol\Lambda\boldsymbol H$ from the same batch, and $\boldsymbol\Lambda$
depends on the predicted class distribution — so it *could* be more
skew-sensitive despite seeing the same marginal. It is **less**:

| | $\beta=0.1$ | $1$ | $100$ | spread |
|---|---|---|---|---|
| EKF $\gamma<1$ | 0.0557 | 0.0568 | 0.0560 | **0.0011** |
| centralized SGD | 0.0766 | 0.0779 | 0.0758 | 0.0021 |

0.0011 is below the 0.0013 threshold; the pooled baseline's 0.0021 is not. And it
holds under drift: at `every25_jump15` the filter is damaged **0.0350 at
$\beta=0.1$ against 0.0370 IID**, twins 0.0557 and 0.0561. Strong skew moves the
filter by less than the noise floor, still or drifting.

**(b) Abrupt against smooth at a matched rate.** Two cells at the same
0.60°/step and near-identical total travel (885° against 899°) — 15° every 25
steps against 0.6° every step. Damage, all four learners:

| | abrupt | smooth |
|---|---|---|
| EKF | 0.0350 | 0.0011 |
| centralized SGD | 0.0509 | 0.0041 |
| ATC | 0.0805 | 0.0086 |
| local only | 0.0321 | 0.0015 |

Continuous motion at that rate is nearly free; the same average delivered as
jumps costs 9–32× more. **This is not the X11-vs-X12 comparison** and does not
contradict it: that one put repeated jumps against *monotone linear* drift at
$\bar\alpha\le0.15$°/step, whereas this "smooth" is a reflecting random walk at
0.60. Monotone linear cannot run at 0.60 for 1500 steps at all — it reaches the
45° cap at $t=75$ — so the two live in different rate regimes.

**The first attempt was invalid, and the reason is D39 arrived at from the other
side.** All three baselines were given one optimiser and one learning rate,
momentum 0.9 at lr 0.05, carried over from a 2.50°/step *drift* condition. The
re-tune, five rates per condition per learner, showed what that cost: 0.015–0.044
for the two momentum methods. For `local_only` the rate was in fact right —
0.05, exactly what X6 chose — and the *optimiser* was wrong, since momentum at
that rate gives an effective step $\eta/(1-\beta)=0.5$, the instability
`sweep_hyperparameters.py` exists to document. It sat at 0.859, essentially
chance, against X6's 0.630.

**Two things fall out of that mistake worth keeping.**

*A broken baseline looks robust.* `local_only`'s damage went from 0.0158 to
**0.0321** when it was fixed — it got *worse* on the drift metric by being
repaired. A learner pinned at chance cannot be damaged much further, so breaking
it made it the most drift-resistant method in the table. Read without the
stationary column beside it, that column says the opposite of the truth. The
existing rule was "a mis-tuned baseline shifts a number"; it should be **a
mis-tuned baseline can invert an ordering**.

*And repairing it did not remove the compression, only the pathology.* At 0.0321
`local_only` is still the smallest damage in the (b) table, and that is still not
robustness. X11 shows it without a new run: `x11_every25_jump15` is this exact
drift under an IID partition, and there `local_only` is damaged **0.0709**,
the largest of the three learners the two sweeps share — against 0.0505 for
centralized SGD and 0.0518 for ATC. Adding skew *lowered* its damage,
0.0709 → 0.0321, while the drift was held fixed. What moved was its floor,
0.1379 → 0.6266.

The mechanism is that damage is a paired difference in a **bounded** coordinate.
Pairing removes the stationary gap, which is what it is for, but error rate is
capped — for ten classes a uniform guess errs at 0.9 — so a fixed degradation
compresses as the base rises: 0.0321 from 0.6266 and 0.0350 from 0.0557 are not
the same quantity. At $\beta=0.1$ each node holds a narrow slice of the label
space and `local_only` never pools, so its error is already dominated by classes
it structurally cannot predict, and rotating the features perturbs a boundary it
never fit.

**No normalisation fixes this, which is why the claim is kept negative.**
Dividing by headroom to chance inverts the ordering one way (`local_only` 0.118,
ATC 0.100, SGD 0.062, EKF 0.042); the ratio drifting/stationary
inverts it the other (`local_only` 1.05$\times$ against 1.6–1.8$\times$). Error
rate carries no scale that makes one of the three canonical. So the rule is
narrower than "normalise it": **damage is comparable only between learners whose
stationary levels are comparable**, and where they are not, the stationary level
is reported beside it and no ranking is claimed. Figure 33(b) prints the base
under each bar for that reason.

The rest of the (b) table is consistent with this once read that way.
`centralized_sgd` pools, so skew never reaches it and its damage is unchanged
from IID, 0.0505 → 0.0509. ATC is per-agent, so skew does reach it —
stationary 0.0775 → 0.0974, damage 0.0518 → 0.0805. ATC and `local_only` are
both hurt by skew and differ only in that ATC still has room to degrade.

*Correcting it strengthened the claim.* Against ATC at the abrupt cell the
total advantage fell from +0.1143 to +0.0873 — but the whole of that came out of
the *fitting* term (+0.0713 → +0.0418), exactly where a too-large step size would
put it, while tracking rose slightly (+0.0430 → +0.0455). The tracking share went
**38% → 52%**. The invalid run had inflated the headline and understated the part
the project actually claims.

**What this does not establish.** The centralised filter pools, so this says
nothing about how a *diffusion* filter would fare under skew — where each agent
holds a skewed shard and the combine step has to reconcile them, which is
precisely where D50's finding that cooperation pays more under label shift would
bite. That is a phase-5 experiment and the one that matters.

### ✅ D78. A sawtooth does not make momentum a liability — the mechanism is dead

Asked whether a *sawtooth* drift — a sustained monotone ramp punctuated by a
reset — would break the gradient methods where the filter survives. Worth being
precise about why that framing was refused, because refusing it is most of what
made X18 worth running.

**Not a fourth breaker hunt.** X11, X14 and X17 all end the same way: at a shift
the filter and the baselines take nearly the same hit (+0.0594 against +0.0635),
because the rise is set by the shift and no causal method can pre-empt one. The
filter wins on the floor it returns to, not on the transient. A third drift shape
asking the same question would have got the same answer.

So X18 was pointed at a mechanism instead, and the mechanism was about the
*baselines*. Momentum is a **directional memory** — at $\beta=0.9$, roughly
$1/(1-\beta)\approx10$ steps of velocity. On a monotone stretch that is an asset:
the velocity points where the distribution is going, so the method effectively
anticipates. At a reset it is maximally wrong and must unwind before it helps
again. A filter carries no directional state at all: $\boldsymbol Q=q\boldsymbol I$ is isotropic
and the covariance says "I am uncertain", never "I was moving that way". Hence a
falsifiable prediction that does not mention the filter:

> `diffusion_sgd_atc_plain` should gain on `diffusion_sgd_atc` as the reset
> period shortens, with the crossover near momentum's own ten-step horizon.

**The estimator had to be a trend, and the cap is why.** With amplitude pinned at
the $45^{\circ}$ cap, shortening the period shortens the monotone stretch *and*
raises the ramp rate together; holding the rate fixed instead needs
$A=\text{rate}\times P$, which passes the cap by $P=300$. So the axes cannot be
separated — the same bind D74 records for $J$ and $t'$ — and, more restrictively,
**no matched monotone control exists at any damaging rate**: the fastest legal
monotone drift is the cap spread over the horizon, $0.03^{\circ}$/step, which
damages nothing. Being able to sustain motion the cap forbids is what a sawtooth
is *for*, and it is the same reason the comparison it would want is unavailable.
The estimator was therefore fixed in advance as the per-seed slope of the gap
`damage(plain) - damage(atc)` on $\log_2$ ramp rate, read across
$P\in\{300,100,50,30,20\}$.

**The result is a null.** Ten seeds, one shared stationary twin, damage against
that twin:

| period | ramp °/step | EKF | centralized SGD | ATC ($\beta$=0.9) | ATC plain | gap |
|---|---|---|---|---|---|---|
| 300 | 0.15 | **0.0236** | 0.0317 | 0.0327 | 0.0323 | −0.0004 |
| 100 | 0.45 | **0.0260** | 0.0343 | 0.0356 | 0.0346 | −0.0009 |
| 50 | 0.90 | **0.0263** | 0.0339 | 0.0349 | 0.0344 | −0.0005 |
| 30 | 1.50 | **0.0262** | 0.0339 | 0.0340 | 0.0349 | +0.0009 |
| 20 | 2.25 | **0.0272** | 0.0343 | 0.0346 | 0.0363 | +0.0016 |

Slope **+0.00052 ± 0.00037 per doubling of rate** ($t=1.41$, $p=0.19$). Not the
predicted sign, and not significant. Removing momentum neither helps nor hurts
more as resets come faster: **there is no interaction between momentum and reset
frequency**, which is the cleanest possible refutation of the mechanism.

**Why the mechanism failed, which is not the same as it being wrong.** Even the
shortest period gives a **19-step monotone ramp** against momentum's ~10-step
horizon, so the velocity is repaid before the reset invalidates it. The predicted
crossover sits at a period *below* the rate ceiling: $P=20$ already runs at a mean
$4.28^{\circ}$/step, above the $\approx3^{\circ}$/step X14 established, and going
shorter means a faster ramp still. So this is not evidence that directional memory
is irrelevant — it is evidence that **the cap and the rate ceiling between them
forbid the regime where it would matter.** Testing it needs a smaller amplitude,
which is a different experiment, not more seeds of this one.

**Faster resets cost everyone a little, and nobody differentially.** From $P=300$
to $P=20$ — a 15× rate change, 4 resets against 74 — every learner moves the same
way: EKF +0.0036 ($p=0.002$), plain +0.0039 ($p=0.007$), centralized
SGD +0.0026 ($p=0.15$), ATC +0.0019 ($p=0.31$). Two clear
significance, two do not, but the *spread* between them is smaller than the
common movement. The filter holds its ≈0.008 lead across the entire axis,
unchanged, which is what a null here was defined in advance to mean: its
advantage is structural, not sawtooth-specific.

**⚠ The five-seed reading of this experiment was wrong in two ways, and both
were confident.** At $n=5$ the slope was +0.00112 ± 0.00045 ($p=0.068$) and read
as a near-significant *reversal*; ATC appeared **immune** to the axis
(−0.0007, $p=0.82$), which invited the story that momentum buffers the reset. Five
more seeds halved the slope and moved $p$ to 0.19, and ATC's immunity became
+0.0019 ($p=0.31$) — an ordinary member of the common upward move. The per-seed
slopes show why: seeds 0–3 gave +0.0018, +0.0009, +0.0019, +0.0015 and the five
added seeds averaged −0.00009. **A four-of-five sign agreement at $n=5$ was not
evidence of anything.** The rule to carry: a wrong-signed trend at $0.05<p<0.10$
is not a weaker version of a result, it is an absence of one, and the cheapest
way to find that out is more seeds rather than more prose. X18's second pass cost
under six hours (D63's exactness guarantee meant seeds 0–4 reproduced
bit-for-bit, so only the new five carried information).

**What this does not establish.** Nothing about a *diffusion* filter, as ever.
And nothing about directional memory in general — only that at
$\beta=0.9$, with a monotone stretch never shorter than twice its horizon, it
neither helps nor hurts. See [[D74]] for the coupled-axis bind this shares, and
[[D77]] for why the gap here *is* rankable: the two ATC floors differ by
1.09× (0.0790 against 0.0863), nothing like the compression that made
`local_only`'s damage unreadable.

### ✅ D79. Covariance sharing buys nothing; the adapt step is where the information is lost

X19, the diffusion filter's first measurement: three conditions (stationary,
linear 0.03°/step, `every25_jump15`) crossed with two graphs (complete, ER
$p=0.3$), five seeds, ten cells. Every diffusion cell lands between the
centralised filter and `local_only`, which is the only ordering the design
permits — each agent sees $1/N$ of the data and the drift is global, so a
diffusion filter that beat the centralised one would be a bug.

**Result 1: the expensive variant buys nothing.** The gap `full` → `mean-only`,
paired by seed:

| condition | complete | ER |
|---|---|---|
| stationary | +0.0003 | +0.0002 |
| linear 0.03 | +0.0006 | +0.0005 |
| `every25_jump15` | +0.0007 | +0.0006 |

All six positive ($p$ between 0.016 and 0.067), so the effect is probably real,
and all six are **below the 0.0013 threshold** this project uses everywhere else.
For that, full sharing pays 8 459 372 scalars per link per step against 2 908 —
a factor of **2909**. The note's justification for measuring the ceiling first was
that the gap "is worth knowing before deciding whether to pay it". It is: the
price of not shipping covariances is nothing, and `eq:cov_local` is not a
compromise but simply the right choice.

**Result 2: sparsity is nearly free too.** ER minus complete is +0.0002 to
+0.0003 for both diffusion variants, against +0.0010 to +0.0023 for ATC,
across a drop from 45 edges to 11. The filter is *less* sensitive to connectivity
than the gradient baseline. (The centralised filter and `local_only` are
identical across topologies to the last digit, neither being able to see the
graph — a harness check passing.)

**Result 3, and the problem: the centralised filter's advantage does not survive
decentralisation.**

| condition | centralised | diff-EKF full | ATC (mom. 0.9) |
|---|---|---|---|
| stationary | **0.0561** | 0.0789 | 0.0785 |
| linear 0.03 | **0.0656** | 0.0948 | 0.0951 |
| `every25_jump15` | **0.0931** | 0.1341 | **0.1289** |

Diffusing the belief costs +0.0228 → +0.0293 → +0.0411 as the condition hardens,
every one at $p<0.001$. The filter ties ATC when still and under linear
drift and **loses to it under abrupt drift**, on level and on damage (0.0553
against 0.0504) — while the centralised filter beats ATC by 0.036 in that
same cell.

**The mechanism, which the derivation already implied.** With $\boldsymbol\Omega=\boldsymbol
P^{-1}$, the centralised filter does $\boldsymbol\Omega\mathrel{+}=\sum_{v}\boldsymbol\Delta_v$
while a local adapt does $\boldsymbol\Omega_v\mathrel{+}=\boldsymbol\Delta_v$. The combine then
averages *covariances*, and an average of $N$ covariances each carrying one
agent's information still carries about one agent's information. So the diffusion
filter accumulates information at $1/N$ the centralised rate — permanently, not
as a transient — and its belief is roughly $N$ times too diffuse. Two
consequences, and the second matches the observed pattern:

* the gain $\boldsymbol K=\boldsymbol P\boldsymbol H^{\mathsf T}(\cdot)^{-1}$ is too large, so each update
  over-corrects on four samples of noise;
* **$q$ is mis-scaled.** It was tuned so the process noise added per step
  balances the centralised influx $N\boldsymbol\Delta$; against $\boldsymbol\Delta$ alone the
  filter forgets about $N$ times too fast *relative to what it learns*. That
  predicts a penalty growing with drift severity, which is what +0.0228 →
  +0.0293 → +0.0411 is.

**⚠ No combine rule can repair this, and that is the structural point.** Not
`eq:cov_combine`, not CI, not anything: CI computes $\boldsymbol\Omega=\sum_u
\omega_u\boldsymbol\Omega_u$, a weighted *average* of information rather than a sum, so
it recovers none of the missing factor $N$ and costs an inverse per fusion
besides. **You cannot fuse your way to information nobody gathered.** The fix has
to be in the adapt step, which is exactly why `prop:complete_graph` is stated for
$\mathcal M_{v,t}=\mathcal V$ and holds for no local-adapt variant.

So X19 measured the wrong axis as its ceiling. The combine axis is the cheap one
— it buys nothing, and the deployable variant is therefore free. The axis that
matters is `adapt_scope`, and `diffusion_ekf_onehop` already implements it.

**⚠ Two things make these numbers provisional, and both were errors of mine.**

*The baseline was not payload-matched.* X19's ATC carries momentum 0.9 with
`mix_optimizer_state: momentum`, so it transmits $2p$ per link while
`diffusion_ekf` transmits $p$. D29 exists to prevent exactly this, and X1 carries
`diffusion_sgd_atc_plain` so that phase 5 can state the claim against the matched
arm. In X18's stationary twin `atc_plain` settled at 0.0863 against `atc`'s
0.0790, so at **equal communication** the diffusion filter plausibly *wins* where
X19 reports a tie. Result 3's headline may not survive the correction.

*The filter carried the centralised tuning.* That was deliberate — the X14
discipline, so a shortfall is attributable — but the mechanism above says $q$ is
the parameter the local adapt mis-scales, and it names the direction. Until $q$
and $\sigma_0^2$ are re-swept for the diffusion information rate, "diffusion
loses under abrupt drift" cannot be distinguished from "the filter was tuned for
ten times the data". [[D77]]'s lesson, arrived at from the other side.

> ⚠ **The direction was wrong, and the re-tune says so.** The steady-state
> argument above predicted $q\approx q_{\text{cent}}/N=6\times10^{-6}$. X20's
> first grid, at `every25_jump15` on ER over three seeds, finds the error
> falling **monotonically as $q$ rises**, at every $\sigma_0^2$ --- 0.4557 at
> $6\times10^{-8}$ down to 0.1173 at $6\times10^{-4}$, the top of the grid --- so
> the optimum is at least an order of magnitude *above* the centralised value
> rather than an order below it. The mechanism's *existence* is not in question
> (the information deficit is arithmetic), but its consequence for $q$ is, and
> this note's prediction is withdrawn pending the widened grid.
>
> Two observations while it re-runs. The re-tuned error of **0.1173** is below
> X19's `diffusion_ekf` at 0.1351 *and* below ATC's 0.1312 in that same
> cell, so the headline above --- that the filter loses to ATC under abrupt
> drift --- is looking like the tuning artefact this note warned it might be. And
> the selected point has $q$ at 60% of $\sigma_0^2$ per step: a filter that
> almost entirely discards its own history. Tuning is compensating for a
> miscalibrated covariance by refusing to trust it, which is evidence *for*
> `combine_exponent` being the structural fix rather than a large $q$.

**A costing correction, found while checking the above.** The note prices the
one-hop adapt step at $O(pq')$ per link. At $p=2908$, $q=10$, $n=4$ that is
122 136 scalars — but the raw measurements it is derived from are $n(d+1)=788$,
and the receiver already gets $\boldsymbol\psi_u=\boldsymbol m_{u,t|t-1}$ in the same message, so
it can recompute $\boldsymbol H_u,\boldsymbol G_u,\boldsymbol s_u$ itself. **Exchanging the measurements
is 155× cheaper than exchanging their information factors, and exactly
equivalent.** ⚠ *Superseded in part by [[D92]]: the one-hop combine needs a
second message, so the whole-step saving is 18×, not 155×.* This is not an
accident of the config: raw data wins whenever
$n(d+1)<pnq$, i.e. $p>(d+1)/q\approx20$, which holds for any over-parameterised
model. The costs of a one-hop adapt are compute (each agent linearises its
neighbours' data too) and privacy (raw data leaves the node) — not bandwidth.

### ✅ D80. The filter is the worst-calibrated method on the page, and $\gamma$ is why

Every run since phase 1 has logged `nll`, `brier`, `ece`, `overconfidence` and
`mean_confidence`, alongside `e_agree` and `theta_mean_norm_sq`. **Nobody had
read them.** X19's numbers were sitting on disk the whole time, and the first
look at them is uncomfortable. Settled values at `every25_jump15` on ER:

| | error | ECE | overconfidence | mean conf. | $\lVert\boldsymbol\theta\rVert^2$ | $E_{\text{agree}}$ |
|---|---|---|---|---|---|---|
| centralised EKF | 0.0931 | 0.0310 | −0.0299 | 0.877 | 84.8 | 0 |
| diff-EKF, full | 0.1345 | 0.0842 | −0.0839 | 0.782 | 34.9 | 0.0114 |
| diff-EKF, mean-only | 0.1351 | 0.0854 | −0.0851 | 0.780 | 34.8 | 0.0118 |
| ATC | 0.1312 | **0.0148** | −0.0028 | 0.866 | 79.6 | 0.0124 |
| local only | 0.2097 | 0.0340 | +0.0269 | 0.817 | 63.7 | 18.1 |

**ATC is the best-calibrated method here and the diffusion filter is the
worst** — six times ATC's ECE — which is an uncomfortable result for a
Bayesian method to have been sitting on. And the sign is the opposite of the one
the note predicts: the filter is **under**-confident, claiming 0.78 where it
delivers 0.87, not over-confident.

**It is not disagreement.** The filter's agents agree as closely as ATC's
($E_{\text{agree}}$ 0.0118 against 0.0124; max pairwise distance 0.245 against
0.237), so "the agents have diverged and averaging blurs them" is ruled out.

**It is the parameter norm, and $\gamma$ explains it.** Confidence tracks
$\lVert\boldsymbol\theta\rVert^2$ monotonically across all five methods, and the
diffusion filter's is 34.8 against ATC's 79.6. Smaller weights give smaller
logits, and a softmax on smaller logits is closer to uniform. At $\gamma=0.9995$
the mean is multiplied by $\gamma^{1500}=0.47$ over a run *absent information* —
D26's point that $\gamma$ is L2 weight decay written in state-space form, not
forgetting. The centralised filter shrugs this off and reaches
$\lVert\boldsymbol\theta\rVert^2=84.5$ because it gathers ten agents' information per step
to push back; the diffusion filter, with $1/N$ of it, loses the tug-of-war.

**Same $\gamma$, same shrinkage, different capacity to counteract it.** That is a
third observable of [[D79]]'s information deficit, in a place nobody was looking,
and it arrived from a metric that was being computed and discarded.

**⚠ X20 does not sweep $\gamma$.** It re-tunes $q$ and $\sigma_0^2$ on exactly the
argument above — that the information deficit mis-scales them — and holds
$\gamma$ at 0.9995, a value X13 chose for a filter seeing ten times the data. So
the "re-tuned" setting is tuned in two of three dimensions.

**⚠ But $\gamma=1$ is not obviously the answer, and the two knobs interact.**
$\gamma$ acts on both moments: $\boldsymbol m\leftarrow\gamma\boldsymbol m$ and $\boldsymbol P\leftarrow
\gamma^2\boldsymbol P+\boldsymbol Q$. So $\gamma^2<1$ is a *contraction on the covariance*, and
besides the information update it is the only one. Setting $\gamma=1$ removes it.
X20's grid already located the divergence cliff at $q=6\times10^{-3}$ *with* that
contraction present; without it the cliff moves down, and the selected
$q=6\times10^{-4}$ may land the wrong side of it. $\gamma$ and $q$ therefore have
to be swept **jointly**, which is X21.

**⚠ And the whole table is plugin calibration.** `calibration.score` works from
$\boldsymbol\mu(\text{logits})$ at the predictive mean; the covariance never enters. So
"the filter is badly calibrated" is a claim about its **point estimate**, which
any method has, and says nothing yet about whether its *belief* is calibrated.
`metrics/predictive.py` has `logit_variance`, `probit_probabilities` and
`sampled_probabilities` built and tested, and nothing calls them. Scoring the
belief is P5.11 proper and remains undone — which means the filter's central
claim, that it knows what it does not know, is still unmeasured.

### ✅ D81. The diffusion filter beats tuned ATC at half its bandwidth — X19's deficit was tuning

X20: the same three conditions and two graphs as X19, five seeds, with the filter
at the $(q,\sigma_0^2)$ its own grid selected rather than the centralised filter's.

**The deficit was tuning, and it was mis-scaled in the direction the data had to
tell us.** Re-tuning alone moves every cell, and the gain grows with drift
severity exactly as [[D79]]'s mechanism predicts:

| | stationary | linear 0.03 | 15° every 25 |
|---|---|---|---|
| X19 → X20 | −0.0066 | −0.0122 | **−0.0178** |
| $p$ | 0.001 | 0.000 | 0.001 |

⚠ Note the stationary column: the setting was chosen on the *harshest* condition
and still improved the easiest one. No trade was made, which is the first thing
that had to be checked and was not guaranteed.

**At matched bandwidth the filter wins everywhere, and it beats the stronger
baseline at half the bandwidth.**

| | vs `atc_plain` ($p$) | vs `atc` ($2p$) |
|---|---|---|
| stationary | −0.048 | −0.0059 |
| linear 0.03 | −0.075 | −0.0120 |
| 15° every 25 | −0.082 | **−0.0123** |

every cell at $p\le0.011$. X19 reported the filter *losing* to ATC under
abrupt drift, 0.1351 against 0.1312; it now wins, 0.1189 against 0.1312. The
headline of D79 is withdrawn, and the mechanism that predicted it was a tuning
artefact is what corrected it.

⚠ **State the $2p$ comparison, not the matched one, as the result.**
`atc_plain` is genuinely weak here — 0.2049, barely better than `local_only`'s
0.2125 — so the matched-bandwidth margin is large partly because the matched
baseline is poor. The defensible claim is the harder one: *the filter beats the
strongest tuned baseline while sending half as much*.

**One-hop helps, and precisely where the theory says it should:**

| | complete | ER |
|---|---|---|
| stationary | +0.0031 (ns) | −0.0022 |
| linear | −0.0003 (ns) | −0.0059 |
| abrupt | −0.0089 | **−0.0123** |

More on the sparse graph than the complete one, and more as the drift hardens. On
a complete graph the mean combine already reaches every agent through the
estimates, so exchanging measurements adds little; on ER it adds a lot.

⚠ `diffusion_ekf_onehop_mean` is **not** exact on a complete graph, and the
reason matters: exactness needs a one-hop adapt *and a common predictive prior*,
and only full covariance sharing maintains the latter. With mean-only sharing the
covariances diverge from the first step, so the agents linearise at different
points. `prop:complete_graph` applies to `diffusion_ekf_onehop`, not to this.

**The result most worth arguing about: one-hop is *less damaged by drift than the
centralised filter*.** 0.0071 against 0.0095 at linear, 0.0352 against 0.0370 at
abrupt — while its absolute error is worse by 0.013–0.017 in every cell.

Both are true and they are not in tension, because damage is
$e^{\text{drift}}-e^{\text{still}}$: a paired difference that deliberately removes
fitting ability in order to isolate tracking. A method can track better from a
worse starting point, and this one does. The floors differ by 1.27–1.35×, inside
[[D77]]'s bound, so the comparison is legitimate rather than a compression
artefact.

The mechanism is the same arithmetic as everywhere else in this story, read
forwards instead of backwards: **less accumulated information means a larger
$\boldsymbol P$, a larger gain, and faster adaptation.** The centralised filter's data
advantage makes it more confident and therefore more sluggish. That is a real
trade rather than a defect, and it is the first thing measured here that a
distributed method does *better* than its centralised reference.

⚠ **It does not mean the distributed filter beats the centralised one.** On
error — the objective — the centralised filter wins all six cells. Damage is a
derived statistic, not a performance measure, and nothing here overturns the
expectation that pooling more data estimates better. The one qualification worth
keeping is that the guarantee is not a theorem here: an EKF is a linearised
approximation, not an optimal estimator, so data-richness does not formally force
dominance on every statistic one might compute. On the one that matters, it does.

**The tuning is fair, and the optima genuinely differ by an order of magnitude.**
Each filter sits at its own selection: the centralised one at X13's
$q=6\times10^{-5}$, the diffusion one at X20's $6\times10^{-4}$. X13's grid shows
$6\times10^{-4}$ would *hurt* the centralised filter badly (0.0630 at its own
choice, against 0.0939 by $10^{-3}$), so this is not a case of one arm carrying a
stale setting.

**And $\gamma$ matters less than [[D80]] expected.** $\lVert\boldsymbol\theta\rVert^2$
went 34.8 → 58.1 for a local adapt and 139.3 for one-hop, against ATC's 79.6;
ECE fell 0.085 → 0.038 → **0.020**, so the one-hop filter is now better
calibrated than the centralised one (0.031) and approaching ATC's 0.0148. A
larger $q$ was already doing much of $\gamma$'s job, and gathering more
information did the rest — which is D80's mechanism confirmed, and a reason to
expect X21 to find less than D80 claimed it would.

---

## 2026-09-12 — Phase 5, the diffusion filter's tuning

### ✅ D82. Three axes tuned in turn is not three axes tuned — but here they were

**The gap.** X20 swept $(q,\sigma_0^2)$ with $\gamma$ pinned at the value X13
chose for the *centralised* filter. X21 then swept $(\gamma,q)$ with $\sigma_0^2$
pinned at X20's selection. Every pass was joint in two axes and blind in the
third, and the axis it pinned was pinned at a value chosen by a pass that had
itself pinned something. That is coordinate descent, and coordinate descent finds
a joint optimum only when the axes separate.

**They do not separate, and X20's own grid says so.** The $\sigma_0^2$ argmin
moves with $q$:

| $q$ | argmin $\sigma_0^2$ | spread across the row |
|---|---|---|
| $6\times10^{-4}$ | $0.001$ | 0.0018 |
| $6\times10^{-5}$ | $0.1$ | 0.0125 |
| $6\times10^{-6}$ | $0.1$ | 0.0645 |

The justification for pinning $\sigma_0^2$ in X21 was that its row is flat. It is
flat *at $q=6\times10^{-4}$* — a property of where we happened to be standing, not
of the axis. Two decades down, the same row spans 0.0645, fifty times the 0.0013
threshold this project treats as noise.

**What is actually untested is the $(\gamma,\sigma_0^2)$ corner.** Neither sweep
ever varied both. And the argument that it does not matter is the argument X21
already used about $\sigma_0^2$, which the table above disposes of.

**Why this is not paranoia about a knob.** $\gamma$ and $\sigma_0^2$ act on the
same object from opposite ends: $\sigma_0^2$ sets $\boldsymbol P_0$, and $\gamma^2$
contracts $\boldsymbol P$ at every step thereafter. A large prior under a strong
contraction and a small prior under none can reach the same steady-state
covariance by different routes, so the two axes are coupled through the quantity
that decides the gain. Whether that coupling is strong enough to move the argmin
is exactly what X23 measures.

**Prediction, recorded before the run.** The selection will not move far: $q$'s
argmin was stable at $6\times10^{-4}$ across all four $\gamma$ in X21, which is
the axis with the steepest surface, and stability there is the best single
predictor that the rest holds. The case worth watching is $\gamma=1$, where the
covariance has no contraction at all and a large $\sigma_0^2$ has nothing pulling
it back — if the surface has a second basin, it is there.

---

**Measured (X23, 40 cells, no divergences).** The joint argmin is
$(\gamma,q,\sigma_0^2) = (0.9995,\ 6\times10^{-4},\ 10^{-3})$ at **0.1173** —
*exactly* the setting coordinate descent selected, to four decimals. Sweeping in
turn cost nothing here.

**The $(\gamma,\sigma_0^2)$ corner is empty, which is the part worth keeping.**
$\sigma_0^2$ behaves identically at all four $\gamma$:

| $q$ | $\sigma_0^2$ spread, across the four $\gamma$ | argmin $\sigma_0^2$ |
|---|---|---|
| $6\times10^{-4}$ | 0.0015 – 0.0029 | $10^{-3}$ at every $\gamma$ |
| $6\times10^{-5}$ | 0.0103 – 0.0132 | $0.1$ at every $\gamma$ |

So $\sigma_0^2$ couples to $q$ and **not** to $\gamma$. The flip that made the
axis look dangerous is entirely a $q$ effect, and $\gamma$ does not modulate it.
The axis X21 pinned was pinnable — which is now measured rather than assumed, and
that is the whole return on the two hours.

**The $\gamma=1$ basin exists and is shallow.** At $q=6\times10^{-5}$,
$\sigma_0^2=0.1$ it reaches 0.1190, +0.0017 from the best cell — real, above the
threshold, and beaten. Predicting where it would be was right; predicting it
would matter was not.

**The selection is a plateau, not a peak.** Seven cells sit within 0.0013 of the
best, spanning $\gamma\in\{0.9995,0.999\}$ and $\sigma_0^2$ across two decades,
while the seed-to-seed spread within a cell is 0.0006–0.0023 — the same size as
the gaps being ranked. Three seeds separate the *grid*; they do not separate a
plateau. Hence `--tie-break`, which re-runs the tied cells at five seeds in their
own directories. Its result is [[D83]].

**What does not change.** [[D79]]'s deficit is untouched: this is the filter's own
tuning, and $\alpha$ — the $c=(N/\lvert\mathcal M_v\rvert)^{\alpha}$ correction —
was **0 in every X23 cell**. The parameter norm shows it: 58.7 at the selected
setting against the centralised filter's 84.8, with `diffusion_ekf_onehop_mean`
overshooting to 139.3 when it genuinely gathers more information. A larger $q$
bought part of the norm back (34.8 → 58.7 from X19 to X20) but stopped well
short, which is X22's motivation stated in one number.

**If the prediction holds**, coordinate descent found the joint optimum, X20 and
X21's settings stand unchanged, and every diffusion result since X20 keeps its
tuning. **If it fails**, the filter has been running mis-tuned since X20 and the
X22 extrapolation sweep would have measured $\alpha$ at the wrong operating
point — which is why X23 runs before X22 rather than after it.

Grid: $\gamma\in\{1,0.9999,0.9995,0.999\}$ × $q\in\{6\times10^{-4},6\times10^{-5}\}$
× $\sigma_0^2\in\{0.1,0.03,0.01,0.003,0.001\}$, 40 cells. The larger $q$ columns
are excluded on measurement, not on taste: X21 found $q=6\times10^{-3}$ destroyed
at *every* $\gamma$ — error 0.43 to 0.89 against a chance level of 0.9, with
$\lVert\boldsymbol\theta\rVert^2$ reaching $1.3\times10^{7}$ — and only one of those four
cells tripped the trust-region guard, so "diverged" understates it. $6\times10^{-6}$
is uniformly poor at every $\gamma$. See [[D79]] for why the filter needs a $q$ ten
times the centralised one at all.

### ✅ D83. The tie-break: a flat plateau, and a selected value that was biased low

X23's grid left seven cells within the 0.0013 threshold. Re-run at five seeds
(`run_diffusion_joint.py --tie-break`), two things came out, and only one of them
is about $\sigma_0^2$.

**The argmin nominally moved, and the move is noise.** At three seeds the best
cell was $\sigma_0^2=10^{-3}$; at five it is $\sigma_0^2=0.1$. Paired across the
common seeds the difference is $-0.00079$ with $\mathrm{sd}=0.00253$, so
$t=-0.69$ on five pairs — and the sign flips three times across the five:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $\sigma_0^2=0.1$ minus $10^{-3}$ | $+0.00037$ | $+0.00265$ | $-0.00074$ | $-0.00404$ | $-0.00217$ |

All seven cells still lie within 0.0017 of each other. This is a flat plateau
behaving like one: **any of the seven is a defensible operating point, and none is
preferred.** The variance is dominated by a *seed* effect common to every cell —
seed 4 is the worst in all seven rows (0.1224 to 0.1262), seed 3 the best in
four — rather than by anything distinguishing the cells.

**The selected value was biased low, which is the part that matters.** The grid
picked the cell with the luckiest three seeds and reported its three-seed mean,
0.1173. The same configuration on five seeds is **0.1189**. That is the winner's
curse in its plainest form: selecting on a noisy statistic and then reporting the
statistic you selected on. The gap, 0.0016, is above the noise threshold.

X20's main pass had already reported 0.1189 for this configuration, and the
tie-break reproduces it **bit-for-bit on every seed** — two runs under different
names, `x20_every25_jump15_erdos_renyi` and `tb_x23_g0p9995_q0p0006_s0p001`,
agreeing to six decimals on all five. A determinism check obtained for free, and
confirmation that the headline number in `results.md` was always the honest one.

**The rule this leaves.** A tuning grid's own minimum is a *selection*, not a
*measurement*. Quote it to identify which cell won; quote a separate run of that
cell — at the reporting seed count — for what the configuration achieves. Where
the two appear side by side, as in the mis-tuning penalty of
the paper draft, the unselected arm is unbiased and the selected one is
not, so the difference is overstated unless both come from re-runs.

**Consequence for the plan.** P5.24's mis-tuned arm must be a re-run at the
reporting seed count, never the tuning grid's cell, for exactly this reason.

### ✅ D84. `adapt_rounds` stays at 1: the communication model is one hop

**Decision.** $L=1$. Agent $v$'s adapt step sees $\mathcal M_v = \mathcal N_v^{\mathrm c}$
and no further. `adapt_rounds` remains implemented and validated, but no
experiment will sweep it.

**Alternative rejected.** $L>1$, the $L$-hop neighbourhood by boolean
reachability, up to $L \ge \mathrm{diam}(\mathcal G)$ where every agent holds every
measurement.

**Why.** The setting is that an agent talks to its neighbours. $L>1$ quietly
replaces that with $L$ rounds of communication per *time step* — $L$ times the
latency before any agent may update, and a payload that grows with the
neighbourhood it reaches. Bandwidth is the filter's claim against ATC
(D29, D81); spending it $L$-fold to recover information is the trade the method
exists to avoid.

And the endpoint is a reductio, which P5.19 already recorded: at
$L \ge \mathrm{diam}(\mathcal G)$ the algorithm *is* the centralised filter, reached
by flooding. "We match the centralised filter" is empty when the method has
become it. On our own graphs that limit is not far away — ER $p=0.3$ at $N=10$
has median diameter 4, and a star has diameter 2 — so the interesting range is
short as well as expensive.

**Consequence if undone.** P5.17 (multi-round flooding), P5.19 (source-tagged
flooding at $\mathrm{diam}(\mathcal G)$) and P5.20 (channel filters on a spanning
tree) become live again. They are now out of scope by decision rather than
merely unrun, which is a different thing and should be stated as such if a
reviewer asks why the ladder stops.

**What stays open.** One-hop *measurement* exchange is not affected — that is
$L=1$ and is `adapt_scope: one_hop`, which D85 keeps.

**⚠ Addendum (2026-09-15, [[D92]]).** One-hop is $L=1$ in evidence but already
**two messages per step**: the sender's predictive mean with its raw batch before
the update, then $\boldsymbol\psi$ for the combine. So the latency this note charges to
$L>1$ starts at the one-hop adapt itself — one-hop doubles it against the local
adapt — and $L$ hops would need $L+1$ messages. The decision is unaffected; "one
round per step" was only ever true of the local adapt.

### ✅ D85. Both `adapt_scope` values are carried; $\beta$ is not a knob on either

**Decision.** `local` and `one_hop` are both reported, neither is "the" default.

**Why.** They are closer to two methods than to one method with a setting. They
differ in what crosses the link (a parameter vector against a parameter vector
plus measurements), in compute, and in privacy — and the evidence does not
separate them cleanly either, since one-hop wins on error (0.1066 against 0.1189)
and on ECE (0.0200 against 0.0384) while carrying a parameter norm of 139.3
against the centralised filter's 84.8, i.e. it overshoots. Picking a default would
assert a preference the measurements do not support.

**⚠ And $\beta$ reaches neither of them.** `combine_exponent` is read inside a
single branch of `combine()`:

```python
if self.covariance_sharing == "full":
    total.add_(cov_psi[other], alpha=weight ** self.combine_exponent)
else:
    state.extras["P"] = cov_psi[node]      # beta never touched
```

$\beta$ scales the *covariance combine*, and under local sharing there is no
covariance combine — each agent keeps its own $\boldsymbol P$. Both deployable learners
(`diffusion_ekf`, `diffusion_ekf_onehop_mean`) are local-sharing, so sweeping
$\beta$ over them would multiply runtime and return identical rows. Caught before
X22 rather than after, which is the only reason it is cheap.

**Where $\beta$ does live, and why it is worth a run anyway.**
`diffusion_ekf_full` and `diffusion_ekf_onehop`. X19 measured full sharing at
$\beta=1$ and found it buys +0.0002 to +0.0007 for 2909× the bandwidth — nothing.
But $\beta=1$ is `lem:conservative`, the worst case, which assumes the neighbours'
errors coincide. So the open question is sharper than "does covariance sharing
help": *was full sharing wasted because what it shipped was maximally
pessimistic?* $\beta=2$ assumes independence and divides the combined covariance
by about $\lvert\mathcal M_v\rvert$. If full sharing pays anywhere, it pays there.

That is X24, and $\alpha$ and $\beta$ must be swept **jointly** for the reason
[[D82]] establishes: $\alpha$ inflates the information entering the adapt step and
$\beta$ deflates the covariance leaving the combine, so they are substitutes to
first order and a coordinate pass would measure one at the other's arbitrary
value. Memory forbids folding it into X22 — three diffusion filters do not fit in
one process, and full sharing needs a second set of covariances live during the
mix.

### ✅ D86. The parameter norm is a diagnostic, not a target

**Decision.** $\lVert\boldsymbol\theta\rVert^2$ is tracked and reported; accuracy and
ECE decide. No experiment optimises the norm, and no tuning selects on it.

**Why.** [[D80]] made the norm a load-bearing observable — confidence tracks it
across every method measured — and X22 was framed partly as "does $\alpha$ close
the gap from 58.7 to the centralised filter's 84.8". That framing is wrong, and
the user's objection is the reason: **the centralised filter is not the correct
teacher.** The diffusing agent genuinely holds less information, so a matching
norm would mean it had become as confident as an estimator that has seen $N$ times
the data — which is a defect, not a success. It would be claiming confidence it
has not earned, and [[D80]] says ECE is exactly where that shows.

**How to read it.** The norm says whether a correction is doing mechanically what
it was designed to do. It does not say whether doing that is good. A move from
58.7 toward 84.8 confirms $\alpha$ bites; only error and ECE say whether the
bite helps.

### ✅ D87. Multiplying by $N$ does not work: the deficit is information, not bookkeeping

**The question, asked by the user several sessions before it could be answered.**
"Can we multiply the diff-EKF by $N$ to fix this? An agent might not know all the
agents, but it is fair to assume we can provide how many there are." It is the
right question — [[D79]] says each agent accumulates $\boldsymbol\Delta_v$ where the
centralised filter accumulates $\sum_v\boldsymbol\Delta_v$, and $N$ is exactly the missing
factor.

`information_exponent` $\alpha$ implements it as a family:
$c=(N/\lvert\mathcal M_v\rvert)^{\alpha}$ scaling $\bar{\boldsymbol B}$ by $\sqrt c$ and
the score by $c$, with $\alpha=0$ claiming nothing and $\alpha=1$ claiming the
whole network's worth. X22 swept it at five values on both adapt scopes.

**Measured: $\alpha=0$ wins, and $\alpha$ hurts monotonically.**

| $\alpha$ | `diffusion_ekf` | `onehop_mean` |
|---|---|---|
| **0** | **0.1173** | **0.1077** |
| 0.25 | 0.1232 | 0.1119 |
| 0.5 | 0.1306 | 0.1185 |
| 0.75 | 0.1686 | 0.1276 |
| 1 | **0.8941** | 0.1344 |

Significant from the first step — paired over three seeds, $\alpha=0.25$ costs
$+0.0059$ ($t=2.93$) on the local adapt and $+0.0043$ ($t=3.99$) on one-hop. And
$\alpha$ is bounded by its own meaning, so an argmin at $0$ is a real answer
rather than a truncated grid.

At $\alpha=1$ the local-adapt filter does not degrade, it is destroyed: 0.8941
against a chance level of 0.9.

⚠ **"$\alpha=1$ is multiplying by $N$" holds for the local adapt only.** The factor
is $c=N/\lvert\mathcal M_v\rvert$ — what is still *missing*. A local adapt has
$\lvert\mathcal M_v\rvert=1$, so $c=N=10$ and the phrase is literal. One-hop has
already gathered $\approx3.7$ agents' worth on ER $p=0.3$, so $c\approx2.7$.
The same $\alpha$ therefore means a different extrapolation on each scope — which
is the reason the sweep carried both, and the reason the two columns are not
comparable at fixed $\alpha$ the way a shared knob would be.

**The mechanism is in the parameter norm.**

| $\alpha$ | $\lVert\boldsymbol\theta\rVert^2$, local | $\lVert\boldsymbol\theta\rVert^2$, one-hop |
|---|---|---|
| 0 | 58.7 | 134.6 |
| 0.5 | 86.7 | 177.8 |
| 0.75 | 1 094.9 | 215.4 |
| 1 | $3.81\times10^{6}$ | 262.1 |

$\alpha$ does mechanically what it was designed to do — inflate the claimed
information, shrink $\boldsymbol P$, raise the gain — and that is the whole problem.
⚠ *"Raise the gain" has the direction wrong — a smaller $\boldsymbol P$ gives a smaller
gain. See [[D92]] for what actually makes the step grow.*
Scaling one agent's information by $N$ does not manufacture $N$ agents' worth of
*independent* evidence; it makes the filter confident about evidence it never
received, and a Gauss–Newton step taken under an over-small covariance walks
further than the linearisation supports.

**This is [[D79]]'s lesson from the other side, and it is worth stating as a pair.**
Information that was never gathered cannot be *fused* into existence — that was
`rem:no_combine_fix`, about the combine. It cannot be *asserted* into existence
either. The $1/N$ deficit is a shortfall of real evidence, not a missing constant,
and no rescaling at either end of the step reaches it. What closes it is gathering
more: one-hop beats local at every $\alpha$, by 0.0096 to 0.76.

**Calibration is the one place $\alpha$ helps, and not enough.** On the local
adapt ECE falls 0.0379 → 0.0134 and overconfidence moves $-0.0362$ → $+0.0130$,
so $\alpha$ genuinely cures the under-confidence [[D80]] identified — but the best
ECE sits on a model at chance, and the $\alpha=0.5$ trade is $+0.0133$ error
(ten times the threshold) for $-0.0102$ ECE. On one-hop ECE is flat
across the whole range (0.0193–0.0209): $\alpha$ buys nothing and costs error. On
all three criteria the user set in advance — accuracy, calibration, norm-matching
as a diagnostic ([[D86]]) — the answer is $\alpha=0$.

**⚠ And the run that exposed the guard.** Every X22 cell was recorded `ok`,
including the destroyed one: the mean reached 323× its initial norm while
`trust_region_ratio` was 1e6, so nothing fired. That is precisely the failure
[[D61]] introduced the guard to catch, waved through because the threshold only ever
caught overflow-scale blowup — which finiteness already catches. Tightened to
**50** on 2026-09-13: healthy runs measured between 1.3× and 5.5×, the centralised
filter at 1.5×, so 50 sits an order of magnitude above the worst legitimate belief
and six times below this one.

X22's cells are **not** re-run under the new guard. The $\alpha=1$ cell would now
stop at a `_diverged` marker partway, and "error 0.8941 with
$\lVert\boldsymbol\theta\rVert^2=3.8\times10^6$" is the more informative record of what
$\alpha=1$ does. The numbers stand; only their marker would change.

### ✅ D88. The conservative bound is not wasteful — it is approximately tight

**The suspicion.** X19 measured full covariance sharing against mean-only and
found it buys nothing for 2909× the bandwidth, and [[D79]] concluded the combine
axis is empty. But that rested on the covariance being *worth* having, and
`eq:cov_combine` ships $\sum_u a_{vu}\boldsymbol P^{\psi}_u$ — `lem:conservative`, the
bound that holds for any cross-correlation precisely because it assumes the
neighbours' errors **coincide**. A filter handed a worst-case covariance runs a
smaller gain than its evidence deserves. So: was full sharing *wasted* rather than
useless? $\beta=2$ assumes independent errors and divides the combined covariance
by about $\lvert\mathcal M_v\rvert$ — the very factor D79 says the belief is
inflated by.

**Measured: no. $\beta=1$ wins, decisively, everywhere.**

| | $\beta=1$ | $\beta=1.5$ | $\beta=2$ |
|---|---|---|---|
| `diffusion_ekf_full`, $\alpha=0$ | **0.1155** | 0.2616 | 0.3336 |
| `diffusion_ekf_full`, $\alpha=0.5$ | 0.1272 | 0.1849 | 0.2021 |
| `diffusion_ekf_onehop`, $\alpha=0$ | **0.1085** | 0.1713 | 0.1872 |
| `diffusion_ekf_onehop`, $\alpha=0.5$ | 0.1163 | 0.1564 | 0.1695 |

Paired, $\beta=1.5$ costs $+0.0401$ to $+0.1460$ and $\beta=2$ costs $+0.0532$ to
$+0.2181$, at $t$ from 7.8 to **42.5**. Nothing marginal about it.

**So the agents' errors really do nearly coincide**, which is what the
conservative bound assumes and what makes it approximately tight. That is not a
disappointment; it is a measurement of how little independent information
diffusion actually moves. Ten agents mixing every step, all tracking the same
$\boldsymbol\theta^{\star}$, end up with errors correlated closely enough that treating
them as independent is badly wrong. **This answers P5.14 from the side nobody
expected** — the worry recorded there was that the bound might be "wildly loose",
in which case the conservative framing would be doing less work than it appears.
The opposite holds.

**⚠ Two predictions of mine, both wrong, both recorded because the errors are
instructive.**

*First: I expected the far corner $(\alpha=1,\beta=2)$ to diverge.* The opposite —
$(\alpha=1,\beta=1)$ diverged in both variants, and $\beta\ge1.5$ **rescues** it:

    diffusion_ekf_full     beta=1: DIV   beta=1.5: 0.1571   beta=2: 0.1652
    diffusion_ekf_onehop   beta=1: DIV   beta=1.5: 0.1450   beta=2: 0.1549

$\beta>1$ shrinks $\boldsymbol P$, which shrinks the gain, which offsets $\alpha=1$'s
tenfold score inflation. **A real interaction, and the justification for sweeping
jointly after all**: an $\alpha$ line at $\beta=1$ reports a divergence that the
other axis silently removes, and a $\beta$ line at $\alpha=1$ reports $\beta>1$ as
*necessary* when at $\alpha=0$ it is ruinous. [[D82]]'s rule earned its keep here
on a case where the mechanism was known in advance.

*Second: I called $\beta=2$ "the overconfident end".* In *reported* covariance it
is. In effect it is the reverse — overconfidence runs $-0.0354$ → $-0.3057$ as
$\beta$ rises, i.e. the filter becomes far more **under**-confident. Shrinking
$\boldsymbol P$ does not merely change what the filter claims; it changes how fast the
filter learns. Smaller gain, sluggish tracking, a smaller $\lVert\boldsymbol\theta\rVert$,
and [[D80]] says confidence tracks the norm. The dynamic consequence dominates the
static one, and both point the same way. **A covariance knob in a filter is never
only a reporting knob** — it sits inside the gain, and anything that changes the
gain changes the trajectory.

**Full sharing, at the re-tuned setting, refines X19 slightly.**

| adapt scope | full sharing | mean-only | difference |
|---|---|---|---|
| local | 0.1155 | 0.1173 | $-0.0018$, $t=-6.88$ |
| one-hop | 0.1085 | 0.1077 | $+0.0008$, $t=0.97$ (ns) |

At the centralised tuning, full sharing bought nothing. Re-tuned it buys a small
but statistically clear 0.0018 for the local adapt — above the 0.0013 threshold —
and nothing for one-hop. The headline survives, since 0.0018 for 2909× the
bandwidth is still a bad trade, but "buys nothing" becomes "buys 0.0018 where it
helps at all", and that is the honest form.

**The guard paid for itself on its first run.** Both divergences were caught
mid-run at 67× the initial norm, at steps 436 and 534. Under the old $10^6$
threshold they would have completed and reported a number from a wrecked belief —
exactly what X22's $\alpha=1$ cell did before the tightening.

**Net: $\alpha=0$, $\beta=1$.** Both free corrections, motivated by the same
mechanism and aimed at opposite ends of the step, are rejected on measurement.

### ✅ D89. Covariance sharing is useless only while the agents are exchangeable

**The claim being tested.** [[D79]] concluded the combine axis is empty: X19 measured
full covariance sharing against mean-only at +0.0002 to +0.0007 for 2909× the
bandwidth, and X24 confirmed it at the re-tuned setting (−0.0018 for the local
adapt, nothing for one-hop). Both were measured on **IID shards**.

**Measured under Dirichlet skew, the conclusion reverses.**

| $\beta_{\mathrm{dir}}$ | full sharing | mean-only | paired difference |
|---|---|---|---|
| 100 (≈IID) | 0.0732 | 0.0742 | $-0.0010$, $t=-2.60$ |
| **0.1 (severe)** | **0.0823** | **0.1137** | $\mathbf{-0.0314}$, $t=-4.61$ |

Thirty times the noise threshold, and larger than the entire
centralised-versus-diffusion gap. **The mechanism is exchangeability.** On IID
shards every agent's $\boldsymbol P$ is nearly the same matrix, so shipping it is
redundant — the mean already carries everything the neighbour knows. Under skew
the agents hold different label distributions, so their Fisher information points
in genuinely different directions, and $\boldsymbol P$ carries what $\boldsymbol\psi$ cannot.

⚠ **And the two repairs are substitutes, not complements.** For the *one-hop*
adapt, full sharing buys nothing even under severe skew ($+0.0009$, ns). One-hop
already gathers the neighbours' measurements, so it reconstructs the missing
directions directly instead of needing the covariance to carry them. Pay for one
or the other, never both.

**The $\beta$ hypothesis this experiment was built on is refuted.** The prediction
was that skew decorrelates the agents' errors, so `lem:conservative` would go
loose and $\beta>1$ would become live. It does not move at all:

| learner | skew | $\beta=2$ minus $\beta=1$ |
|---|---|---|
| full sharing | 0.1 | $+0.1341$, $t=9.34$ |
| full sharing | 100 | $+0.1396$, $t=7.59$ |
| one-hop, full | 0.1 | $+0.0402$, $t=5.03$ |
| one-hop, full | 100 | $+0.0414$, $t=7.52$ |

Skew changes the penalty by 0.004 on a 0.134 effect. **The agents' errors coincide
because they mix at every step, not because their data are alike.** That is a
stronger conclusion than the one being sought: $\beta=1$ is correct for any
diffusion algorithm, structurally, and not merely for exchangeable shards.
[[D88]]'s result generalises rather than being conditional.

**Skew sensitivity, against X6's yardstick.** Spread of settled error across
$\beta_{\mathrm{dir}}\in\{0.1,1,100\}$:

| learner | spread |
|---|---|
| centralised EKF | 0.0019 |
| centralised SGD | 0.0031 |
| one-hop diffusion | 0.0163 |
| ATC | 0.0171 |
| **local-adapt diffusion** | **0.0395** |
| local only | 0.4895 |

[[D77]] is confirmed exactly as stated: the centralised filter barely moves, so
X17 said nothing about a diffusion filter. And the diffusion filter is **more**
skew-sensitive than ATC, which is not the direction one would guess for a
method that mixes a whole belief rather than a point.

**One-hop's value is almost entirely a skew effect** — $-0.0274$ ($t=-3.74$) at
skew 0.1 against $-0.0042$ ($t=-4.96$) at skew 100, a 6.5× swing. Where the shards
are exchangeable there is little to gather that the mean does not already carry.

**⚠ Do not write "the filter" for these cells.** Against $2\psi$ ATC at severe
skew the two variants disagree in *sign* — the mean-only local adapt **loses** by
$+0.0196$ ($t=3.01$) while one-hop **wins** by $-0.0078$ ($t=-8.22$):

| $\beta_{\mathrm{dir}}$ | local vs ATC | one-hop vs ATC |
|---|---|---|
| 0.1 | $+0.0196$, $t=+3.01$ | $-0.0078$, $t=-8.22$ |
| 1 | $-0.0052$ (ns) | $-0.0086$, $t=-7.45$ |
| 100 | $-0.0028$, $t=-2.87$ | $-0.0070$, $t=-14.03$ |

X20's "all six cells" was measured with the mean-only variant on IID data and
holds there. At severe skew it fails, and only the one-hop adapt carries the claim
— the same conclusion this note reaches from the covariance side, since one-hop and
full sharing turn out to be substitutes.

**And the centralised line is flat, not falling.** Its three points are 0.0544,
0.0563, 0.0558 — non-monotone, paired $t=0.97$ across three seeds, with a seed sd
at skew 100 of 0.0039, nearly three times the apparent slope. It *must* be flat:
the centralised filter pools every agent's data, so the union it trains on does not
depend on how the labels were partitioned. Figure 38(a) therefore draws it as
markers without a connecting line, with a seed-range band on every series.

**Damage under drift at skew 0.1**, abrupt: centralised 0.0358 < one-hop 0.0431 <
centralised SGD 0.0513 < local diffusion 0.0604 ≈ ATC 0.0638.
⚠ `local_only` shows the *lowest* damage of all, 0.0306, and is excluded: its floor
is 0.6215, **11.4×** the centralised filter's, far outside [[D77]]'s 2.5× rule. The
other five sit within 2.09× and are rankable.

### 🔄 D90. `atc_plain` has been under-tuned since X20, so the matched-bandwidth margin is inflated

**The defect.** X20 built its matched-bandwidth arm as
`{**PLAIN, "lr": rates[BASELINE["name"]]}` — `atc_plain` carrying the *momentum*
arm's selected rate. Plain SGD at $\eta=0.01$ takes an effective step of
0.01; momentum 0.9 at the same $\eta$ takes $\eta/(1-\beta)=0.10$. **Ten times
larger.** `local_only` was given its own rate in the same cell, so the omission is
inconsistent as well as wrong.

X17's docstring states the principle plainly and predates the mistake: momentum at
lr 0.05 gives an effective step of 0.5, "at which a single agent lands at chance
while the same agent at lr 0.005 reaches 0.188". The methods differ in how far one
update travels, so they do not share an optimum — which is the whole reason
`sweep_hyperparameters.py` exists.

**What it affects.** X20 reported the filter beating `atc_plain` by 0.048–0.082
and, to its credit, declined to headline that number: "`atc_plain` is weak enough
that the matched margin flatters us", choosing the $2\psi$ comparison instead. The
instinct was right and the diagnosis was wrong. `atc_plain` is not weak because
plain SGD is weak; it is weak because it was handed a rate chosen for a method
with ten times its effective step. At abrupt/ER it scored 0.2007 against
`local_only`'s 0.2097 — a *cooperating* method barely beating a lone agent, which
should have been read as a tuning failure rather than as a property of the method.

**What it does not affect.** The headline claim rests on the $2\psi$ comparison
against momentum ATC, which is correctly tuned and unaffected. D81 stands.

**What is now missing.** X25 carried no `atc_plain` at all, so the
matched-bandwidth comparison **under skew** is unmeasured — and skew is where
[[D89]] finds the filter losing to the $2\psi$ arm. Whether it also loses at
matched bandwidth is the open question, and it is the one a reviewer will ask.

**The fix is cheap**, because the data stream is independent of which learners are
attached — verified by X25's `lr` cells reproducing the main cells' baseline
numbers to twelve decimals on shared seeds. So `atc_plain` can run in its own
cells, at its own selected rate, and be compared paired against what already
exists. SGD-only cells are minutes.

### ✅ D91. The matched-bandwidth claim survives, at half the margin, and only one-hop survives skew

**What D90 predicted, confirmed.** Given its own grid, `atc_plain` selects
$\eta=0.05$ at **every one of the eight conditions** --- five times the 0.01 X20
handed it from the momentum arm. The effective-step argument gave the direction;
the size was not obvious.

| condition | tuned, $\eta=0.05$ | X20's, $\eta=0.01$ | recovered |
|---|---|---|---|
| stationary | 0.0859 | 0.1213 | 0.0354 |
| linear 0.03 | 0.1094 | 0.1587 | 0.0493 |
| $15^{\circ}$ every 25 | 0.1408 | 0.2007 | 0.0600 |

So the arm was 0.035 to 0.060 worse than the method is. Its "barely beats
`local_only`" behaviour was a learning rate, not a property of plain SGD, and
[[D90]]'s diagnosis is closed.

**The filter still wins at matched bandwidth --- and by *more* than against the
$2\psi$ arm.**

| condition | `diffusion_ekf` | `atc_plain` | gap | $t$ |
|---|---|---|---|---|
| stationary | 0.0736 | 0.0859 | $-0.0123$ | $-10.2$ |
| linear 0.03 | 0.0844 | 0.1094 | $-0.0250$ | $-24.2$ |
| $15^{\circ}$ every 25 | 0.1189 | 0.1408 | $-0.0218$ | $-24.3$ |

This needs stating carefully, because two numbers move in opposite directions.
X20's margin against `atc_plain` was 0.048--0.082 and was **inflated** --- the
honest figure is 0.012--0.025. But that is *larger* than the margin against
momentum ATC (0.0059--0.0125), because momentum genuinely helps ATC
everywhere ($-0.005$ to $-0.021$). The $2\psi$ arm is the **stronger** baseline,
not merely the more expensive one.

**So the claim to headline is unchanged and now properly supported**: the filter
beats the stronger baseline at half its bandwidth, and the matched baseline by
about twice that margin. Reporting the $2\psi$ comparison remains right, for the
reason X20 gave and not the one it gave it for.

**Under severe skew the filter loses at matched bandwidth too.**

| condition | gap | $t$ |
|---|---|---|
| skew 0.1 | $+0.0073$ | $1.66$ |
| skew 1 | $-0.0105$ | $-3.12$ |
| skew 100 | $-0.0114$ | $-20.8$ |
| skew, abrupt | $-0.0046$ | $-1.02$ |
| skew, smooth | $+0.0023$ | $0.38$ |

[[D89]]'s loss was not an artefact of comparing against a $2\psi$ arm: the mean-only
filter loses to diffusion SGD at *equal communication* under severe skew.
⚠ **But $t=1.66$ on three seeds is suggestive, not established.** The IID
cells run $t=10$ to $24$; this one is not in that class and should not be written
as a finding without more seeds.

**One-hop wins everywhere, and by most where the task is hardest.**

| condition | `onehop_mean` | `atc_plain` | gap | $t$ |
|---|---|---|---|---|
| $15^{\circ}$ every 25 | 0.1066 | 0.1408 | $-0.0341$ | $-19.7$ |
| skew 0.1 | 0.0863 | 0.1064 | $-0.0200$ | $-6.1$ |
| **skew + abrupt** | **0.1295** | **0.1788** | $\mathbf{-0.0493}$ | $-6.4$ |
| skew + smooth | 0.0856 | 0.1153 | $-0.0297$ | $-30.8$ |

Its best cell is skew crossed with abrupt drift --- the hardest condition in the
benchmark --- at $-0.0493$ against `atc_plain` (⚠ not its matched arm: one-hop
sends $2.27\psi$, see [[D92]]). That is the strongest case the
one-hop variant has, and it is the cell nobody would have looked at first.

**Net.** At equal communication the filter beats diffusion SGD on every
IID condition and on mild skew; at severe skew only the one-hop adapt does.
Combined with [[D89]] --- where covariance sharing pays 0.0314 under skew but
nothing for one-hop --- the picture is consistent: **once the shards stop being
exchangeable, the mean alone is not enough, and the two ways of fixing that are
substitutes.**

### ✅ D92. One-hop sends two messages a step, and its payload was short by one ψ

**Found by the user**, reviewing the deck's payload column: the 788-scalar increment
was labelled as neighbours sending $(\boldsymbol H,\boldsymbol R,\boldsymbol y)$, which it is not — it is the
raw batch. Tracing what a receiver needs to rebuild a block from a raw batch turned
up the larger error.

**The code defers the one-hop update.** `adapt` returns the *predictive* mean
$\boldsymbol\theta_u^-$ ("what travels is the prior plus the information to act on it");
`combine` then replaces $\boldsymbol\psi$ with each agent's *updated* mean from
`_one_hop_update` and mixes those. In one process that is one call. On a network it
is two messages:

1. **before the update** — $\boldsymbol\theta_u^-$ with the raw batch, $n(d+1)=788$,
   because the receiver rebuilds $u$'s block at $\boldsymbol\theta_u^-$, the linearisation
   point the implementation uses. It cannot already hold it: $\boldsymbol\theta_u^-$ comes
   from $u$'s previous combine, which depends on $u$'s neighbours, not $v$'s.
2. **after it** — $\boldsymbol\psi_u^+$ for the combine, which $v$ cannot compute without
   $u$'s $\boldsymbol P$.

| variant | messages | payload | ×ψ |
|---|---|---|---|
| `diffusion_ekf` | 1 | 2 908 | 1.00 |
| `diffusion_ekf_onehop_mean` | **2** | **6 604** | **2.271** |
| `diffusion_ekf_full` | 1 | 4 232 594 | 1 455.50 |
| `diffusion_ekf_onehop` | **2** | **4 236 290** | **1 456.77** |
| `diffusion_sgd_atc` (momentum) | 1 | 5 816 | 2.00 |

The information-factor path was costed correctly all along — $\boldsymbol\psi+\boldsymbol B+\boldsymbol g$ =
122 136, because a $\boldsymbol B$ block is already linearised and needs no
$\boldsymbol\theta_u^-$. Only the raw-sample path lost a ψ. So raw samples are **32×**
cheaper than factors in the first message (3 696 against 119 228) and **18×** over a
whole step — not 151× or 155×. The condition under which raw samples win,
$n(d+1)<p\,nq$, is unchanged: the $\boldsymbol\theta_u^-$ that must accompany the batch is
matched by the $\boldsymbol g$ that accompanies $\boldsymbol B$.

**What changes.** One-hop mean-only sends 14% more than momentum ATC, not 27%
more than plain ATC. Its results survive, restated: its nearest baseline by
bandwidth is momentum ATC at 2ψ, which it beats at every condition measured
(0.1066 against 0.1312 on IID abrupt; 0.0863 against 0.0941 at severe skew) —
so "beats momentum ATC at 1.14× its bandwidth", not "at less bandwidth", and
not "beats the matched arm", since `atc_plain` at 1ψ is not one-hop's matched arm
([[D91]]). It also doubles one-hop's latency against the local adapt ([[D84]]).

**Amended 2026-09-17: the carried variant inverts this paragraph's conclusion.**
The arithmetic above is right for the *sender* point, which is what one-hop meant
when this note was written. [[D99]] adopted the receiver point and [[D100]]
confirmed it, and there one-hop mean-only sends **3 696** scalars per link per step:
$0.64\times$ momentum ATC's 5 816, and $1.27\times$ `atc_plain`'s 2 908. Three
consequences, all favourable:

* "beats momentum ATC at $1.14\times$ its bandwidth, **not** at less bandwidth"
  becomes *beats it at $0.64\times$* — that is, at less. The phrasing this note
  explicitly ruled out is now the correct one.
* **The nearest baseline by bandwidth flips** from momentum ATC to `atc_plain`:
  one-hop sits 788 scalars above the cheap arm and 2 120 below the costly one, where
  at the sender point it was above both.
* [[D91]]'s "`atc_plain` is not one-hop's matched arm" weakens accordingly — still
  not matched at $1.27\times$, but near enough that the comparison is now worth
  making, and one-hop wins it at every condition measured.

The one-hop *results* are unchanged by this; what changed is the variant the paper
carries and therefore which baseline it should be priced against.

**Compute is unaffected, but one claim is withdrawn.** "One-hop costs no extra
Jacobians" is true of the simulation, which reuses each sender's block in-process,
and false of the raw-sample protocol the payload column describes, where the
receiver must re-linearise its neighbours' samples. The excluded cost is small at
this size — an estimated ~10⁶ flops against ~5×10⁹ for the Woodbury update — so the
3.8× ratio is close to the total, but the claim as stated was wrong. ⚠ *Measured
in [[D93]]: the Jacobians are 13% of the one-hop step, and one-hop costs 2.0×,
not 3.8× — the estimate here was wrong on both counts.*

**And from the same review: "a smaller $\boldsymbol P$ is a larger gain" was backwards.** A
smaller $\boldsymbol P$ gives a smaller $\boldsymbol K$. What $\alpha$ does ([[D87]]) is scale the
score by $c$ while the covariance shrinks by much less than $c$ wherever the prior
dominates, so the step grows roughly $c$-fold in exactly the directions with least
data — where the linearisation is least trustworthy.

### ✅ D93. One-hop linearises at mixed points, and the compute column is now measured

Two findings from a second external review of the deck, one about what one-hop
computes and one about what it costs.

**Mixed linearisation points.** A receiver rebuilds each neighbour's block at the
*sender's* predictive mean $\boldsymbol\theta_u^-$, sums the blocks with its own, and applies
the sum to its own prior $(\boldsymbol\theta_v^-,\boldsymbol P_v^-)$. For a nonlinear network,
blocks linearised at different points and summed into one update are **not** one
EKF update at a common point. It is exact when $\boldsymbol\theta_u^-=\boldsymbol\theta_v^-$
for every $u\in\mathcal M_v$ — consensus, which holds on a complete graph with a
common predictive prior, i.e. `prop:complete_graph`'s condition — and exact for a
linear observation model. Away from consensus it is a well-defined approximation.
The deck's "why one-hop evidence sharing is valid" overstated it.

**And the choice was a simulation convenience that costs a ψ in deployment.** The
sender forms its block once and every neighbour reuses it, which is cheap in one
process. A deployment that ships raw batches re-linearises at the receiver anyway,
so it could do so at the receiver's *own* $\boldsymbol\theta_v^-$: one linearisation point
— the textbook diffusion EKF, where neighbours send measurements and the
receiver linearises its own model — and no $\boldsymbol\theta_u^-$ in the first message, so
788 + 2 908 = 3 696 per link per step, below momentum ATC again. On a complete
graph with a common prior the two coincide, so the exactness gate cannot tell them
apart; on ER they differ, and X20–X26's one-hop numbers would not carry over.
❓ **Open:** whether to implement the receiver-point variant and re-run. ⚠ *Implemented
in [[D94]], alongside the sender point rather than replacing it; the comparison is X27.*

**Measured compute replaces the analytic ratios.** Per agent per step, float64 on
the RTX 4070 Laptop, median of 25 after warm-up, Jacobian reconstruction included:

| variant | ms | × local | flop model |
|---|---|---|---|
| `diffusion_ekf` | 14.0 | 1.00 | 1.00 |
| `diffusion_ekf_onehop_mean` | 28.7 | **2.04** | 3.8 |
| `diffusion_ekf_full` | 19.0 | **1.35** | 1.05 |
| `diffusion_ekf_onehop` | 33.7 | 2.40 | 3.8 |

The flop model is wrong **in both directions**. A wider Woodbury block parallelises
far better on a GPU than its flops suggest — $m=148$ takes 2.04× the time of
$m=40$ for 3.7× the flops — so one-hop is about half as expensive as claimed.
Covariance mixing is memory-bound rather than flop-bound — each dense
$p\times p$ add streams 64.5 MiB — so full sharing is a third more expensive than
claimed. And the Jacobians [[D92]] estimated at ~10⁶ flops and called negligible
are **13.3%** of the one-hop step: per-sample Jacobians are launch- and
overhead-bound, so flops were the wrong unit. The review said "remove or measure";
measuring overturned the estimate, which is the argument for measuring.

These are one GPU's ratios for isolated per-agent kernels, not end-to-end sweep
times, and a CPU would order them differently.

### 🔄 D94. Both linearisation points are implemented, and the ledger now counts scalars

Decided 2026-09-15: build the receiver point next to the sender point and compare
them, rather than choose between them on argument ([[D93]]'s open question).

**The switch.** `linearization_point` $\in\{$`sender`, `receiver`$\}$, one-hop
only, and pinned by the learner name like the two axes:
`diffusion_ekf_onehop_mean_receiver` and `diffusion_ekf_onehop_receiver` are the
existing one-hop variants at the receiver's point. Every other name is `sender`,
the default, so X20–X26 reproduce unchanged. A receiver pools every reachable
batch and calls `information_pair` once at its own $\boldsymbol\theta_v^-$; that function
sums over samples, so the result is the concatenation of per-agent blocks at one
point — and on a complete graph it is the centralised filter's pooled call
verbatim. `receiver` under a local adapt is refused: one batch has one point, so
the setting would name a variant that does not exist.

**What the tests pin.** The complete-graph identity for all four one-hop variants,
to 1e-10 in the unit test and 1e-12 through the runner. Both points pass, which is
exactly why that gate cannot choose between them. On a ring the two agree at the
first step — a common prior makes every $\boldsymbol\theta_u^-$ equal — and part from the
second. The ledger per variant, and the refusal when a config contradicts the name.

**How far apart, and at what cost.** A probe on the tests' small network ($p=349$,
6-ring): $\max|\text{sender}-\text{receiver}|$ is 1.4e-17 after step 1, 1.2e-4
after step 2 and 1.6e-4 by step 5, against an agent spread of 1e-2 to 3e-2 —
about 1% of the disagreement that causes it, and not growing over five steps.
Whether that is 1% of anything in the *error* is what X27 measures. Compute at
$p=2908$, $N=10$, ER $p=0.3$ (one draw, mean $|\mathcal M_v|=5.0$), float64
on the RTX 4070, median of 15 network steps: sender 36.1 ms per agent, receiver
37.6 (+4%). The receiver linearises $\sum_v|\mathcal M_v|$ batches where the
*simulated* sender linearises $N$; a *deployed* sender re-linearises its
neighbours' batches too ([[D92]]), so the receiver's figure is the honest one for
both.

**The ledger, fixed.** `comm_scalars_per_step` counted $p$-vectors: full sharing
at $p$ further vectors — all of $p^2$, lower triangle included — and one-hop at
$q'+1=10$ further vectors, ignoring $n$ entirely. It now counts scalars per
direction as a deployment sends them: $\boldsymbol\psi$, $p$; full sharing adds the upper
triangle, $p(p+1)/2$; one-hop adds the raw batch $n(d+1)$ — recorded from the
data, since $n$ belongs to the environment — plus $\boldsymbol\theta_u^-$ under `sender`.
At $p=2908$, $n=4$, $d=196$, per link per direction:

| variant | before | now |
|---|---|---|
| `diffusion_ekf` | 2 908 | 2 908 |
| `diffusion_ekf_full` | 8 459 372 | 4 232 594 |
| `diffusion_ekf_onehop_mean` | 31 988 | **6 604** |
| `diffusion_ekf_onehop_mean_receiver` | — | **3 696** |
| `diffusion_ekf_onehop` | 8 488 452 | 4 236 290 |
| `diffusion_ekf_onehop_receiver` | — | 4 233 382 |

For `adapt_rounds` $L>1$ it charges one batch per round, a lower bound: a flooding
round forwards every batch newly reached, and how many is a property of the graph.

⚠ **What is on disk.** Every parquet written before this carries the old values in
`cum_scalars_tx` for its full-sharing and one-hop cells, X19–X26. Nothing read
them: `make_figures.py` uses the column only for x1, x2 and x5, which carry no
filter, and figure 39 and the deck price bandwidth from [[D92]]'s formulas.
Re-deriving from the formulas is correct; re-reading the column is not. Separately,
`config_fingerprint` hashes the resolved config, which now has one more field, so
a checkpoint written before this will not resume. Finished runs are unaffected,
and nothing was mid-run.

**X27** pairs the two points on X25's seven cells — three still, two drifting and
their twins — at X25's filter settings and seeds, both learners in one run so each
comparison is paired against the same data stream. The sender arm must reproduce
X25's `diffusion_ekf_onehop_mean` to the digit, since this note changed the
ledger and not the filter, which makes X27 a reproduction check as well. About 7 h
by the probe's step time. ❓ **Open:** whether to run it, and at three seeds or
five. ⚠ *Decided 2026-09-15: five seeds, about 12 h. X25 ran three, so the
reproduction check compares per seed on the three both share.*

### ✅ D95. Mackey–Glass is planned before it is built, and every choice was asked

The supervisor approved the regression task on 2026-09-15. Before any code, every
open question was put to the user with options and a recommendation; the answers,
the work packages and the battery are in `docs/mackey_glass_plan.md`, which is the
authority for the task from here on.

Three answers depart from the recommendation and are worth recording as choices.
**Calibration adds PIT histograms**, beyond NLL, coverage and the variance ratio.
**Heterogeneity is all three kinds** — per-agent $\beta$ offsets, noise $\sigma_v$ and
delay $\tau_v$ — not the recommended first two. **The $\beta$ range is the pilot's to
widen** if roughly 0.1–0.4 proves too narrow. Two are staged: fixed sinusoidal
positions first, a learned embedding if order proves a problem; sensor gain/bias
drift first, $\tau$ drift if time.

Five quantities are deliberately left to the pilot (M0) rather than guessed: the
noise level $\sigma$, the usable $\beta$ range, whether $\boldsymbol R$ is diagonal
($|\rho_1|<0.2$, and watched throughout rather than settled once), whether it is
per-position ($>2\times$ variance across positions), and the centre of the $\boldsymbol R$
grid. The pilot is also a hard gate: one agent alone must trail the pooled learner by
more than three times the seed noise, or the task is hardened before anything else is
built.

One operational rule follows from the repository rather than the task. Several
modules import lazily inside functions, so editing shared source in the tree an MNIST
sweep runs from can change that sweep mid-run. Mackey–Glass is therefore developed in
a separate git worktree and merged only between sweeps, with the full suite green.

### ✅ D96. M0: the task separates methods at every noise level, and the chaos is narrow

The Mackey–Glass pilot (`scripts/pilot_mackey_glass.py`, 54.5 min CPU,
`results/m0_pilot/report.json` in the worktree). Online gradient learners only:
ten solo learners against one pooling all ten agents' blocks, five seeds, $T=1500$,
each optimiser at its own rate chosen on two seeds. Standardisation constants:
mean 0.930075, std 0.226215 ($\beta=0.2$, $10\times10\,000$ samples, seed 0).

**The gate passes everywhere** — solo minus pooled, in multiples of the pooled
learner's seed noise:

| $\sigma$ | SGD | momentum | AdamW | best pooled / floor | persistence |
|---|---|---|---|---|---|
| 0.01 | 3.4× | 21.2× | 11.6× | 5.50 | 0.147 |
| 0.05 | 4.1× | 19.5× | 39.9× | 1.87 | 0.162 |
| 0.10 | 7.4× | 7.7× | 57.7× | 1.52 | 0.203 |

AdamW is the best pooled learner at every $\sigma$ (0.055, 0.093, 0.152), and it
chose $10^{-2}$, **the top of its grid**, every time — M3's AdamW grid must reach
higher. Plain SGD passes but barely at low noise, which is the pilot's argument
for having added AdamW (decision 19).

**The residual structure decides whether the filter's $\boldsymbol R$ is right**, and it
depends on $\sigma$ (a converged offline model, 20 000 held-out blocks):

| $\sigma$ | $\rho_1$ | variance ratio (from pos. 2) | $\boldsymbol R$ centre |
|---|---|---|---|
| 0.01 | **+0.372** | 51.6 (5.4) | $1.58\times10^{-3}$ |
| 0.05 | **−0.206** | 6.4 (3.3) | $6.86\times10^{-3}$ |
| 0.10 | −0.087 | 3.2 (2.8) | $1.93\times10^{-2}$ |

At low noise the model's own error dominates and is smooth across positions
(positive correlation); as noise grows the observation noise takes over, and at
0.05 the sign flips negative mid-block (−0.30). **Only $\sigma=0.1$ meets the
diagonal rule** ($|\rho_1|<0.2$, decision 13). The per-position rule
(ratio $>2\times$, decision 14) fires at **every** $\sigma$: position 1, predicting
from one sample, is the worst by far, and the variance falls steadily to position
31.

**The chaos is confined to $\beta\in[0.20,0.24]$.** The divergence rate of two
histories $10^{-8}$ apart is $+0.0039$ at 0.20, $+0.0074$ at 0.22, $+0.0048$ at
0.24, and zero to four decimals elsewhere across $[0.10,0.40]$: periodic, or a
fixed point at 0.12 (std 0.000). A model converged at $\beta=0.2$ ($\sigma=0.05$)
loses RMSE in both directions — +0.0072 at 0.18, +0.054 at 0.16, +0.032 at 0.22,
+0.077 at 0.24, +0.135 at 0.26 — so a smaller $\beta$ is not an easier task for a
model trained at 0.2, unlike the smoke run's under-trained model suggested.
The amplitude grows with $\beta$ (std 0.13 at 0.16, 0.30 at 0.24).

**What this leaves open.** The noise level; how the drift is placed against a
chaotic window only 0.04 wide — the provisional span of 0.04 fits it exactly
upward from 0.2, but a recurring schedule jumps both ways and would leave it; and
the $\boldsymbol R$ grid, whose centre and per-position shape now come from the table
above at whichever $\sigma$ is chosen. ❓ **Open**, put to the user.

**Decided 2026-09-15**, all three as recommended:

1. **$\sigma=0.1$** — the only level meeting the diagonal rule, with the strongest
   separation (57.7× the seed noise) and the pooled learner still 1.52× above the
   floor. Per-position $\boldsymbol R$ is needed regardless.
2. **$\beta_0=0.22$, span 0.02.** Every cell sits at the centre of the chaotic
   window, the most chaotic point measured ($\lambda=0.0074$), and the full span
   reaches exactly its edges: linear drift runs 0.22→0.24, and recurring jumps of
   15 "degrees" (0.0067) reflect inside $[0.20,0.24]$. No condition leaves chaos,
   so a drift changes the law's parameters, never the kind of dynamics. The
   standardisation constants stay those of $\beta=0.2$ — fixed, never per run.
3. **$\boldsymbol R$ = the pilot's per-position profile × one tuned scale.** The shape is
   measured, the level is tuned — one grid axis, not 31. ⚠ The pilot measured the
   profile at $\beta=0.2$; it is re-measured at the chosen law ($\beta=0.22$,
   $\sigma=0.1$), for the linear AR as well as the Transformer, before M4.

**Re-measured at the chosen law** (`scripts/measure_r_profile.py`, four seeds'
agents fitted, the fifth held out):

| model | RMSE | $\boldsymbol R$ centre | per-position ratio | $\rho_1$ |
|---|---|---|---|---|
| Transformer (offline AdamW) | **0.1526** | $2.20\times10^{-2}$ | 3.69 | −0.118 |
| linear AR(31) (least squares) | 0.1844 | $3.40\times10^{-2}$ | 3.13 | +0.017 |
| persistence | 0.2265 | — | — | — |

Both meet the diagonal rule at this law, and both need a per-position $\boldsymbol R$. The
Transformer beats the linear AR's *optimum* by 0.032 — decision 11's question,
whether the Transformer is needed at all, answered offline before any online run:
it is. The two profiles differ in level by 1.5× and are recorded separately, each
in its own model config, as the centre of its $\boldsymbol R$ grid.

### ✅ D108. M11: drift depends on what changed, not how much — and a confound in the pairing

`scripts/run_m11_sensor_drift.py`, four cells (gain and bias × linear and abrupt) ×
five seeds × six learners, 19:42–01:56 on 2026-09-21/22 (≈6 h 15 min GPU). **Zero
divergences.** Filters carry M4's and M5's selections, gradient learners M3's rates by
schedule, so these are the same learners M6 ran. **Closes D106's gap 5.**

**Gate.** `drift_state` spans exactly what the design asked: gain 1.0000→1.0999
(linear, 1 500 distinct) and 0.9000→1.1000 (abrupt, 7 levels); bias 0.0000→0.0999 and
$-0.1000$→$+0.1000$. `span = 0.1` puts the sensor one observation-noise sd off at full
displacement on both channels. Channel dispatch was verified in the code
(`law_blocks` dispatches on `series.channel` for β, gain and bias, and `generate` does
the same per sample) **and** empirically: bias shifts the mean by exactly $0.1000$ with
a std ratio of $1.0000$, gain scales the spread by $1.0989$, β changes the dynamics
($1.1386$).

**The headline: damage follows the dimensionality of what changed, not the magnitude
of the perturbation.** All three channels were scaled to one $\sigma$. Cell minus
cell, paired per seed — the stationary twin appears in both damages and cancels
exactly, which the next section explains is essential:

| linear schedule | bias − gain | gain − β | bias − β |
|---|---|---|---|
| centralised EKF | $-0.0028$ (3.6σ) | $-0.0018$ (3.9σ) | $-0.0046$ (5.9σ) |
| diffusion, one-hop | $-0.0027$ (4.2σ) | $-0.0021$ (3.8σ) | $-0.0049$ (6.0σ) |
| local only | $-0.0077$ (9.1σ) | $-0.0055$ (7.1σ) | $-0.0132$ (11.0σ) |

All six learners, all five seeds, the same sign: **bias < gain < β**. A scalar offset
is absorbed outright, a scalar multiplier costs about half a law change, and changing
the dynamics costs most. The weakest learner is the most sensitive to *which* channel
drifts (`local_only` spans $-0.0132$ where the filters span $-0.0046$).

⚠ **Under `abrupt` the ladder collapses at the top.** gain − β is $-0.0003$ to
$-0.0017$ at **0.0–0.7σ — not resolved**. Both schedules reflect at the 45° cap and
keep returning, so a bounded amplitude change and a bounded law change cost the same.
Bias stays clearly below both ($-0.0033$ to $-0.0069$, 1.7–2.8σ).

**⚠ The confound: absolute damage is the wrong quantity at five seeds.** Damage is
`cell − stationary twin`, and `_build` draws the held-out set per (seed, channel
value) — `numpy_rng("stream", "eval", key)`, derived from the run's master seed. Seed
3's **twin** draw is the easiest of the five for five of six learners (centralised
0.1342 against 0.1397–0.1411; one-hop 0.1360 against 0.1399–0.1437), which inflates
damage at seed 3 in *every* cell for *every* learner. Because the twin term is common
to all cells:

- **cell-minus-cell differences cancel it exactly** — the ladder above is clean;
- **absolute damage does not**, and its *sign* is not trustworthy.

Concretely: the bias cells' apparently **negative** damage ($-0.0016$ to $-0.0038$, up
to 5.3σ, five of five seeds on two arms) is **not** evidence that drift helps. It
rides on the twin draw and is withdrawn. Measured draw noise: sd $0.0033$ on
persistence RMSE across repeated draws of the same law, against a base of $0.2264$.
Within-cell learner contrasts share one eval set and so cancel the draw — which is
why they resolve far better than damages do.

**The prediction on record is refuted.** The runner's docstring predicted one-hop's
advantage would shrink under a sensor drift, since a neighbour's raw batch informs
about a moving *law* while every agent's sensor moves identically. It does not
collapse:

| condition | median advantage | SE |
|---|---|---|
| β linear | $+0.0019$ | 6.2 |
| β abrupt | $+0.0018$ | 3.8 |
| gain linear | $+0.0008$ | 2.6 |
| gain abrupt | $+0.0010$ | 4.3 |
| bias linear, `current` | $+0.0003$ | 1.3 |
| **bias linear, `canonical`** | $\mathbf{+0.0028}$ | **5.1** |

Under bias it is unresolvable on `current` — but there is no tracking error for anyone
there — while on `canonical` it is $+0.0034$ at 5.1σ, **positive on all five seeds**.
One-hop's fit to the undrifted law degrades far less even where tracking is identical.
D106's own stationary column was already refuting the mechanism: one-hop beats local
adapt 0.1408 to 0.1442 with no drift at all to carry information about. The advantage
is about better-informed agents, not about tracking a moving law.

**`current` and `canonical` measure different things and both are needed.**

| | `canonical` (how far the fit moved) | `current` (tracking error) |
|---|---|---|
| bias | $+0.011$–$0.015$ | $\approx 0$ |
| gain | $+0.014$–$0.016$ | $+0.0025$–$0.0038$ |
| β | $+0.026$–$0.030$ | $+0.0044$–$0.0065$ |

A bias drift is **perfectly tracked, not absent**: the learner absorbs the offset so
completely that predicting under the biased sensor costs nothing, while its fit to the
*unbiased* law degrades by $0.011$–$0.015$ at 4.4–5.1σ. "Bias is inert" would have
been wrong.

**Calibration scales with error rather than disproportionately.** The runner argued a
sensor drift is a mis-specified $\boldsymbol R$ and should stress calibration harder.
It does not: `variance_ratio` damage scales by 0.53–0.56 against RMSE's 0.52–0.58 under
gain. ⚠ Those are 1.1–2.1σ, with spreads exceeding the means — weak either way.

**A lead, not a finding.** The distributed one-hop filter appears to move *less* from
the true law than the centralised one ($+0.0042$ at 3.6σ under β-abrupt, $+0.0010$ at
2.5σ under bias-linear) while centralised stays better on `current`. Resolved in only
2 of 5 cells and $-0.0002$ in a third.

**Seed structure worth carrying.** Seed 3 has the easy *twin* draw and inflates every
damage; seed 4 shows the largest one-hop advantage in all ten comparisons, and that one
is genuine, because the paired contrast cancels the draw. Medians, not means.

**What this does not measure.** One graph, one $N$. Per-agent $\tau$ heterogeneity, the
other half of D106's gap 5, is still unrun. And the abrupt cells cannot rank the
channels at five seeds.

### ✅ D107. The shift transient: a finer cadence, and the confound that nearly inverted it

`scripts/run_m6_shift_cycles.py --device cuda`, one cell (`m6cyc_abrupt`) × five
seeds × four learners at `eval_every = 5`, 12:07–14:27 on 2026-09-21 (≈2 h 20 min
GPU). Zero divergences. It writes no selection and nothing M6 reads. **Closes gap
6 of D106.**

**The finer cadence changed the resolution and nothing else.** M6's abrupt grid is
phase-0-only by construction: its 61 recorded steps carry the phase histogram
$\{0:60,\ 24:1\}$, the lone outlier being the final off-grid evaluation at
$t=1499$. Restricting the new run to phase 0 reproduces M6's abrupt column.

| arm | M6 | finer run, phase 0 | difference |
|---|---|---|---|
| centralised EKF | 0.1408 | 0.1409 | $+0.0001$ |
| diffusion, local adapt | 0.1485 | 0.1488 | $+0.0003$ |
| diffusion, one-hop | 0.1428 | 0.1429 | $+0.0002$ |
| ATC AdamW | 0.1675 | 0.1677 | $+0.0002$ |

An order of magnitude tighter than the 0.0030 seed effect D105 measured. The naive
comparison — all five phases against M6's one — reads as a uniform $-0.0001$ to
$-0.0016$ shift, and that is the phase mix, not the cadence.

**The confound, which is why this note exists.** Detrending each cycle by its own
mean removes the cycle's *level*; it does nothing about a slope *within* the
cycle. Phase 0 leads and phase 20 trails, so while a learner is still converging
from the prior, a globally decaying error curve manufactures a phase profile out
of nothing.

| window | centralised | local adapt | one-hop | ATC AdamW |
|---|---|---|---|---|
| cycles 0–11 | $+0.0241$ | $+0.0477$ | $+0.0270$ | $+0.0822$ |
| all 60 cycles | $+0.0054$ | $+0.0116$ | $+0.0067$ | $+0.0168$ |
| **cycles 30–59** | $+0.0008$ | $+0.0025$ | $+0.0015$ | $+0.0004$ |

The artefact runs to thirty times the effect and **ranks the arms in the opposite
order**, because the slowest converger collects the largest fake wound. Pooling
all 60 cycles reproduces it almost exactly. The fix is to *drop* the early cycles,
not to detrend them — a distinction worth carrying to any phase-aligned figure.

**The window is not doing any work.** Cycles 30–44, cycles 45–59 and a linear
detrend in $t$ across the whole late window return the same four amplitudes to
four decimals.

**The result.** Wound $=$ RMSE at the jump step minus 20 steps later, late window,
spread over five seeds:

| arm | wound | sd |
|---|---|---|
| diffusion, local adapt | $+0.0025$ | 0.0003 |
| diffusion, one-hop | $+0.0015$ | 0.0003 |
| centralised EKF | $+0.0008$ | 0.0002 |
| ATC AdamW | $+0.0004$ | 0.0003 |

**The wound is ordered by how well informed the filter is** — the same ordering
D106 found for settled error, now on a second and independent quantity. AdamW's is
the smallest of the four, and that is not robustness: it sits ≈0.026 RMSE above
every filter at every phase and barely reacts to a $\beta$ step at all. A method
that does not respond to the shift cannot show a response to it.

⚠ $+0.0004$ is small, not absent. Per seed it is $+0.0005$, $+0.0002$, $+0.0000$,
$+0.0007$, $+0.0006$ — about three standard errors from zero. The figure's bars
are the seed *spread*, which is the wider quantity.

**The whole transient lives in the first five steps**: every filter is back at or
below its cycle mean by phase 5. ⚠ That is also the resolution limit. At
`eval_every = 5` the interior of that first interval is unobserved, exactly as
`jump_every = eval_every = 25` hid the cycle in M6. `--eval-every 1` would resolve
it at five times the evaluation cost, and is the natural follow-up if the shape of
the recovery ever matters.

Figure MG12 in `make_mg_results_figures.py`.

**Amended 2026-09-21: the interior is observable after all, and three claims above
are withdrawn.**

The caveat that `--eval-every 1` is needed to see inside the first interval is
**wrong for the prequential metrics**. `rmse`, `mse` and `nll` on the `prequential`
evalset are recorded at *every* step — 1 500 points, not 301 — because prequential
scoring is test-then-train on the incoming stream and needs no held-out set. All 25
phases of every cycle are already on disk. Only the held-out `current` set is tied
to `eval_every`, and that restriction is what the rest of this note measures.

**The transient is one step, not five.** Deviation from the cycle mean at the jump
step, late window, against the seed spread:

| metric | centralised | local adapt | one-hop | ATC AdamW | best $p_1$ |
|---|---|---|---|---|---|
| `rmse` | 5.1σ | 5.0σ | 4.5σ | 0.7σ | 1.5σ |
| `nll` | 6.5σ | 4.6σ | 4.6σ | 0.4σ | 1.9σ |
| `mse` | 4.8σ | 4.8σ | 4.3σ | 0.5σ | 1.5σ |

By phase 1 everything is inside a noise floor of ~0.0010, which phases 10–24 fix
independently. So:

1. ⚠ **"The whole transient lives in the first five steps" is withdrawn.** Five
   steps is the resolution of the `current` grid, not a property of the transient.
   On the per-step stream the instantaneous cost is confined to the jump step.
2. ⚠ **The informed ordering at phase 1 is withdrawn.** It looked like local adapt
   $+0.0020$, one-hop $+0.0012$, centralised $+0.0006$; those are 1.5σ, 0.9σ and
   0.4σ. Five seeds do not support it.
3. ⚠ **A lagged disagreement peak is withdrawn.** `max_pairwise_distance` appears
   to peak five steps *after* the jump rather than at it, but $p_5-p_{20}$ is
   $+0.0028\pm0.0021$ and $+0.0035\pm0.0026$ — about 1.3σ. `e_agree` shows no
   transient at all: the combine absorbs the shift without the agents visibly
   diverging.

**At the jump step the three filters are indistinguishable** ($+0.0043$, $+0.0038$,
$+0.0038$, all $\pm0.0008$) and ATC AdamW shows no resolvable hit. The instantaneous
surprise is the *shift's own size* and is the same for every filter. What separates
them — the 0.0008 / 0.0025 / 0.0015 ordering at 4–8σ on the held-out set — is how far
the **model** is knocked off the new law, not how badly the next prediction misses.

**Calibration takes the same shape, and harder.** These reproduce M6 at phase 0
($+0.0012$, $+0.0041$, $+0.0024$ on `variance_ratio`; $-0.0004$, $-0.0007$,
$-0.0007$ on `coverage_90`), so they are gated exactly as the error figures are.
Wound at the jump step, `current`:

| arm | `variance_ratio` | `predictive_nll` | `coverage_90` |
|---|---|---|---|
| centralised | $+0.0077$ (0.0016) | $+0.0049$ (0.0010) | $-0.0008$ (0.0014) |
| local adapt | $+0.0409$ (0.0051) | $+0.0206$ (0.0026) | $-0.0066$ (0.0017) |
| one-hop | $+0.0245$ (0.0049) | $+0.0125$ (0.0024) | $-0.0046$ (0.0013) |

The same ordering again, on a third independent quantity. Coverage *falls* at the
jump — the interval misses more often while the filter's uncertainty catches up —
and the gradient baselines are absent because they report no predictive covariance.

**Correction to D106's calibration arc.** D106 records "under-confident at
stationary (0.81–0.84), over-confident under linear (1.07–1.14), near-nominal under
abrupt (0.968–1.000, coverage 0.900–0.906)". Recomputed from the M6 parquets over
all three conditions and both evalsets:

| `variance_ratio`, $\gamma=1$ filters | stationary | linear | abrupt |
|---|---|---|---|
| `current` | 0.806–0.842 | 0.835–0.899 | 0.818–0.892 |
| `canonical` | 0.806–0.842 | 1.230–1.373 | 1.092–1.118 |

Only the stationary figure reproduces. **1.07–1.14 is not linear on either
evalset** — it is `canonical` under *abrupt* (1.092–1.118). And **0.968–1.000
appears nowhere**, in any condition, on either evalset; the nearest value in the
study is 0.9688, which is `centralized_ekf_walk`'s realised coverage at *nominal
0.95* in the MG8 cache, not a variance ratio.

**Rechecked 2026-09-21, before retracting anything.** Six routes: a committed script
that computes the arc (none exists); D106's own commit (it carries `design_notes.md`
and nothing else, so no report shipped with it and none was deleted); four
aggregations — settled, whole-run, final point, first half — across both evalsets;
all seven arms and their subsets; the `m6_*_smoke` cells; and node selection (pooled,
a single node, and a `mean` row, which the calibration metrics do not carry at all).

- **1.07–1.14 never reproduces exactly.** The nearest reproductions are `canonical`
  under *abrupt*: $1.092$–$1.118$ settled, $1.080$–$1.134$ whole-run. Whole-run
  canonical *linear* matches the lower bound ($1.069$) but runs to $1.173$.
- **0.968–1.000 reproduces nowhere**, by any route. The largest value in a
  comparable slice is $0.964$.
- **The arc exists only on `canonical`.** On `current` all three conditions sit at
  $0.81$–$0.90$ with linear and abrupt almost equal, so there is no arc to report.
  D106 names no evalset, and that omission is the defect that matters most — the
  same paragraph is true or false depending on a choice it never states.

**The conclusion survives; only the numbers fail.** On `canonical` the ordering is
stationary $0.806$–$0.842$, abrupt $1.092$–$1.118$, linear $1.303$–$1.373$: abrupt
*is* nearest nominal among the drifted conditions, so "the uncertainty model fits
best under the schedule that keeps returning" holds on the measured data. The
paragraph should therefore be **restated with measured values and a named evalset,
not withdrawn**. Recording that here rather than editing D106's claim directly: the
inference is the supervisor's to keep or drop.

### ✅ D106. M6: the main comparison, and an online filter that beats its offline reference

`scripts/run_m6_comparison.py`, six cells (three conditions × two groups) × five
seeds × fourteen learners, 11 h 55 min GPU on 2026-09-20. **Zero divergences in
any cell.** Graph: Erdős–Rényi $p=0.3$ at $N=10$, Metropolis weights. Filters
carry M4's and M5's selections, gradient learners M3's rates, per condition.

**The headline is a ladder, at $\beta=0.22$ — the law the stationary condition
runs, and the level M2 measured.**

| | RMSE | |
|---|---|---|
| noise floor | 0.1000 | unreachable |
| full context | 0.1177 | sees the whole window |
| **centralised filter** | **0.1391** | online, one pass |
| **diffusion, one-hop** | **0.1408** | online, distributed |
| $e^\star$, offline Transformer | 0.1433 | trained to convergence |
| diffusion, local adapt | 0.1442 | |
| best gradient baseline | 0.1628 | AdamW, centralised |
| worst gradient baseline | 0.1899 | local-only |
| persistence | 0.2264 | |

**The budgets are equal by construction, which is what makes this a result rather
than an artefact.** `ReferenceSettings` gives $e^\star$ $40\times375=15\,000$
blocks — "$N\times T$ blocks, the pooled data an online run sees" — and the series
environment sizes each agent's per-step batch from `series.n_blocks = 1`, so the
online budget is $N\times T\times n_b = 10\times1500\times1 = 15\,000$. (The
config's `env.samples_per_node_per_step = 4` is an inert MNIST-inherited field
here; reading it as the block count inflates the online budget fourfold and
invalidates the comparison.) So **an online filter making a single pass beats a
Transformer trained to convergence on identical data, and so does the distributed
one** — each agent seeing a tenth of the stream and exchanging only means.

⚠ Indicative, not a significance test. $e^\star$ is scored on 4 000 held-out
blocks; M6's `current` set is 32 blocks per seed. The 0.0042 margin is about
three standard errors of the filter's own seed spread (0.0028), and $e^\star$
carries no error bar at all. The *ordering* is safe; the margin is not.

**Full covariance sharing buys nothing — on error and on uncertainty.** Paired
per seed, six measurements:

| | stationary | linear | abrupt |
|---|---|---|---|
| one-hop | $+0.0000$ (0.0003) | $-0.0003$ (0.0004) | $-0.0001$ (0.0001) |
| local adapt | $-0.0004$ (0.0005) | $-0.0004$ (0.0007) | $+0.0001$ (0.0005) |

All within $\pm0.0004$, the tightest at $\pm0.0001$, for $p(p+1)/2 = 2\,584\,401$
scalars against a mean's 2 273 — about 1 100× the payload. Calibration moves no
further: the largest paired difference on `variance_ratio`, `coverage_90/95` or
`predictive_nll` is ~0.013 (about 1.5%), and it shifts the *same* direction in
both conditions, so it improves calibration under drift and worsens it at
stationary. D100 held only for error, on one task; this holds for error and
uncertainty, on a second task, second architecture, second problem class.

**One-hop is what makes the filter viable when distributed — the finding is in
the row beneath the one we were watching.** Centralised minus its diffusion form:

| family | stationary | linear | abrupt |
|---|---|---|---|
| filter, one-hop | $+0.0017$ | $+0.0021$ | $+0.0020$ |
| **filter, local adapt** | $+0.0052$ | $+0.0073$ | $+0.0077$ |
| SGD + momentum | $+0.0029$ | $+0.0025$ | $+0.0029$ |
| AdamW | $+0.0031$ | $+0.0028$ | $+0.0018$ |

The prediction that filters lose *less* from decentralising than gradient methods
is **not supported**: one-hop's $+0.0017$–$0.0021$ overlaps SGD and AdamW, and
AdamW is nominally better under abrupt. But without the one-hop adapt the filter
pays two to three times what gradient methods pay. One-hop is not a refinement of
the adapt scope; it is the repair that restores parity.

**Cooperation dwarfs every architectural contrast here**: local-only minus the
diffusion form is $+0.0111$ to $+0.0146$ for both gradient families across all
three conditions, an order above the adapt scope ($+0.0035$–$0.0057$) or the
distance to centralised ($+0.0017$–$0.0021$). ATC's momentum mixing buys
$+0.0081$–$0.0092$ for $2p$ per link rather than $p$ — a datapoint track C wants.

**The $\gamma$ reference arms earned their place** (P5.24). Mis-tuning $\gamma$ to
0.9995 costs the diffusion filter $+0.0097$, $+0.0097$, $+0.0091$ across the three
conditions and the centralised filter only $+0.0018$, $+0.0021$, $+0.0017$ — a
fivefold asymmetry, stable everywhere, and **1.7–2.8× the adapt-scope effect it
would otherwise be confounded with**. A comparison run at one shared setting is
confounded by an effect larger than the architectural difference it measures, in
every regime, not just the one X20 caught it in.

They also produced a result the RMSE table cannot show. Under **stationary** the
$\gamma=1$ filters are under-confident (`variance_ratio` 0.81–0.84, over-covering
at 0.927 against nominal 0.90), and the $\gamma<1$ arm corrects exactly that:
1.0020 and 0.8975, the best-calibrated filter in the study. Under **linear** the
$\gamma=1$ filters are already over-confident (1.07–1.14) and the same shift makes
it worse (1.1386). `predictive_nll` is worse for the $\gamma$ arm in all three
conditions, because better calibration does not pay for $+0.0097$ of error. So
$\gamma=1$ stands — on error everywhere, on likelihood everywhere, on calibration
everywhere except a stationary law.

**Calibration has an arc worth reporting on its own**: under-confident at
stationary (0.81–0.84), over-confident under linear (1.07–1.14), near-nominal
under abrupt (0.968–1.000, coverage 0.900–0.906). The uncertainty model fits best
under the schedule that keeps returning.

⚠ **Corrected 2026-09-21 — see D107.** Recomputing this from the M6 parquets
reproduces only the stationary figure, and a six-route recheck could not source the
other two: 1.07–1.14 is nearest `canonical` under *abrupt* (1.092–1.118), and
0.968–1.000 appears nowhere at all. The arc also exists **only** on `canonical` — on
`current` there is no arc — and this paragraph names no evalset, which is the defect
that matters most. **The conclusion stands:** on `canonical`, abrupt (1.092–1.118)
is nearer nominal than linear (1.303–1.373), so the uncertainty model does fit best
under the schedule that keeps returning. The sentence needs restating with measured
values and a named evalset, not withdrawing.

**Drift costs, paired.** The one-hop filters pay least under linear
($+0.0044$–$0.0047$ against the baselines' $+0.0063$–$0.0108$), but the claim
"every filter pays less than every baseline" is false at the boundary: local adapt
pays $+0.0065$ and `diffusion_atc_adamw` $+0.0063$. Under abrupt every cost is
small with a spread of ~0.005 — `recurring` reflects at the 45° cap (D74), so the
target keeps returning and barely damages anyone.

**What this does not measure.** Four gaps, in the order they matter:

1. **There is no non-cooperating filter arm.** Every filter here communicates;
   even `diffusion_ekf` combines means. So the filter's own cooperation gain — the
   number we can quote for both gradient families — is unmeasured, and "compared
   to not distributing at all?" has no answer. The natural route is an edgeless
   graph: Metropolis weighting gives $a_{vv}=1$ at degree zero, so the combine
   becomes the identity and `diffusion_ekf` becomes a per-agent EKF. It needs **no
   new learner and no new topology**: `disconnected` at `n_components = N` builds
   an edgeless graph — verified, `build_graph(topology="disconnected", n_nodes=10,
   params={"n_components": 10})` returns 0 edges, 10 components, and a Metropolis
   weight diagonal of exactly 1.0000. Only a probe script is new, in the pattern of
   `run_m5_linear_probe.py`: three cells at five seeds, paired per seed against the
   existing `m6_*_a` cells. About three GPU-hours. **This is the next run.**
2. **One graph, one $N$.** ER $p=0.3$ at $N=10$ only; P5.3's topology sweep and
   the $N$ ladder have no Mackey–Glass analogue yet.
3. **M5 tuned on `m_abrupt` alone** and carried to all three conditions. The
   linear probe is the only evidence the setting travels; `m_stationary` was never
   tuned, and the carried setting's ordering held there regardless.
4. **The $e^\star$ comparison rests on a 32-block evaluation set** against
   $e^\star$'s 4 000. Raising `series.eval_blocks` would tighten the headline, at
   the cost of re-running the battery.
5. **One drift channel of three.** `env/series.py` implements `beta`, `gain` and
   `bias`; every cell here drifts $\beta$. Gain and bias move the *sensor* rather
   than the law, so they are a different kind of non-stationarity — closer to the
   image task's prior drift than to its rotation — and neither has been run. They
   are queued as M11 alongside per-agent $\tau$. Until then every claim here is a
   claim about a drifting **law**, not about drift in general.
   **Closed 2026-09-22 by D108** for the channel half: `run_m11_sensor_drift.py` ran
   four cells and found **bias < gain < β** — damage follows the dimensionality of
   what changed, not the size of the perturbation, with a scalar offset absorbed
   outright. ⚠ Per-agent $\tau$ heterogeneity, the other half of this gap, is still
   unrun, so "a claim about a drifting law" now has a measured comparison but the
   heterogeneity question stands.
6. **The within-cycle transient is unobserved.** `jump_every` and `eval_every` are
   both 25 on the abrupt condition, so every recorded step lands on a jump
   boundary: MG11 can rank what the drift *costs* and nothing here shows what a
   single shift *looks like*. `scripts/run_m6_shift_cycles.py` re-runs that
   condition at a finer cadence for exactly this. **Closed 2026-09-21 by D107**,
   which also records why the obvious way to read that run is wrong.

**Amended 2026-09-21: gap 1 is closed.** `scripts/run_m6_isolated.py`, three cells
× five seeds, 1 h GPU, zero divergences. `diffusion_ekf` on an edgeless graph
(`disconnected` at `n_components = N`, where Metropolis weighting gives
$a_{vv}=1$, so the combination matrix is $\boldsymbol I$) is a per-agent EKF that
never communicates — the same learner, settings and seeds as its connected twin,
so the paired difference is the communication and nothing else.

| paired, isolation minus connected | stationary | linear | abrupt |
|---|---|---|---|
| communication alone | $+0.0038$ | $+0.0031$ | $+0.0028$ |
| whole deployable design | $+0.0073$ | $+0.0083$ | $+0.0085$ |
| *for scale:* SGD family | $+0.0120$ | $+0.0137$ | $+0.0118$ |
| *for scale:* AdamW family | $+0.0111$ | $+0.0131$ | $+0.0146$ |

**The decomposition is exactly additive, in all three conditions**, against the
adapt-scope gaps this note measured independently: $0.0038+0.0035=0.0073$,
$0.0031+0.0052=0.0083$, $0.0028+0.0057=0.0085$. Communication and adapt scope
contribute separately.

**And they dissociate under drift.** The adapt-scope term widens
($+0.0035\to+0.0057$) while the communication term *narrows*
($+0.0038\to+0.0028$) — the reverse of what was predicted for the second of them.
**Sharing estimates helps less when the target moves; sharing data helps more.**
That favours the receiver-point one-hop design specifically, whose first message
is the raw batch rather than a belief.

**The headline is now causal rather than merely true.** At $\beta=0.22$ the
isolated filter scores 0.1480 — *above* $e^\star$'s 0.1433 — while the one-hop
form scores 0.1408, below it. So it is the communication that carries the online
filter past a Transformer trained to convergence, not the filtering alone.

Two framings survive together, and the second is the stronger. Cooperation buys
the filter about 60–65% of what it buys a gradient method. But the filter's
*non-communicating* form (0.1480–0.1538) beats the best *communicating* gradient
baseline (0.1628–0.1694) in every condition, by more than cooperation is worth to
either. The method dominates the cooperation; the cooperation then clears
$e^\star$.

### ✅ D105. M5: the tie-break that could not break the tie, and the image task's $q$ pattern inverts

`scripts/run_m5_diffusion.py` (16 cells × 2 seeds, ≈4.3 h GPU), its `--tie-break`
(2 cells × 5 seeds, ≈1.3 h), and `scripts/run_m5_linear_probe.py` (2 cells × 2
learners × 2 seeds, ≈1 h). `results/m5_selection.json`. **Zero divergences in any
of the twenty cells**, at every $q$ including the largest.

**Selected: $\gamma=1$, $q=6\times10^{-7}$, $\sigma_0^2=0.01$** — settled RMSE
0.1428 at five seeds on `m_abrupt`, for `diffusion_ekf_onehop_mean_receiver`, the
variant D99 adopted. Carried unchanged to all four diffusion variants and all three
conditions in M6, as X14's discipline requires.

| $q$ | $\gamma=1,\sigma_0^2=0.01$ | $\gamma=1,10^{-3}$ | $0.9995,0.01$ | $0.9995,10^{-3}$ |
|---|---|---|---|---|
| $6\times10^{-7}$ | **0.1458** | 0.1508 | 0.1562 | 0.1662 |
| $6\times10^{-6}$ | 0.1463 | 0.1486 | 0.1499 | 0.1518 |
| $6\times10^{-5}$ | 0.1536 | 0.1548 | 0.1545 | 0.1546 |
| $6\times10^{-4}$ | 0.2000 | 0.1970 | 0.1974 | 0.1938 |

**The grid was extended downward, and the extension returned a null.** The first
twelve cells put the argmin on the bottom row, so the discipline the docstring
wrote for the upward case was applied symmetrically: one step down, not a blind
decade. Per-decade gains were $0.046$, then $0.0073$, then **$0.0005$** — below
`THRESHOLD`. The extrapolation had predicted $\approx0.001$, so the measurement
was not needed to *guess* the answer; it was needed because M4's grid bottoms out
at the same $q=6\times10^{-6}$, and extending M5's search without extending M4's
would have searched the diffusion filter finer than the baseline it is measured
against. The null is what keeps M6 like-for-like. M4 was therefore left alone.

Note the extension also *narrowed* where the filter is safe: the spread across the
four $\gamma\times\sigma_0^2$ cells widens from 0.0055 at $6\times10^{-6}$ to
0.0204 at $6\times10^{-7}$. Only the best corner improved; the other three got
worse by 0.0022, 0.0063 and 0.0144. That is an optimum flattening, not a trend
with further to run, and it is the argument against a third decade.

**The tie-break did not break the tie.** Two cells fell inside `THRESHOLD`, and at
five seeds both moved by *exactly* $-0.0030$:

| $\gamma$ | $q$ | $\sigma_0^2$ | 2 seeds | 5 seeds | moved |
|---|---|---|---|---|---|
| 1 | $6\times10^{-7}$ | 0.01 | 0.1458 | 0.1428 | $-0.0030$ |
| 1 | $6\times10^{-6}$ | 0.01 | 0.1463 | 0.1433 | $-0.0030$ |

Identical movement is a **seed effect, not a cell effect** — seeds 2–4 are kinder
than 0–1 — and the gap is 0.0005 before and after. The two-seed winner agreed, but
agreement at a resolution where the gap is a quarter of the threshold is not a
separation. Five more seeds would buy nothing; the cells are the same.

**A hypothesis, and its refutation.** The plateau made the choice ours, so it was
argued on robustness: $6\times10^{-7}$ is the least adaptive setting in the grid,
tuned on `m_abrupt` — whose schedule is `recurring`, which **reflects at the
45-degree cap (D74)**, so the target keeps returning to where the filter already
is. That is the condition least able to expose under-adaptation. The prediction was
that on `m_linear`, which marches once to 45 degrees and never returns,
$6\times10^{-7}$ would under-track; and that `diffusion_ekf` (local adapt) would
suffer most, since by D87 an agent holding $1/N$ of the information should want
*more* process noise.

`run_m5_linear_probe.py` measured it. **Both halves were wrong:**

| learner on `m_linear` | $6\times10^{-7}$ | $6\times10^{-6}$ | difference |
|---|---|---|---|
| `diffusion_ekf_onehop_mean_receiver` | 0.1460 | 0.1468 | $+0.0008$ |
| `diffusion_ekf` (local adapt) | **0.1509** | 0.1596 | $+0.0088$ |

The carried variant is tied, and the local-adapt filter — the one predicted to need
more $q$ — prefers the **smaller** value by 0.0088, four times the threshold and
the largest separation in the comparison. Overriding the selection on the
robustness argument would have degraded M6's local-adapt arm by more than most of
the differences M6 exists to measure. The selection stands unedited.

**The finding that outlives the tuning decision: D87's pattern inverts here.** On
the image task the diffusion filter wanted *ten times more* process noise than its
centralised twin, because each agent holds $1/N$ of the information and must stay
adaptive. M5's grid was built to test whether that transfers. It does not, and it
does not merely fail — it reverses. M4's centralised filter chose
$q=6\times10^{-6}$; the diffusion filter prefers $6\times10^{-7}$, and the
local-adapt variant prefers it most strongly of all. The $1/N$ deficit does not
translate into wanting a looser covariance on this task.

**$\gamma=1$ is not a variance-explosion risk here, and that is measured too.**
$\gamma$ acts on the mean, not the covariance (D26): $\boldsymbol P=\gamma^2\boldsymbol P+\boldsymbol Q$
contracts. At the selected setting the filter is calibrated to three decimals —
coverage 0.504/0.901/0.951 against nominal 0.50/0.90/0.95, variance ratio 0.9996 —
and $\lVert\boldsymbol m\rVert^2$ grows 25% over 1500 steps with decelerating increments,
against a trust-region guard at 50× that never fired. The inflation signature does
appear in the grid, but at **large $q$**: at $6\times10^{-4}$ the variance ratio
falls to 0.55 and coverage over-shoots to 0.65/0.97/0.99, and those are the
worst-scoring cells. $\boldsymbol Q$ inflates $\boldsymbol P$; $\gamma$ does not.

$\gamma=0.9995$ at the selected $q$ is both worse-scoring and slightly
over-confident (ratio 1.124, under-covering at 0.882 and 0.935) — the contraction
shrinking $\boldsymbol P$ below what the residuals support. Consistent with X23, which
chose $\gamma=0.9995$ *at $q=6\times10^{-4}$*: both tasks show the same
$\gamma$–$q$ interaction, and differ only in where their optimum sits along it.

**Carried, with its limits stated.** One setting serves four variants and three
conditions, so a shortfall in the local-adapt arm of M6 is attributable to the
adapt scope rather than confounded with tuning — the trade the image task also
made. The probe is the only evidence that the setting travels: the carried variant
scores 0.1460 on `m_linear` against 0.1458 on `m_abrupt`. Nothing yet tests
`m_stationary`, and M6 is where that shows.

### ✅ D104. The model protocol had no device contract, and exactly one model needed one

M5 died on its first cell with `Expected all tensors to be on the same device, but
found at least two devices, cuda:0 and cpu!`, raised at `embed(x) + self.positions`
inside the Transformer's `forward`.

**Why it surfaced only there.** Models here are *functional*: parameters arrive as
an argument and carry their own placement, so a model that owns no tensors runs
correctly wherever it was built. The MLP and the linear AR own nothing — which
is why every image-task run, P5.3 included, has used `device: auto` on a GPU
without trouble. The causal Transformer owns **two registered buffers**, the
sinusoidal positions and the causal mask. Buffers travel with the module, and
`functional_call` substitutes parameters without touching them, so device-resident
parameters met a CPU-resident encoding.

**Why it took until M5.** Every Mackey--Glass run before it was CPU: M3 and
M6's smoke by configuration, M4 deliberately, to leave the GPU free for the
image sweeps. M5 is the first MG run ever to ask for a device.

**The gap was in the protocol.** `Model` declares twelve methods and not one of them
mentions placement; `build_model_from_config` took only a dtype, and the runner moved
`theta0` alone. Device-agnostic construction is correct for a model that owns
nothing and silently wrong for one that does not.

**The fix.** `build_model` now takes a device, `build_model_from_config` resolves
`run.device` through the same `resolve_device` the environment uses — so the model
and the data cannot disagree about where they are — and the Transformer builds its
buffers on that device rather than being moved afterwards. Placement is *also*
applied generically to `_module`, which every model wraps, so a buffer added to any
future model is covered. It is a no-op for the models that hold nothing.

**⚠ That fix then caused a second, different failure — and the test written for the
first one hid it.** With the module on a GPU, `init_params` split: its `*_like`
constructions followed the module to CUDA while the generator-driven weights
stayed on the CPU, and `flatten`'s `cat` raised one line later in `run_one`.
**The MLP carried the identical mix**, so the same change broke the *image* path,
which had been working — undetected only because no image sweep had started since.

`init_params` is now **CPU-only by contract** in every model, and must stay so.
A CUDA tensor needs a CUDA generator, which draws a different random stream;
$\boldsymbol\theta_0$ would then depend on where it was built, breaking reproducibility and the
D9/D18 requirement that every agent start from the *same* vector. The runner moves the
flat vector once, after assembly.

The first version of the test moved every parameter to CUDA itself before using
it, which is exactly why it passed while the split existed — it asserted the fix it
was meant to check. It now asserts the contract: `init_params` returns CPU
tensors whatever the module's device, $\boldsymbol\theta_0$ is bit-identical across devices,
and `forward`/`vjp` run through the runner's real path (`flatten` → `.to(device)` →
`unflatten`) rather than by hand.

**Verified.** Twelve device tests pass with the CUDA cases executing; one real M5
cell runs end to end on the GPU (`ok`, 40 steps, marker written); the suite is
**1 433 passed, 1 skipped**, which is exactly the five added tests and no movement
elsewhere; $p=2273$ is unchanged.

**The lesson, which is the reason this note is long.** Both faults were device-only,
and the suite never left the CPU. The first was invisible because no MG run
had used a GPU; the second because the test written for the first one worked
around it. A unit test that constructs the conditions it is checking for proves
nothing — the verification that finally caught both was *running one real cell of the
actual sweep on the actual device*, which takes sixteen seconds and should have
preceded the first launch.

⚠ **No result is affected.** Every completed MG cell records `device=cpu`, and
the image-task runs use a model with no buffers.

⚠ **The suite could not have caught either fault, which is why the device test now
exists.** Every
series test pinned `"device": "cpu"`, and the only `cuda.is_available()` in the suite
guarded the CUDA-*refusal* test — the one that shows as skipped. No test ever
placed a model on a GPU. `tests/test_model_device.py` now asserts the contract in
both directions, with the CUDA cases skipped where there is no device to fail on.

⚠ **And the pre-flight missed it.** M5's report path, refusal guard, lint and output
encoding were all exercised before launch, on empty data, and the chain was called
"wired and verified end to end" — but no cell had ever been executed on the device
the sweep would actually use. Verifying the plumbing is not verifying the run.

### ✅ D103. P5.3: the sharing gap does not widen as connectivity falls, and one-hop's value is not monotone in degree

`run_diffusion_topology.py`, 12.6 h over 8 cells × 5 seeds (plus 0.6 h tuning), on
path, ring, ER $p=0.3$ and complete. The first sweep to run one-hop at the
**receiver** point natively (D99). Gradient baselines re-tuned per topology (D39,
D77); the filter carries X20's selection unchanged, which is the X14 discipline.

**Both stated hypotheses are refuted.**

*"Mean-only sharing loses more when information has to travel further, so the two
variants should separate here if they separate anywhere"* — they do not separate.
Full minus mean-only, paired by seed:

| | local adapt | one-hop (receiver) |
|---|---|---|
| path | −0.0010 ($t=-10.4$) | +0.0001 (ns) |
| ring | −0.0015 ($t=-5.1$) | +0.0004 (ns) |
| ER 0.3 | −0.0010 ($t=-1.4$) | +0.0003 (ns) |
| complete | −0.0018 ($t=-4.0$) | +0.0000 |

The local-adapt benefit is flat across the whole axis and nominally **largest on the
complete graph** — the opposite of the prediction. Connectivity is not what makes
covariance sharing pay; heterogeneity is (D89, D101).

*"One-hop's value should scale with degree"* — it is **non-monotone**, peaking at
intermediate connectivity: −0.0006 (ns) on a path, −0.0038 ($t=-4.1$) on a ring,
−0.0028 ($t=-2.7$) at ER 0.3, and **+0.0031 (ns)** on a complete graph, where
one combine step already reaches consensus. A path's degree-1 endpoints gather
little, so more reach is not monotonically more value.

**[[D100]]'s null holds at every topology.** Full sharing adds nothing on top of a
one-hop adapt anywhere on the axis, including the sparsest. This note set the ring
up as the place it should break — one-hop reaches two neighbours there, so the
"fresh evidence has already entered the adapt step" mechanism is at its weakest —
and it did not break at the ring or at the path either.

**Three correctness checks passed.**

1. `centralized_ekf_gamma`, `centralized_sgd` and `local_only` never read the graph,
   and produce **0.0e+00** spread across all four topologies. The graph does not leak
   into the data path.
2. On the complete graph the two one-hop variants are identical by construction, and
   full minus mean-only is exactly $+0.0000$ with zero variance. ⚠ The report prints
   $t=\infty$ there; read it as degenerate, not as significance.
3. The complete-graph one-hop-minus-local figure reproduces X20's **+0.0031** to four
   decimals — correctly, since the sender and receiver points coincide on a complete
   graph. That was the number flagged as most at risk of flipping at the receiver
   point (results.md §1283); it did not flip.

**At matched-ish bandwidth** one-hop beats `atc_plain` by −0.0152 to −0.0176
($t=-9$ to $-15$) on path, ring and ER 0.3, and ties on complete (−0.0001, ns)
where ATC reaches consensus in one step and *is* centralized.

⚠ **The sparse point was originally ER $p=0.15$, and that was my error.** At
$N=10$ the connectivity threshold is $\ln(n)/n=0.230$, so 0.15 sits below it: the
builder resampled for a connected draw and gave up after 20 attempts, mid-sweep.
The deeper fault is that a draw which *does* succeed is conditioned on a rare event
and is no longer a sample from ER(10, 0.15) — the cell would not have meant what its
label said, so the crash was the lucky outcome. `path` replaced it: connected by
construction and genuinely sparser than a ring in spectral gap. The tuning pass had
slipped through on a lucky draw at seeds 0–1, which is why it was not caught earlier.

**Cost rises with degree**, as one-hop's mechanism predicts: 86, 96 and 138 min for
the group-A cells at path, ER 0.3 and complete. The tuning pass showed the
reverse ordering, but that was machine contention rather than topology — it runs only
the gradient baselines, which never re-linearise a neighbour's batch.

### ✅ D102. M4: the centralised filter's four knobs, and an axis that flattened rather than ran out

`scripts/run_m4_centralised.py`, 336.6 min CPU: $\gamma\times q\times\sigma_0^2\times$
the scale of $\boldsymbol R$, **crossed**, 24 cells × 2 seeds on the abrupt condition and
carried everywhere (the X14 discipline). Selected: $\gamma=1$, $q=6\times10^{-6}$,
$\sigma_0^2=0.01$, $\boldsymbol R\times1$, at settled RMSE **0.1429**.

**The $q$ argmin sits on the grid's bottom edge, and that is not a reason to extend
it.** The axis flattened:

| | $q=6\times10^{-4}$ | $6\times10^{-5}$ | $6\times10^{-6}$ |
|---|---|---|---|
| mean over the 8 slices | 0.1580 | 0.1459 | **0.1453** |
| best slice | 0.1543 | 0.1449 | **0.1429** |

The first decade buys 0.0121; the second buys **0.0006**, and in **three of eight**
slices $6\times10^{-6}$ is already *worse* than $6\times10^{-5}$ (0.1465 against
0.1456; 0.1487 against 0.1453; 0.1452 against 0.1451). A further decade would buy
less than the noise we decline to interpret elsewhere. **Decided: carry
$6\times10^{-6}$, sweep lower only if a later result makes $q$ look load-bearing** —
the same reasoning as the AdamW tie in [[D98]].

**The axes do not separate, which is why the grid was crossed.** $\gamma$ interacts
with $q$: at $q=6\times10^{-4}$ the contracting $\gamma=0.9995$ wins **all four**
slices, while at $6\times10^{-5}$ and $6\times10^{-6}$ the random walk $\gamma=1$
wins **all eight**. $\boldsymbol R$ interacts the same way — $\times2$ is better at high $q$,
$\times1$ at low. X23 found coordinate descent sufficient on the image task; that was
a measurement there and does not transfer, and here it would have missed the
interaction.

⚠ **The selection is not sharply determined.** The top four cells span 0.1429 to
0.1447 — a range of 0.0018, the same order as the $q$ flattening. M6 should carry
this setting as "the best of a flat region", not as a located optimum.

⚠ **The run exited non-zero on a cosmetic fault in its own report.** A warning-sign
glyph in the edge check raised `UnicodeEncodeError` under the cp1252 encoding of a
redirected log, *after* every cell had been computed and `m4_selection.json` written.
The real report came from `--report-only` against the corrected file. Recorded in
`howto.md`; the traceback's line numbers were also garbled, because the file had been
edited while the run was in flight — the second hazard that file documents.

### ✅ D101. X25+: the skew cells at five seeds, and `atc_plain` under skew at last

`run_skew_topup.py`, 231 min on the RTX 4070. Seeds 3 and 4 in new cells carrying
X25's exact settings, pooled with X25's 0--2, plus `atc_plain` at X26's selected
rate — the arm X25 never carried, which is what D90 recorded as missing. Seven
cells; the drifting pair and $\beta_c=2$ were deliberately left at three seeds,
because the schedule names the *still* cells as where the equal-bandwidth claim
lives.

**The pooling is exact, not merely matched.** The sender-point one-hop filter rides
along as a reproduction check and agrees with X27 to **0.0e+00 on every shared
seed**, at all three skews. The data stream depends on configuration and seed alone,
never on which learners are attached (X26 established that to twelve decimals), so
new cells at new seeds are the same experiment rather than a comparable one.

| paired at $\beta_{\mathrm{dir}}$ | 0.1 | 1 | 100 |
|---|---|---|---|
| local adapt vs `atc_plain` (equal $\psi$) | **+0.0068**, $t=2.43$ ($p=0.07$) | −0.0114, $t=-5.88$ | −0.0130, $t=-12.10$ |
| local adapt vs momentum ATC ($2\psi$) | +0.0211, $t=5.11$ | −0.0044 (ns) | −0.0042, $t=-4.16$ |
| one-hop vs momentum ATC ($2\psi$) | −0.0063, $t=-5.63$ | −0.0075, $t=-6.88$ | −0.0062, $t=-11.13$ |
| one-hop vs local adapt | −0.0274, $t=-6.40$ | −0.0031 (ns) | −0.0021 (ns) |

**What it settled.** X25's sharpest claim — that the mean-only local-adapt filter
loses to `atc_plain` at equal bandwidth under severe skew — was $t=1.66$ on three
seeds. At five it is $t=2.43$, $p=0.07$. The effect size barely moved (+0.0073 to
+0.0068) while the statistic firmed, which is what a real effect does on gaining
seeds rather than what an artefact does. It remains **marginal, not established**.

**What it changed**, and both are corrections rather than additions:

* **"One-hop is worth 6.5× more under skew" does not survive.** At five seeds
  one-hop's advantage over the local adapt is −0.0274 ($p=0.003$) at severe skew and
  **indistinguishable from zero** at 1 and 100 ($p=0.20$, $p=0.21$). The ratio was
  an artefact of a small denominator that three seeds made look significant. The
  honest statement is that one-hop's value *is* a skew effect.
* **The sender-point substitution number flipped sign and stayed a tie**: full
  sharing on top of one-hop reads −0.0009 ($p=0.55$) where three seeds gave
  $+0.0009$. It now agrees in sign with the receiver point's −0.0019, and [[D100]]
  is amended accordingly — the sign difference was the seed count, not the
  linearisation point.

⚠ **`atc_plain` is weak in absolute terms at severe skew** (0.1046, against the
filter's 0.1114 and `local_only`'s 0.6266), so the equal-bandwidth row compares two
poor performers there. The $2\psi$ comparison against momentum ATC remains the
defensible headline, and one-hop wins it at every skew.

### ✅ D100. X28: at the receiver point too, one-hop leaves covariance sharing nothing to do

`run_receiver_full.py`, 1 h 58 min on the RTX 4070: two cells at the ends of X25's
skew axis, `diffusion_ekf_onehop_receiver` (one-hop, **full** sharing, receiver
point) at $\beta_c=1$, five seeds, $T=1500$. Paired by seed against X27's mean-only
receiver cells already on disk — the same configuration and the same stream, so the
pairing is exact rather than merely matched.

| cell | full sharing | mean-only | full − mean | $t$ | $p$ | seeds favouring full |
|---|---|---|---|---|---|---|
| still, $\beta_{\mathrm{dir}}=0.1$ | 0.0761 | 0.0779 | −0.0019 | −1.61 | 0.18 | 4/5 |
| still, $\beta_{\mathrm{dir}}=100$ | 0.0692 | 0.0693 | −0.0001 | −0.45 | 0.67 | 3/5 |

**The substitution survives the change of linearisation point.** X25 measured the
two repairs as substitutes at the sender point (+0.0009, ns, at three seeds, under
severe skew);
[[D99]] then established that the receiver point is a *different update*, so that
result could not simply be inherited. Re-measured, it holds: full sharing is
nominally ahead at both ends of the axis and significant at neither, for **1 145×**
the bandwidth — 4 233 382 scalars per link per direction against 3 696, on
[[D94]]'s ledger.

**What gives the null its force is the contrast, not the $p$-value.** The same
sharing is worth **−0.0314 ($t=-4.61$)** to the *local-adapt* filter under severe
skew (X25). Beside a one-hop adapt it is worth −0.0019 — seventeen times smaller,
and inside the noise. The mechanism reads the same at both linearisation points:
once fresh neighbour evidence enters the adapt step, the neighbours' accumulated
uncertainty has nothing left to contribute.

⚠ **Recorded as a tie, not a win.** Per seed at $\beta_{\mathrm{dir}}=0.1$: −0.0048,
−0.0010, **+0.0020**, −0.0024, −0.0031. The direction is consistent in four of five,
but the effect does not clear noise at $n=5$.

**Amended 2026-09-17.** This note first read the receiver point as differing in sign
from the sender point's $+0.0009$, and declined to call that a reversal. X25+ has
since put the sender point on five seeds too, where it reads **−0.0009 ($p=0.55$)**:
the sign difference was the three-seed estimate, not the linearisation point. Both
points now agree — a tie, leaning the same way, at either one ([[D101]]).

⚠ **Not a claim that covariance sharing is useless.** It is useless *on top of a
one-hop adapt*. Where the adapt step stays local, sharing is the repair that works,
and X25 measured it doing so.

$\beta_c=2$ was not topped up here: [[D88]] and X25 both priced it as ruinous, and
skew did not make it live.

**Consequence.** Mean-only at the receiver point stays the deployable variant — now
on a measurement taken at the linearisation point the paper actually carries, rather
than on an analogy from the one it abandoned.

### ✅ D99. X27: the receiver point is never worse, and wins where the agents disagree

Seven cells, five seeds, X25's conditions and settings, both linearisation points in
one run so every comparison is paired against the same data stream
(`run_linearization_point.py`; 9 h on the RTX 4070).

| cell | sender | receiver | receiver − sender | $t$ | ATC (2ψ) |
|---|---|---|---|---|---|
| still, $\beta_{\mathrm{dir}}=0.1$ | 0.0840 | 0.0779 | **−0.0060** | −2.55 | 0.0941 |
| still, $\beta_{\mathrm{dir}}=1$ | 0.0734 | 0.0728 | −0.0005 | −0.98 (ns) | 0.0831 |
| still, $\beta_{\mathrm{dir}}=100$ | 0.0705 | 0.0693 | **−0.0012** | −5.19 | 0.0771 |
| abrupt + severe skew | 0.1281 | 0.1189 | **−0.0091** | −2.54 | 0.1579 |
| smooth + severe skew | 0.0834 | 0.0792 | −0.0043 | −1.80 (ns) | 0.1024 |

**The receiver point is never worse, and its margin grows with the cell's
difficulty.** It is largest in the hardest cell — abrupt drift on severe skew,
−0.0091 — and statistically flat at moderate skew. The near-IID cell's −0.0012 at
$t=-5.19$ is tiny but consistent across every seed.

⚠ **The obvious explanation was tested and does not hold.** A first version of this
note claimed the margin "tracks disagreement", since the two points differ *only*
by the inter-agent spread in the predictive mean (D93). Measured on the recorded
$E_{\mathrm{agree}}$, the correlation between the margin and the disagreement is
**+0.44** over the five distinct cells — the wrong sign for that story, and
meaningless at $n=5$ in any case. The two cells with the *highest* disagreement
(0.409, 0.412 at $\beta_{\mathrm{dir}}=100$ and $1$) have the *smallest* margins;
the ordering the margin actually follows is the ordering of the error. Nor does the
receiver point systematically reduce disagreement: the ratio to the sender's
straddles one (0.977 to 1.027).

So the defensible claim is narrower: **the receiver point is never worse, wins most
where the task is hardest, and X27 does not isolate why.** Separating difficulty
from disagreement needs cells that vary one while holding the other — which these
do not.

**And it is the cheaper one**: 3 696 scalars per link per step against 6 604, since
$\boldsymbol\theta_u^-$ need not travel (D94). Better *and* 1.8× cheaper, so the receiver
point is the one to carry forward — the textbook diffusion EKF, which is also
what the note should have specified all along.

**Three reproduction checks passed.** The sender arm reproduces X25 **exactly**
(0.0e+00 on every shared seed, all five cells): D94 changed the ledger, not the
filter. Both control cells reproduce the still severe-skew cell exactly, as the same
law on the same seeds must. Damage (drifting minus its twin) is +0.0441 abrupt and
−0.0006 smooth for the sender, +0.0410 and +0.0013 for the receiver.

⚠ **X20–X26's one-hop numbers stand as published** — they are the sender point, and
this note does not restate them. What changes is which variant the paper *carries*:
`diffusion_ekf_onehop_mean_receiver` from here on.

**Decided 2026-09-16: the receiver point is the carried variant.** Better in five
cells of five, better calibrated, 44% cheaper on the wire, and the formulation the
literature states. Every sweep from here carries it; the published sender-point
numbers stay as they are, labelled as such.

One inherited result does **not** transfer automatically. X25's finding that full
covariance sharing adds nothing on top of a one-hop adapt (+0.0009, ns, under
severe skew, against −0.0314 for the *local*-adapt filter) was measured at the
sender point. The receiver point is a different update, so **X28** re-measures it:
`diffusion_ekf_onehop_receiver` at $\beta_c=1$ on the ends of the skew axis, paired
by seed against X27's mean-only receiver cells, which are already on disk at the
same conditions and seeds.

### ✅ D98. M3: the rates are stable across conditions, and the abrupt anomaly was one jump draw

`scripts/run_m3_rates.py`, 255 min CPU for the first pass and 140 min to re-run the
five abrupt cells, 15 cells (3 conditions × 5 rates) × 2 seeds, all seven gradient
baselines per cell, none diverged as a cell. Selected rate per learner per
condition, on settled RMSE on the held-out current set:

| learner | stationary | linear | abrupt | selected rate |
|---|---|---|---|---|
| centralised AdamW | **0.1634** | **0.1700** | **0.1682** | $3\times10^{-3}$ |
| ATC AdamW | 0.1680 | 0.1728 | 0.1704 | $3\times10^{-3}$ |
| centralised SGD | 0.1781 | 0.1836 | 0.1826 | $3\times10^{-5}$ |
| momentum ATC | 0.1810 | 0.1869 | 0.1850 | $1\times10^{-5}$ |
| plain ATC | 0.1865 | 0.1953 | 0.1915 | $3\times10^{-5}$ |
| local AdamW | 0.1784 | 0.1850 | 0.1825 | $3\times10^{-3}$ ⚠ |
| local only | 0.1900 | 0.2000 | 0.1959 | $3\times10^{-5}$ |

The abrupt column is the **re-run** under the per-seed jump draw; stationary and
linear are unchanged and were served from cache, so the two halves of this table
come from the same computation on different days.

**Every learner selects the same rate in all three conditions**, and only one lands
on a grid edge — so per-condition tuning here *confirms* stability rather than
changing the choice. That is worth having measured rather than assumed, which is
the whole of D77 and D90. **AdamW beats the SGD family for every learner type**, by
0.010 to 0.015: decision 19 earned its place.

⚠ **That one exception is nominal.** Under the re-run, local AdamW's abrupt argmin
moves to $10^{-3}$, the bottom of the grid, but its curve there reads
$\{10^{-3}\!:\,0.1825,\ 3\times10^{-3}\!:\,0.1827\}$ — a gap of $0.0002$, far inside
seed noise, so the selection is *indifferent* rather than relocated and
$3\times10^{-3}$ is carried. Extending the grid downward is not a one-line change:
a cell pairs an SGD rate and an AdamW rate at the same index, so a sixth rate costs
a sixth cell for both families. **Decided 2026-09-16: carry $3\times10^{-3}$**, and
sweep a sixth rate only if a later result makes that rate look load-bearing.

**The NLL-scale grid was the right one.** Rates at $10^{-4}$ and above diverge for
the SGD family, and the optimum is bracketed at $3\times10^{-5}$ — about 700×
below the pilot's MSE-scale rates, as the scale argument predicted.

⚠ **A diverged regression learner records NaN rather than raising.** That is better
than the classification path, where one divergence discards the whole cell — but
NaN compares False both ways, so the first version of the selection could have
picked a diverged rate as "best". Non-finite is now mapped to infinity. The table
above is from the corrected code; the sweep's own end-of-run printout used the old.

**The abrupt condition scored *better* than stationary for all seven learners**
(0.1610 against 0.1634 for centralised AdamW; 0.1821 against 0.1900 for local
only). Drift does not make a task easier, so the condition was measured rather
than reasoned about:

| condition | mean $\beta$ | min | max | fraction below 0.22 |
|---|---|---|---|---|
| stationary | 0.2200 | 0.2200 | 0.2200 | 0% |
| linear | 0.2300 | 0.2200 | 0.2400 | 0% |
| **abrupt**, one fixed draw | **0.2111** | 0.2000 | 0.2333 | **73%** |
| **abrupt**, jump seed per run seed | 0.2193 | 0.2000 | 0.2400 | 46% |

The abrupt run sat at a *gentler law* for most of its length, and a gentler law is
an easier task: M2 measured $e^\star$ rising monotonically from 0.1342 at
$\beta=0.20$ to 0.1467 at 0.24.

**✅ Resolved by the re-run** (140 min CPU, 2026-09-16). With the jump seed derived
from the run seed the realised law centres on the channel's own centre — 0.2191 and
0.2196 on the two seeds, pooled 0.2193 against a nominal 0.22, with 46% of steps
below it rather than 73% — and the reflection now reaches the full window (max
0.2400, not 0.2333). Every abrupt number rose, by 0.0072 to 0.0138, and **all seven
learners now score worse under abrupt drift than stationary**, which is what drift
must do. Abrupt still sits just under linear (0.1682 against 0.1700 for centralised
AdamW), and that ordering is the sensible one: the linear schedule *ends* at 0.24,
the hardest law in the window, while the abrupt one reflects about 0.22.

⚠ **The first explanation here was wrong, and is kept because the correction is the
point.** It claimed `recurring` reflects at the band edges and is therefore biased
downward from the centre — a systematic defect in the schedule. Measuring twelve
jump seeds refutes it: mean displacement **−0.38°, s.d. 12.0°, range −20° to +19°**.
The schedule is symmetric in distribution; **seed 0 simply drew a low walk**. With
$T/t'=60$ jumps, a run's *time-average* $\beta$ is itself a random variable with a
12° spread, so any single jump seed lands somewhere in $[0.211, 0.229]$.

That is the real finding, and it is about the **estimator, not the schedule**: on
this task the drifting cell's mean law varies by seed as much as the drift effect
being measured, so a damage figure from one jump seed compares two different mean
laws. It does not arise on MNIST, where rotation is symmetric about a neutral 0 and
the task's difficulty is (to first order) even in the angle; here the channel's
centre is the interesting law and either side is a different difficulty.

**Decided 2026-09-16: `jump_seed` varies with the run seed.** Each run seed draws
its own jump sequence, so the realised mean law averages out across the five seeds
instead of being one fixed offset, and what the damage metric measures is the
drift rather than the draw. The schedule and the one-twin-per-cell design are
unchanged; the twin still shares every history and noise draw with its drifting
run, since those come from streams that do not depend on the drift.

The user's earlier choice ("start at the band edge") was made against the wrong
diagnosis and does not survive it: starting at an edge biases the walk *upward*
instead. The two alternatives considered and not taken — a twin at each cell's own
realised mean $\beta$, and reading damage against $e^\star(\bar\beta)$ from D97's
line — both remove the confound for cells already run, but the first makes the twin
depend on the drift draw and the second changes what "damage" means relative to
MNIST's twin-based definition.

M3's five abrupt cells have been re-run under the new rule; the stationary and
linear cells were served from cache and their selections are unaffected. The
superseded cells are kept in `results/_superseded_m3_abrupt_fixed_jump_seed/`,
because the "one fixed draw" row above is theirs and the comparison is the point.

### ✅ D97. M2: the offline reference across the chaotic window, and why a line beats the points

`scripts/run_m2_references.py`, 23.7 min CPU, `results/m2_references.json` in the
worktree. At each of 13 $\beta$ levels on $[0.20,0.24]$ (a 1/300 grid that contains
every value the abrupt schedule can reach, and covers the linear one at twice that
resolution): the Transformer trained offline on one run's whole budget (15 000
blocks at a fixed law), epoch chosen on 2 000 validation blocks, scored on 4 000.

| $\beta$ | 0.2000 | 0.2100 | 0.2200 | 0.2300 | 0.2400 |
|---|---|---|---|---|---|
| $e^\star$ | 0.1342 | 0.1405 | 0.1433 | 0.1442 | 0.1467 |
| full context | 0.1135 | 0.1173 | 0.1177 | 0.1170 | 0.1184 |
| persistence | 0.2033 | 0.2151 | 0.2264 | 0.2378 | 0.2480 |

(all 13 levels in the JSON; noise floor 0.1 throughout.)

**The epoch cap had to rise first.** At the original 40 the first level selected
epoch 37 — still improving at the cap, so not the converged reference $e^\star$ is
defined as. At a ceiling of 100 with patience 5 every level stopped on its own,
between epochs 26 and 58.

**What it says.** The reference degrades by about 0.0125 across the window while
persistence degrades by 0.045: the drift makes the *trivial* predictor much worse
and a good one only slightly worse, so the task stays learnable across the whole
span. $e^\star$ sits 1.34–1.47× above the noise floor.

⚠ *See [[D98]]: the abrupt condition these references are read against scores
**better** than stationary, which needs explaining before any damage figure on this
task is believed.*

**Use the line, not the points.** A least-squares fit gives $e^\star(\beta)\approx
0.0816+0.274\,\beta$ with residual s.d. 0.0015 — and that residual is single-run
training noise, not structure: $\beta=0.2267$ came out *below* its neighbour at
0.2233. A gap measured against one run's reference would inherit 0.0015 of noise
that has nothing to do with the method; measured against the fitted line it does
not. **Proposed:** report gaps against the linear fit, with the 13 points drawn
for honesty. The alternative — several reference seeds per level — costs 13 more
CPU-minutes per seed and buys the same thing less cleanly.
