# Research Work Plan — Distributed Online Learning Benchmark

**Project.** Build a benchmark for distributed online learning over a graph, establish first-order baselines on it, and prepare the ground for the diffusion EKF (Diff-EKF).

**Status.** Planning document. No implementation yet. The open questions of §10 were resolved on 2026-07-30; §10 records the decisions and the reasoning.

**Companion documents.**
- `IMPLEMENTATION.md` — repository layout, module responsibilities, interfaces, test suite, figure specifications, tooling. This document says *what* and *why*; that one says *how it is built*.
- Research note: *Distributed Online Bayesian Learning of Deep Neural Networks — A Diffusion Extended Kalman Filtering Formulation.* Referenced throughout for the Class L/C distinction, the complete-graph exactness proposition, and baseline (B3).

---

## 1. Objective and scope

A network of $N$ agents sits on a communication graph. At each time step every agent receives a *small* number of labelled samples. The agents must cooperatively learn one shared classifier, online, without a fusion centre. We want a benchmark that makes this setting measurable, and two first-order baselines on it.

**In scope**

1. The environment: graph, per-node data streams, sparse arrival, stationary and non-stationary (rotating-MNIST) regimes.
2. A model abstraction not tied to the MLP.
3. Two learning methods: centralized online SGD and distributed online SGD with neighbour communication.
4. Measurement: correct-classification rate for the centralized run, mean error rate across nodes for the distributed run, both referenced against a centralized *offline* classifier.

**Out of scope now, but designed for**

5. Diff-EKF as a third learning method. Nothing in phases 1–4 may make this expensive to add.

**Non-goals.** Beating state of the art on MNIST. Distributed *systems* engineering — the network is simulated in one process. Large models.

---

## 2. Research questions

The benchmark exists to answer these, in order of importance.

**Q1 — What does decentralization cost?** How much worse is distributed online SGD than centralized online SGD, at equal communication, as a function of graph connectivity?

**Q2 — Does communication help at all?** How much better is distributed than purely local training, and does the answer change under distribution shift?

**Q3 — What happens under drift?** Does the gap to the reference classifier widen when the data distribution rotates, and how quickly does each method recover from an abrupt change?

**Q4 — Where does sparsity bite?** With only $n$ samples per node per step, at what point does the online signal become too weak to learn from?

**Q5 (phase 5) — Does a second-order Bayesian update earn its keep?** At the *same* one-exchange-per-step communication, does Diff-EKF beat distributed SGD, and does its posterior covariance give usable uncertainty?

Q5 is the point of the whole exercise. Q1–Q4 exist to build an instrument that can answer it credibly.

---

## 3. The learning methods

All agents share one parameter vector $\bm\theta\in\mathbb R^p$ and one architecture. Every method is expressed as an **adapt** step (use local data) followed by a **combine** step (use neighbours), so that Diff-EKF later differs from distributed SGD *only* in the adapt step.

### 3.1 Centralized online SGD (upper reference)

A single logical learner sees the pooled batch $\mathcal D_t=\bigcup_v\mathcal D_t^v$ at every step and takes one optimizer step:

$$\bm\theta_t=\bm\theta_{t-1}-\eta\,\nabla L(\bm\theta_{t-1};\mathcal D_t)$$

This is the upper reference *for the online setting*. It is not the same thing as the offline reference classifier of §5.1, which sees the whole dataset many times.

### 3.2 Distributed online SGD

**Primary: adapt-then-combine (ATC) diffusion.**

$$\bm\psi_t^{v}=\bm\theta_{t-1}^{v}-\eta\,\nabla L\!\left(\bm\theta_{t-1}^{v};\mathcal D_t^{v}\right),\qquad
\bm\theta_t^{v}=\sum_{j\in\mathcal N_v\cup\{v\}}a_{vj}\,\bm\psi_t^{j},\qquad \sum_j a_{vj}=1$$

Each agent takes one gradient step on its own data, then averages the result with its one-hop neighbours. **Parameters are mixed, not gradients** — that is what drives the agents toward agreement. Communication is one $p$-vector per link per step.

**Secondary: combine-then-adapt (CTA).** Olshevskyi et al. [1], their eq. (17):

$$\bm\theta^i(t+1)=\sum_{j\in\mathcal N^{+}(i)}\bm W_{ij}\,\bm\theta^j(t)\;-\;\alpha_t\, f\!\left(\bigl\{\widehat{\nabla J_i(\bm\theta)}(b)\bigr\}_{b=1}^{B}\right)$$

with Metropolis–Hastings weights, where the gradient is evaluated at the *pre-combine* parameters. This is the Nedić–Ozdaglar consensus-plus-local-gradient form [2].

**Why ATC is primary.** The Diff-EKF is ATC. If the SGD baseline were CTA, the phase-5 comparison would confound *EKF vs SGD* with *ATC vs CTA* — and ATC generally has the better mean-square performance [3], so the baseline would be handicapped in a way a reviewer will notice. With both as ATC, the two methods differ in the adapt step alone, at identical communication, and any Diff-EKF advantage is attributable to the second-order update rather than to a larger payload. Experiment X1b measures the ATC/CTA difference so that the choice is reported rather than assumed.

**Empirical support for parameter mixing.** [1] compare eq. (17) against a "D-naive" variant that runs $K$ rounds of consensus on *gradients* instead: D-naive needs roughly twice the message-passing rounds to reach the same MSE (their Fig. 2a). Parameter mixing is both cheaper and better behaved.

### 3.3 Local-only (lower reference)

Each agent trains on its own data with no communication. The gap between this and the distributed method *is* the value of cooperation, and it is the cleanest answer to Q2.

### 3.4 Optimizer state under diffusion

Adaptive optimizers carry per-node state, and whether the combine step mixes the moments as well as the parameters is a real fork. [1] settle it empirically (their Fig. 2a): **D-Adam converges quickly and then diverges**, because local momentum drifts apart across agents, while **D-AMSGrad — which runs consensus on the momentum terms too — is their best distributed method.**

Therefore:

- **plain SGD** for the exactness check X0, where the algebra must come out exact;
- **SGD with momentum, momentum also mixed**, as the primary configuration for X1–X6;
- **AdamW with its moments mixed**, only as a documented secondary.

Never an adaptive optimizer with unmixed state — that is a known failure mode, not an open question. The centralized reference uses whichever optimizer the distributed runs use, for symmetry.

### 3.5 What Diff-EKF will add (phase 5)

Same combine step, same communication. The adapt step becomes an EKF measurement update: the correction is scaled by a running posterior covariance instead of a fixed learning rate, and that covariance is carried forward as an uncertainty estimate. The reason to build the benchmark carefully now is that this substitution should require no change to the environment, the metrics, or the evaluation protocol.

---

## 4. Environment specification

### 4.1 Graph

$N$ agents on a fixed, connected, undirected communication graph $\mathcal G^{\mathrm c}$. Combination weights $a_{vu}\ge0$ respecting the graph, row-stochastic ($\sum_u a_{vu}=1$), with positive self-weight. Metropolis weights are the default; relative-degree is the alternative.

Topologies: complete, ring, path, 2-D grid, star, Erdős–Rényi, Watts–Strogatz, plus disconnected as a negative control.

The relevant scalar summary is the **spectral gap** $1-\rho$, with $\rho=\|\bm A-\tfrac1N\mathbf 1\mathbf 1^{\mathsf T}\|_2$. It is the natural x-axis for Q1: the complete graph has $\rho=0$ and should reproduce centralized behaviour, while a path has $\rho$ near 1 and should be the worst case.

A second graph $\mathcal G^{\mathrm d}$ (data coupling, over which a predictor's forward pass would exchange information) is **empty for this phase**. MNIST with an MLP is Class L in the research note's terms: every agent's forward pass is purely local. $\mathcal G^{\mathrm d}$ becomes non-trivial only when the project moves to GNNs.

### 4.2 Sparse data arrival

Two independent knobs:

- **$n$ — samples per node per step**, small (1–8), **default 2**. This is the "not a huge amount of data" requirement.
- **$\pi_{\text{lab}}$ — label availability**, the probability that a given agent has a labelled sample at a given step. With $\pi_{\text{lab}}<1$ some agents idle on some steps. This matters because both diffusion SGD and Diff-EKF handle it gracefully — an agent with no label still benefits from the combine step — and because it is a realistic feature of the target applications. Default 1.0 in phase 1; a sweep axis in phase 4.

Each agent draws from its own **disjoint** shard of the MNIST training set, assigned once at setup. No sample is seen by two agents, and no sample is reused across time steps unless explicitly enabled.

**The shard budget binds the horizon, and it is easy to violate by accident.** With 60k images and $N$ agents, each shard holds $60000/N$ samples, which at $n$ per step lasts $60000/(Nn)$ steps. The defaults $N=10$, $n=2$ give a 6000-sample shard and a 3000-step ceiling; the default horizon is $T=1500$, comfortably inside it. Two consequences worth stating, because both have already caught us once:

- $N=10$, $n=4$, $T=2000$ — the combination that appears in early drafts of the runtime table — needs 8000 samples per agent and does **not** fit. `test_stream` asserts $Nn T\le 60000$ whenever `allow_epochs` is false, and this is the assertion that catches it.
- Scaling $N$ shortens the run rather than lengthening it. $N=100$ at $n=2$ leaves 600 samples per agent, i.e. 300 steps, which is too short for the drift experiments. Reaching $N=50$–$100$ therefore requires either `allow_epochs`, or $n=1$, or accepting a shorter horizon — a decision to be taken at the time, not assumed now (§10.1).

### 4.3 Stationary and non-stationary regimes

**Simulation 1 — stationary.** Images presented unmodified.

**Simulation 2 — rotating.** At step $t$ every image is rotated by $\alpha t$ degrees. The per-step rate $\alpha$ must be small so that drift is slow relative to the information rate — that is the regime the Diff-EKF assumptions target, and the regime in which tracking is meaningful rather than hopeless. But the per-step rate is not the only thing that has to be small.

**The total rotation is capped at $45°$, and this is a hard constraint rather than a preference.** Cumulative rotation $\alpha T$ is what determines whether the task stays well-posed, and it is easy to get wrong: $\alpha=0.2°$/step over $T=1500$ is $300°$ of total rotation, at which point the benchmark is measuring something other than what it claims to. Past roughly $\pm45°$ MNIST accuracy degrades sharply for reasons that have nothing to do with decentralization; near $180°$ a 6 *is* a 9 and a 2 is close to a 5, so the labels themselves stop being well defined. A reference error $e^\star$ computed there measures label ambiguity, and the headline gap $\bar e_t-e^\star$ becomes uninterpretable — every method looks equally bad, and the experiment answers nothing.

Therefore $\alpha$ is derived from the horizon rather than chosen independently:

$$\alpha = \frac{45°}{T},\qquad\text{so } T=1500 \implies \alpha = 0.03°/\text{step}.$$

This keeps the *cumulative* drift inside the well-posed range while leaving the per-step drift far slower than the information rate, which is the assumption that matters for tracking. The configuration exposes `drift.total_degrees` (default 45) rather than a bare $\alpha$, so the constraint cannot be violated by changing $T$ alone.

Two further schedules answer questions linear drift cannot:

- **piecewise** — abrupt jumps at known times. An abrupt change makes the *adaptation transient* measurable, which is the cleanest possible test of tracking (Q3). Jump size is $15°$, well inside the well-posed range.
- **sinusoidal** — the distribution returns to previously seen states, exposing forgetting. Amplitude $\pm30°$, so that the extremes are still comfortably inside the cap and the return to a previously seen state is a genuine return rather than a wrap-around.

**Global vs per-node drift.** Start global: all agents rotate identically, so a single shared model remains the correct object. Per-node drift — agents rotating at different rates — is harder and more interesting, and is where a single shared $\bm\theta$ starts to be the wrong assumption and the hierarchical shared/local extension becomes motivated. Keep it configurable; default global.

### 4.4 Data partition

IID across agents by default, so drift is the only source of non-stationarity and Q1 is not confounded. Dirichlet label-skew (parameter $\beta$) as a configurable axis for X6 — non-IID is where distributed learning becomes genuinely difficult and where the value of communication is largest.

### 4.5 Shared initialization

All agents start from the *same* $\bm\theta_0$. Diff-EKF requires a common prior — independently initialized agents do not represent the same Bayesian model — and for SGD it removes a confound from the comparison. One seed, broadcast.

### 4.6 Input representation and the parameter budget

Phase 5 wants a model small enough for a dense $p\times p$ covariance, and the phases-1–4 experiments should use *that* model, so the eventual Diff-EKF comparison is like-for-like rather than run on a different architecture. The target is $p\lesssim3\times10^{3}$.

That budget cannot be met by shrinking the hidden layer of a 784-input MLP. With $784h+h+10h+10\le3000$ the hidden width is forced to $h\approx3$, which is not a classifier. **The input dimension is what has to come down, not the width.** MNIST images are therefore **downsampled to $14\times14$** (196 inputs), giving a $196$–$14$–$10$ MLP with

$$p = 196\cdot14 + 14 + 14\cdot10 + 10 = 2908 \;\lesssim\; 3\times10^{3},$$

which is a real classifier and fits a dense covariance ($2908^2\approx8.5\times10^{6}$ entries, a few tens of MB in float64).

**Order of operations matters.** Rotation is applied to the full-resolution image and downsampling second. Rotating a $14\times14$ image directly destroys far more information than rotating at $28\times28$ and then pooling, and it makes the transform resolution-dependent in a way that would differ between train and eval if either path ever changed. `data/transforms.py` owns the composed `rotate → downsample` pipeline and is the single implementation used by training, by the evaluation sets, and by the offline reference classifier.

The full-size $784$–$128$–$10$ MLP ($p\approx1.0\times10^{5}$) is kept as a configurable model and is used for the stationary sanity runs, so that the cost of the small architecture is measured rather than assumed. It is not the model the headline figures are built on.

---

## 5. Measurement

### 5.1 The offline reference classifier

A one-time, standard offline training run on the full MNIST training set to convergence, same architecture. This yields the reference error rate $e^\star$ against which every online method is measured. It is a fixed asset, computed once and cached, not part of any experiment run.

Under drift it must be recomputed per rotation level, otherwise the measured gap conflates *decentralization cost* with *drift cost*.

"Per rotation level" needs a grid, since the rotation is continuous in $t$. With the $45°$ cap of §4.3 the grid is **every $5°$ over $[0°,45°]$, giving ten cached references**, plus the $\pm30°$ range for the sinusoidal schedule (which reuses the same grid by symmetry, since rotation by $-\phi$ and $+\phi$ are statistically equivalent over the full training set). $e^\star$ at an intermediate rotation is linearly interpolated between the two nearest grid points; the residual error is far below the between-seed spread of the online methods, and is checked once rather than assumed. Ten offline trainings of a small MLP is a few minutes total, which is what makes the cap of §4.3 a budget decision as well as a validity one — the uncapped $300°$ range would have needed sixty.

### 5.2 Metrics

**Centralized run.** Correct classification rate $\mathrm{acc}_t\in[0,1]$.

**Distributed run.** Mean error rate across agents,

$$\bar e_t=\frac1N\sum_{v=1}^{N}\bigl(1-\mathrm{acc}_{v,t}\bigr),$$

reported against $e^\star$. The **headline number is the gap** $\bar e_t-e^\star$, since that is what answers Q1.

**Also reported, because they are cheap and diagnose failures the mean hides:**

- **Spread across agents:** $\max_v e_{v,t}-\min_v e_{v,t}$, and the standard deviation. Good mean with terrible spread is not a working method.
- **Parameter disagreement:** $\frac1N\sum_v\|\bm\theta^v_t-\bar{\bm\theta}_t\|^2$. This is $E_{\text{agree}}$ from the research note and is the direct check on whether the combine step is doing its job.
- **Deviation from the centralized learner:** $E_{\text{cent}}(t)=\frac1N\sum_v\|\bm\theta^v_t-\bm\theta^{\mathrm C}_t\|^2$, where $\bm\theta^{\mathrm C}_t$ is the centralized run *on the same stream*. This is $E_{\text{cent}}$ of the research note §7.3, and it is the phase-1 analogue of the quantity the Diff-EKF will be judged on. It is only well defined if both learners see an identical stream, which is why §6.1 runs them together in one process rather than matching them by seed.
- **Communication:** cumulative scalars transmitted and rounds of exchange. Every curve is plotted against this as well as against $t$ — a method that sends more per step wins any per-step plot for uninteresting reasons.
- **Adaptation transient** (drift runs): steps to return within $\epsilon$ of the pre-shift gap after an abrupt change.

### 5.3 Evaluation under drift — the thing that is easy to get wrong

At step $t$ the data distribution is rotated by $\alpha t$. The evaluation set must be rotated by the **same** $\alpha t$; otherwise performance is being measured on a distribution the model was never asked to fit, and every method will appear to fail.

Maintain three evaluation sets:

1. **Current** — rotated by $\alpha t$. The headline metric.
2. **Backward** — rotated by $\alpha t'$ for an earlier $t'$. Measures forgetting.
3. **Canonical** — unrotated. Interpretable across all runs and comparable to published MNIST numbers.

The reference classifier gets the same treatment.

### 5.4 Protocol

- **Prequential (test-then-train).** At each step, evaluate on the incoming batch *before* training on it, then train. Cheap, unbiased, and gives a per-step signal. The standard online-learning protocol, and the default logged metric.
- **Periodic full evaluation.** Every $K$ steps (e.g. $K=25$), evaluate every agent on the full held-out test set. Expensive, and this is what goes in the figures.
- **Seeds.** Every experiment runs over at least 5 seeds; report mean and band. Initialization, shard assignment, stream order, and graph realization are separately seeded so any one can be held fixed.

---

## 6. Experiments

| # | Name | Setup | Question |
|---|---|---|---|
| **X0** | Exactness check | complete graph, uniform weights, plain SGD, float64 | Does ATC diffusion reproduce centralized SGD exactly? (§7.1) |
| **X1** | Stationary baseline | ring, $N=10$, $n=2$, $T=1500$, $196$–$14$–$10$ MLP, no drift | Q1, Q2 — do all methods learn, what is the gap? |
| **X1b** | ATC vs CTA | ring + grid, stationary | Which diffusion ordering, and how much does it matter? |
| **X2** | Rotating baseline | as X1, linear rotation to $45^\circ$ total ($\alpha=0.03^\circ$/step) | Q3 — does the gap widen, does local-only collapse? |
| **X3** | Topology sweep | complete / grid / ring / path, stationary | Q1 — gap vs spectral gap: the price of connectivity |
| **X4** | Sparsity sweep | $n\in\{1,2,4,8\}$, $\pi_{\text{lab}}\in\{0.25,0.5,1.0\}$, $T=750$ | Q4 — where does the sparse regime hurt? |
| **X5** | Abrupt shift | piecewise rotation, $15^\circ$ jump at $t=500$ | Q3 — adaptation transient, the cleanest tracking test |
| **X6** | Non-IID | Dirichlet label skew, $\beta\in\{0.1,1,\infty\}$ | Q2 — does cooperation still work when agents see different classes? |

X0, X1, X1b, X2 are the phase-3 deliverable. X3–X6 are phase 4.

X1b should report parameter disagreement alongside error rate, since ATC and CTA can reach similar mean accuracy while differing noticeably in how tightly the agents agree.

X4 runs at $T=750$ rather than 1500 because its largest cell, $n=8$, consumes $10\times8\times750=60000$ samples — the entire training set, exactly. Holding the horizon fixed across cells is what makes the heatmap comparable; letting each cell run until its shard is exhausted would confound sparsity with run length.

### 6.1 One environment, several learners

Every experiment instantiates **one** `Environment` and runs all of its learners against it simultaneously, stepping them in lockstep over the same $t$. Centralized SGD consumes the union $\bigcup_v\mathcal D_t^v$ of exactly the batches the distributed learners receive individually.

This is a deliberate departure from the more obvious design of one run per learner, matched afterwards by seed, and it buys three things:

1. **X0 becomes exact by construction.** The exactness check compares centralized SGD against ATC diffusion on *the same* samples in the same order. Matching two separate runs by seed makes that identity contingent on every RNG draw lining up — a property that is true until someone adds a `torch.randn` somewhere, at which point the check fails for a reason unrelated to what it is testing.
2. **$E_{\text{cent}}$ becomes available in phase 1** (§5.2), instead of requiring a re-run and a parameter-trajectory join.
3. **Every cross-method comparison is paired.** Differences between methods are not inflated by stream variation, so fewer seeds are needed for the same resolution.

The loop skeleton is unchanged in shape — `adapt` then `combine`, learner-agnostic — so the phase-5 substitution of Diff-EKF is unaffected. `IMPLEMENTATION.md` §3 gives the revised loop.

---

## 7. Validity checks

These are what make a result trustworthy, and they come before any experiment is believed.

### 7.1 The exactness check (highest value — build this first)

On a **complete** graph with uniform weights $a_{vu}=1/N$ and plain SGD, ATC diffusion is algebraically identical to centralized SGD on the pooled batch:

$$\sum_v \tfrac1N\Bigl(\bm\theta_{t-1}-\eta\nabla L(\bm\theta_{t-1};\mathcal D^v_t)\Bigr)
=\bm\theta_{t-1}-\eta\,\tfrac1N\sum_v\nabla L(\bm\theta_{t-1};\mathcal D^v_t)$$

given a common $\bm\theta_{t-1}$. In float64 the two runs must agree to about $10^{-12}$.

**The identity has preconditions, and they must be pinned in the X0 config rather than left to chance.** The right-hand side is the centralized step only if the pooled gradient equals the average of the per-agent gradients, and that requires:

- **equal batch sizes across agents** — the average of per-agent means equals the pooled mean only when every $|\mathcal D^v_t|$ is the same, so $\pi_{\text{lab}}=1$ and uniform $n$ are mandatory, not defaults;
- **`mean` loss reduction at every agent**, with the centralized learner reducing over all $Nn$ pooled samples;
- **plain SGD**, no momentum and no weight decay — any optimizer state makes the two trajectories diverge legitimately;
- **a common $\bm\theta_{t-1}$**, which holds at $t=0$ by shared initialization and is then preserved inductively by the identity itself.

Violating the first of these produces a small, plausible, non-zero residual rather than an obvious failure, which is precisely the failure mode the test exists to catch.

This single check catches weight-normalization errors, initialization mismatches, batch-partition errors, and loss-reduction (mean vs sum) errors — every one of which otherwise produces a plausible-looking but wrong curve. It is also the direct analogue of the Diff-EKF's complete-graph exactness proposition, so the same harness is reused in phase 5 — where, note, the proposition additionally requires the **one-hop adapt step** $\mathcal M_{v,t}=\mathcal V$ and not the local one (research note, Prop. 1). The SGD case needs no such distinction, but the interface must carry it from the start; see `IMPLEMENTATION.md` §13.12.

### 7.2 Supporting checks

- $N=1$: all methods coincide.
- Weights are row-stochastic; the graph is connected; the complete graph has spectral gap 1.
- Drift with $\alpha=0$ is identical to stationary.
- No leakage: training shards, per-agent shards, and the test set are disjoint.
- Exactly-once: over a run, no sample is consumed twice unless epochs are enabled.
- Reproducibility: the same seed gives the same logged metrics.

How these are implemented: `IMPLEMENTATION.md` §9.

---

## 8. Phases and milestones

Effort figures assume one person working part-time; treat them as relative sizes rather than commitments.

| Phase | Focus | Research milestone | ~Effort |
|---|---|---|---|
| **0** | Scaffolding | Repository runs; MNIST loads | 2 days |
| **1** | Environment | Figure showing each agent's received digits at $t=0,100,500$ under both regimes | 1 week |
| **2** | Models, reference classifier, metrics | Offline MLP at expected MNIST accuracy; $e^\star$ cached | 1 week |
| **3** | Learners, first results | X0 passes; X1, X1b, X2 produce the headline figure | 1.5 weeks |
| **4** | Sweeps and write-up | X3–X6; a short results memo ready for advisors | 1 week |
| **5** | Diff-EKF | Separate plan. Entry condition: the compatibility checklist is complete and X0–X2 reproduce | — |

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Sparse arrival makes online SGD too noisy to learn | X1 shows nothing | Sweep $n$ early in phase 3, not phase 4; consider a small local replay buffer as an option |
| MLP too large for a phase-5 dense covariance | Phase 5 stalls or needs a different model | **Resolved (§4.6):** $14\times14$ inputs, $196$–$14$–$10$ MLP, $p=2908$, carried through all phases |
| Drift too fast relative to learning | Every method fails, no signal | **Resolved (§4.3):** total rotation capped at $45°$, $\alpha$ derived from $T$. Still confirm in a short pilot that reference-at-$t$ accuracy stays high while a frozen $\bm\theta_0$ degrades measurably |
| Horizon silently exceeds the shard budget | Samples reused without `allow_epochs`; "exactly once" violated | $NnT\le60000$ asserted in `test_stream`; defaults chosen to satisfy it with margin (§4.2) |
| Evaluation-set mismatch under drift | Silently wrong conclusions | Three-eval-set design (§5.3) plus an explicit test |
| Adaptive-optimizer state diverges across agents | Distributed runs look worse than they are | Resolved: mix the moments (§3.4). Plain SGD for X0 |
| ATC/CTA chosen inconsistently between SGD and Diff-EKF | Phase-5 comparison confounded | ATC primary for both; CTA a labelled variant only |
| Results not reproducible across machines | Wasted debugging | Separable seeds, pinned versions, determinism flags |

---

## 10. Decisions taken, and what remains open

### 10.1 Resolved (2026-07-30)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| 1 | **$N$ and horizon.** Is $N=10$ the target scale, or should the benchmark reach $N=50$–$100$? | Start at $N=10$, $n=2$, $T=1500$. Raise $N$ only if measured runtime allows, and only after re-checking the shard budget — scaling $N$ *shortens* the feasible horizon (§4.2). | §4.2, §6 |
| 3 | **Parameter budget.** How small may the MLP be for a dense phase-5 covariance? | Downsample inputs to $14\times14$ and use a $196$–$14$–$10$ MLP, $p=2908$. Dense covariance is then affordable and phases 1–4 run on the same architecture phase 5 will use. Shrinking the hidden width alone cannot reach the budget. | §4.6 |
| 5 | **ATC vs CTA.** | **ATC is primary**, for both diffusion SGD and Diff-EKF, so the phase-5 comparison isolates the adapt step. CTA is a labelled variant measured in X1b and reported, not assumed. | §3.2, X1b |

Two further decisions were taken at the same time that were not previously listed as open questions, because the problems only became visible when the numbers were checked:

- **Total rotation is capped at $45°$**, with $\alpha=45°/T$ derived rather than chosen. Cumulative drift, not per-step drift, is what determines whether the task stays well-posed. (§4.3)
- **All learners in an experiment share one environment instance** and are stepped in lockstep, rather than being run separately and matched by seed. This makes X0 exact by construction and yields $E_{\text{cent}}$ in phase 1. (§6.1)

### 10.2 Still open

2. **Non-IID.** Is label skew across agents in scope for the first paper, or is IID sufficient? X6 and the Dirichlet $\beta$ axis are built either way, so this is a question about what gets written up, not about what gets implemented. Decide before phase 4.
4. **Per-node drift.** Is the interesting story "all agents drift together" (a shared model stays correct) or "agents drift differently" (a shared model becomes wrong, motivating the hierarchical shared/local extension of the research note §L5)? Default remains global drift; `drift_scope` is configurable so the question can be answered empirically rather than settled in advance. Decide before phase 4.

---

## 11. References

[1] R. Olshevskyi, Z. Zhao, K. Chan, G. Verma, A. Swami, S. Segarra, "Fully Distributed Online Training of Graph Neural Networks in Networked Systems," arXiv:2412.06105, Dec 2024. Code: `github.com/RostyslavUA/fdTrainGNN`.
Relevant for: the parameter-mixing update (their eq. 17), Metropolis–Hastings weights (eq. 16), the D-Adam vs D-AMSGrad momentum finding (Fig. 2a), and the communication ledger (Table I). Their §III-A derives fully distributed backpropagation for GCNNs and becomes the reference when this project reaches GNN models.

[2] A. Nedić, A. Ozdaglar, "Distributed subgradient methods for multi-agent optimization," IEEE TAC, vol. 54, no. 1, 2009.

[3] A. H. Sayed, "Adaptation, learning, and optimization over networks," Foundations and Trends in Machine Learning, vol. 7, no. 4–5, 2014.

[4] Companion research note: *Distributed Online Bayesian Learning of Deep Neural Networks — A Diffusion Extended Kalman Filtering Formulation.*
