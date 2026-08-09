# Research Work Plan — Distributed Online Learning Benchmark

**Project.** Build a benchmark for distributed online learning over a graph, establish first-order baselines on it, and prepare the ground for the diffusion EKF (Diff-EKF).

**Status.** Phases 0-4 complete: the environment, the learners, the runner, the full experiment grid X0-X7, and figures F1-F10. Phase 5 (Diff-EKF) is next. The open questions of §10 were resolved on 2026-07-30 and 2026-08-05; §10 records the decisions and the reasoning.

**Amended 2026-08-05** after the first full X1 run: §3.6 (hyperparameters are measured, not assumed), §3.7 ($n$ is an axis), §4.2 (why $n=4$, $T=1500$), §10.1b.

**Amended 2026-08-08:** X7 added — the sinusoidal schedule revisits states, which is what makes forgetting measurable at all.

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

**Why ATC is primary.** The Diff-EKF is ATC. If the SGD baseline were CTA, the phase-5 comparison would confound *EKF vs SGD* with *ATC vs CTA*. That reason stands alone and is why the choice holds.

An earlier draft added "and ATC generally has the better mean-square performance [3], so the baseline would be handicapped". **Measured on this benchmark, that is too strong** (design note D40): ATC wins 47 of 50 grid cells, so the sign is real, but at each ordering's own optimum the difference is around 0.001 against a seed spread of 0.002–0.012. ATC's advantage here is *robustness to step size*, not a better optimum — it separates sharply only in the unstable region (0.118 vs 0.251 at momentum lr 0.2). Mechanically that fits: ATC averages after stepping, so the combine step damps a too-large update, while CTA averages first and the damping arrives before the step it would have absorbed. With both as ATC, the two methods differ in the adapt step alone, at identical communication, and any Diff-EKF advantage is attributable to the second-order update rather than to a larger payload. Experiment X1b measures the ATC/CTA difference so that the choice is reported rather than assumed.

**Empirical support for parameter mixing.** [1] compare eq. (17) against a "D-naive" variant that runs $K$ rounds of consensus on *gradients* instead: D-naive needs roughly twice the message-passing rounds to reach the same MSE (their Fig. 2a). Parameter mixing is both cheaper and better behaved.

### 3.3 Local-only (lower reference)

Each agent trains on its own data with no communication. The gap between this and the distributed method *is* the value of cooperation, and it is the cleanest answer to Q2.

### 3.4 Optimizer state under diffusion

Adaptive optimizers carry per-node state, and whether the combine step mixes the moments as well as the parameters is a real fork. [1] settle it empirically (their Fig. 2a): **D-Adam converges quickly and then diverges**, because local momentum drifts apart across agents, while **D-AMSGrad — which runs consensus on the momentum terms too — is their best distributed method.**

Therefore:

- **plain SGD** for the exactness check X0, where the algebra must come out exact;
- **SGD with momentum, momentum also mixed**, or plain SGD, for X1–X6 — *whichever the tuning sweep selects per method* (see §3.6);
- **AdamW with its moments mixed**, only as a documented secondary.

Never an adaptive optimizer with unmixed state — that is a known failure mode, not an open question. The centralized reference uses whichever optimizer the distributed runs use, for symmetry.

### 3.5 What Diff-EKF will add (phase 5)

Same combine step, same communication. The adapt step becomes an EKF measurement update: the correction is scaled by a running posterior covariance instead of a fixed learning rate, and that covariance is carried forward as an uncertainty estimate. The reason to build the benchmark carefully now is that this substitution should require no change to the environment, the metrics, or the evaluation protocol.

---

### 3.6 Hyperparameters are measured, not assumed (added 2026-08-05)

An earlier draft of this plan fixed **SGD momentum 0.9 at lr 0.05** as the X1–X6
primary. The first full X1 run showed that configuration is unstable at $n = 2$:
the effective step $\eta/(1-\beta) = 0.5$ is past the stability edge for a batch
of two, `local_only` landed at chance (0.897), and — the giveaway —
`diffusion_sgd_atc_plain` finished *ahead of* `centralized_sgd`, which pools every
agent's samples and cannot legitimately be beaten by a distributed method.

Nearly the whole "cooperation is essential" gap was therefore an optimizer
artefact. Averaging over $N=10$ agents cancelled the noise that killed the lone
agent, so the diffusion methods survived a setting the baseline could not.

**Every method is now tuned on its own grid before any comparison is drawn**
(`scripts/sweep_hyperparameters.py`, design note D39): lr $\times$ $n$ $\times$
optimizer, scored on the *held-out* set over the last 100 steps, two seeds. The
per-method optimum is reported alongside the result. Selection is on held-out
rather than prequential error because prequential is what the tuning gets
reported against, and choosing on it would select for that estimate's noise.

At matched optimizer the expected ordering is restored:
centralized $\le$ ATC $\le$ local-only in every cell of the grid.

### 3.7 $n$ is an axis, not a hyperparameter (added 2026-08-05)

The sweep prefers $n = 8$–$10$ for every method — but $n$ does not move them
uniformly. Each method at its own optimum:

| $n$ | centralized | ATC | local only | cooperation gap |
|---|---|---|---|---|
| 2 | 0.105 | 0.109 | 0.259 | **0.150** |
| 4 | 0.094 | 0.096 | 0.177 | **0.081** |
| 6 | 0.090 | 0.093 | 0.147 | 0.054 |
| 8 | 0.078 | 0.090 | 0.134 | 0.044 |
| 10 | 0.081 | 0.089 | 0.127 | **0.038** |

Centralized moves 0.010 across the range; `local_only` moves 0.132. **The
cooperation gap collapses sixfold.** Mechanically this is expected — cooperation
is variance reduction and so is a larger batch, so they buy the same thing and
the more each agent sees alone, the less its neighbours add.

So choosing $n$ to minimise error would quietly minimise the effect Q2 exists to
measure. $n$ belongs on an axis, which is what **X4** already does. The headline
experiments fix $n = 4$.

The **pooling gap** (ATC vs centralized) is 0.003 at $n \le 6$, widening to 0.011
at $n = 8$: diffusion recovers almost all of pooling's advantage on a ring, which
is itself a result worth reporting.

## 4. Environment specification

### 4.1 Graph

$N$ agents on a fixed, connected, undirected communication graph $\mathcal G^{\mathrm c}$. Combination weights $a_{vu}\ge0$ respecting the graph, row-stochastic ($\sum_u a_{vu}=1$), with positive self-weight. Metropolis weights are the default; relative-degree is the alternative.

Topologies: complete, ring, path, 2-D grid, star, Erdős–Rényi, Watts–Strogatz, plus disconnected as a negative control.

The relevant scalar summary is the **spectral gap** $1-\rho$, with $\rho=\|\bm A-\tfrac1N\mathbf 1\mathbf 1^{\mathsf T}\|_2$. It is the natural x-axis for Q1: the complete graph has $\rho=0$ and should reproduce centralized behaviour, while a path has $\rho$ near 1 and should be the worst case.

**This definition requires $\bm A$ to be doubly stochastic, and that condition is load-bearing.** The term $\tfrac1N\mathbf1\mathbf1^{\mathsf T}$ is the projector onto the consensus direction only when $\mathbf 1$ is a *left* eigenvector of $\bm A$ as well as a right one. Metropolis weights satisfy this; relative-degree and uniform weights do so only on a **regular** graph. Off that case $\rho>1$ and the "gap" comes out negative — on a 10-agent star with relative-degree weights it is $-1.56$, while the star in fact mixes faster than a ring because the hub aggregates the network in one hop. The quantity is then wrong in sign and in ranking, not merely imprecise.

The benchmark therefore reports two numbers. The **spectral gap** above is used wherever it is defined, which includes all of X3 since that sweep uses Metropolis weights. The **mixing gap** $1-\mathrm{SLEM}$, where SLEM is the second-largest eigenvalue modulus of $\bm A$, is valid for any row-stochastic matrix and coincides with the spectral gap whenever both exist. `env/graph.py` raises rather than returning the undefined value, so a non-Metropolis sweep fails loudly instead of producing a plausible figure.

A second graph $\mathcal G^{\mathrm d}$ (data coupling, over which a predictor's forward pass would exchange information) is **empty for this phase**. MNIST with an MLP is Class L in the research note's terms: every agent's forward pass is purely local. $\mathcal G^{\mathrm d}$ becomes non-trivial only when the project moves to GNNs.

### 4.2 Sparse data arrival

Two independent knobs:

- **$n$ — samples per node per step**, small (1–10), **default 4** (was 2; see §3.7 and the horizon note below). This is the "not a huge amount of data" requirement.
- **$\pi_{\text{lab}}$ — label availability**, the probability that a given agent has a labelled sample at a given step. With $\pi_{\text{lab}}<1$ some agents idle on some steps. This matters because both diffusion SGD and Diff-EKF handle it gracefully — an agent with no label still benefits from the combine step — and because it is a realistic feature of the target applications. Default 1.0 in phase 1; a sweep axis in phase 4.

  **An idle step consumes no data.** The agent receives nothing and its shard is untouched, so $\pi_{\text{lab}}$ controls only how often an update happens, never how fast the data runs out. The alternative — delivering the samples unlabelled — destroys them, since no method here is semi-supervised and shards are finite. It would also make the maximum horizon independent of $\pi_{\text{lab}}$, foreclosing the natural follow-up to Q4: whether a sparse-label regime merely learns *slower* or genuinely *cannot* learn, which can only be answered by running it longer. See `design_notes.md` D24.

Each agent draws from its own **disjoint** shard of the MNIST training set, assigned once at setup. No sample is seen by two agents, and no sample is reused across time steps unless explicitly enabled.

**The shard budget binds the horizon, and it is easy to violate by accident.** With 60k images and $N$ agents, each shard holds $60000/N$ samples, which at $n$ per step lasts $60000/(Nn)$ steps. The defaults $N=10$, $n=2$ give a 6000-sample shard and a 3000-step ceiling; the default horizon is $T=1500$, comfortably inside it. Two consequences worth stating, because both have already caught us once:

- $N=10$, $n=4$, $T=2000$ — the combination that appears in early drafts of the runtime table — needs 8000 samples per agent and does **not** fit. `test_stream` asserts $Nn T\le 60000$ whenever `allow_epochs` is false, and this is the assertion that catches it.
- Scaling $N$ shortens the run rather than lengthening it. $N=100$ at $n=2$ leaves 600 samples per agent, i.e. 300 steps, which is too short for the drift experiments. Reaching $N=50$–$100$ therefore requires either `allow_epochs`, or $n=1$, or accepting a shorter horizon — a decision to be taken at the time, not assumed now (§10.1).

**Why the headline runs use $n = 4$, $T = 1500$.** At $N = 10$ that is exactly
60 000 — the largest $n$ that keeps the full horizon. The horizon is not free to
shrink, for three reasons beyond the data budget:

1. **$\alpha$ is derived** as `total_degrees / T` (§4.3), so $T$ sets the *drift
   rate*, not just the duration. Shortening $T$ does not truncate a run — it
   changes what the data looks like at step $t$. Two experiments at different
   horizons are therefore not comparable, which is why one horizon is used across
   X1, X1b, X2 and X5.
2. **X5's change point is at $t = 500$** and F7 plots
   $[t^\ast-50,\, t^\ast+300]$, so $T \ge 800$ is a floor for the adaptation
   transient to exist at all. (The change point *is* movable, and moving it costs
   only a re-run of X5 — no other experiment reads `change_points`. It is held
   fixed because the alternative buys a larger $n$, which §3.7 argues against on
   scientific rather than budgetary grounds.)
3. The methods converge by $t \approx 700$ at $n = 2$; the reported number is the
   tail after that. A horizon ending near convergence reports a transient.

$n = 10$ at $T = 600$ would fit the budget and lower every method's error by
about 0.01 — and shrink the cooperation gap from 0.081 to 0.038. That is the
trade being declined.

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

A one-time, standard offline training run on the full MNIST training set to convergence, same architecture. This yields the reference error rate $e^\star$ against which every online method is measured. It is a fixed asset, computed once and cached, not part of any experiment run — it carries its own seed, so re-running an experiment can never silently retrain the thing it is measured against.

**Convergence is verified, not assumed.** "To convergence" is an assertion until something checks it. By default a `validation_size` slice (5000) is held out of the *training* split, the best epoch is selected on it, and the test set is scored exactly once at the end. Early-stopping on the test set would leak into the very quantity every gap is measured against — invisibly, and always in the direction that flatters the online methods. The test error is still recorded per epoch, for inspection only.

Each run reports whether the selected epoch was inside the budget. At a 20-epoch budget, 9 of 16 rotation levels selected the *final* epoch, meaning the budget rather than convergence decided where training stopped; the budget is now 100 and `all_converged` reports the outcome rather than assuming it. This matters directionally: an under-trained $e^\star$ is too high, so every gap $\bar e_t - e^\star$ comes out too small.

Two further choices are configurable rather than fixed, so the comparison can be measured (`design_notes.md` D33):

- **How the per-rotation models relate.** `shared_seed` (default) trains each level independently from a common $\bm\theta_0$; `independent_seeds` uses a different $\bm\theta_0$ per level; `warm_start` initialises each level from the previous, which is ~3× cheaper but makes $e^\star(45°)$ depend on having passed through $40°$.
- **The train/validation split**, via `reference.validation_size`, and the epoch budget via `reference.epochs`.

Under drift it must be recomputed per rotation level, otherwise the measured gap conflates *decentralization cost* with *drift cost*.

"Per rotation level" needs a grid, since the rotation is continuous in $t$. The grid is **every $5°$ over $[-30°, +45°]$ — 16 levels**, which is the union of every rotation the configured schedules actually visit: linear reaches $[0°,45°]$, piecewise $[0°,15°]$, sinusoidal $[-30°,+30°]$. Nothing extrapolates.

An earlier version of this plan proposed ten levels over $[0°,45°]$ and mirrored the negatives, on the assumption that $e^\star(-\varphi) = e^\star(+\varphi)$ by symmetry. That is plausible but not free — $+15°$ and $-15°$ are genuinely different image distributions — so the full union is trained instead and the symmetry is **measured**: the largest mismatch across the grid is reported by `Reference.symmetry_error()`. The assumption turned out to hold to within 0.004 in error rate, so the cheaper grid would have been defensible; it is now a measured fact rather than a hope, at a cost of six extra trainings.

$e^\star$ at an intermediate rotation is **linearly interpolated** between the two nearest grid points rather than rounded to the nearest — a $5°$ grid with nearest-neighbour lookup would put a sawtooth into the gap curve at the same scale as the effect being measured. A rotation outside the grid raises rather than extrapolating: a gap against an extrapolated reference is not a measurement.

Sixteen offline trainings of a small MLP is a few minutes total, which is what makes the cap of §4.3 a budget decision as well as a validity one — the uncapped $300°$ range would have needed sixty.

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
| **X1** | Stationary baseline | ring, $N=10$, $n=4$, $T=1500$, $196$–$14$–$10$ MLP, no drift | Q1, Q2 — do all methods learn, what is the gap? |
| **X1b** | ATC vs CTA | ring + grid, stationary | Which diffusion ordering, and how much does it matter? |
| **X2** | Rotating baseline | as X1, linear rotation to $45^\circ$ total ($\alpha=0.03^\circ$/step) | Q3 — does the gap widen, does local-only collapse? |
| **X3** | Topology sweep | complete / grid / ring / path, stationary, **lr re-tuned per topology** | Q1 — gap vs spectral gap: the price of connectivity |
| **X4** | Sparsity sweep | $n\in\{1,2,4,8\}$, $\pi_{\text{lab}}\in\{0.25,0.5,1.0\}$, $T=750$ | Q4 — where does the sparse regime hurt? |
| **X5** | Abrupt shift | piecewise rotation, $15^\circ$ jump at $t=500$ | Q3 — adaptation transient, the cleanest tracking test |
| **X6** | Non-IID | Dirichlet label skew, $\beta\in\{0.1,1,\infty\}$ | Q2 — does cooperation still work when agents see different classes? |
| **X7** | Forgetting | sinusoidal rotation, amplitude $30^\circ$, period 500, `backward` evalset enabled | Q3 — does a method lose what it learned at a rotation it has left? |

X0, X1, X1b, X2 are the phase-3 deliverable. X3-X7 are phase 4.

**X7 is the only experiment whose schedule revisits states**, and therefore the only one
where forgetting is a meaningful question. Under `linear` the rotation never returns, so a
backward probe would ask about a state the model will never face again; under `stationary`
there is no earlier state at all. Measured coverage: the probe is defined for 97% of
evaluation steps under sinusoidal, 67% under linear and 0% under stationary. X7 also enables
the `backward` evalset explicitly -- it is not in the default set, and without it the
experiment measures nothing it exists for.

**Every experiment runs at the per-method tuned settings** from
`scripts/sweep_hyperparameters.py` (§3.6), never at a shared assumed default.
The tuning is reported with the results.

X1b should report parameter disagreement alongside error rate, since ATC and CTA can reach similar mean accuracy while differing noticeably in how tightly the agents agree.

**X1b is tuned on its own grid**, with ATC and CTA in the same cells. Its
design is "identical settings, only the ordering differs", so inheriting ATC's
optimum would report a tuning difference as an ordering difference if CTA's
optimum sits elsewhere. Both the matched comparison and each-at-its-best are
reported.

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
| **An assumed lr flatters the cooperative methods** | The headline gap is an optimizer artefact, not a learning result — and the baseline looks catastrophic for a tuning reason | **Materialised once (§3.6).** Resolved: per-method tuning before any comparison; the sanity check is that no distributed method may beat `centralized_sgd` at matched settings |
| ATC/CTA chosen inconsistently between SGD and Diff-EKF | Phase-5 comparison confounded | ATC primary for both; CTA a labelled variant only |
| Results not reproducible across machines | Wasted debugging | Separable seeds, pinned versions, determinism flags |

---

## 10. Decisions taken, and what remains open

### 10.1 Resolved (2026-07-30)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| 1 | **$N$ and horizon.** Is $N=10$ the target scale, or should the benchmark reach $N=50$–$100$? | Start at $N=10$, $n=4$ (raised from 2 on 2026-08-05, §3.7), $T=1500$. Raise $N$ only if measured runtime allows, and only after re-checking the shard budget — scaling $N$ *shortens* the feasible horizon (§4.2). | §4.2, §6 |
| 3 | **Parameter budget.** How small may the MLP be for a dense phase-5 covariance? | Downsample inputs to $14\times14$ and use a $196$–$14$–$10$ MLP, $p=2908$. Dense covariance is then affordable and phases 1–4 run on the same architecture phase 5 will use. Shrinking the hidden width alone cannot reach the budget. | §4.6 |
| 5 | **ATC vs CTA.** | **ATC is primary**, for both diffusion SGD and Diff-EKF, so the phase-5 comparison isolates the adapt step. CTA is a labelled variant measured in X1b and reported, not assumed. | §3.2, X1b |

Two further decisions were taken at the same time that were not previously listed as open questions, because the problems only became visible when the numbers were checked:

- **Total rotation is capped at $45°$**, with $\alpha=45°/T$ derived rather than chosen. Cumulative drift, not per-step drift, is what determines whether the task stays well-posed. (§4.3)
- **All learners in an experiment share one environment instance** and are stepped in lockstep, rather than being run separately and matched by seed. This makes X0 exact by construction and yields $E_{\text{cent}}$ in phase 1. (§6.1)

### 10.1b Resolved (2026-08-05)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| 6 | **How are lr, $n$ and the optimizer chosen?** | Measured per method on a held-out grid, not assumed. Forced by the X1 instability that made `local_only` land at chance and a distributed method beat centralized. | §3.6, D39 |
| 7 | **What $n$ for the headline runs?** | $n=4$, $T=1500$ — the largest $n$ keeping the full horizon. Larger $n$ lowers every error but collapses the cooperation gap sixfold, so $n$ is an axis (X4), not a value to optimise. | §3.7, §4.2 |
| 8 | **Is CTA tuned separately?** | Yes — its own grid with ATC in the same cells, so F8 reports an ordering difference rather than a tuning difference. | §6, X1b |
| 9 | **Is lr re-tuned per topology?** | Yes, and now **measured**: the effect is real but small. Six of seven topologies select the same cell as the ring (momentum, lr 0.01); only `complete` differs, and structurally -- one combine reaches full consensus there, so ATC *is* centralized and inherits its preference for a large plain step. Better mixing buys tolerance to an *oversized* step (spread 0.34-0.59 at lr 0.2) rather than a better optimum (0.093-0.103 at lr 0.01). | §10.2 item 5, X3, `results.md` |

### 10.1c Resolved (2026-08-06)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| 10 | **Phase 4 before phase 5, or the filter first?** | **Finish X3, X4 and X6 before Diff-EKF.** The filter is a different mathematics and a different learning paradigm; keeping it out of the backpropagation baselines until those are fully characterised avoids mixing the two. Re-running X4 and X6 later to include the filter is accepted -- there is no deadline. | §6, §8 |
| 11 | **Where does CUDA help?** | Nowhere in phases 1-4 -- it is 0.69-0.92x, *slower*, because p=2908 with batches of 4-40 is too small to amortise kernel launches. It is a 14x win on the dense p x p covariance, so phase 5 depends on it. `run.device` now refuses anything but `cpu` rather than accepting a value it ignores. | D43 |

### 10.1d Resolved (2026-08-09)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| 12 | **What does halving the message actually cost?** | **0.007-0.046, and the cost has a shape.** Across the tuned X4 plane it grows ~2.5x toward the sparse corner along each axis; across three decades of label skew it is **flat** (0.015 / 0.013 / 0.015). Both X4 axes control how much signal one step carries, and the extra $p$ scalars buy momentum -- worth most where each gradient is noisiest, worth nothing extra when gradients are merely *different* from a neighbour's. **The second half of the message pays under sparsity, not under heterogeneity**, which bounds a $p$-per-link Diff-EKF at ~0.046 worst case and says where to look for it. | D45, `results.md` §9.2, §10, F6a panel 4 |
| 13 | **How is a quantity with a construction-fixed sign protected?** | **By a test, not by looking at the figure.** F6b's penalty and F6a's payload cost are both non-negative by construction and both produced impossible values (-0.042; exactly 0.000 in all twelve cells) that rendered as unremarkable cells. Now asserted in `tests/test_figures.py`, and `make_summary_docx` raises rather than printing a negative penalty. | D44 |

### 10.2 Still open

2. **Non-IID.** Is label skew across agents in scope for the first paper, or is IID sufficient? X6 and the Dirichlet $\beta$ axis are built either way, so this is a question about what gets written up, not about what gets implemented. Decide before phase 4.
4. **Per-node drift.** Is the interesting story "all agents drift together" (a shared model stays correct) or "agents drift differently" (a shared model becomes wrong, motivating the hierarchical shared/local extension of the research note §L5)? Default remains global drift; `drift_scope` is configurable so the question can be answered empirically rather than settled in advance. Decide before phase 4.

5. **Does the optimal lr depend on topology?** Plausibly yes: a denser graph
averages over more neighbours per round, which cuts gradient noise further — the
same mechanism that let ATC survive an lr that killed `local_only` (§3.6). So a
complete graph may tolerate a higher lr than a ring. Tuning on the ring and
holding it fixed would make F3 partly measure "how well does the ring's lr suit a
star" rather than connectivity alone. **Resolved 2026-08-05: re-tune per
topology.** `local_only` and `centralized_sgd` ignore the graph entirely, so the
topology axis only needs the diffusion learners — roughly an hour. To be run
after the tuned X1/X1b/X2/X5 and before phase 5, so Diff-EKF starts knowing which
topology to use as primary rather than inheriting the ring by default.

---

## 11. References

[1] R. Olshevskyi, Z. Zhao, K. Chan, G. Verma, A. Swami, S. Segarra, "Fully Distributed Online Training of Graph Neural Networks in Networked Systems," arXiv:2412.06105, Dec 2024. Code: `github.com/RostyslavUA/fdTrainGNN`.
Relevant for: the parameter-mixing update (their eq. 17), Metropolis–Hastings weights (eq. 16), the D-Adam vs D-AMSGrad momentum finding (Fig. 2a), and the communication ledger (Table I). Their §III-A derives fully distributed backpropagation for GCNNs and becomes the reference when this project reaches GNN models.

[2] A. Nedić, A. Ozdaglar, "Distributed subgradient methods for multi-agent optimization," IEEE TAC, vol. 54, no. 1, 2009.

[3] A. H. Sayed, "Adaptation, learning, and optimization over networks," Foundations and Trends in Machine Learning, vol. 7, no. 4–5, 2014.

[4] Companion research note: *Distributed Online Bayesian Learning of Deep Neural Networks — A Diffusion Extended Kalman Filtering Formulation.*
