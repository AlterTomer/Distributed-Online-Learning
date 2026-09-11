# Phase 5: what the diffusion filter still has to be tested on

Every experiment X0–X18 measured methods that either pool the data or share only
a parameter vector. The diffusion filter is the first method that holds a
*belief* per agent and mixes it, so most of those experiments have an analogue
here that asks a genuinely different question — and a few of them ask a question
that could not be posed before at all.

This file is the checklist. Tick an item when its run has landed **and** its
result is written up; a run whose numbers nobody has read is not done. As each
lands it becomes a row in [`experiments.md`](experiments.md), which stays the
operational index — this file is the plan, that one is the record.

**Two learners are the subject throughout**: `diffusion_ekf` (mean-only,
deployable) and `diffusion_ekf_full` (covariance sharing, the ceiling). The gap
between them is the price of not shipping covariances and is the point of nearly
every row below. `diffusion_ekf_onehop` is a fixture, not a competitor: it is not
tuned and appears only in the gate.

**Baselines in every cell**: `centralized_ekf_gamma` (the reference the diffusion
filter should approach), `diffusion_sgd_atc`, and `local_only` where the
cooperation gap is the question. Carrying the centralised filter is what makes
each result readable as "how much of the centralised filter does diffusion
recover", which is the phase-5 claim.

**Hyperparameters**: X13's selected setting — `transition: scalar`, γ=0.9995,
q=6e-5, σ₀²=0.01 — carried unchanged, the X14 discipline. Re-tune only if a
result's signature is mis-tuning (belief too diffuse, damage dominated by the
fitting term), and then report both, the X15 pattern. Each agent sees $1/N$ of
the information at the same forgetting rate, so σ₀² is the first knob to suspect.

**Cost**: measured from a 60-step smoke run carrying both diffusion variants plus
the centralised filter and ATC — roughly **1 h per cell at five seeds**, and
3.3 GiB of GPU. Memory is what binds: each variant holds $N$ covariances at
64.5 MiB, and full sharing needs a second set live during the mix. Three
diffusion filters in one process will not fit; two do.

---

## Gate — before anything is believed

- [x] **P5.0a** Complete-graph exactness as a unit test. `diffusion_ekf_onehop`
  reproduces the centralised filter to 1e-10 in mean *and* covariance, and
  `diffusion_ekf_full` with a local adapt does **not** — both halves asserted, so
  the positive one cannot pass vacuously. `tests/test_learners.py`.
- [x] **P5.0b** The same identity through the *runner*'s own dispatch. Residual
  **4.4e-16** — machine epsilon, and comparable to X0's 1.7e-15, which was not
  the expectation: a Cholesky solve and a $p\times p$ subtraction per step could
  reasonably have accumulated more than a weighted mean of gradients does. It
  does not, because both filters run the *same* `woodbury_update` on the same
  concatenated information in the same order. Both local-adapt variants miss by
  0.12, so the negative half is nowhere near marginal.
  `tests/test_exactness.py` §3.

## First measurement

- [x] **P5.1** *(X19)* **Stationary plus two drift rates, on ER $p=0.3$ and the
  complete graph.** ✅ Run and written up (D79). Covariance sharing buys +0.0002
  to +0.0007 for 2909× the bandwidth — below threshold, so mean-only is free.
  Sparsity is nearly free too. But diffusing the belief costs +0.0228 → +0.0411
  as drift hardens, and the filter loses to ATC under abrupt drift. **Provisional
  in two ways** (unmatched baseline, centralised tuning) — see P5.1a and P5.1b.
  Original scope:
  `scripts/run_diffusion_ekf.py --lr`, then without the flag. Does the diffusion
  filter recover the centralised one, and what does mean-only sharing cost
  against full sharing? The complete graph earns its place here for a reason it
  never did for SGD — with a *local* adapt the filter is closest to, but still
  not equal to, the centralised one, so that cell isolates what the local adapt
  costs, with graph sparsity removed; the ER cells then add what sparsity costs
  on top. 6 cells + 4 stationary twins.

  ⚠ The harsh condition is **not** monotone, and cannot be: over $T=1500$ the
  45° cap makes 0.03°/step the fastest legal linear drift, while X14 put the
  gradient methods' break at 0.038–0.044. No legal monotone rate reaches even
  the mild end of where methods fail, so the harsh end is delivered as jumps
  (`every25_jump15`, 0.60°/step), which X11 and X17 have already characterised.
  The same bind X18 documents.

## Repairing the first measurement — before anything else

X19's two defects are mine, not the method's, and both are cheap to close. Until
they are, no later row can be interpreted: every one of them would inherit an
unmatched baseline and a filter tuned for ten times the data it has.

- [ ] **P5.1a** *(X20)* **Re-tune $q$ and $\sigma_0^2$ for the diffusion
  information rate.** Script: `run_diffusion_tuning.py --tune`, a 5x5 grid, two
  decades either side of the predicted $6\times10^{-6}$ and including the
  centralised $6\times10^{-5}$ so "no change" is expressible. It refuses to
  select an argmin that lands on a grid edge. The mechanism in D79 names $q$ as the mis-scaled parameter
  and gives the direction: it was chosen to balance an influx of $N\bm\Delta$ per
  step and now faces $\bm\Delta$, so it should fall by roughly $N$. The grid spans
  wider than that argument, because a scaling argument that predicts the answer is
  the worst reason to only look where it points.
- [ ] **P5.1b** Add `diffusion_sgd_atc_plain` and re-read at **matched
  bandwidth**. Folded into `run_diffusion_tuning.py`'s main pass. X19 compared a filter sending $p$ against an ATC sending $2p$,
  which is precisely the pairing D29 exists to prevent. In X18's stationary twin
  `atc_plain` was 0.0863 against `atc`'s 0.0790, so the correction may turn a tie
  into a win.
- [ ] **P5.1c** **Promote one-hop from fixture to method** and measure it on ER.
  Done as a learner: `diffusion_ekf_onehop_mean` (one-hop adapt, mean-only
  combine) — the deployable pairing, since X19 showed the combine axis is empty.
  It rides in `run_diffusion_tuning.py`'s main pass.
  D79 shows the deficit is in the adapt step and no combine rule can reach it;
  one-hop is the only implemented thing that can. ⚠ Implement its exchange as
  **raw measurements, not information factors** — 788 scalars against 122 136,
  exactly equivalent since the receiver already gets the sender's linearisation
  point. Its real costs are compute and privacy, not bandwidth.

## Every earlier experiment that has an analogue

Ordered by what each would change if it came out badly, not by experiment number.

- [ ] **P5.2** *(X6/X17 analogue)* **Dirichlet label skew.** ⚠ **The one that
  matters most.** D77 states plainly that X17 says nothing about a diffusion
  filter: the centralised filter pools, so skew barely reaches it, and every
  "the filter is robust to heterogeneity" reading of X17 is really "pooling
  defused the skew". Here each agent holds its own skewed shard and the combine
  step has to reconcile beliefs formed from different label distributions. This
  is where D50's finding — cooperation pays *more* under label shift — would
  bite, and where covariance sharing could plausibly matter most, since an agent
  that has seen only three classes should be *confident* about those and
  uncertain elsewhere, which is information a mean-only exchange throws away.
  β ∈ {0.1, 1, 100}, stationary and drifting.
- [ ] **P5.3** *(X3 analogue)* **Topology and the spectral gap.** Whether the
  covariance-sharing gap widens as connectivity falls. Mean-only sharing loses
  more when information has to travel further, so the two variants should
  separate here if they separate anywhere. Learning rate re-tuned per topology
  for the SGD arm, as X3 does.
- [ ] **P5.4** *(X4 analogue)* **Sparse labels, $n$ × $\pi_{\text{lab}}$.** An
  unlabelled agent passes its prediction through and still takes part in the
  combine — one of diffusion's more attractive properties, and untested for a
  filter. A belief that only ever loses confidence between labels behaves quite
  differently from a parameter that simply stops moving.
- [ ] **P5.5** *(X9/X16 analogue)* **The break rate.** Where does the diffusion
  filter stop tracking, as a rate? The centralised filter survives to
  0.064°/step against 0.038–0.044 for the gradient methods; the question is how
  much of that 1.5× survives decentralisation. This is the only place the
  filter's advantage is a *rate* rather than an error difference.
- [ ] **P5.6** *(X11 analogue)* **Repeated abrupt shifts, $J$ × $t'$.** Recovery
  after a shift, when each agent must re-linearise and then reconcile.
- [ ] **P5.7** *(X8 analogue)* **Heterogeneous drift.** Agents drifting
  *differently* is the case where per-agent beliefs should beat a pooled one —
  the centralised filter must average incompatible states, while diffusion can
  hold them apart. ⚠ One of only two rows below where the diffusion filter could
  plausibly **beat** the centralised reference rather than approach it; worth
  running earlier than its number suggests if the earlier rows are dull.
- [ ] **P5.8** *(X10 analogue)* **Prior drift / label shift.** The other row
  where per-agent beliefs might win, and where D50 measured cooperation paying
  most.
- [ ] **P5.9** *(X14 analogue)* **The generalisation grid.** Only once the
  cheaper rows have shown the effect is real and where it lives — 21 conditions
  × 2 diffusion variants is expensive and is the wrong place to discover a
  tuning problem.
- [ ] **P5.10** *(X7, X12, X18 analogues)* **The remaining drift shapes** —
  sinusoidal, constant-rate linear, sawtooth. Low priority: X18 established that
  the shape of the drift has not once changed the ordering between methods, and
  there is no reason to expect diffusion to be the first exception. Run them to
  close the grid, not to learn something.

## Algorithmic paths out of the information deficit

D79's finding is that a local adapt gathers $\bm\Delta_v$ where the centralised
filter sums $\sum_v\bm\Delta_v$, and that **no combine rule reaches that** —
averaging information is not summing it. So every candidate below is a change to
what an agent *gathers* or to how it *reports its confidence*, never to the fusion
rule. Ordered by value per unit of work.

- [ ] **P5.15 — the covariance is never told the mean was averaged.** ⭐ The
  cheapest idea on this page and the most directly aimed at the mechanism. After
  $\bm m_v\leftarrow\sum_u a_{vu}\bm\psi_u$, the *estimate* has been averaged over
  $|\mathcal M_v|$ agents but the *covariance* still describes one agent's
  evidence. Two extremes bracket the truth:

  $$\bm P_v\leftarrow\sum_u a_{vu}\bm P^{\psi}_u \quad\text{(perfect correlation, what we do)}$$
  $$\bm P_v\leftarrow\sum_u a_{vu}^2\,\bm P^{\psi}_u \quad\text{(independent errors)}$$

  The first is `lem:conservative`, tight exactly when the neighbours' errors
  coincide. The second is what independence gives, and is smaller by roughly
  $|\mathcal M_v|$ — which is the same factor D79 says the belief is inflated by.
  **It costs no extra communication at all**: one line in `combine`. Interpolate
  with $\bm P_v\leftarrow\sum_u a_{vu}^{\beta}\bm P^{\psi}_u$, $\beta\in[1,2]$.

  ⚠ The risk is real and is the reason to measure before adopting: the errors
  *are* correlated through shared history, so $\beta=2$ is over-confident and an
  over-confident EKF can diverge (D61). **P5.14 is the prerequisite** — it
  measures where in $[1,2]$ the truth actually sits by comparing reported variance
  against realised squared error.

- [ ] **P5.16 — LO-FI: diagonal-plus-low-rank precision.** Represent
  $\bm\Lambda_v\approx\bm D_v+\bm W_v\bm W_v^{\trans}$, $\bm W_v\in\mathbb
  R^{p\times L}$ (Chang et al., CoLLAs 2023). Memory $O(pL)$ instead of $O(p^2)$;
  the only dense inverse is $L\times L$.

  **This does not fix the information deficit** — it changes how the covariance is
  *represented*, not what enters it — and at $p=2908$ we do not yet have the
  memory problem it solves. Its value is elsewhere, and it is real:

  * $\bm J^{\trans}\bm\Delta\bm J=\bm B\bm B^{\trans}$ is low-rank *before* any
    approximation, and $\bm B$ is precisely what a one-hop exchange already ships.
    The two compose: append received blocks to $\bm W$, compress. One-hop first is
    not wasted work.
  * It reopens the **prior-corrected information combine**, $\bm\Omega_v\leftarrow
    \sum_{u}\bm\Omega_u-(|\mathcal M|-1)\bm\Omega_{\text{prior}}$, which D79
    dismissed as $O(p^3)$. That is true only in dense form; in factored precision
    form there is no $p\times p$ inverse anywhere. Shipping $\bm W_u$ costs $Lp
    \approx 29\,000$ scalars at $L=10$ against a full covariance's 8.4 M.
  * It answers the obvious reviewer objection to an $O(p^2)$-per-agent method.

  ⚠ **The blocker is data incest, and it must be solved rather than assumed
  away.** Summing *accumulated* precisions double-counts evidence that already
  travelled — which is why conservative fusion and \ac{ci} exist at all. Averaging
  is incest-safe and transfers nothing; summing transfers everything and
  double-counts. One-hop threads the needle because it sums *measurement*
  information, fresh each step and entering each neighbourhood exactly once, which
  is why it is exact on a complete graph rather than merely better. A LO-FI
  precision exchange has no such guarantee and would need channel filters or
  equivalent bookkeeping.

  ⚠ Second risk, specific to us: truncation **discards** information and we are
  already short of it. But $p=2908$ is the one scale where this is checkable — the
  exact full-covariance filter is computable here, so rank $L$ can be scored
  against ground truth. That is the right moment to adopt an approximation, before
  it becomes the only option.

- [ ] **P5.17 — multi-round information flooding, and the reductio it forces.**
  One round of one-hop gives $\mathcal M_v=\N_v\cup\{v\}$. $L$ rounds with
  **source-tagged** blocks (so each $\bm\Delta_u$ is counted once) give the
  $L$-hop neighbourhood, and at $L=\operatorname{diam}(\G)$ that is $\V$ — i.e.
  **exactly the centralised filter**. Our \ac{er} $p=0.3$ graph has diameter 2–3,
  so two or three rounds would close the deficit entirely, and with the raw-sample
  encoding the payload is a few $p$ per link.

  ⚠ **Worth writing down precisely because it is the honest reductio.** At that
  point every agent holds every agent's data and the method *is* the centralised
  filter, replicated. `communication.py` already records the uncomfortable fact
  this lands on: at these dimensions, shipping raw samples to a fusion centre is
  *less* traffic than ring diffusion. So flooding cannot be the answer we argue
  for — it would mean the decentralised paradigm is justified by privacy and data
  sovereignty, not by bandwidth. Useful as an **upper bound to measure against**,
  not as a method to propose.

- [ ] **P5.18 — channel filters.** The textbook incest fix: track per-link what
  has already been shared and subtract it. At $p=2908$ each channel filter is
  another covariance per link, so this is almost certainly infeasible here.
  Recorded so that "why not just do decentralised data fusion properly" has an
  answer.

## The ladder: preserve the information first, then cut the price

The rows above are fixes to a cheap method. This section is the opposite
strategy, and the one the note already used once when it measured
`eq:cov_combine` before `eq:cov_local`: **build the most information-preserving
decentralised scheme that exists, establish what is achievable, then remove
capability until it breaks.** A ceiling that has been measured makes every later
reduction informed; a ceiling that was assumed makes all of them guesses.

**The obstacle is not bandwidth, it is double counting.** Information is additive
in the precision domain, so preserving it means *summing*, and summing evidence
that already travelled counts it twice — data incest. Every scheme below is
distinguished by how it earns the right to sum.

- [ ] **P5.19 — the ceiling: source-tagged flooding at $\operatorname{diam}(\G)$
  rounds.** Exactly the centralised filter (P5.17), and included here **only as the
  measured upper bound**, not as a proposal. Establishes what "no information lost"
  is worth in error terms, so every row below is a measured fraction of a real
  number rather than of a hope.

- [ ] **P5.20 — channel filters on a spanning tree.** ⭐ The textbook answer, and
  the strongest *genuinely decentralised* scheme that is exact. Each link carries a
  filter tracking the information already common to its two endpoints, and each
  agent fuses $\bm\Lambda_v\leftarrow\bm\Lambda_v+\sum_u(\bm\Lambda_u-\bm\Lambda_{
  \text{chan}(v,u)})$ — subtracting exactly what would otherwise be counted twice.
  **Exact on an acyclic network**, which is why the tree matters: on a graph with
  cycles the same evidence returns by two paths and the subtraction no longer
  accounts for it.

  Cost is why it was never considered here: one covariance per link, 64.5 MiB
  each, 22 directed links on our \ac{er} graph — about 1.4 GB. **With P5.16's
  rank-$L$ factors that becomes $O(pL)$ per link, roughly 29 000 scalars**, and
  the scheme moves from infeasible to routine. This is the strongest reason to do
  LO-FI, and a better one than memory.

  ⚠ Requires a spanning tree, so it gives up some of the graph's connectivity to
  buy exactness — a real trade to measure, not a technicality.

- [ ] **P5.21 — consensus on the information *increments*, rescaled by $N$.**
  Average consensus converges to $\frac1N\sum_u\bm\Delta_u$; multiplying by $N$
  recovers the sum. **Incest-free by construction**, because what is averaged is
  this step's increment — fresh, and entering the average exactly once — rather
  than an accumulated precision. Works on any connected graph, no tree and no
  per-link state.

  $L$ rounds trades accuracy for bandwidth continuously, which is what makes it
  the natural rung to cut: $L\to\infty$ is exact, $L=1$ is one-hop with a
  rescaling. Needs $N$ known globally, which is a mild assumption but should be
  stated. The factor grows by one block per round, so it wants P5.16's compression
  to stay bounded.

- [x] **P5.22 — one-hop with an $N/|\mathcal M_v|$ rescaling. BUILT** as
  `information_exponent`, $c=(N/|\mathcal M_v|)^{\alpha}$ with $\alpha\in[0,1]$.
  Scales **both** the information and the score: shrinking $\bm P^{\psi}$ alone
  would make every update $c$ times too small, so the filter would report a
  confident belief it never moved toward. Default $\alpha=0$ changes nothing. The cheapest rung
  and a one-line change: $\sum_{u\in\mathcal M_v}\bm\Delta_u$ is a sum over
  $k$ agents, so $\frac{N}{k}\sum_{u\in\mathcal M_v}\bm\Delta_u$ is an unbiased
  estimator of the network total under exchangeability.

  ⚠ It **claims confidence it has not gathered** — an extrapolation, not evidence
  — so it shares P5.15's risk of over-confidence and needs the same P5.14
  measurement first. Note that P5.15 and this are the same correction in the two
  domains: both tell the covariance that the estimate is better than one agent's
  data implies.

### Checked against Cattivelli & Sayed (TSP 2010), and one claim withdrawn

The diffusion LMS paper defines **two** combination matrices, and they are
exactly our two axes: $c_{lk}$ "determine which nodes $l$ should share their
**measurements** with node $k$" in the incremental step, and $a_{lk}$ determine
which share their **intermediate estimates** in the diffusion step. So
`adapt_scope` is $\bm C$ and the Metropolis weights are $\bm A$.

**Withdrawn:** the claim that our local adapt is a *deviation* from the canonical
method. It is not. $\bm C=\bm I$ is an explicitly named and studied special case
— "the \ac{atc} algorithm without measurement exchange" — noted as the mode
originally proposed for least-squares adaptive networks. `diffusion_ekf` is a
recognised variant, not an idiosyncrasy.

**What does hold, and is stronger than the framing claim was:** the paper
*proves* that measurement exchange is never worse. Under equal regressor
covariance and noise variance across nodes and a stated weight choice, "the
algorithm that uses measurement exchange will have equal or lower network
\ac{msd} than the algorithm without measurement exchange". That is a theorem
about the population quantity, not a simulation result, and it is direct
theoretical support for pursuing one-hop.

**And one observation that sharpens D79.** The paper notes that with $\bm C=\bm
I$, \ac{atc} still "uses measurements available at the *neighbors* of node $k$" —
because adapting locally and then averaging estimates carries the neighbours'
measurement influence into the estimate. So even our local-adapt filter already
gets neighbourhood information *into the mean*. What it does not get is that
information into the **covariance**, which is precisely D79's mechanism, and
precisely why P5.15's free correction is aimed at the right place.

### Checked against the Kalman paper too — and one-hop is the canonical algorithm

Cattivelli & Sayed, IEEE TAC 55(9) 2010, the paper `\cite{cattivelli2010}`
actually points at. Four things it settles, and they change what the note may
claim.

**1. The incremental step loops over neighbours.** Algorithm 1's Step 1 reads
"for every neighboring node $l\in\N_k$, repeat ... end". Algorithm 2, the
information form, does the same. **So one-hop *is* the diffusion Kalman filter**,
and `diffusion_ekf` with a local adapt is not the canonical algorithm — unlike
the LMS case, where $\bm C=\bm I$ is a named variant. `diffusion_ekf_onehop_mean`
should be read as the method and the local-adapt one as our reduction of it.

**2. ⚠ The note's imported stability analysis is analysis of one-hop.** The
detectability condition is stated as "if every node were to use a conventional
Kalman filter on the measurements *from its neighborhood*, its estimate would
converge". Section IV's Lyapunov recursion and steady-state \ac{msd} are derived
for Algorithm 1. So `sec:assumptions`' appeal to this analysis does not currently
describe the variant X19 measured — a correction the note needs.

**3. The authors flag D79's mechanism themselves.** Of the propagated
covariances: *"these matrices do not represent the covariances of the state
estimation errors any longer, since the diffusion update is not taken into
account in the recursions for these matrices."* That is exactly D79, stated for
the linear case in 2010. Our contribution is not the phenomenon — it is measuring
what it costs for a nonlinear filter on a real network, where the deficit turned
out to be 0.023–0.041 and to grow with drift.

**4. Section IV derives the true covariance — but it is not implementable.**
⚠ Corrected from an earlier claim here. Their eq. (32) is a Lyapunov recursion
over the *augmented* error vector collecting every node: an $Np\times Np$ matrix,
6.8 GB at our size, and it needs the true model matrices. It computes theoretical
\ac{msd}; an agent cannot propagate it. So the interpolation
$\sum_u a_{vu}^{\beta}\bm P^{\psi}_u$ stays the practical route, and what the
paper contributes here is confirmation of the phenomenon rather than a runnable
correction.

**5. Algorithm 1 exchanges raw measurements**, $\{\bm H_l,\bm R_l,\bm y_l\}$, plus
$\bm\psi_l$ — not information factors. Algorithm 2 exchanges
$\bm H^{\trans}\bm R^{-1}\bm H$ and $\bm H^{\trans}\bm R^{-1}\bm y$, which is what
our one-hop implements. The paper treats them as alternatives and notes Algorithm
1 can send a Cholesky factor to economise, which independently vindicates the
finding that at $p\gg d$ the measurements are the cheaper encoding.

## Built in this pass

Two mechanisms, both landed with tests, both available to X20 and everything
after it. Neither needed a new learner class: they are dials on `DiffusionEKF`.

- [x] **`combine_exponent`** — `eq:combine_exponent`, $\bm P_v\leftarrow\sum_u
  a_{vu}^{\beta}\bm P^{\psi}_u$ with $\beta\in[1,2]$. P5.15, implemented.
  Verified numerically: on a complete graph with uniform weights, $\beta=2$
  divides the covariance by exactly $N$ — which is the factor D79 says the belief
  is inflated by, arrived at independently. **Costs no communication at all.**
  Default stays $\beta=1$, so nothing changes until it is asked for.
- [x] **`adapt_rounds`** — the measurement set becomes the $L$-hop neighbourhood.
  $L=1$ is the canonical diffusion Kalman filter's incremental step; $L\ge
  \operatorname{diam}(\G)$ makes it the whole vertex set, so the filter equals the
  centralised one on *any* connected graph. P5.19 and P5.21 collapse into this one
  parameter, and P5.22's rescaling is `combine_exponent` in the other domain.

  **Reachability is boolean, and that is the incest guard**: each agent's
  $\bm\Delta_u$ enters once however many paths carry it, which is source-tagged
  flooding expressed as a set rather than a sum along paths. Tested directly on a
  complete graph at three rounds, where the naive version would multiply-count
  everything.

## Deferred, deliberately, with the reason

Not forgotten — these are here so the decision is visible rather than implicit.

- **LO-FI, diagonal-plus-low-rank precision (P5.16).** A large change — a new
  representation touching every covariance operation — that does **not** address
  the deficit X20 is about to measure. It should follow the combine decision, not
  precede it: if `adapt_rounds` closes the gap, LO-FI becomes the scalability
  story and is the right answer to "your filter needs $O(p^2)$ per agent"; if it
  does not, LO-FI's factored form is the route to P5.20 and becomes algorithmic
  rather than engineering. Either way the argument for it is *stronger* after
  X20, and building it now would be building against a moving target.
- **Channel filters (P5.20).** Needs LO-FI to be feasible at all — one covariance
  per link is 1.4 GB dense — so it is blocked behind P5.16 by construction.
- **The exact network covariance recursion.** ⚠ **Withdrawn as an implementation
  target.** Cattivelli & Sayed's eq. (32) is a Lyapunov recursion over the
  *augmented* error vector across all nodes: $Np\times Np$, which is 6.8 GB here,
  and it requires the true model matrices. It is an analysis tool for computing
  theoretical \ac{msd}, not something an agent can propagate. An earlier note in
  this file claimed we could "implement what the analysis says is correct" — that
  was wrong, and `combine_exponent` is the practical route instead.
- **`diffusion_ekf_full` in future comparisons.** X19 priced covariance sharing
  at $+0.0002$ to $+0.0007$ for $2909\times$ the bandwidth. Carrying it further
  costs 1.3 GiB to re-confirm a null.

## Every drift schedule, and where the filter meets it

The benchmark supports seven schedules. This table exists so that "have we tested
all the drifts?" is a question with a checkable answer rather than a recollection
— and writing it out immediately found one that no plan item named.

| schedule | what it is | covered by | status |
|---|---|---|---|
| `stationary` | no drift; the twin every damage number is measured against | X20 main pass | ⏳ running |
| `linear` | constant rate, monotone | X20 at 0.03°/step — **the fastest legal monotone rate**, since the 45° cap over T=1500 forbids more | ⏳ running |
| `recurring` | a jump of fixed size in a fresh random direction every $t'$ steps | X20 at `every25_jump15`; **P5.6** for the full $J\times t'$ grid | ⏳ / open |
| `ramp` | accelerating rate, so one run sweeps the rate axis and names a break point | **P5.5** | open |
| `sinusoidal` | smooth reversal; the only schedule that *revisits* states | **P5.10** | open |
| `sawtooth` | monotone ramp punctuated by a reset — the shape none of the others reach | **P5.10** | open |
| `piecewise` | a step function at named change points | **⚠ nothing** — see below | **gap** |

- [ ] **P5.23 — `piecewise` has no plan item.** Found by writing the table above,
  which is the point of writing it. It is the only schedule where the change
  points are *specified* rather than periodic or random, so it is the one that can
  place a shift exactly where a filter is most vulnerable — mid-transient, or
  immediately after the covariance has contracted. `recurring` cannot express
  that, because its jumps are evenly spaced by construction.

  Low priority as a *drift shape* — X18 established that shape has not once
  changed the ordering between methods — but potentially high as an *adversarial*
  probe once the tuning question is settled, because a filter operating one decade
  below a divergence cliff (X20's selected point) is exactly the kind of thing a
  well-placed shift could push over. Worth one cell, not a grid.

⚠ **Two schedules are covered only at a single setting** by X20: `linear` at one
rate and `recurring` at one $(J,t')$. That is deliberate — X20 is about tuning and
the adapt step, not about drift shape — but it means X20 cannot say anything about
how the diffusion filter's advantage varies *within* a schedule. P5.5, P5.6 and
P5.10 are what close that, and they should be read as completing this table rather
than as separate curiosities.

## Questions only this method can be asked

Nothing in X0–X18 could pose these, because no earlier method held a belief per
agent. They are the reason phase 5 is not just "the same experiments again".

- [ ] **P5.11** **Does the belief stay calibrated?** The filter reports
  predictive uncertainty and no SGD baseline can. `logit_covariance` exists and
  has never been scored against outcomes. A tracking filter whose error bars are
  wrong is a worse scientific object than a point estimator that never claimed
  any — and the delta-method approximation is known to be systematically
  over-confident, so the question is how badly, not whether.
- [ ] **P5.12** **$E_{\text{agree}}$ and $E_{\text{cent}}$ over time.** The
  disagreement metrics exist and have been identically zero for every pooled
  method and meaningless for `local_only`. This is the first method for which
  they measure something: whether agents' beliefs converge, and whether they
  converge to the centralised one or merely to each other.
- [ ] **P5.13** **The communication-matched claim (M2).** Accuracy against
  *scalars sent*, not against $t$. `diffusion_ekf` sends $p$ per link against
  ATC-with-momentum's $2p$, so the filter should be compared at half the
  bandwidth — while `diffusion_ekf_full` sends 2908×, which is the honest cost of
  the ceiling and belongs on the same axis. This is the claim the ledger was
  built for and the one a reviewer will press hardest.
- [ ] **P5.14** **Is the conservative bound actually conservative?** `lem:
  conservative` proves $\sum_u a_{vu}\bm P^{\psi}_u$ upper-bounds the combined
  estimate's error covariance for any cross-correlation, with equality when the
  errors coincide. Measurable: compare the reported variance against the realised
  squared error across seeds. If the bound is wildly loose the "conservative"
  framing is doing less work than it appears to; if it is tight, the neighbours'
  errors really are nearly perfectly correlated, which is itself a finding about
  how much independent information diffusion is actually moving.
