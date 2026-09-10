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
  information rate.** The mechanism in D79 names $q$ as the mis-scaled parameter
  and gives the direction: it was chosen to balance an influx of $N\bm\Delta$ per
  step and now faces $\bm\Delta$, so it should fall by roughly $N$. The grid spans
  wider than that argument, because a scaling argument that predicts the answer is
  the worst reason to only look where it points.
- [ ] **P5.1b** Add `diffusion_sgd_atc_plain` and re-read at **matched
  bandwidth**. X19 compared a filter sending $p$ against an ATC sending $2p$,
  which is precisely the pairing D29 exists to prevent. In X18's stationary twin
  `atc_plain` was 0.0863 against `atc`'s 0.0790, so the correction may turn a tie
  into a win.
- [ ] **P5.1c** **Promote one-hop from fixture to method** and measure it on ER.
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
