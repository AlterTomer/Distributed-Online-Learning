# Track C — reducing communication: the design (C0)

**Status: draft, 2026-09-27; C1 built the same day ([[D135]]).** Written ahead of Track C's calendar slot (Nov 8–28,
`schedule.md`) so that the building work is not also a design exercise. Every
decision still open is marked ❓ and collected in §9; nothing here is built yet.

The supervisor's verdict on 2026-09-15 was that **communication is the main
weakness a reviewer will press on**. Every diffusion method here, filter or SGD,
sends $O(p)$ per link per step, and the obvious objection is that the filter is
never compared with *compressed* decentralised methods. Track C's headline is
**error against cumulative bits**, with every compressor also given to the
baselines.

---

## 1. The claim Track C can make, and the one it cannot

**Can:** at a matched bit budget, where does the filter stand against the gradient
baselines, and does its advantage survive, shrink or grow as the budget falls?
Either answer is publishable. A filter whose lead holds down to a few bits per
parameter is a strong result. One that loses its lead first is still worth
reporting, because it says where the belief costs more to ship than it is worth.

**Cannot:** "the filter is communication-efficient" in absolute terms. Centralised
SGD's notional cost of shipping raw samples to a centre is *less* traffic than
ring diffusion at the defaults (`metrics/communication.py`, the ledger's own
docstring). Track C is a comparison between decentralised methods at matched bits,
not a defence of the paradigm.

## 2. What crosses a link today

Per link per step (`metrics/communication.py`). Bytes assume float64 parameters and
the task's native data format:

| learner | message | MNIST ($p=2908$) | Mackey–Glass ($p=2273$) |
|---|---|---|---|
| `diffusion_ekf` (local adapt, mean-only) | $\boldsymbol\psi$ | 23 264 B | 18 184 B |
| `…onehop_mean_receiver` | $\boldsymbol\psi$ + the raw batch | 24 052 B | $\boldsymbol\psi$ + one block |
| `diffusion_sgd_atc_plain` | $\boldsymbol\psi$ | 23 264 B | 18 184 B |
| `diffusion_sgd_atc` (momentum mixed) | $\boldsymbol\psi$, momentum | 46 528 B | 36 368 B |
| `diffusion_atc_adamw` (both moments mixed) | $\boldsymbol\psi$, $\boldsymbol m$, $\boldsymbol v$ | 69 792 B | 54 552 B |
| `diffusion_ekf_full` | $\boldsymbol\psi$, packed $\boldsymbol P$ | 33.9 MB | 20.7 MB |

Two kinds of payload, and they should be treated differently:

* **Parameter-like vectors** ($\boldsymbol\psi$, the optimiser moments, $\boldsymbol P$). Large,
  and each is averaged into the receiver's state by the combine, which
  tolerates, and to some degree averages out, per-message error. These are what
  Track C compresses.
* **Data** — one-hop's raw batch: 788 B on MNIST (8-bit pixels) and one 32-sample
  block on Mackey–Glass. Already a few percent of one $\boldsymbol\psi$. ❓ **C-1:**
  leave data uncompressed (proposed), or quantise it too? Compressing it saves at
  most ~3% and would change what the adapt step sees, which is a different
  experiment.

**Where the hook goes.** `Intermediate` (`learners/base.py`) is already the
message: frozen, produced by `adapt`, consumed by `combine`, with `payload_vectors`
feeding the ledger. A compressor therefore sits in `simulate.py` as a **channel**
between the two calls — `intermediates → channel.transmit → combine` — and no
learner's algebra changes. Compression is **sender-side broadcast**: one encoded
message per agent per step, the same bits to every neighbour, which is what
CHOCO's public copies require (§4.2) and what the ledger's per-link count assumes.

## 3. The bits ledger (C1)

`CommunicationCost` counts scalars. It becomes a count of **bits per message**:

$$\text{bits} = \sum_{\text{vectors}} \bigl(\text{values sent}\times b + \text{index bits} + \text{side information}\bigr) + \text{header},$$

where $b$ is the value precision, index bits are $\lceil\log_2 p\rceil$ per value for a
top-$k$ message (or a seed, for random-$k$), and side information is any scale or
norm a quantiser ships (e.g. one float32 per vector for scaled quantisation). An
event-triggered agent that stays silent costs a one-bit flag, or nothing if silence
is implicit (❓ C-2, proposed: count the flag).

**The reference precision matters for how the headline reads.** The filter
*computes* in float64, which the covariance recursion needs (see §8), but nothing
requires the link to carry float64. A reviewer will call float64 transmission a
strawman, so the uncompressed reference should be **float32** ($32p$ bits), with
float64 shown only as the as-run figure.
❓ **C-3:** float32 as the reference (proposed), or float64 as run?

## 4. The compressors (C1–C3), ranked

Each is applied **identically to every diffusing learner** — filter, plain ATC,
momentum ATC, ATC AdamW — to every parameter-like vector each one sends.

### 4.1 Lower precision, no memory (C1)

Send $\boldsymbol\psi$ at $b\in\{32,16,8,4\}$ bits: float32, float16/bfloat16, then scaled
uniform quantisation with **stochastic rounding** (unbiased) below 16.

For the filter this has a specific consequence. The combine averages a neighbour's
quantised $\boldsymbol\psi$, so quantisation error enters like extra measurement noise
that $\boldsymbol P$ does not know about. At 16 bits it should be invisible. At 4 bits it
is a candidate source of over-confidence, and it has a filter-only remedy: add the
quantiser's known variance as process noise after the combine (for scaled uniform
rounding with step $\Delta$ it is at most $\Delta^2/4$ per coordinate). A gradient
method has no belief to put it in. ❓ **C-4:** build that compensated variant as a
separate arm? Proposed yes: it is the one place a compressor could favour the filter
*by design*, and it should be measured rather than claimed.

### 4.2 Differences against public copies (C2): the CHOCO construction

Every agent keeps a public copy $\hat{\boldsymbol x}_u$ of each neighbour's vector and of its
own. It sends $\boldsymbol q_v = Q(\boldsymbol x_v - \hat{\boldsymbol x}_v)$, and everyone adds
$\boldsymbol q_v$ to $\hat{\boldsymbol x}_v$. Compression error is carried in the copy and corrected
on the next step rather than accumulated. A plain difference without the public copy
drifts, which is why the copy is the construction and not an option (CHOCO-gossip [1];
CHOCO-SGD for deep networks [2]).

**Memory:** $(\lvert\mathcal N_v\rvert+1)$ copies of each sent vector per agent, about
0.1 MB for $\boldsymbol\psi$ on ER 0.3. Negligible beside $\boldsymbol P$.

**The fork for the filter is how the combine reads the copies.** CHOCO replaces the
combine by a gossip step $\boldsymbol x_v \leftarrow \boldsymbol x_v + \gamma\sum_u a_{vu}(\hat{\boldsymbol x}_u - \hat{\boldsymbol x}_v)$
with a consensus step size $\gamma<1$ that it needs for stability. That changes the
effective combination matrix from $\boldsymbol A$ to $(1-\gamma)\boldsymbol I + \gamma\boldsymbol A$. It is still
doubly stochastic, but it mixes more slowly, and for the filter it detaches the mean
combine from the covariance weights that the conservative bound (`lem:conservative`)
assumes.
❓ **C-5:** (a) CHOCO's γ-step for every learner, one more knob tuned per learner (D77);
(b) the exact combine evaluated on the copies, $\boldsymbol\psi_v \leftarrow \sum_u a_{vu}\hat{\boldsymbol x}_u$, which keeps
the filter's algebra but has no convergence guarantee under aggressive $Q$; or (c)
both, with (b) as the filter-natural arm. Proposed (c). Under (b), does an agent combine its own exact $\boldsymbol\psi_v$ (as C1's channel does) or its own copy $\hat{\boldsymbol x}_v$ (as CHOCO does)? The exact own term keeps information the copy has not caught up with; proposed exact, with the copy variant as an ablation. Read the adapt–compress–
then–combine (ACTC) diffusion strategy from Sayed's group first [3, 4], since it may
already settle this for diffusion specifically.

### 4.3 Sparsification and entropy coding on the public copy (C2)

⚠ **Corrected 2026-09-27 (D136).** This section used to add "an error-feedback memory"
on top of §4.2's public copies. With a public copy that is double counting. The copy
advances only by what was sent, so $\boldsymbol x_v-\hat{\boldsymbol x}_v$ *already* holds
every untransmitted change, and a separate residual adds the same error a second
time. Take $\boldsymbol x$ constant at $\boldsymbol c$, $\hat{\boldsymbol x}_0=\boldsymbol 0$, and a
quantiser that drops everything for a while: the residual grows $\boldsymbol c, 2\boldsymbol c,
3\boldsymbol c,\dots$ until it clears the threshold, and the copy then jumps *past*
$\boldsymbol c$. The rule is $\boldsymbol e_v=\boldsymbol x_v-\hat{\boldsymbol x}_v$, with the copy as the only
memory. A separate residual belongs to schemes *without* a shared reference, which is
where [5, 6] use it.

Two ways to make $\boldsymbol e_v$ cheap, both sender-side and both applied to every learner:

* **Top-$k$ / random-$k$**: send $k$ coordinates of $\boldsymbol e_v$, with indices, or a seed
  for random-$k$. A fixed rate.
* **Dead zone + entropy coding** (the codec of `Diff_EKF_Huffman_Communication_Summary`,
  corrected): $q_i=\operatorname{round}(e_i/\Delta)$, so every change smaller than
  $\Delta/2$ is a zero; then run-length code the zeros and Huffman-code the run lengths
  and the nonzero amplitudes, with canonical tables fixed in advance. The coding is
  lossless, so it costs nothing in accuracy. A variable rate: quiet periods cost little,
  and the rate rises after drift without a drift detector. The bits column records the
  **actual encoded length**, including run lengths, scales and headers, so `Channel`
  returns a per-message length rather than a fixed count. Per-block scales, one per
  layer, are preferable to one global scale, because layers differ in range.

**The filter has a scale SGD lacks:** its own uncertainty. It can rank coordinates by
$\lvert e_i\rvert/\sqrt{P_{ii}}$ for top-$k$, or, for the dead zone, keep the plain rule's
*average* threshold $\Delta/2$ and redistribute it by uncertainty:
$\lvert e_i\rvert<\tfrac{\Delta}{2}\sqrt{P_{ii}}/\overline{\sqrt{P}}$, tighter where the filter
is sure and looser where it is not. (A raw $\kappa\sqrt{P_{ii}}$ is no comparison: with
$\sqrt{P_{ii}}\approx0.1$, far above a step's change, it sends almost nothing and lets the
copy lag by hundreds of steps, a rate bought with distortion; D136.) Either way the
decision is the sender's, and amplitudes still travel in absolute units, so a receiver never needs
$\boldsymbol P$, which a mean-only filter does not send. ❓ **C-6:** what does ATC rank by?
Plain magnitude for SGD, and $\lvert e_i\rvert/\sqrt{v_i}$ for ATC AdamW, its own
second-moment scale. **Both rules for the filter**, so an uncertainty-scaled gain cannot
be credited to sparsity alone.

**Why this could favour the filter by design, and must be measured first.** The
filter's gain shrinks as $\boldsymbol P$ contracts, so in a quiet period its successive means
barely move and $\boldsymbol e$ is mostly zeros. ATC's constant step keeps moving it by
about $\eta\lVert\boldsymbol g\rVert$ for ever, and AdamW normalises its steps and never
quiets. The *adaptive rate after drift* is **not** filter-specific: ATC's gradients grow
after a shift too. `scripts/run_delta_entropy_probe.py` (D136) measures, open-loop on
real messages, the entropy each learner's quantised differences would need at matched
distortion, under both dead-zone rules, and the rate around each jump. **C2's codec is
built only if the probe shows the filter's differences are materially more compressible
than the baselines', or that the rate genuinely rises after a jump and then decays.**

**Codebook discipline.** A fixed Huffman table built on the runs it is then scored on
understates the rate. ✅ **C-10, decided 2026-10-01 (D137):** tables are built on seeds
100–104 and the step chosen on 200–204, both at the full horizon; seeds 0–4 are the
report, paired with the existing cells. Seeds 5–9 are not clean (top-ups used them). The
entropy bound and the cross-entropy gap are reported beside the actual coded length. The
whole training protocol, one global step multiplier and the frontier first, is D137.

### 4.4 Event-triggered communication (C3)

Transmit only when the public copy has fallen too far behind:
$(\boldsymbol\psi_v-\hat{\boldsymbol\psi}_v)^{\mathsf T}\boldsymbol P_v^{-1}(\boldsymbol\psi_v-\hat{\boldsymbol\psi}_v)>\eta$. That is claim (M4)'s
"communication scheduling" made concrete. $\boldsymbol P^{-1}$ in full is $O(p^3)$, so the
trigger uses the diagonal, $\sum_i\delta_i^2/P_{ii}$, which is $O(p)$ (❓ C-7: diagonal
proposed; a Woodbury form of the full metric is possible but costly). ATC's trigger is
$\lVert\boldsymbol\psi_v-\hat{\boldsymbol\psi}_v\rVert^2>\eta$. Both are compared against a **periodic-$K$** baseline
at matched average bits, since a trigger that merely sends less often has to beat
sending less often on a clock.

### 4.5 Compressing full sharing — proposed out of scope

Full sharing's covariance is 1 455× a mean on MNIST. The schedule sketched a
low-rank difference against a public copy of each neighbour's $\boldsymbol P$: exact through
the adapt step (rank ≤ $nq$ = 40), approximate through the combine, which mixes in
covariances the receiver does not track. But the evidence argues against building it:

* On IID data full sharing buys nothing, on both tasks and both adapt scopes (D79,
  D100, D106).
* Under severe skew it helps local adapt, but one-hop matches that help at 1.03× a
  mean's bandwidth, and the two are substitutes (D89).

A compressed full-sharing arm would therefore be competing with one-hop, which is
already cheap. ❓ **C-8:** proposed: report full sharing at its uncompressed cost,
state why it is not compressed, and leave the low-rank scheme as future work.

### 4.6 Shrinking $p$ — out of scope

Filtering a subset of the parameters, such as the last layer, changes the *method*
rather than the channel. It belongs with the scaling section of the note, not here.

## 5. Fairness rules

1. **Same compressor, same budget, every diffusing learner**, applied to every
   parameter-like vector it sends. ATC AdamW compresses three vectors, so at a given
   compressor it spends 3× the filter's bits. The error-against-bits axis charges
   that honestly instead of hiding it.
2. **Rates re-tuned under compression, or not.** Compression adds noise to the
   combine, and a baseline carrying its uncompressed rate may be mis-tuned: the D77
   trap. ❓ **C-9:** re-tune the gradient baselines at each compression level
   (correct, costs a `--lr` pass per level), or re-tune at a few levels and
   interpolate. Proposed: re-tune at every level of the headline grid, and carry
   rates only within a family's own sweep. The filters carry their selections, as
   everywhere (the X14 discipline). The compensated arm of §4.1 is the filter's only
   compression-aware setting.
3. **The pairing survives.** Compression draws from its own seed stream (random-$k$,
   stochastic rounding), so an uncompressed cell and its compressed twin share every
   data and graph draw. The existing `assert_paired_runs` gate applies unchanged if
   the channel lives in config, not in the learner.

## 6. What is reported

* **Error against cumulative bits**, per learner, one curve per compressor level
  (settled error at each level, then the whole trajectory).
* **Bits to reach a target error**, the tabular reading. The target is set relative
  to each task's centralised filter, so it transfers between tasks.
* **Calibration under compression** for the filters: `variance_ratio` and coverage on
  Mackey–Glass, ECE and κ\* on MNIST (D120). Quantisation that turns the filter
  over-confident is a finding, and §4.1's compensated arm is its test.
* The ledger itself: bits per link per step for every arm, printed, never only
  plotted.

## 7. The experiments (C4)

| cell | tasks | compressors | seeds |
|---|---|---|---|
| **C4a** precision sweep | MNIST IID, MNIST skew 0.1, MG stationary | float32, float16, 8-bit, 4-bit (stochastic rounding) | 5 |
| **C4b** public copies + sparsity | the same three | top-$k$ at $k/p\in\{10\%,1\%\}$ × 8-bit values, and the dead-zone codec at three $\Delta$ (plain and, for filters, $\sqrt{P_{ii}}$-scaled); CHOCO γ-step and exact-combine arms. **Gated on the delta probe** (§4.3) | 5 |
| **C4c** event trigger | MNIST IID, MG stationary | three thresholds η, periodic-$K$ at matched bits | 5 |

Skew 0.1 is in because that is where one-hop's advantage lives (D89). A compressor
that erased one-hop's lead there would matter more than one that erased it on IID
data. The learners are one-hop (receiver), local adapt, the centralised filter as the
reference line, ATC plain, momentum and AdamW, and `local_only`. Named questions go
into a D-note before any C4 run (D118), not here.

**Cost, order of magnitude only:** a cell costs what its uncompressed twin does
(encoding is $O(p)$ against the filter's $O(p^2q')$), so C4 is roughly
(levels × tasks) main-comparison cells. That is about 18 for C4a and 12 for C4b,
plus a `--lr` pass per level (❓ C-9), about 40–60 GPU-h in all. To be priced properly
once the grid is agreed.

## 8. Working precision is not link precision

The filter *computes* in float64 for its covariance recursion (D58, `filter.md` §5):
$\boldsymbol P-\boldsymbol A\boldsymbol S^{-1}\boldsymbol A^{\mathsf T}$ subtracts two nearly equal matrices. The Joseph
form, which is what usually protects positive definiteness, is not used (D62), and
the source paper reports positive definiteness lost within a few hundred
single-precision steps against runs of 1 500. None of that constrains the link:
§4.1 is about what is *sent*. A float32 or mixed-precision *working* precision is
a separate question about compute, not communication. It has never been measured
on this code (see the note to the user, 2026-09-27) and does not belong in Track C.

## 9. Decisions open (for the user)

| # | question | proposed |
|---|---|---|
| C-1 | Compress one-hop's raw data too? | No: ≤3% of a message, and it changes what the adapt step sees |
| C-2 | Does a silent event-triggered agent cost a flag bit? | Yes, one bit |
| C-3 | Reference precision for the headline | float32; float64 shown as run |
| C-4 | A quantisation-compensated filter arm (process noise = quantiser variance)? | Yes, as a separate arm |
| C-5 | How the combine reads public copies | ✅ **Decided 2026-10-03 (D138):** the exact combine on the copies, own term exact; no CHOCO arm unless C4 shows instability |
| C-6 | What ATC ranks top-$k$ / sets its dead zone by | Magnitude for SGD, $\lvert\delta\rvert/\sqrt v$ for AdamW; both rankings for the filter |
| C-7 | The trigger's metric | Diagonal $\sum\delta_i^2/P_{ii}$ |
| C-8 | Compress full sharing? | ✅ **Decided 2026-10-03 (D138):** out of scope for now, reported uncompressed; added to the end of the schedule if time allows |
| C-9 | Re-tune baselines per compression level? | ✅ **Decided 2026-10-03 (D138):** yes, at every c, on calibration seeds 100–104 |
| C-10 | What the Huffman tables are built on | ✅ **Decided 2026-10-01 (D137):** calibrate on seeds 100–104, validate on 200–204, report on 0–4, all at the full horizon; entropy bound and cross-entropy gap beside the coded length |
| C-11 | Build C2's codec at all? | **Probed 2026-09-29 (D136): neither condition holds on either task**, so not as a filter-specific contribution; the filter's per-message advantage is structural (one vector against two or three). The user's offline-trained codec, reviewed and revised 2026-10-01 (D137), is built instead as a codec for every learner: C4b's dead-zone arm |

**Before C1 starts:** read [3] and [4] and check whether ACTC's analysis covers a
combine with a non-scalar (covariance) weight, which would bear on C-5; then [1], [2]
for the public-copy construction and [5], [6] for error feedback.

## 10. Reading list (verified 2026-09-27)

Compression in decentralised learning:

1. A. Koloskova, S. U. Stich, M. Jaggi, "Decentralized Stochastic Optimization and Gossip
   Algorithms with Compressed Communication," *ICML* 2019, PMLR 97:3478–3487.
   CHOCO-gossip and CHOCO-SGD: public copies, arbitrary compressors.
2. A. Koloskova, T. Lin, S. U. Stich, M. Jaggi, "Decentralized Deep Learning with
   Arbitrary Communication Compression," *ICLR* 2020 (arXiv:1907.09356). CHOCO-SGD on
   non-convex deep nets and non-IID data; code at `github.com/epfml/ChocoSGD`.
3. M. Carpentiero, V. Matta, A. H. Sayed, "Distributed Adaptive Learning Under
   Communication Constraints," *IEEE Open Journal of Signal Processing*, vol. 5,
   pp. 321–358, 2024 (arXiv:2112.02129). ACTC, adapt–compress–then–combine: the
   diffusion-native construction, streaming data.
4. M. Carpentiero, V. Matta, A. H. Sayed, "Compressed Regression over Adaptive
   Networks," *IEEE Trans. Signal and Information Processing over Networks*, vol. 10,
   pp. 851–867, 2024 (arXiv:2304.03638). ACTC's mean-square error split into the
   uncompressed term plus a compression term: the closest analysis to a filter.
5. S. U. Stich, J.-B. Cordonnier, M. Jaggi, "Sparsified SGD with Memory," *NeurIPS*
   2018 (arXiv:1809.07599). Top-$k$ with error compensation.
6. S. P. Karimireddy, Q. Rebjock, S. U. Stich, M. Jaggi, "Error Feedback Fixes SignSGD
   and other Gradient Compression Schemes," *ICML* 2019, PMLR 97:3252–3261.

The filter's numerics (for §8 and the float32 probe, D133):

7. G. V. Puskorius, L. A. Feldkamp, "Parameter-Based Kalman Filter Training: Theory and
   Implementation," ch. 2 of S. Haykin (ed.), *Kalman Filtering and Neural Networks*,
   Wiley, 2001. EKF training of neural networks in practice, including numerical care.
8. G. J. Bierman, *Factorization Methods for Discrete Sequential Estimation*, Academic
   Press, 1977 (Dover reprint). The standard reference on covariance-form instability
   and square-root filters; the place to look for what precision the recursion needs.

Already in the Diff-EKF note's bibliography and underlying the combine: Cattivelli and
Sayed, diffusion LMS (*IEEE TSP* 58(3), 2010) and diffusion Kalman filtering (*IEEE TAC*
55(9), 2010); Singhal and Wu (NeurIPS 1988) and Puskorius and Feldkamp (*IEEE TNN* 5(2),
1994) for EKF training.
