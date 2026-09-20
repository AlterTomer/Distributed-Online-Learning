# Schedule to ICML 2027: three tracks

Drafted 2026-09-15, after the supervisor reviewed the Diff-EKF results. His
verdict: the results are good, the Mackey–Glass regression task is approved, and
**communication is the main weakness a reviewer will press on**.

| | track | goal |
|---|---|---|
| **A** | Finish MNIST | Close the experiments the paper needs; decide the rest explicitly |
| **B** | Mackey–Glass regression | The same battery of tests on a second task, likelihood and architecture (P5.25) |
| **C** | Reduce communication | Error against **bits** sent, with every compressor also given to the baselines |

**Order: A, then B.** Decided 2026-09-15: MNIST is finished before the new task
starts, so B's pilot and build wait for A. Where C sits is open (see the calendar).

**Fixed points.** The ICML deadline is **22 January 2027**. Writing starts about
mid-October **from the theory, not the results**, so experiments continue in
parallel with writing; the proposed experiment freeze is **27 December 2026**,
leaving four weeks to write the results sections. One GPU (RTX 4070 laptop, 8 GB):
sweeps run one at a time.

**All four diffusion variants stay in every comparison**, full covariance sharing
included — decided 2026-09-15; it is an important variant, not only a diagnostic.
The cost is memory: full sharing needs its own process per cell (X25's group B),
and at $N=50$ it would hold two sets of fifty 64.5 MiB covariances during the
combine, 6.4 GiB, which does not fit beside everything else on 8 GB. Hence
$N\in\{20,30\}$ below.

---

## Track A — finish MNIST

**Tier 1 — the paper needs these**

| item | what | GPU | status |
|---|---|---|---|
| X27 | Sender vs receiver linearisation point, 5 seeds; picks the headline one-hop variant | ~12 h | ⏳ running |
| X25+ | X25's still skew cells, full-sharing cells at $\beta=1$, and `atc_plain`, topped up to 5 seeds. The equal-bandwidth loss at $\beta_{\mathrm{dir}}=0.1$ is $t=1.66$ on 3 | ~7 h | queued |
| P5.3 | Topology and spectral gap (X3 analogue): ring and a sparser ER beside the existing ER 0.3 and complete graphs, baselines re-tuned per topology. One-hop's value should scale with degree | ~14 h | queued |
| N>10 | $N\in\{20,30\}$, stationary and abrupt, all variants | ~16 h | queued |
| P5.7 | Heterogeneous drift (X8 analogue) — one of the two rows where per-agent beliefs could **beat** the centralised filter | ~8 h | queued |
| P5.11, P5.14 | Calibration, and whether `lem:conservative` is conservative in practice: new evaluation metrics, then a re-run of X20's ER cells | 1–2 days code, ~6 h | open |
| P5.12 | $E_{\text{agree}}$, $E_{\text{cent}}$ over time — already recorded, analysis only | — | open |
| P5.24 | Mis-tuned arm: the number is in hand (+0.0161, $t=6.3$); carry it in the figures | — | open |
| figures | Diff-EKF folder builder with a computed `SUMMARY.md` | — | open |

About 63 GPU-hours. At one sweep launched a day, with scripts, pre-flights and
write-ups done while the previous sweep runs: **about ten days, to roughly 27
September.**

**Tier 2 — cheap and informative** (~12 GPU-hours, about three days):
P5.4 sparse labels $n\times\pi_{\text{lab}}$ (X4 analogue — an unlabelled agent
still combines, untested for a filter); P5.5 break rate (X9/X16 analogue — the
only place the filter's advantage is a *rate*: the centralised filter survives to
0.064°/step against 0.038–0.044 for gradient methods).

**Tier 3 — decided 2026-09-15:**

| item | analogue of | what it asks | GPU | decision |
|---|---|---|---|---|
| P5.8 | X10 | Prior drift / label shift instead of covariate shift — the other row where per-agent beliefs could win | ~8 h | **kept** |
| P5.23 | — | One `piecewise` cell placing a shift mid-transient, as an adversarial probe | ~3 h | **kept** |
| P5.6 | X11 | Repeated abrupt shifts on a $J\times t'$ grid. X11 found longer intervals leave *bigger* wounds (0.209 at $J=30$, $t'=200$) | ~3 days | deferred |
| P5.9 | X14 | The generalisation grid: 21 conditions ($t'\in\{2,\dots,200\}\times J\in\{5,15,20,30\}$, three linear rates, a low-sample block). X14 found the centralised filter less damaged than ATC in 21 of 21 | ~4 days | deferred |
| P5.10 | X7, X12, X18 | The remaining shapes: sinusoidal (the only schedule that revisits states, so the only one where forgetting is well-posed), linear at several rates, sawtooth | ~10 h | deferred |

Deferred means "if the buffer allows", not cut. The reasons: P5.6 is nearly a subset
of P5.9 — X11's twelve cells sit almost entirely inside X14's jump grid — and the
benchmark already holds a great deal of rotation evidence; X18 established that
drift *shape* has never changed the ordering between methods, which covers P5.10.

**Track A in total:** tier 1 (~63 GPU-hours), tier 2 (~12), P5.8 and P5.23 (~11) —
about 86 GPU-hours, **finishing about 3 October**, or about 7 October if a sweep
has to be re-run.

## Track B — Mackey–Glass regression (P5.25)

Configuration as settled in the design doc (v3): one realisation per agent of the
same law, blockwise causal many-to-many on non-overlapping blocks, $L=32$,
$n_b\in\{1,\dots,4\}$, one causal Transformer block with $d_{\text{model}}=16$,
$p=2273$, Gaussian observations at a chosen $\sigma$, drift in $\beta_t$.

| step | what | MNIST analogue | cost |
|---|---|---|---|
| B0 | **Pilot, gating the build**: headroom above the noise floor, and within-block residual correlation | — | ½–1 day, CPU |
| B1 | Data layer: DDE generator per (schedule, seed), $\beta_t$ drift, stationary twins as separate integrations | env + drift | 3–4 days |
| B2 | Model: causal Transformer with per-sample Jacobian; `DatasetSpec` generalised beyond images | `mlp.py` | 3–4 days |
| B3 | Metrics: prequential RMSE against the noise floor, predictive NLL and interval coverage | error, ECE | 1–2 days |
| B4 | Gates: ATC = centralised SGD on $K_N$; a linear head makes the EKF an exact Kalman filter, so `prop:complete_graph` holds with no linearisation error | X0, P5.0 | 1 day |
| B5 | Tuning: SGD rates; centralised EKF $(\gamma,q,\sigma_0^2)$; the diffusion filter tuned jointly | X13, X20, X23, X26 | ~1 day GPU |
| B6 | Main comparison: stationary, smooth and abrupt $\beta$ drift with twins; all four variants against ATC (plain and momentum), local-only, centralised SGD and EKF | X19, X20, X26 | ~1 day GPU |
| B7 | The bracket's prediction, $\beta_c\in\{1,2\}$: if $\beta_c>1$ wins here and lost on MNIST, that is a mechanistic result | X24 | ~6 h |
| B8 | Heterogeneity, the other end of the bracket: per-agent deviations in $\tau$ or sensor gain | X25 | ~8 h |
| B9 | Calibration — scoreable directly under a Gaussian | P5.11, P5.14 | analysis |
| B10 | Whichever Track-A tier-1 rows carry over: topology, $N$, heterogeneous drift | P5.3, P5.7 | ~2 days GPU |

Carried over without re-running: $\alpha$ (X22), rejected monotonically on both
adapt scopes. **About four to five weeks**: one and a half to two of build, the
rest runs. B0 can extend it if the task proves too easy.

## Track C — reduce communication

### Where the communication actually stands

| variant | scalars / link / step | bytes (float64 $\boldsymbol\psi$, 8-bit pixels) | vs one $\boldsymbol\psi$ |
|---|---|---|---|
| local, mean-only | 2 908 | 23 264 | 1.00 |
| **one-hop, mean-only, receiver point** | 3 696 | **24 052** | **1.03** |
| momentum ATC | 5 816 | 46 528 | 2.00 |
| local, full | 4 232 594 | 33.9 MB | 1 455 |

A raw MNIST batch is 788 *bytes*, not 788 doubles; on Mackey–Glass a block is 32
samples against $p=2273$. So evidence exchange is nearly free, and the expensive
variant is full sharing. Every diffusion method, filter or SGD, still sends $O(p)$
per link per step, and a reviewer will ask why the filter is not compared with
*compressed* decentralised methods. Track C answers that.

### Options, ranked

1. **Count bits, and send $\boldsymbol\psi$ at lower precision.** The filter runs in
   float64 (D61), but what crosses the link need not; rounding enters like extra
   measurement noise.
2. **Differences with memory.** Every agent keeps a public copy
   $\hat{\boldsymbol\psi}_u$ of each neighbour's estimate and of its own, sends
   $Q(\boldsymbol\psi_v-\hat{\boldsymbol\psi}_v)$, and everyone adds it to $\hat{\boldsymbol\psi}_v$, so
   compression error is corrected rather than accumulated — the CHOCO-gossip
   construction (Koloskova, Stich and Jaggi, ICML 2019); Sayed's group has an
   adapt–compress–then–combine diffusion variant, to be read first. A plain
   difference without the public copy drifts.
3. **Sparsification with error feedback**, top-$k$ or random-$k$. The
   filter-specific form ranks coordinates by $|\delta_i|/\sqrt{P_{ii}}$, the
   change in units of the filter's own uncertainty — a scale SGD does not have.
4. **Event-triggered communication**: transmit when
   $(\boldsymbol\psi_v-\hat{\boldsymbol\psi}_v)^{\mathsf T}\boldsymbol P^{-1}(\boldsymbol\psi_v-\hat{\boldsymbol\psi}_v)>\eta$,
   against a periodic-$K$ baseline. Claim (M4)'s "communication scheduling" made
   concrete.
5. **Compressing full sharing**, now that it stays in the comparison. The adapt
   step changes $\boldsymbol P$ by rank at most $nq=40$ and the predict step is
   deterministic, so without the combine a 40-column factor — $116\,320$ scalars
   against $4\,229\,686$, 36× less — would reconstruct it exactly. The combine
   mixes in the sender's *own* neighbours' covariances, which the receiver does
   not track, so the step-to-step change is full rank and the reconstruction is
   inexact. Candidates: a public copy of each neighbour's $\boldsymbol P$ with a low-rank
   (or low-rank-plus-diagonal) difference and error feedback — exact at the
   adapt step, approximate at the combine, and costing
   $\lvert\mathcal N_v\rvert$ extra dense covariances per agent; or sending
   $\boldsymbol P$ every $K$ steps only. Worth a design note before building.
6. **Shrink $p$ itself** by filtering fewer parameters (the note's scaling
   section).

Every compressor is also applied to ATC, and the headline plot is error against
cumulative bits.

| step | what | cost |
|---|---|---|
| C0 | Design note: CHOCO, ACTC; compressor interface; bits ledger; the full-sharing option | 2 days |
| C1 | Bits ledger; $\boldsymbol\psi$ in float32/float16 | 1 day |
| C2 | Public copies and difference transmission with a pluggable $Q$, error feedback; also for ATC | 4–5 days |
| C3 | Event trigger and periodic-$K$ baseline | 2 days |
| C4 | Error against bits on MNIST (IID and severe skew), then on Mackey–Glass | ~3 days GPU |

About **two and a half to three weeks.**

---

## Calendar

| dates | work |
|---|---|
| **Sep 15–27** | A, tier 1 |
| **Sep 28–Oct 3** | A, tier 2, then P5.8 and P5.23 — **MNIST done** |
| **Oct 4–Nov 7** | B: pilot, build, battery |
| mid-Oct | **writing starts, from the theory** — experiments continue |
| **Nov 8–Nov 28** | C |
| **Nov 29–Dec 27** | buffer: seed top-ups, gaps the draft exposes, supervisor's requests, then deferred tier-3 items |
| **Dec 27** | experiment freeze |
| **Jan 22, 2027** | ICML deadline |

**C goes last** — decided 2026-09-15, to push on the regression task first. The
alternative considered was C's MNIST half right after A, which would have moved B
to about Oct 18–Nov 21.

**Risks, in order.** (1) B0 says the task is too easy: the fixes are ordered in
the design doc and cost days. (2) The queue: tier 1 alone is ~63 GPU-hours, so
estimates assume a sweep is launched as soon as the previous one ends. (3) X27
favours the receiver point, which changes the one-hop variant every later sweep
carries; that is why it runs first.
