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
| X27 | Sender vs receiver linearisation point, 5 seeds; picks the headline one-hop variant | 9 h | ✅ done — receiver adopted ([[D99]]) |
| X28 | Does full sharing still buy nothing once one-hop linearises at the **receiver** point? Two cells at the ends of the skew axis, paired against X27's mean-only cells | 2 h | ✅ done — it does not, at either end ([[D100]]) |
| X25+ | X25's still skew cells, full-sharing cells at $\beta=1$, and `atc_plain`, topped up to 5 seeds. The equal-bandwidth loss at $\beta_{\mathrm{dir}}=0.1$ is $t=1.66$ on 3 | 3.9 h | ✅ done — the loss firms to $t=2.43$ on five seeds, and `atc_plain` under skew is no longer unmeasured ([[D101]]) |
| P5.3 | Topology and spectral gap (X3 analogue): path, ring, the existing ER 0.3 and complete, baselines re-tuned per topology. (A sparser ER was the original plan; at $N=10$ it sits below the $\ln(n)/n=0.230$ connectivity threshold, so `path` supplies the sparse end instead.) One-hop's value should scale with degree | ~14 h | ✅ done — and **both hypotheses are refuted**: the covariance-sharing gap does *not* widen as connectivity falls (flat across the axis, nominally largest on the complete graph), and one-hop's value is non-monotone in degree, peaking at intermediate connectivity. D100's null holds at every topology ([[D103]]) |
| N>10 | $N\in\{10,20,30\}$ at a common $T=500$, $n=4$, ER $p$ matched to the $N=10$ mixing gap and each seed's draw conditioned into $0.119\pm0.02$ (≤3 draws); stationary and abrupt; every diffusion filter in its own process. Full sharing at $N=10$ by default, at $N=20,30$ only with `--full-sharing` ([[D114]]). `scripts/run_network_size.py` | ~16 h (est.) | memory smoke passed 2026-09-26, every default cell fits; `--lr` next |
| P5.7 | Heterogeneous drift (X8 analogue) — one of the two rows where per-agent beliefs could **beat** the centralised filter | ~8 h | ✅ done — **refuted**: diffusion does not close its gap (all four changes null). Heterogeneity costs every shared model ~0.009, entirely as misfit, and the ordering is unchanged; diffusion agents never leave consensus, so they cannot personalise ([[D115]]) |
| P5.11, P5.14 | Calibration, and whether `lem:conservative` is conservative in practice: new evaluation metrics, then a re-run of X20's ER cells | 1–2 days code, ~6 h | open |
| P5.12 | $E_{\text{agree}}$, $E_{\text{cent}}$ over time — already recorded, analysis only | — | open |
| P5.24 | Mis-tuned arm: the number is in hand (+0.0161, $t=6.3$); carry it in the figures | — | open |
| figures | Diff-EKF folder builder with a computed `SUMMARY.md` | — | open |
| AdamW | Add AdamW arms to the main MNIST cells, **horizontally** — see below | ~20–30 h (est.) | queued |

About 63 GPU-hours, plus the AdamW pass. At one sweep launched a day, with
scripts, pre-flights and write-ups done while the previous sweep runs: **about ten
days, to roughly 27 September.**

**The AdamW pass** (added 2026-09-25). Mackey--Glass added AdamW as a baseline
family (decision 19) and it is the *strongest* gradient baseline there — centralised
AdamW 0.1628 against centralised SGD's 0.1750 (D106). MNIST has no AdamW arm at all,
so the two tasks are compared against different baseline sets, which is the first
thing a reviewer will notice. It is a **config change, not a port**: the image
branch already validates `optimizer: "adamw"` (`OPTIMIZERS` in `utils/config.py`)
and `optim_state.py` knows the kind.

Three constraints, in the order they bite:

1. **Its own rate grid.** The existing sweeps use $[0.2, 0.05, 0.01, 0.005, 0.001]$,
   which is SGD-shaped; AdamW wants roughly $10^{-4}$ to $10^{-2}$. So each
   experiment needs a second `--lr` grid, not an extra row in the current one.
2. **Moments must be mixed.** `optimizer != "sgd"` with
   `mix_optimizer_state: "none"` raises — unmixed adaptive state diverges across
   agents. That puts a diffusion AdamW arm at $3p$ per link against momentum ATC's
   $2p$, which Track C's bandwidth column has to carry.
3. **Do it horizontally, one experiment at a time**, merging each run's `json` and
   `parquet` into the existing outputs so the figures redraw over the union. Adding
   AdamW to one experiment first would leave the baseline set differing *between*
   experiments — a new asymmetry in place of the one the pass removes. P5.7 ships
   without AdamW for exactly this reason and joins the pass with the others.

⚠ **Gate the merge.** A rerun must differ from the recorded cells in the learner
list and nothing else. Include one *existing* SGD arm in each AdamW run as a
reproduction check: if it does not reproduce its recorded numbers, something is
learner-dependent and the outputs are not poolable. Seeds derive by keyed hash of
stream name (D8) precisely so this holds, but D102 is the standing reminder that a
model-level setting can split cells and destroy seed-pairing silently.

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

**Planned in full in `docs/mackey_glass_plan.md`** (D95): 25 decisions, eight work
packages, the M-series battery M0–M14 at about 90 GPU-hours. The table below is the
original sketch and is kept for the record.

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

**The many-to-one comparison** (added 2026-09-25). A reviewer will ask why the task
predicts at every causal position instead of reading 31 samples and predicting the
32nd. First raised in the 2026-09-23 review of `diff_ekf_summary.pdf` (point 2) and
never recorded until now. The comparison has to answer two separate questions, and
they need different arms:

| item | what | answers | cost | status |
|---|---|---|---|---|
| M2O-a | Score the **last position alone** ($\hat x_L$, full 31-sample context) of a many-to-many run | "is the headline inflated by the short-context positions?" | one metric (`rmse_last`) + a run that records it | open |
| M2O-b | **Supervision-matched** many-to-one: windows strided by **1**, only the last position scored, so each step still consumes 31 fresh targets | "why not many-to-one?" — the honest version | build 1–2 days (est.); GPU sized at pre-flight | open |
| M2O-c | **Sample-matched** many-to-one: windows strided by $L$, one target per block | the rank-1 starvation that motivated the design | small, on M2O-b's plumbing | open |

⚠ **M2O-a is not free from existing results.** The series runs record `rmse` (every
position) and `rmse_full_context` (positions $\ge16$, a whole delay of context), not
the last position alone. `rmse_full_context` already answers the question in its
weaker form and should be quoted meanwhile. The proper answer needs `rmse_last` added
and a run that records it: the cheapest is a fresh many-to-many reference cell run
**alongside M2O-b**, at the same seeds, which M2O-b needs as its comparator anyway.

**M2O-b is the experiment for reviewers; M2O-c is not.** Striding by $L$ with one
target per block cuts supervision from 31 to 1 and the information increment to rank
one per block, so its loss is foregone and a reviewer would call it a strawman. Keep it
only as a labelled illustration. Striding by 1 is the fair competitor, for three
reasons:

1. **No temporal data incest.** Every sample is still a target exactly once. The
   incest the non-overlap rule guards against (`phase5_plan.md`, P5.25) arises only
   when *every* position of an overlapping window is scored.
2. **The same information per step.** 31 targets, rank at most 31, and the same
   Woodbury width $m=31$.
3. **It may be more accurate.** Every target is predicted from a full 31-sample
   context, where many-to-many's position $i$ sees only $i$ samples.

The design also removes the confound flagged on 2026-09-23. All of M2O-b's targets
sit at one position, so the filter's per-position $\boldsymbol R$ profile (which
spans 3.69×) has nothing to exploit, and neither design carries a filter-specific
advantage the other lacks.

**What decides it is compute, and the resource note prices it.** With
`Parameterized_Communication_Memory_Compute_Comparison.pdf` §4.6, stride 1 costs the
filter $31(2C_F+C_B)\approx124\,C_F$ against $2C_F+31\,C_B\approx64\,C_F$ in $C_J$
(about 2×), **nothing extra in $C_W$**, and the gradient baselines about 31× in
forward and backward passes. So if M2O-b matches or beats many-to-many on accuracy,
the paper's case for many-to-many is its cost: decisive for the baselines, modest
for the filter. That is still a defensible argument, but it must be found out here
rather than from a reviewer.

**Order.** Stationary first, at the main law. Add an abrupt-$\beta$ cell only if the
stationary comparison separates the designs.

**M12b — the coupling contrast on the abrupt schedule** (added 2026-09-26). M12's
three `independent` cells ran abrupt (a linear ramp cannot express independence), so
nothing could be subtracted from them, and their only instrument, the additivity
residual, was withdrawn for both bias pairs because it subtracts D108's withdrawn
D(bias, abrupt) ([[D109]]). M12b adds `correlated` and `anti` for the three pairs on the
same `recurring` schedule — six cells, each its pair's `independent` config with only
the coupling overridden — so **independent − correlated** and **anti − correlated
(abrupt)** become cell-minus-cell contrasts with no twin and no withdrawn term. The
primary channel's jumps come from their own seed sub-stream under every coupling, and
the report checks that path is identical per seed rather than assuming it.

| item | cells | GPU | status |
|---|---|---|---|
| M12b | 3 pairs × {correlated, anti}, abrupt, 5 seeds | ~9 h (6/9 of M12's 13.75 h) | runner written 2026-09-26 (`scripts/run_m12b_abrupt_coupling.py`, mg branch); after N>10 |

⚠ Abrupt has returned nulls five times at five seeds. These contrasts avoid the reason
(each seed's twin wandering on its own excursion), so they should resolve better — a
prediction, not a guarantee. Implementation touches the series data
layer (a window stride), the model output (the last position only, i.e. the
selection $\boldsymbol C_{u,t}=\boldsymbol e_q^{\mathsf T}$ of the Diff-EKF note), and
$\boldsymbol R$ (one variance, the last position's).

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

## After A and B — one codebase

Decided 2026-09-19, and **gated: this starts only once both MNIST and
Mackey–Glass are finished**, not before. The two tracks grew on separate
branches and the point is to end with one codebase rather than two that share a
name.

The git half is nearly free. Since the merge base (`f379326`) the only file both
branches have touched is `docs/figures.md`, where each added a `## 15.` — MG1–MG8
on `mg-task`, the diffusion figures 36–39 on `x15-state-model-settled`. Those two
sections have to be interleaved by hand. Nothing else conflicts: the MG work
added modules and edited shared ones the MNIST branch has left alone.

The real work is the redundancy afterwards. Candidates, best first:

1. **The model wrappers.** `mlp.py`, `linear_ar.py` and `transformer.py` each
   repeat the same nine delegations to `models/functional.py` — `names`,
   `shapes`, `flatten`, `unflatten`, `param_groups`, `vjp`, `jvp`, `jacobian`,
   `per_sample_jacobian` — with identical bodies. They belong in `models/base.py`
   once, leaving each class only its `_build_module`, `init_params` and
   `summary`.
2. **Classification against regression.** `metrics/classification.py` beside
   `metrics/regression.py`, and `evaluation/evalsets.py` and `reference.py`
   beside their `series_` counterparts. Measure how much of each series pair is
   genuinely series-specific before merging them — the split may be load-bearing.
3. **The four registries** (`data`, `models`, `learners`, `likelihoods`) share a
   shape. Whether that is worth abstracting is a judgement call, not an obvious
   win, and it is listed last for that reason.

Two constraints. The D104 device contract — `init_params` is CPU-only, modules
carry their own buffers — arrived on `mg-task` but touches `mlp.py`, so it lands
on the MNIST path too; `tests/test_model_device.py` guards it and must come
across with it. And the figure builders stay untracked: the merge must not add
any `scripts/make_*.py` or `scripts/plot_*.py` to the public repo.

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
