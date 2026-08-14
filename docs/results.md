# Results

Measured numbers, the settings that produced them, and what they support. Every
figure of this document is reproducible from `results/` by
`scripts/make_figures.py`; the tuning tables come from
`scripts/sweep_hyperparameters.py --report`.

If this document and the data disagree, the data is right and this is stale.

---

## 1. X0 — the exactness check

On a complete graph with uniform weights and plain SGD in float64, ATC diffusion
is algebraically identical to centralized SGD.

| quantity | value |
|---|---|
| worst residual over 50 steps | **1.7e-15** (target 1e-12) |
| error rates, both learners | identical to 4 decimals |
| $E_\text{agree}$ | $\approx 7\times10^{-31}$ |

**The floor is arithmetic, not algebra.** Combining $N$ identical vectors returns
them perturbed by about one ulp — not because the weights are wrong ($\sum_u
a_{vu}$ is exactly 1.0 in float64 at every $N$ tested) but because the partial
sums inside the matmul are not representable even when the summands and the total
are. At $N=2$ the fixed point is exact; at $N \ge 8$ it is not. That is why the
target is 1e-12 rather than zero.

**What X0 certifies is broader than plain SGD.** The identity survives any
optimizer *linear in the gradients*:

| optimizer | mixing | residual | |
|---|---|---|---|
| plain SGD | — | 9.99e-16 | exact |
| momentum $\beta{=}0.9$ | mixed | 1.33e-15 | exact |
| momentum $\beta{=}0.9$ | not mixed | 9.99e-16 | exact |
| AdamW | all | 0.76 | **breaks** |

30 steps, complete graph, float64, tuned lr 0.01. The exact rows are float64
accumulation and are not stable to two digits across torch versions or learning
rates; AdamW's residual scales with the step size (5.24 at lr 0.05). Only the
gap against the 1e-12 tolerance is a claim.

AdamW breaking is the positive control: without a case that fails, a test that
always passes is indistinguishable from one that checks nothing. **It is not a
convergence result** — the identity fails on the first step because Adam's second
moment is nonlinear in $\bm g$, which is a different phenomenon from the D-Adam
divergence in `WORKPLAN.md` §3.4. No experiment here runs AdamW to convergence:
the tuning sweep grid is `optimizer` $\in$ {`sgd`, `sgd_momentum`} only.

---

## 2. The tuning sweep

Run because the first full X1 exposed a confound serious enough to invalidate the
headline (see §6). Grid: lr $\in$ {0.2, 0.05, 0.01, 0.005, 0.001} $\times$
$n \in$ {2, 4, 6, 8, 10} $\times$ optimizer $\in$ {sgd, sgd_momentum}, two seeds,
$T=600$ on the ring. Scored on the held-out set over the last 100 steps,
counts-then-divide.

### 2.1 The selected settings

At $n = 4$, each method's own optimum:

| learner | optimizer | lr | held-out error |
|---|---|---|---|
| `centralized_sgd` | sgd_momentum 0.9 | 0.01 | 0.0937 |
| `diffusion_sgd_atc` | sgd_momentum 0.9 | 0.01 | 0.0964 |
| `diffusion_sgd_cta` | sgd_momentum 0.9 | 0.01 | 0.0974 |
| `diffusion_sgd_atc_plain` | plain sgd | 0.20 | 0.1133 |
| `local_only` | plain sgd | 0.05 | 0.1773 |

Three things this settles.

**Everything that averages wants the same cell.** Centralized, ATC and CTA all
select momentum at lr 0.01. Convenient for F8: matched settings *are* each
method's optimum, so the ordering comparison needs no caveat.

**`local_only` is the outlier**, wanting plain SGD at a step five times larger.
It has the noisiest gradient and no averaging to damp it, so momentum's
accumulation is what it cannot afford — the same mechanism as the original bug,
now handled by tuning rather than by hitting it.

**The payload-matched variant costs 0.017.** `atc_plain` at $p$ per link reaches
0.1133 against momentum ATC's 0.0964 at $2p$. That is the measured price of the
matched-communication comparison, and it is the number **Diff-EKF must beat** —
phase 5's claim is stated at $p$ per link, so `atc_plain` is its real competitor,
not the stronger momentum variant.

### 2.2 $n$ changes the answer, not just the accuracy

Each method at its own optimum for that $n$:

| $n$ | centralized | ATC | local only | cooperation gap | pooling gap |
|---|---|---|---|---|---|
| 2 | 0.1051 | 0.1086 | 0.2589 | **0.1503** | 0.0035 |
| 4 | 0.0937 | 0.0964 | 0.1773 | **0.0810** | 0.0027 |
| 6 | 0.0895 | 0.0925 | 0.1470 | 0.0544 | 0.0030 |
| 8 | 0.0781 | 0.0896 | 0.1341 | 0.0444 | 0.0115 |
| 10 | 0.0807 | 0.0890 | 0.1270 | **0.0380** | 0.0083 |

- **cooperation gap** $= e(\texttt{local\_only}) - e(\texttt{ATC})$ — what
  talking to neighbours is worth.
- **pooling gap** $= e(\texttt{ATC}) - e(\texttt{centralized})$ — what a fusion
  centre is worth on top.

Centralized moves 0.010 across the whole range; `local_only` moves 0.132. **The
cooperation gap collapses sixfold.** Cooperation is variance reduction and so is
a larger batch — they buy the same thing, so the more an agent sees alone, the
less its neighbours add. Choosing $n$ to minimise error would therefore minimise
the effect Q2 exists to measure, which is why $n$ is an axis (X4) and the
headline runs fix $n=4$.

**The pooling gap is small**: 0.003 at $n \le 6$. Diffusion on a ring recovers
almost all of pooling's advantage, which is a result in its own right.

*Caveat.* Two seeds, spreads 0.002–0.025. The trend across $n$ is far larger than
the noise; individual ~0.01 comparisons are not. Centralized at $n=8$ (0.0781)
versus $n=10$ (0.0807) is non-monotonic and almost certainly noise. Two seeds is
right for *selecting* hyperparameters, not for quoting results — the headline
runs use five and report ±1 s.d.

### 2.3 Momentum is not the problem; the effective step is

At lr 0.01 momentum *beats* plain SGD (0.094 vs 0.203 for centralized at $n=4$).
At lr 0.05 with $n=2$ it is catastrophic. What matters is
$\eta/(1-\beta)$: at $\beta = 0.9$ that multiplies the step tenfold, so lr 0.05
becomes an effective 0.5 and lr 0.01 an effective 0.1.

---

## 3. X1, X1b, X2, X5 — the headline runs

At the tuned settings, $n=4$, $T=1500$, ring $N=10$, five seeds. Held-out error
over the last 100 steps, +/-1 s.d. across seeds.

### 3.1 X1 — stationary

| learner | error | +/-1 s.d. |
|---|---|---|
| `centralized_sgd` | **0.0754** | 0.0035 |
| `diffusion_sgd_atc` | 0.0768 | 0.0034 |
| `diffusion_sgd_atc_plain` | 0.0902 | 0.0034 |
| `local_only` | 0.1350 | 0.0035 |

The ordering is correct throughout, which is the check that the D39 confound is
gone. Three numbers, read against the seed noise:

**Diffusion on a ring is statistically indistinguishable from pooling.** The
pooling gap is 0.0014 against a seed s.d. of 0.0035 -- *smaller than the noise*.
Ten agents exchanging one $2p$ message with two neighbours per step recover
essentially everything a fusion centre holding all the data would get.

This reframes what phase 5 has to show. There is almost no headroom above ATC on
stationary MNIST, so **Diff-EKF's case cannot be made on stationary accuracy** --
it has to come from tracking under drift, from communication efficiency, or from
calibrated uncertainty.

**Cooperation is worth 0.058**, 17x the noise. Unambiguous, and now a learning
result rather than the optimizer artefact it was before, because `local_only`
runs at its own optimum.

**The payload price is 0.013**, about 4x the noise. `atc_plain` at $p$ per link
against momentum ATC at $2p$.

### 3.2 X1b — ATC vs CTA at full length

| learner | error | +/-1 s.d. |
|---|---|---|
| `diffusion_sgd_atc` | 0.0768 | 0.0034 |
| `diffusion_sgd_cta` | 0.0774 | 0.0033 |

A difference of 0.0006 against a s.d. of 0.0034 — indistinguishable, exactly as
the $T=600$ grid predicted. D40's conclusion holds at full horizon with five
seeds: **ATC's advantage is robustness to step size, not accuracy at the right
one.**

### 3.3 X2 — linear rotation to 45 degrees

| learner | $t$ 1400-1500 | drift cost vs stationary |
|---|---|---|
| `centralized_sgd` | 0.0912 | +0.016 |
| `diffusion_sgd_atc` | 0.0932 | +0.016 |
| ATC (payload-matched) | 0.1120 | **+0.022** |
| `local_only` | 0.1667 | **+0.032** |

**Drift cost tracks how much averaging a method gets.** Centralized and the
momentum ATC both pay 0.016; the payload-matched variant pays 0.022 and the lone
agent 0.032. The variant that carries no momentum has less inertia to smooth a
moving target, and the agent with no neighbours has none at all.

Every method still *improves* through the run, so at $\alpha = 0.03$ deg/step the
drift is slow relative to the information rate — the regime the Diff-EKF
assumptions target, and the regime where tracking is meaningful rather than
hopeless (WORKPLAN 4.3).

**Drift costs the cooperative methods 0.016 and `local_only` 0.032** — twice as
much. The pooling gap stays at 0.002, so rotation does not widen the
diffusion-vs-centralized gap; it widens the cooperation gap. Answering Q3: the
lone agent degrades faster, and communication is what buys the resilience.

### 3.4 X5 — abrupt 15-degree shift at $t = 500$

| learner | before | spike | final | final vs before |
|---|---|---|---|---|
| `centralized_sgd` | 0.1005 | +0.0737 | 0.0738 | **−0.027** |
| `diffusion_sgd_atc` | 0.1034 | +0.0734 | 0.0753 | **−0.028** |
| ATC (payload-matched) | 0.1237 | +0.0811 | 0.0879 | **−0.036** |
| `local_only` | 0.1870 | +0.0742 | 0.1327 | **−0.054** |

**The shift costs every method the same 0.073–0.081**, regardless of how much
data it pools or how much it communicates — so an abrupt change is a shared tax,
not something cooperation protects against. What differs is the level each
recovers to.

All four end *better* than their pre-shift error (the negative final column).
That is not the shift being harmless: the methods were still improving at
$t=500$, so recovery and ongoing learning overlap.

The transient itself is the measurement, which is why F7 zooms on
$[t^\ast-50,\ t^\ast+300]$ rather than quoting endpoints. All three recover
within ~200 steps, and the *relative* ordering is unchanged throughout — the
shift does not advantage any method.

### 3.5 F2 — the curves cross, under drift as well

Measured at matched bandwidth, prequential error over the last 20 evaluation
points before each budget:

| budget (scalars) | stationary | rotating 45° |
|---|---|---|
| $10^{7}$ | payload-matched +0.064 | payload-matched +0.067 |
| $2\times10^{7}$ | payload-matched +0.013 | payload-matched +0.009 |
| $5\times10^{7}$ | payload-matched +0.014 | payload-matched +0.008 |
| $10^{8}$ | **ATC (2p) +0.009** | **ATC (2p) +0.024** |

The same shape in both regimes, so **the crossing is not an artefact of the
stationary task**. Under drift the momentum variant's eventual win at high
bandwidth is *larger* (0.024 against 0.009): momentum pays off more when the
target is moving, but only once bandwidth is plentiful.

This panel could not be drawn until X2 was re-run. Its config listed three
learners while X1 listed four, so the payload-matched variant — the series this
figure exists for — was simply absent from the rotating panel. X5 had the same
omission and was re-run with it too.

**It is a genuine crossing, not a restatement of §3.1**: at equal *time* momentum
ATC wins by 0.013, at equal *bandwidth* the payload-matched variant wins until
quite late. Which one is "better" depends on whether the clock or the network is
the binding constraint.

It is also the shape phase 5 needs. Diff-EKF's claim is stated at identical
communication, so F2 is the axis on which it has to be argued -- and §3.1 already
showed there is almost no headroom on the time axis.

---

## 4. The ATC vs CTA tuning grid

Full grid with both orderings in the same cells. **ATC wins 47 of 50** — the
sign is not chance — but at each one's optimum the difference is an order of
magnitude below seed noise (0.0002–0.0018 against spreads of 0.002–0.012).

ATC separates sharply only where the step is too large: 0.118 vs 0.251 at
momentum lr 0.2, $n=10$.

**So ATC's advantage is robustness to step size, not accuracy at the right one.**
ATC averages after stepping, so the combine step damps a too-large update; CTA
averages first and the damping arrives before the step it would have absorbed.

ATC remains primary because Diff-EKF is ATC and matching the ordering removes a
confound from phase 5 — a reason that does not depend on CTA being worse. See
design note D40.

---

## 5. Communication

At $N=10$ on a ring, $p = 2908$:

| learner | per link | scalars/step | relative |
|---|---|---|---|
| `diffusion_sgd_atc_plain` | $p$ | 58 160 | 1.0× |
| `diffusion_sgd_atc` | $2p$ | 116 320 | 2.0× |
| `diffusion_ekf` (local) | $p$ | 58 160 | 1.0× |
| `diffusion_ekf` (one_hop) | $p(q'{+}1)$ | 581 600 | 10.0× |
| `centralized_sgd` | — | 0 on this axis | — |
| `local_only` | 0 | 0 | — |

Centralized logs zero **not because it is free** — it ships every sample to a
centre — but because that cost is off the axis F2 measures. It is drawn as a
horizontal reference rather than a point at $x=0$, which would read as "free and
this good" (design note D30).

---

## 6. A confound that was caught, and how

Recorded because the failure mode is more instructive than the fix.

The planned primary was SGD momentum 0.9 at lr 0.05, $n=2$. The first full X1 run
gave:

| learner | optimizer | held-out error |
|---|---|---|
| `diffusion_sgd_atc_plain` | plain SGD | **0.095** |
| `centralized_sgd` | momentum | 0.117 |
| `diffusion_sgd_atc` | momentum | 0.146 |
| `local_only` | momentum | 0.897 |

`local_only` at chance for ten classes would have been reported as "cooperation
is essential". The mechanism is even real — averaging over $N=10$ agents cuts
gradient noise like a tenfold batch increase, which is exactly why the diffusion
methods survived a setting the lone agent could not.

**The giveaway was not the suspicious number.** It was that
`diffusion_sgd_atc_plain` finished *ahead of* `centralized_sgd`. Centralized pools
every agent's samples each step, so on a complete graph ATC equals it exactly
(X0) and on a ring it can only be a perturbation of it. A distributed method
above the pooled bound means something other than the method is driving the
numbers — and the learner labels happened to correlate with the optimizer.

Isolating it on a single agent, varying only the optimizer:

| lr | momentum | $n$ | held-out error |
|---|---|---|---|
| 0.05 | 0.9 | 2 | **0.898** |
| 0.05 | 0.0 | 2 | 0.194 |
| 0.01 | 0.9 | 2 | 0.356 |
| 0.005 | 0.9 | 2 | 0.188 |
| 0.05 | 0.9 | 20 | 0.188 |

Not divergence — $\|\bm\theta\|^2$ stayed comparable to the other methods — but a
settle into near-uniform output, mean confidence 0.154 against a floor of 0.1.

**The standing check this leaves behind:** no distributed method may beat
`centralized_sgd` at matched settings. It is now in the risk table of
`WORKPLAN.md` §9.

---

## 7. X3 — the topology tuning sweep

350 cells (7 topologies x 5 lr x 2 optimizers x 2 seeds), `diffusion_sgd_atc`
only: `centralized_sgd` and `local_only` never read the graph, so their tuning
transfers from the ring and including them would have wasted 60% of the compute.

### 7.1 Does a denser graph tolerate a larger step?

Held-out error at each lr under momentum, ordered by connectivity:

| topology | $1-\rho$ | lr 0.005 | **lr 0.01** | lr 0.05 | lr 0.2 |
|---|---|---|---|---|---|
| path | 0.033 | 0.1092 | 0.0982 | 0.1282 | 0.4689 |
| grid2d | 0.096 | 0.1075 | 0.0955 | 0.1212 | 0.3925 |
| star | 0.100 | 0.1118 | 0.1032 | **0.1812** | **0.5906** |
| ring | 0.127 | 0.1073 | 0.0954 | 0.1101 | 0.4094 |
| watts_strogatz | 0.197 | 0.1065 | 0.0937 | 0.1032 | 0.3790 |
| erdos_renyi | 0.417 | 0.1067 | 0.0943 | 0.1088 | 0.3861 |
| complete | 1.000 | 0.1063 | 0.0933 | 0.1069 | **0.3406** |

**Yes, and exactly where predicted -- at the large learning rates.** At lr 0.2 the
spread across topologies is 0.34 to 0.59; at the optimum it is 0.093 to 0.103.
Better mixing buys tolerance to an *oversized* step, not a better optimum. Same
shape as the ATC-vs-CTA finding (D40).

**But the optimum barely moves.** Six of seven topologies select momentum at
lr 0.01, identical to the ring. Only `complete` differs (plain SGD at lr 0.2),
and structurally: one combine step reaches full consensus there, so ATC *is*
centralized (the X0 identity) and inherits its preference for a large plain step.

So the re-tuning was worth doing -- it turned an assumption into a measurement --
but it changes almost nothing in F3, which is the good outcome: the connectivity
axis is not contaminated by shifting learning rates.

### 7.2 `star` is an outlier, and F3 has to account for it

`star` is the **worst topology at every learning rate**, despite a *higher*
spectral gap (0.100) than `grid2d` (0.096), and it degrades far more at large lr
(0.59 against 0.39).

If the final gap were a clean function of $1-\rho$, star and grid2d would sit
together. They do not. The likely reason is that Metropolis weights on a star
give the hub a very different weight profile from the leaves, so **the spectral
gap understates how badly a star mixes information** -- every path runs through
one node, and $1-\rho$ does not see that.

**A correction to my first reading.** I proposed plotting against the mixing gap
$1-\mathrm{SLEM}$ alongside $1-\rho$, expecting it to separate star from grid2d.
It cannot: Metropolis weights are *symmetric*, so the second-largest eigenvalue
modulus **is** the spectral gap, and the two are identical to machine precision
for every topology here -- that would have been two panels of the same plot.

On the first three topologies to finish, `star` had the **highest** spectral gap
and the **worst** gap by 5x, at essentially the same $1-\rho$ as grid2d.

**I then overclaimed, and §7.3 corrects it.** From three points I wrote that "the
spectral gap does not predict the price of decentralization". With all seven
measured that is false: $1-\rho$ predicts with Spearman $-0.786$, $p = 0.036$, in
the expected direction. What survives is the weaker and more useful statement —
it is *not the best* predictor, and it specifically mis-ranks the star.

### 7.3 What does predict it: the mean self-weight

All seven topologies, transient window, gap paired within seed:

| topology | gap | s.d. | $1-\rho$ | $\bar a_{vv}$ |
|---|---|---|---|---|
| complete | 0.0000 | 0.0000 | 1.000 | 0.100 |
| watts_strogatz | 0.0005 | 0.0006 | 0.197 | 0.285 |
| erdos_renyi | 0.0013 | 0.0009 | 0.417 | 0.236 |
| grid2d | 0.0034 | 0.0012 | 0.096 | 0.317 |
| ring | 0.0040 | 0.0009 | 0.127 | 0.333 |
| path | 0.0093 | 0.0013 | 0.033 | 0.400 |
| **star** | **0.0182** | 0.0023 | 0.100 | **0.820** |

| predictor | Spearman | exact $p$ | Pearson vs $\log$ gap |
|---|---|---|---|
| spectral gap $1-\rho$ | $-0.786$ | **0.048** | $-0.619$ |
| **mean self-weight $\bar a_{vv}$** | $+0.964$ | **0.0028** | $+0.765$ |

All seven candidates, and how they moved as the last two topologies landed:

| candidate | $\rho$ at $n{=}5$ | $\rho$ at $n{=}7$ | exact $p$ |
|---|---|---|---|
| mean self-weight | +1.000 | **+0.964** | **0.0028** |
| Kirchhoff index | +0.700 | **+0.857** | **0.014** |
| max self-weight | +0.600 | +0.679 | 0.094 |
| spectral gap | −0.500 | **−0.786** | **0.048** |
| degree variance | +0.100 | +0.252 | 0.585 |
| min self-weight | −0.051 | +0.436 | 0.328 |
| diameter | −0.051 | +0.436 | 0.328 |

**`min self-weight` and `diameter` flipped sign** between five and seven points
($-0.051 \to +0.436$). That is what noise looks like at this sample size, and it
is the best available warning about how far any of these can be trusted.

**The $p$ values are exact permutation tests**, enumerating all $7! = 5040$
relabellings. An earlier draft of this section quoted scipy's defaults (0.0005
and 0.036), which use a $t$-approximation that is anti-conservative at $n=7$. The
correction matters for the spectral gap: 0.048 rather than 0.036 puts it *just*
inside the conventional threshold, which is much weaker support than first
reported and is a further reason to present both predictors rather than declare
a winner.

**Reading the signs.** They are opposite and both expected. A larger spectral gap
means better connectivity, so it should come with a *smaller* gap -- negative.
A larger self-weight means each agent keeps more of its own estimate and mixes
*less*, so it should come with a *larger* gap -- positive. What matters is
$|\rho|$ together with the sign matching the expected direction; ranking these
candidates by signed value alone would put `degree variance` (+0.252) above the
spectral gap, which is backwards.

**Both work, in the expected direction.** Higher connectivity, lower price of
decentralization. The spectral gap is significant at $p=0.036$ — so F3's premise
stands, and my three-point claim that it "does not predict" was wrong.

**The self-weight is the better predictor, and it places the star correctly.**
Ranked by $1-\rho$, star sits **3rd of 7** while measuring **worst of 7**; ranked
by $\bar a_{vv}$ it sits 7th, exactly where it belongs.

**Why $\bar a_{vv}$ is the mechanically right quantity.** It is the average
fraction of its own estimate an agent *keeps*, so $1-\bar a_{vv}$ is literally
how much mixing happens per round. Star follows immediately: Metropolis gives each
leaf a hub-weight of $1/(1+9)=0.1$, so **every leaf retains 90% of its own
estimate** and the network barely mixes despite a diameter of 2. The spectral gap
describes the *asymptotic rate* of consensus; over a 1500-step online run what
matters is mixing *per step*, and those come apart on exactly this kind of graph.

### 7.3.1 The out-of-sample test, and its result

The predictor was identified on five topologies while two were still running, so
the prediction for those two was recorded in advance:

| topology | predicted | measured | |
|---|---|---|---|
| complete | $\approx 0$ exactly, by the X0 identity | **0.0000** | correct, but see below |
| erdos_renyi | $< 0.0005$ (below watts_strogatz) | **0.0013** | **wrong** |

**Only one of these was a test at all.**

`complete` was never one. On a complete graph with uniform weights, one combine
step averages over *every* agent, so ATC reduces algebraically to centralized
SGD -- the X0 identity, which is proven rather than measured. The gap is forced
to zero **by mathematics, whatever any predictor says**; a completely wrong
theory would also get it right. Its 0.0000 confirms the pipeline is sound and
says nothing about $\bar a_{vv}$.

`erdos_renyi` was the real test, because nothing forces its value. The predictor
said: self-weight 0.236, below watts_strogatz's 0.285, therefore a *smaller* gap
than 0.0005. Measured **0.0013** -- roughly triple, on the wrong side. That single
inversion is the whole departure from $\rho = +1$.

**So the predictor has faced exactly one genuine out-of-sample test, and failed
it.** That is the honest accounting.

So: $\bar a_{vv}$ fits this set substantially better than $1-\rho$ and is
mechanically motivated, but it is **not validated out of sample** — the one
genuine test it faced, it failed. Reported as the better of two descriptive
correlations on seven graphs, not as a law.

A further caveat noted when the prediction was made: `erdos_renyi` and `complete`
do not *discriminate* between the two predictors — both order them the same way.
The discriminating case is `star` alone, which is a single point.

### 7.4 A free correctness check inside X3

On the complete graph with plain SGD, ATC must equal centralized *exactly* -- it
is the X0 identity. X3 runs both in the same run, so it reproduces the check on
the production path rather than only in the test suite. Measured at 60 steps:
centralized, ATC and atc_plain all at 0.16110, agreeing to five decimals.

---

## 8. CUDA

Measured on an RTX 4070 Laptop against 10 CPU threads:

| workload | batch | CPU | CUDA | |
|---|---|---|---|---|
| one agent's gradient | 4 | 1.34 ms | 1.93 ms | **0.69x** |
| one agent's gradient | 40 | 1.58 ms | 2.04 ms | **0.77x** |
| one agent's gradient | 400 | 2.16 ms | 1.61 ms | 1.34x |
| reference trainer, one epoch | 128 | 0.57 s | 0.62 s | **0.92x** |
| **dense $p\times p$ matmul** | — | **120 ms** | **8.6 ms** | **14x** |

**Nothing in phases 1-4 benefits.** At $p = 2908$ with batches of 4-40, kernel
launch overhead exceeds the arithmetic; the crossover is near batch 400, an order
of magnitude above anything these experiments use. Even the longest CPU job in
the project -- the 20-minute reference trainer -- comes out slower.

**Phase 5 inverts it.** A dense covariance is 8.5M entries per agent and the EKF
update needs several $p \times p$ products per agent per step. 120 ms against
8.6 ms across 10 agents and 1500 steps is the difference between days and
minutes; CUDA is what makes the dense filter feasible, which is the assumption
$p=2908$ was chosen under.

`run.device` was validated but never used -- `cuda` would have been accepted and
silently ignored. It now raises, naming where CUDA does pay. Phase 5 removes the
guard when it wires the device through. See design note D43.

---

## 9. X4 — where sparsity bites (Q4)

$n \in \{1,2,4,8\}$ crossed with $\pi_\text{lab} \in \{0.25, 0.5, 1.0\}$,
$T = 750$, five seeds, at the **headline tuning** ($n{=}4$, $\pi_\text{lab}{=}1$).
Held-out error over the last 100 steps.

| $n$ | $\pi_\text{lab}$ | centralized | ATC | ATC (payload-matched) | local only | coop gap | pool gap |
|---|---|---|---|---|---|---|---|
| 1 | 0.25 | 0.4035 | 0.1722 | 0.2921 | 0.5847 | 0.413 | **−0.231** |
| 1 | 0.5 | 0.1904 | 0.1489 | 0.2513 | 0.5118 | 0.363 | **−0.041** |
| 1 | 1.0 | 0.1276 | 0.1352 | 0.2480 | 0.4309 | 0.296 | +0.008 |
| 2 | 0.25 | 0.1908 | 0.1453 | 0.1797 | 0.3971 | 0.252 | **−0.046** |
| 2 | 0.5 | 0.1289 | 0.1220 | 0.1658 | 0.3047 | 0.183 | −0.007 |
| 2 | 1.0 | 0.1038 | 0.1068 | 0.1418 | 0.2406 | 0.134 | +0.003 |
| 4 | 0.25 | 0.1369 | 0.1317 | 0.1303 | 0.2724 | 0.141 | −0.005 |
| 4 | 0.5 | 0.1058 | 0.1090 | 0.1200 | 0.2008 | 0.092 | +0.003 |
| 4 | 1.0 | 0.0886 | 0.0905 | 0.1081 | 0.1606 | 0.070 | +0.002 |
| 8 | 0.25 | 0.1109 | 0.1252 | 0.1178 | 0.2143 | 0.089 | +0.014 |
| 8 | 0.5 | 0.0926 | 0.1015 | 0.1030 | 0.1572 | 0.056 | +0.009 |
| 8 | 1.0 | 0.0847 | 0.0859 | 0.0864 | 0.1271 | 0.041 | +0.001 |

*Reading the payload-matched column here.* This table fixes every learner at its
own headline setting, and ATC's headline carries momentum while the
payload-matched variant's does not — so the column is a snapshot of one
particular pair of settings, not a tuned comparison. That is why it can go
**negative** at $n \in \{4, 8\}$, $\pi_\text{lab} = 0.25$: momentum's
$\eta/(1-\beta) = 10\eta$ effective step is too large there, and dropping it
helps. The properly tuned payload cost is §9.2's, and it is positive everywhere.

### 9.1 The negative pooling gaps: mostly a step-size artefact

Five cells show ATC *beating* centralized — the sanity check of §6 firing again.
Investigated rather than reported.

**Most of it is an effective-step artefact.** At $\pi_\text{lab} < 1$ most agents
are idle, and an idle agent contributes its unchanged $\bm\theta$ to the combine
step. So ATC's update is
$\bm\theta - \frac{\eta}{N}\sum_{v\ \text{active}} \nabla L_v$ — an effective step
of $\eta \times n_\text{active}/N$, while centralized takes the full $\eta$ on the
pooled batch. At $\pi_\text{lab} = 0.25$ that is a **4x** difference, and the two
methods are therefore not running at comparable step sizes.

Swept finely at $n{=}1$, $\pi_\text{lab}{=}0.25$:

| lr | centralized | ATC |
|---|---|---|
| 0.01 | 0.3767 | **0.1613** |
| 0.005 | 0.2437 | 0.2030 |
| 0.0025 | **0.1927** | — |
| 0.001 | 0.2137 | 0.5397 |

Centralized's optimum is lr 0.0025 and ATC's is lr 0.01 — **exactly the predicted
factor of 4**, which confirms the mechanism quantitatively.

**But a residual survives correct tuning.** At each method's own optimum the gap
is $0.1613 - 0.1927 = -0.031$, not zero. Of the raw $-0.232$, roughly $-0.20$ is
tuning and $-0.031$ is real.

**What the residual is.** Diffusion maintains $N$ trajectories that the combine
step continuously averages, so it gets **implicit iterate averaging** — a
variance-reduction device a single centralized trajectory does not have. In a
high-noise regime (a pooled batch of ~2.5 samples) that is worth something real.
It is *not* evidence that diffusion uses data better, and a centralized baseline
with Polyak averaging would probably close much of it. Worth stating because
"distributed beats centralized" is a claim a reader will not expect.

### 9.2 The tuned grid (F6a)

432 cells: 12 $(n, \pi_\text{lab})$ combinations $\times$ 9 learning rates
(0.001 to 1.0) $\times$ 2 optimizers $\times$ 2 seeds. Each method at **its own
optimum in each cell**, and every optimum is interior to the grid (§9.3).

| $n$ | $\pi_\text{lab}$ | centralized | ATC | ATC (payload-matched) | local only | coop gap | pool gap | payload cost |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.25 | 0.1907 | 0.1698 | 0.2160 | 0.5394 | 0.370 | **−0.021** | 0.0461 |
| 1 | 0.50 | 0.1602 | 0.1521 | 0.1787 | 0.4080 | 0.256 | −0.008 | 0.0266 |
| 1 | 1.00 | 0.1168 | 0.1226 | 0.1374 | 0.2883 | 0.166 | +0.006 | 0.0148 |
| 2 | 0.25 | 0.1544 | 0.1445 | 0.1863 | 0.4144 | 0.270 | −0.010 | 0.0418 |
| 2 | 0.50 | 0.1223 | 0.1183 | 0.1395 | 0.2965 | 0.178 | −0.004 | 0.0213 |
| 2 | 1.00 | 0.1077 | 0.1091 | 0.1210 | 0.2010 | 0.092 | +0.001 | 0.0120 |
| 4 | 0.25 | 0.1220 | 0.1165 | 0.1291 | 0.2846 | 0.168 | −0.006 | 0.0126 |
| 4 | 0.50 | 0.1038 | 0.1039 | 0.1247 | 0.2035 | 0.100 | +0.000 | 0.0209 |
| 4 | 1.00 | 0.0881 | 0.0904 | 0.1042 | 0.1555 | 0.065 | +0.002 | 0.0138 |
| 8 | 0.25 | 0.1096 | 0.0998 | 0.1210 | 0.2218 | 0.122 | −0.010 | 0.0212 |
| 8 | 0.50 | 0.0896 | 0.0903 | 0.1000 | 0.1608 | 0.070 | +0.001 | 0.0097 |
| 8 | 1.00 | 0.0764 | 0.0806 | 0.0871 | 0.1271 | 0.047 | +0.004 | 0.0065 |

*How the payload-matched variant is tuned.* Both names map to one class, and the
sweep sets the optimizer for every learner it runs — so left unconstrained the
variant picks momentum and becomes numerically identical to ATC, making the
payload cost exactly 0.000 in all twelve cells. That is not a small effect but a
definition being overridden: carrying no optimizer state is precisely what makes
the message $p$ per link rather than $2p$. Its optimum is therefore taken
**within the plain-SGD arm only** (`make_figures.x4_tuned`).

**Tuning accounts for about 90% of the apparent effect.** At $n{=}1,
\pi_\text{lab}{=}0.25$ the fixed-tuning table showed $-0.232$; properly tuned it
is $-0.021$.

**The residual is structured, not scattered.** The pooling gap is negative in
every $\pi_\text{lab} < 1$ cell, positive in every $\pi_\text{lab} = 1$ cell, and
shrinks monotonically as $\pi_\text{lab}$ rises. That gradient toward the noisiest
corner is the signature predicted for **implicit iterate averaging**: diffusion
maintains $N$ trajectories under continuous averaging, a variance-reduction
device a single centralized trajectory does not have, and it should pay exactly
where the per-step gradient is noisiest. Residual mis-tuning would have left the
negative cells scattered.

It remains a small effect (≤0.021) and it is *not* evidence that diffusion uses
data better. A centralized baseline with Polyak averaging would likely close much
of it — worth stating, because "distributed beats pooled data" is a claim no
reader expects.

**The payload cost rises as the problem gets sparser.** Averaged over the other
axis it runs 0.0118 → 0.0196 → 0.0304 as $\pi_\text{lab}$ falls from 1.0 to 0.25,
and 0.0125 → 0.0292 as $n$ falls from 8 to 1 — about **2.5x** along each. Both
axes control the same thing: how much signal one step carries. The extra $p$
scalars buy momentum, and momentum's job is to accumulate a consistent direction
out of noisy gradients, so it is worth most exactly where each gradient is worst.
The trend is clean in the marginals but not cell-by-cell — $n{=}4$ is
non-monotone in $\pi_\text{lab}$ (0.0126 / 0.0209 / 0.0138), which at two sweep
seeds is inside the noise.

Contrast §10: under label skew the payload cost is **flat** (0.015 / 0.013 /
0.015). Skew changes *what* an agent sees, not how often it updates — and
momentum only compensates for the latter. Taken together: the $2p$ message earns
its second half under sparsity, not under heterogeneity. For phase 5 that bounds
what a $p$-per-link Diff-EKF gives up — at most ~0.046, in the sparsest corner of
the plane.

### 9.3 The cost of not re-tuning (F6b)

$\text{penalty} = e(\text{headline lr}) - e(\text{best lr for that cell})$.

| method | worst penalty | where | median |
|---|---|---|---|
| `centralized_sgd` | **0.183** | $n{=}1$, $\pi{=}0.25$ | 0.007 |
| `local_only` | 0.151 | $n{=}1$, $\pi{=}1.0$ | 0.004 |
| ATC (payload-matched) | 0.110 | $n{=}1$, $\pi{=}0.5$ | 0.000 |
| **`diffusion_sgd_atc`** | **0.028** | $n{=}8$, $\pi{=}0.25$ | 0.004 |

**ATC's worst case is 6.5x smaller than centralized's**, and its penalty never
exceeds 0.028 anywhere on the plane while centralized reaches 0.183 and
`local_only` 0.151.

*Both terms come from the tuning sweep, not from the X4 runs.* A penalty defined
as "headline minus the minimum over a grid that contains the headline" cannot be
negative, so a negative value is a bug rather than a finding. Two produced them
and both are fixed: lr 0.2 was missing from the grid (so the "headline" for a
method tuned there was being compared against a minimum taken without it), and
the two terms were drawn from different estimators — a five-seed X4 number minus
a two-seed sweep number, which gave penalties as low as −0.042. With one
estimator on both sides, **0 of 48 cells are negative**. The five-seed X4 runs
remain what is quoted everywhere else; they are simply not what *this* difference
can be built from.

**The mechanism is automatic step-size scaling, not damping.** An earlier draft
here said "the combine step damps an oversized update"; testing separated three
candidates and that was not the one that survived.

| candidate | verdict |
|---|---|
| ATC's error-vs-lr curve is flatter | **not robustly** — the measure depends on the grid (below) |
| ATC's optimum moves less across the plane | **yes** — 1.0 decades against 1.9 for centralized and 1.7 for `local_only` |

*Why flatness was dropped.* "Cost of being one grid step off the per-cell
optimum," averaged over the twelve cells within each method's own optimizer arm:

| grid | centralized | ATC | ATC (payload-matched) | local only |
|---|---|---|---|---|
| full (lr ≤ 1.0) | 0.025 | 0.013 | 0.040 | 0.073 |
| without 1.0, 0.5 | 0.025 | 0.013 | 0.033 | 0.073 |
| without 1.0, 0.5, 0.2 | 0.013 | 0.012 | 0.070 | 0.042 |

Widening the grid changes nothing for three of the four — their optima sit far
from the top, so their neighbours are untouched. Trimming it down to 0.05
*halves* centralized's number and closes the gap to 0.013/0.012. A quantity that
reorders when the grid is trimmed is not measuring the method, so it is not the
mechanism — even though on the full grid it happens to favour ATC.

The optimum's *location* is stable, and why it barely moves is visible in the
full momentum-arm profile at $n{=}1,\ \pi_\text{lab}{=}0.25$:

| lr | centralized | ATC |
|---|---|---|
| 0.0025 | **0.208** ← optimum | 0.321 |
| 0.005 | 0.259 | 0.207 |
| **0.01** ← headline | 0.374 | **0.170** ← optimum |
| 0.02 | 0.726 | 0.199 |
| 0.05 | 0.896 | 0.302 |

ATC's optimum is *still the headline value*; centralized's has moved by 4x. With
~2.5 of 10 agents active, an idle agent contributes its unchanged $\bm\theta$ to
the combine, so ATC's effective step is $\eta \cdot n_\text{active}/N \approx
\eta/4$ **automatically** — exactly the reduction a smaller batch calls for.
Centralized applies the full $\eta$ however many agents held labels, so its
nominal optimum must move to compensate.

*A caveat on the per-cell argmin, and one that was tested and withdrawn.*

**The grid ceiling does not bind — checked, not assumed.** An earlier draft here
warned that the payload-matched variant's optimum sat at lr 0.2, the largest rate
then swept, in 7 of 12 cells, so its tuned errors were probably *pessimistic* and
its 0.6-decade span was "the grid's width, not the method's". The reasoning was
sound — it runs plain SGD, and momentum's $\eta/(1-\beta) = 10\eta$ means a plain
learner needs roughly 10x the nominal rate for the same effective step, which
should push its optimum past the edge — but the conclusion was wrong.

The grid was extended to **lr 0.5 and 1.0** (192 further cells, 432 per tag) to
settle it. **Neither new rate wins a single cell for any method.** The optima
stay at 0.2 in 7 cells and 0.05 in 5, and every reported number in §9.2 and §9.3
is unchanged to four decimals. So:

- the payload costs are **not** pessimistic; they are measured;
- the 0.6-decade span is the **method's**, and survives a grid 0.7 decades wider
  — the payload-matched variant genuinely has the narrowest optimum range here,
  which the ceiling argument had dismissed as an artefact;
- centralized's 1.9-decade span is likewise no longer a lower bound. It still
  picks 0.2 at $n{=}8$, $\pi \in \{0.5, 1.0\}$, but 0.2 is now interior.

The $10\eta$ heuristic was not wrong so much as over-read: ATC's optima run
0.005–0.05, so it predicts a band of 0.05–0.5, and the plain variant's 0.05–0.2
sits inside it — at the low end. Predicting a *band* was fine; reading the top of
that band as "at or past the edge" was the error.

**The identity of the best lr in one cell is barely determined.** At
$n{=}1,\pi{=}0.25$ the grid picks plain SGD at lr 0.02 (0.191) for centralized
while a finer sweep picked momentum at lr 0.0025 (0.193) — two optimizer arms
within 0.002 of each other, inside the ±0.023 seed spread. The penalties (0.183
against 0.028) are an order of magnitude clear of that noise; the argmin itself
is not.

The two baselines fail in different places, which is itself informative.
Centralized is worst in the sparse corner, where its pooled batch shrinks and its
step size is far too large for it. `local_only` is worst at $\pi_\text{lab}=1$,
$n=1$, where it updates every step on a single sample and the accumulated
momentum has nothing damping it.

### 9.4 What X4 answers regardless

The two robust readings do not depend on the cross-method comparison:

**Sparsity bites hardest on the lone agent.** `local_only` runs from 0.127
($n{=}8$, $\pi{=}1$) to 0.585 ($n{=}1$, $\pi{=}0.25$) — a factor of 4.6. ATC moves
only 0.086 to 0.172, a factor of 2.

**The cooperation gap is a strong function of both axes**, from 0.041 in the
densest cell to 0.413 in the sparsest — a tenfold range. Combined with §2.2's
finding that it collapses sixfold in $n$ alone, this settles that sparsity is not
a nuisance axis: it substantially determines the answer to Q2.

---

## 10. X6 — non-IID (Q2)

Dirichlet label skew, $T = 1500$, three seeds, shard sizes held equal so only the
*composition* varies.

| $\beta$ | centralized | ATC | ATC (payload-matched) | local only | coop gap | payload cost |
|---|---|---|---|---|---|---|
| 0.1 (strong skew) | 0.0779 | 0.1025 | 0.1171 | **0.6292** | **0.527** | 0.015 |
| 1.0 (mild) | 0.0769 | 0.0799 | 0.0925 | 0.2727 | 0.193 | 0.013 |
| 100 (~IID) | 0.0795 | 0.0804 | 0.0951 | 0.1420 | 0.062 | 0.015 |

**The payload cost is flat across skew** — 0.015, 0.013, 0.015 at five seeds,
with no trend at all. Halving the message from $2p$ to $p$ costs the same whether
agents see the same classes or very different ones. So Diff-EKF's competitor sits
at 0.1171 even in the hardest regime: only 0.015 behind the $2p$ variant, and
still 0.51 ahead of a lone agent.

**Contrast this with sparsity, where the payload cost is *not* flat** (§9.2): it
grows about 2.5x as $\pi_\text{lab}$ falls. Skew changes *what* each agent sees;
sparsity changes *how often* it updates — and momentum, which is what the extra
$p$ buys, only compensates for the latter.

**This is the clearest answer to Q2 in the whole benchmark.** The cooperation gap
runs 0.062 → 0.527, an **8.5x increase** from IID to strong skew. Under
$\beta = 0.1$ an agent nominally touches 5.5 of the ten digits but has
meaningful data on about three (perplexity 2.9; `environment.md` §3), so alone it cannot learn
the rest at all — `local_only` sits at 0.63, near chance — while the same agent
inside a diffusion network reaches 0.103.

**Diffusion recovers nearly all of it.** ATC is 0.026 behind centralized at
$\beta = 0.1$, against 0.001 at IID — so skew does cost diffusion something, but
it closes 0.49 of the 0.54 that separates local-only from pooled data.

**Centralized is flat across $\beta$** (0.080, 0.078, 0.082), exactly as it should
be: it pools every agent's samples, so the partition is invisible to it. A free
correctness check that the skew is applied to the *partition* and not leaking
into the data path.

---

## 11. X7 — forgetting under a revisiting schedule (Q3)

Sinusoidal rotation, amplitude 30°, period 500, $T = 1500$ (three full periods),
five seeds. The `backward` evalset is enabled explicitly — it is not in the
default set, and without it this experiment measures nothing it exists for.

**The only schedule where forgetting is a well-posed question.** Under `linear`
the rotation never returns, so a backward probe asks about a state the model will
never face again; under `stationary` there is no earlier state at all. Measured
coverage of the probe: **97 %** of evaluation steps here, 67 % under linear, 0 %
under stationary.

### 11.1 The instantaneous gap is dominated by phase, not by forgetting

The raw backward-minus-current gap **oscillates with the drift phase**, swinging
roughly ±0.05, and the answer you get depends entirely on how much of a period
the averaging window covers:

| window | periods covered | centralized | ATC | local only |
|---|---|---|---|---|
| $[1400,1500)$ | 0.2 | **+0.0161** | +0.0160 | +0.0056 |
| $[1250,1500)$ | 0.5 | −0.0029 | −0.0033 | −0.0164 |
| $[1000,1500)$ | 1.0 | **−0.0035** | −0.0038 | −0.0194 |
| $[500,1500)$ | 2.0 | −0.0033 | −0.0036 | −0.0197 |

**The sign flips.** A fifth of a period gives $+0.016$; a whole period gives
$-0.0035$. At one and two periods the estimate is stable, so **any scalar summary
of forgetting must average over a whole number of periods** — a constraint that
did not arise for any other experiment, because no other schedule is periodic.
F10 draws the cycle mean as a dashed line for exactly this reason: without it the
peaks read as forgetting.

*Not a difficulty artefact.* $e^\star$ varies only 0.0075 across the whole
$\pm30°$ range, far too little to explain a 0.02 swing.

### 11.2 Nobody forgets; the lone agent lags

Over one full period, paired within seed:

| learner | gap | ±1 s.d. | $\sigma$ |
|---|---|---|---|
| `centralized_sgd` | −0.0035 | 0.0025 | 1.4 |
| `diffusion_sgd_atc` | −0.0038 | 0.0024 | 1.6 |
| ATC (payload-matched) | −0.0020 | 0.0019 | 1.1 |
| `local_only` | **−0.0194** | 0.0019 | **10.3** |

**No method shows measurable forgetting.** The three cooperative learners sit at
1.1–1.6 σ from zero — consistent with none.

**`local_only` is significantly *negative* at 10 σ**, which is the interesting
result. It is **better on a rotation it has left than on the one in front of it**.
That is not forgetting; it is **lag**. A slow learner's parameters trail the
world, so they fit where the world *was*. Its absolute error is far worse
everywhere (0.181 against 0.105), so this is not an advantage — it is the
signature of a method that cannot keep up.

**What this means for phase 5.** Forgetting is *not* currently a problem any
method has, so a filter that manages plasticity well cannot win by fixing one.
What X7 does provide is a clean instrument and a baseline of "no forgetting, and
lag at 10 σ for the uncooperative method" — so if the Diff-EKF's forgetting
factor λ trades tracking against retention, this is where that trade becomes
visible.

---

## 12. What is not yet measured

| | |
|---|---|
| X3 topology sweep | phase 4, with lr **re-tuned per topology** — a denser graph averages more and may tolerate a larger step, so holding the ring's lr fixed would confound F3's connectivity axis |
| X4 sparsity sweep | phase 4; the $(n, \pi_\text{lab})$ axes, which §2.2 shows matter more than expected |
| X6 non-IID | phase 4 |
| Diff-EKF | phase 5; its competitor is `atc_plain` at 0.1133, not momentum ATC |
