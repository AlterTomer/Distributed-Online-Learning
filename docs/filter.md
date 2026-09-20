# The filter

The centralised EKF: what it computes, which choices were made and why, and
what is deliberately left to the diffusion version.

Its own document rather than a section of `learners.md` because the four SGD
methods share one update rule and differ in what they transmit, while the filter
differs from all of them in *kind* — it carries a covariance, its step size is
derived rather than tuned, and its correctness argument is about moments rather
than about averaging commuting with a linear map.

Source of record: `Distributed_Online_Bayesian_Learning_DNN_DEKF.tex`. Equation
numbers below refer to it. If this document and the code disagree, the code is
right and this is stale.

---

## 1. What is being estimated

The filter never represents $\boldsymbol\theta_t$. It represents a Gaussian belief about
it: a mean $\boldsymbol m_{t|s}$ and covariance $\boldsymbol P_{t|s}$, where the first index is
the time of the *state* and the second the time up to which *data* have been
used (eq 10). Two instances matter per step: $(\boldsymbol m_{t-1|t-1},\boldsymbol P_{t-1|t-1})$
left by the previous update, and $(\boldsymbol m_{t|t-1},\boldsymbol P_{t|t-1})$ after the
prediction step but before the new data.

The parameters are a slowly varying latent state (eq 9):

$$\boldsymbol\theta_t = \boldsymbol F_t\boldsymbol\theta_{t-1} + \boldsymbol w_t,\qquad \boldsymbol w_t\sim\mathcal N(\boldsymbol 0,\boldsymbol Q_t)$$

All the nonlinearity is in the observation map; the dynamics are
linear-Gaussian.

## 2. Two state models, and why both are implemented

### The γ family — `transition: scalar`

$\boldsymbol F_t=\gamma\boldsymbol I$ with $0<\gamma\le1$ (eq 11–12):

$$\boldsymbol m_{t|t-1}=\gamma\,\boldsymbol m_{t-1|t-1},\qquad \boldsymbol P_{t|t-1}=\gamma^{2}\boldsymbol P_{t-1|t-1}+\boldsymbol Q_t$$

**γ is not a forgetting factor**, and the paper is emphatic about this.
Forgetting means loosening the prior so new data count for relatively more, and
$\gamma^2\le1$ *contracts* the covariance. All the loosening comes from
$\boldsymbol Q_t$, with γ working against it. What γ actually does is shrink the mean
toward the origin — $L_2$ weight decay in state-space form.

**So the γ family requires $\boldsymbol Q_t\succ\boldsymbol 0$.** With $\boldsymbol Q_t=\boldsymbol 0$ the
covariance contracts monotonically and the filter stops learning (remark on
covariance collapse).

### The λ family — `transition: identity`

$\boldsymbol F_t=\boldsymbol I$, $\boldsymbol Q_t=\boldsymbol 0$, and the second moment is inflated directly
(eq 13–14):

$$\boldsymbol m_{t|t-1}=\boldsymbol m_{t-1|t-1},\qquad \boldsymbol P_{t|t-1}=\lambda^{-1}\boldsymbol P_{t-1|t-1},\qquad 0<\lambda\le1$$

Now λ governs adaptivity alone. The mean passing through is **not** the estimate
freezing: it is the correct one-step forecast for a driftless random walk, since
$\mathbb E[\boldsymbol w_t]=\boldsymbol 0$. The estimate moves in the *measurement* update.
Between two data points the best guess is unchanged while confidence decays —
the parameters are believed to drift, but in no known direction, so the mean
cannot anticipate it and only the covariance records that it happened.

### Why they are not the same, and what γ = 1 buys

The families differ in **two** ways, and γ = 1 separates them:

| | mean | covariance loosening |
|---|---|---|
| γ < 1 | shrunk toward origin | **additive** ($+\boldsymbol Q$) |
| **γ = 1** (random walk) | unchanged | **additive** ($+\boldsymbol Q$) |
| λ < 1 | unchanged | **multiplicative** ($\lambda^{-1}\times$) |

So γ = 1 is *not* a third model — it is the boundary of the γ grid, and
`transition: identity` is exactly how the config expresses it. Including it in
the sweep gives two clean comparisons for free:

* **γ = 1 against γ < 1** — does shrinking the mean help? Isolated from any
  question about adaptivity.
* **γ = 1 against λ** — additive against multiplicative loosening, both on two
  hyperparameters, a matched-budget comparison.

**Exactly one of λ < 1 and $\boldsymbol Q\succ\boldsymbol 0$ may be active.** They are two
parameterisations of one effect; tuning both makes the pair unidentifiable and a
sweep would wander along a ridge rather than find an optimum.

**Two cautions about shrinking toward the origin,** worth stating because they
are specific to networks. The origin is where a DNN computes approximately the
constant zero map, so γ < 1 pulls toward a degenerate model rather than an
uninformative one. And under the positive-rescaling symmetry of ReLU layers or
the scale invariance of LayerNorm, multiplying all weights by γ can leave the
computed function almost unchanged while moving $\boldsymbol\theta$ a long way — so the
strength of the shrinkage is not well defined as an operation on the function.

## 3. The observation model

MNIST classification is case (b) of eq 21. The network outputs logits
$\boldsymbol h_{v,t}$; then

$$\boldsymbol\pi=\mathrm{softmax}(\boldsymbol h),\qquad \boldsymbol\nu=\boldsymbol y-\boldsymbol\pi,\qquad \boldsymbol\Lambda=\mathrm{diag}(\boldsymbol\pi)-\boldsymbol\pi\boldsymbol\pi^{\top}$$

with $\boldsymbol y$ one-hot. $\boldsymbol\Lambda\succeq\boldsymbol 0$ and $\boldsymbol\Lambda\boldsymbol 1=\boldsymbol 0$, so
its rank is at most $K-1 = 9$ — that singularity encodes the shift invariance of
the softmax and is a feature, not a defect.

**Not a Gaussian surrogate.** Writing $\boldsymbol y=\mathrm{softmax}(\boldsymbol h)+\boldsymbol\varepsilon$
with Gaussian $\boldsymbol\varepsilon$ and reusing the regression machinery fails three
ways: the residual is bounded and cannot be Gaussian; its entries sum to zero so
its covariance is singular and $\boldsymbol R^{-1}$ does not exist, making the
information increment *undefined* rather than approximate; and its variance is a
deterministic function of $\boldsymbol\pi$, so a free $\boldsymbol R$ discards known
heteroscedasticity. The exponential-family form avoids all three, and the update
it produces is the GGN step for cross-entropy.

**The Gaussian assumption lives in parameter space, not observation space.**
What is taken to be Gaussian is the belief about $\boldsymbol\theta_t$, never $\boldsymbol y$.
The update consumes only the first two conditional moments of the likelihood, so
the recursion is a Gaussian assumed-density filter.

## 4. The update

### Batching: one stacked update per step

The paper assumes one observation per agent per step; the benchmark serves
$n = 4$. They are stacked into a single update — $\bar{\boldsymbol H}$ of shape
$(nK)\times p$ with block-diagonal $\bar{\boldsymbol\Lambda}$ — which is eq 34 applied
within an agent rather than across agents.

**Not four sequential rank-9 updates.** Those are not the same operation: they
relinearise between samples, giving the filter four linearisations per step
where every SGD baseline gets one. That would flatter the filter on the axis
the comparison is about.

### Information form

Agent $v$'s linearised likelihood contributes (eq 35)

$$\Delta\boldsymbol\Omega_{v,t}=\boldsymbol H_{v,t}^{\top}\boldsymbol\Lambda_{v,t}\boldsymbol H_{v,t},\qquad \Delta\boldsymbol\xi_{v,t}=\boldsymbol H_{v,t}^{\top}\left[\boldsymbol\nu_{v,t}+\boldsymbol\Lambda_{v,t}\boldsymbol H_{v,t}\boldsymbol m_{t|t-1}\right]$$

and the centralised update is a sum (eq 36). Collecting terms gives the
innovation form (eq 37):

$$\boldsymbol m_{t|t}=\boldsymbol m_{t|t-1}+\boldsymbol P_{t|t}\sum_v \boldsymbol H_{v,t}^{\top}\boldsymbol\nu_{v,t}$$

**This is a preconditioned gradient step.** The sum is exactly the negative
gradient of the log-loss over agents, and $\boldsymbol P_{t|t}$ is a running inverse GGN
— the sense in which the EKF is the online natural gradient. Constrain $\boldsymbol P$
to diagonal and it becomes an adaptive-gradient method structurally. That is a
useful sanity check and also a caution: much of any benefit may come from the
preconditioner rather than the Bayesian interpretation, which is why an ablation
against a tuned adaptive-gradient baseline is mandatory rather than optional.

### Woodbury, and why it is not optional

The information increment is low rank: $\boldsymbol\Lambda=\boldsymbol G\boldsymbol G^{\top}$ with
$\mathrm{rank}\boldsymbol\Lambda\le K-1$, so $\Delta\boldsymbol\Omega=\boldsymbol B\boldsymbol B^{\top}$
with $\boldsymbol B=\boldsymbol H^{\top}\boldsymbol G\in\mathbb R^{p\times q'}$ (eq 38). Stacked over
$N=10$ agents and $n=4$ samples the total rank is $Nn(K-1)=360$ against
$p = 2908$.

Inverting $\boldsymbol P^{-1}$ directly is $O(p^3)\approx2.5\times10^{10}$ flops per
step, about an hour of pure inversion per seed. Woodbury inverts a
$360\times360$ instead and leaves $O(p^2q')$ as the dominant term. Same answer,
three orders of magnitude cheaper.

## 5. Numerical care

The recursion loses positive definiteness easily, and three defences are used
together:

* **float64.** In single precision the paper reports PD lost within a few
  hundred steps; the benchmark's runs are 1500. At $p=2908$ a covariance is
  68 MB in float64, which is affordable for one belief.
* **Symmetrise every step**: $\boldsymbol P\leftarrow\tfrac12(\boldsymbol P+\boldsymbol P^{\top})$.
  Without it the recursion drifts out of symmetry within a few hundred steps.
* **A per-step $O(p)$ guard** on the mean staying finite and the variances
  staying positive, so divergence stops the run instead of reaching the metrics
  as NaN.

**The Joseph form is not among them, and the reason is worth stating.** Written
the usual way,

$$\boldsymbol P^+ = (\boldsymbol I-\boldsymbol K\bar{\boldsymbol B}^{\top})\,\boldsymbol P\,
            (\boldsymbol I-\boldsymbol K\bar{\boldsymbol B}^{\top})^{\top} + \boldsymbol K\boldsymbol K^{\top},$$

it forms the $p\times p$ matrix $\boldsymbol I-\boldsymbol K\bar{\boldsymbol B}^{\top}$ and multiplies it
by $\boldsymbol P$ — which is $O(p^3)$, exactly the cost §4 used Woodbury to avoid.
Expanding the product instead keeps every term $O(p^2q')$, but the expansion
telescopes:

$$\boldsymbol P - \boldsymbol K\boldsymbol A^{\top} - \boldsymbol A\boldsymbol K^{\top} + \boldsymbol K\boldsymbol S\boldsymbol K^{\top}
  = \boldsymbol P - \boldsymbol A\boldsymbol S^{-1}\boldsymbol A^{\top},$$

with $\boldsymbol A=\boldsymbol P\bar{\boldsymbol B}$ and $\boldsymbol K=\boldsymbol A\boldsymbol S^{-1}$ — the short form again.
That is not a coincidence; the two are algebraically identical and always were.
The Joseph form's value is **numerical**, and it comes precisely from evaluating
the un-expanded product, which is the version that costs $O(p^3)$. So the choice
here is not "Joseph or short form" but "Joseph or Woodbury", and at $p=2908$ over
1500 steps the cheap form plus float64 plus symmetrisation is what makes the
sweep affordable. Positive definiteness then becomes an empirical claim rather
than a structural one — which is why it is soaked over a full-length run rather
than asserted (design note D62).

With these, a loss of positive definiteness means a genuine bug rather than
accumulated rounding — which is the point of paying for float64.

**CUDA.** Design note D43 measured CUDA 0.69× on the SGD path at this model size
and 14× on dense covariance operations, and closed `run.device` to `cpu` with
the note that phase 5 would reopen it. This is that moment: the filter is
dominated by exactly the operations CUDA wins. Measured over a full 1500-step
run at $p=2908$: **561s per seed on CPU against 123s on CUDA**, bit-identical.
Opening the device exposed three tensors built without one — the optimizer's
momentum buffers, the mixing matrix, and the empty batch an idle agent returns.
The last would have failed only on steps where no agent had a label.

### 5.1 What the checks actually assert

Every claim above is tested rather than argued, and two of the tests earn their
keep by having failed:

| check | tolerance | what it caught |
|---|---|---|
| Woodbury vs a direct $p\times p$ information-form inverse | 1.6e-15 | — |
| linear probe + Gaussian vs an exact Kalman filter, in **gain** form | 2.3e-14 mean, 3.6e-15 covariance | the score/innovation bug (D60) |
| Cholesky over 1500 steps at $p=2908$, five hyperparameter corners | holds throughout | — |
| $\boldsymbol P-\boldsymbol P^+\succeq\boldsymbol0$ | $>-10^{-9}$ | — |
| $\gamma=1$ vs `transition: identity` | bitwise equal | — |
| $\boldsymbol\Lambda=\mathrm{Cov}(\boldsymbol s)$, sampled | 5e-3 at $2\times10^5$ draws | — |

The exactness check is the filter's analogue of X0, and it is worth being
precise about why it has teeth. A linear probe makes $\boldsymbol h(\boldsymbol\theta)=\boldsymbol
H\boldsymbol\theta$ **exactly**, so the linearisation has no remainder; with Gaussian
observations the model is precisely the one the Kalman filter is derived for, and
the two must agree to floating point. The reference is written in the *gain* form
$\boldsymbol K=\boldsymbol P\boldsymbol H^\top\boldsymbol S^{-1}$, which shares no algebra with the Woodbury
update — so agreement means both are right, not that one echoes the other.

### 5.2 $\sigma_0^2$ is a trust region

The mean update is a Gauss–Newton step and $\boldsymbol P$ bounds its size, so too large
a prior does not converge slowly — it **diverges**, on the first step, while the
covariance stays perfectly well conditioned. Measured at $p=2908$:

| $\sigma_0^2$ | first $\lVert\Delta\boldsymbol m\rVert$ | ÷ $\lVert\boldsymbol\theta_0\rVert$ | outcome |
|---|---|---|---|
| 1.0   | 7.91  | 1.29  | diverges by step 25; $10^{113}$ by step 100 |
| 0.1   | 1.51  | 0.25  | stable |
| 0.01  | 0.26  | 0.042 | stable |
| 0.001 | 0.041 | 0.007 | stable |

So the sweep is bounded above at $10^{-1}$, and the filter carries an $O(p)$
per-step guard that raises on a non-finite mean or a non-positive variance. A
diverged cell is then reported as diverged rather than averaging into a seed mean
as NaN (design note D61).

## 6. Hyperparameters

| | γ family | λ family |
|---|---|---|
| `transition` | `scalar` | `identity` |
| `gamma` | swept in $(0,1]$ | 1 (forced) |
| `lambda_forget` | 1 | swept in $(0,1)$ |
| `process_noise_q` | swept, $>0$ | 0 |
| `prior_scale` | swept | swept |

$\boldsymbol P_0=\sigma_0^2\boldsymbol I$ acts as an initial learning rate and the paper calls
it **the most sensitive hyperparameter of the method**. It is swept exactly as
the SGD learning rates were, because D39 exists precisely because an untuned
comparison invalidated a headline once already.

## 7. What this reports that no SGD baseline can

The predictive covariance (eq 47):

$$\boldsymbol\Sigma^{\mathrm{pred}}_{v,t}\approx\boldsymbol H_{v,t}\boldsymbol P_{t|t}\boldsymbol H_{v,t}^{\top}+\boldsymbol R_{v,t}$$

Logged from the first implementation alongside the ECE, Brier and
overconfidence the protocol already computes for every learner. It is a
delta-method approximation and **systematically over-confident**, because it
ignores both the linearisation remainder and model misspecification — so
recording it from day one makes the calibration claim measurable rather than
assertable.

## 8. The diffusion version

Built, in `learners/diffusion_ekf.py`. It shares this document's `information_pair`
and `woodbury_update` rather than reimplementing them, so the complete-graph
identity below is an identity *within* one implementation rather than an agreement
between two — which is what makes it usable as a correctness gate.

Two independent axes, each pinned by a learner name so a config cannot contradict
the variant it asked for:

* **What the combine step sends.** `diffusion_ekf` sends the mean only (eq 45,
  the standard diffusion choice, $O(p)$ per link — 23 kB); `diffusion_ekf_full`
  sends the mean and the covariance (eq 46, $O(p^2)$ — 68 MB). The second is
  undeployable and is measured first anyway: the two are not competing designs to
  choose between on cost, they are an upper bound and a candidate, and the gap
  between them is the price of not shipping covariances.

  ⚠ Eq 46 is **not** covariance intersection, and an earlier version of this
  document and of the research note both said it was. CI combines in the
  *information* domain, $\boldsymbol P^{-1}=\sum_i\omega_i\boldsymbol P_i^{-1}$, and weights the
  fused mean by the information matrices; eq 46 averages covariances under the
  same $a_{vu}$ that weight the mean. The conservativeness CI was being cited for
  holds anyway and is proved directly — a convex combination of consistent
  covariances bounds the covariance of the convexly combined estimate for *any*
  cross-correlation, by Jensen, with equality when the errors coincide. CI would
  give a strictly tighter bound (harmonic below arithmetic) at the cost of an
  inverse per fusion.

* **The adapt scope.** `local` (no communication) or `one_hop` (neighbours
  exchange their raw labelled batches, $n(d+1)$ scalars, and each receiver
  rebuilds $(\boldsymbol B_{u,t},\boldsymbol H_{u,t}^{\top}\boldsymbol s_{u,t})$ itself — 32× cheaper
  than shipping the pair, D92).
  Complete-graph exactness — the filter's analogue of X0 — holds **only** for
  one-hop, because on $K_N$ that makes the measurement set the whole vertex set,
  which is the hypothesis of the exactness proposition. Under a local adapt each
  agent updates on its own data and the combine averages covariances, which is
  not the same as summing information, so the two filters differ even on a
  complete graph.

  `diffusion_ekf_onehop` therefore exists as a **fixture, not a competitor**: it
  is not tuned and not swept. `tests/test_learners.py` asserts both halves — that
  one-hop reproduces the centralised filter to 1e-10, and that local does not —
  so the positive test cannot pass vacuously. What one-hop buys on a *sparse*
  graph was measured from X20 on; see `docs/results.md`.

* **The linearisation point** (one-hop only; D93, D94). A receiver can rebuild
  a neighbour's block at the *sender's* predictive mean $\boldsymbol\theta_{u,t}^-$ —
  `sender`, what X20–X26 ran, which sums blocks from different points into one
  update and needs $\boldsymbol\theta_{u,t}^-$ in the first message, 6 604 scalars per
  link per direction — or at its *own* $\boldsymbol\theta_{v,t}^-$ — `receiver`, one
  point per update, the textbook diffusion EKF, and 3 696. They coincide
  whenever the agents' predictive means agree, so on a complete graph both pass
  the exactness gate and only a sparse graph separates them. The `*_receiver`
  learner names select it.

**Memory, not compute, is what binds.** Each agent holds a $p\times p$
covariance: 64.5 MiB at $p=2908$ in float64, so ten agents cost 645 MiB, and full
sharing needs a second set live during the mix because every $\boldsymbol P^{\psi}_u$ must
survive until the last $\boldsymbol P_{v,t|t}$ is written. A measured run carrying both
diffusion variants and the centralised filter peaked at **3.3 GiB**. Compute is
**not** centralised-equal, as this section used to say: measured per agent per
step (RTX 4070, float64; D93), a local adapt takes 14.0 ms against 64.7 ms for
one pooled centralised update, so ten agents cost about twice the centralised
filter — small Woodbury blocks underuse the GPU — and one-hop costs 2.04× a local
adapt.
