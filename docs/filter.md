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

The filter never represents $\bm\theta_t$. It represents a Gaussian belief about
it: a mean $\bm m_{t|s}$ and covariance $\bm P_{t|s}$, where the first index is
the time of the *state* and the second the time up to which *data* have been
used (eq 10). Two instances matter per step: $(\bm m_{t-1|t-1},\bm P_{t-1|t-1})$
left by the previous update, and $(\bm m_{t|t-1},\bm P_{t|t-1})$ after the
prediction step but before the new data.

The parameters are a slowly varying latent state (eq 9):

$$\bm\theta_t = \bm F_t\bm\theta_{t-1} + \bm w_t,\qquad \bm w_t\sim\mathcal N(\bm 0,\bm Q_t)$$

All the nonlinearity is in the observation map; the dynamics are
linear-Gaussian.

## 2. Two state models, and why both are implemented

### The γ family — `transition: scalar`

$\bm F_t=\gamma\bm I$ with $0<\gamma\le1$ (eq 11–12):

$$\bm m_{t|t-1}=\gamma\,\bm m_{t-1|t-1},\qquad \bm P_{t|t-1}=\gamma^{2}\bm P_{t-1|t-1}+\bm Q_t$$

**γ is not a forgetting factor**, and the paper is emphatic about this.
Forgetting means loosening the prior so new data count for relatively more, and
$\gamma^2\le1$ *contracts* the covariance. All the loosening comes from
$\bm Q_t$, with γ working against it. What γ actually does is shrink the mean
toward the origin — $L_2$ weight decay in state-space form.

**So the γ family requires $\bm Q_t\succ\bm 0$.** With $\bm Q_t=\bm 0$ the
covariance contracts monotonically and the filter stops learning (remark on
covariance collapse).

### The λ family — `transition: identity`

$\bm F_t=\bm I$, $\bm Q_t=\bm 0$, and the second moment is inflated directly
(eq 13–14):

$$\bm m_{t|t-1}=\bm m_{t-1|t-1},\qquad \bm P_{t|t-1}=\lambda^{-1}\bm P_{t-1|t-1},\qquad 0<\lambda\le1$$

Now λ governs adaptivity alone. The mean passing through is **not** the estimate
freezing: it is the correct one-step forecast for a driftless random walk, since
$\mathbb E[\bm w_t]=\bm 0$. The estimate moves in the *measurement* update.
Between two data points the best guess is unchanged while confidence decays —
the parameters are believed to drift, but in no known direction, so the mean
cannot anticipate it and only the covariance records that it happened.

### Why they are not the same, and what γ = 1 buys

The families differ in **two** ways, and γ = 1 separates them:

| | mean | covariance loosening |
|---|---|---|
| γ < 1 | shrunk toward origin | **additive** ($+\bm Q$) |
| **γ = 1** (random walk) | unchanged | **additive** ($+\bm Q$) |
| λ < 1 | unchanged | **multiplicative** ($\lambda^{-1}\times$) |

So γ = 1 is *not* a third model — it is the boundary of the γ grid, and
`transition: identity` is exactly how the config expresses it. Including it in
the sweep gives two clean comparisons for free:

* **γ = 1 against γ < 1** — does shrinking the mean help? Isolated from any
  question about adaptivity.
* **γ = 1 against λ** — additive against multiplicative loosening, both on two
  hyperparameters, a matched-budget comparison.

**Exactly one of λ < 1 and $\bm Q\succ\bm 0$ may be active.** They are two
parameterisations of one effect; tuning both makes the pair unidentifiable and a
sweep would wander along a ridge rather than find an optimum.

**Two cautions about shrinking toward the origin,** worth stating because they
are specific to networks. The origin is where a DNN computes approximately the
constant zero map, so γ < 1 pulls toward a degenerate model rather than an
uninformative one. And under the positive-rescaling symmetry of ReLU layers or
the scale invariance of LayerNorm, multiplying all weights by γ can leave the
computed function almost unchanged while moving $\bm\theta$ a long way — so the
strength of the shrinkage is not well defined as an operation on the function.

## 3. The observation model

MNIST classification is case (b) of eq 21. The network outputs logits
$\bm h_{v,t}$; then

$$\bm\pi=\operatorname{softmax}(\bm h),\qquad \bm\nu=\bm y-\bm\pi,\qquad \bm\Lambda=\operatorname{diag}(\bm\pi)-\bm\pi\bm\pi^{\top}$$

with $\bm y$ one-hot. $\bm\Lambda\succeq\bm 0$ and $\bm\Lambda\bm 1=\bm 0$, so
its rank is at most $K-1 = 9$ — that singularity encodes the shift invariance of
the softmax and is a feature, not a defect.

**Not a Gaussian surrogate.** Writing $\bm y=\operatorname{softmax}(\bm h)+\bm\varepsilon$
with Gaussian $\bm\varepsilon$ and reusing the regression machinery fails three
ways: the residual is bounded and cannot be Gaussian; its entries sum to zero so
its covariance is singular and $\bm R^{-1}$ does not exist, making the
information increment *undefined* rather than approximate; and its variance is a
deterministic function of $\bm\pi$, so a free $\bm R$ discards known
heteroscedasticity. The exponential-family form avoids all three, and the update
it produces is the GGN step for cross-entropy.

**The Gaussian assumption lives in parameter space, not observation space.**
What is taken to be Gaussian is the belief about $\bm\theta_t$, never $\bm y$.
The update consumes only the first two conditional moments of the likelihood, so
the recursion is a Gaussian assumed-density filter.

## 4. The update

### Batching: one stacked update per step

The paper assumes one observation per agent per step; the benchmark serves
$n = 4$. They are stacked into a single update — $\bar{\bm H}$ of shape
$(nK)\times p$ with block-diagonal $\bar{\bm\Lambda}$ — which is eq 34 applied
within an agent rather than across agents.

**Not four sequential rank-9 updates.** Those are not the same operation: they
relinearise between samples, giving the filter four linearisations per step
where every SGD baseline gets one. That would flatter the filter on the axis
the comparison is about.

### Information form

Agent $v$'s linearised likelihood contributes (eq 35)

$$\Delta\bm\Omega_{v,t}=\bm H_{v,t}^{\top}\bm\Lambda_{v,t}\bm H_{v,t},\qquad \Delta\bm\xi_{v,t}=\bm H_{v,t}^{\top}\left[\bm\nu_{v,t}+\bm\Lambda_{v,t}\bm H_{v,t}\bm m_{t|t-1}\right]$$

and the centralised update is a sum (eq 36). Collecting terms gives the
innovation form (eq 37):

$$\bm m_{t|t}=\bm m_{t|t-1}+\bm P_{t|t}\sum_v \bm H_{v,t}^{\top}\bm\nu_{v,t}$$

**This is a preconditioned gradient step.** The sum is exactly the negative
gradient of the log-loss over agents, and $\bm P_{t|t}$ is a running inverse GGN
— the sense in which the EKF is the online natural gradient. Constrain $\bm P$
to diagonal and it becomes an adaptive-gradient method structurally. That is a
useful sanity check and also a caution: much of any benefit may come from the
preconditioner rather than the Bayesian interpretation, which is why an ablation
against a tuned adaptive-gradient baseline is mandatory rather than optional.

### Woodbury, and why it is not optional

The information increment is low rank: $\bm\Lambda=\bm G\bm G^{\top}$ with
$\operatorname{rank}\bm\Lambda\le K-1$, so $\Delta\bm\Omega=\bm B\bm B^{\top}$
with $\bm B=\bm H^{\top}\bm G\in\mathbb R^{p\times q'}$ (eq 38). Stacked over
$N=10$ agents and $n=4$ samples the total rank is $Nn(K-1)=360$ against
$p = 2908$.

Inverting $\bm P^{-1}$ directly is $O(p^3)\approx2.5\times10^{10}$ flops per
step, about an hour of pure inversion per seed. Woodbury inverts a
$360\times360$ instead and leaves $O(p^2q')$ as the dominant term. Same answer,
three orders of magnitude cheaper.

## 5. Numerical care

The recursion loses positive definiteness easily, and three defences are used
together:

* **float64.** In single precision the paper reports PD lost within a few
  hundred steps; the benchmark's runs are 1500. At $p=2908$ a covariance is
  68 MB in float64, which is affordable for one belief.
* **Symmetrise every step**: $\bm P\leftarrow\tfrac12(\bm P+\bm P^{\top})$.
  Without it the recursion drifts out of symmetry within a few hundred steps.
* **A per-step $O(p)$ guard** on the mean staying finite and the variances
  staying positive, so divergence stops the run instead of reaching the metrics
  as NaN.

**The Joseph form is not among them, and the reason is worth stating.** Written
the usual way,

$$\bm P^+ = (\bm I-\bm K\bar{\bm B}^{\top})\,\bm P\,
            (\bm I-\bm K\bar{\bm B}^{\top})^{\top} + \bm K\bm K^{\top},$$

it forms the $p\times p$ matrix $\bm I-\bm K\bar{\bm B}^{\top}$ and multiplies it
by $\bm P$ — which is $O(p^3)$, exactly the cost §4 used Woodbury to avoid.
Expanding the product instead keeps every term $O(p^2q')$, but the expansion
telescopes:

$$\bm P - \bm K\bm A^{\top} - \bm A\bm K^{\top} + \bm K\bm S\bm K^{\top}
  = \bm P - \bm A\bm S^{-1}\bm A^{\top},$$

with $\bm A=\bm P\bar{\bm B}$ and $\bm K=\bm A\bm S^{-1}$ — the short form again.
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
dominated by exactly the operations CUDA wins.

## 6. Hyperparameters

| | γ family | λ family |
|---|---|---|
| `transition` | `scalar` | `identity` |
| `gamma` | swept in $(0,1]$ | 1 (forced) |
| `lambda_forget` | 1 | swept in $(0,1)$ |
| `process_noise_q` | swept, $>0$ | 0 |
| `prior_scale` | swept | swept |

$\bm P_0=\sigma_0^2\bm I$ acts as an initial learning rate and the paper calls
it **the most sensitive hyperparameter of the method**. It is swept exactly as
the SGD learning rates were, because D39 exists precisely because an untuned
comparison invalidated a headline once already.

## 7. What this reports that no SGD baseline can

The predictive covariance (eq 47):

$$\bm\Sigma^{\mathrm{pred}}_{v,t}\approx\bm H_{v,t}\bm P_{t|t}\bm H_{v,t}^{\top}+\bm R_{v,t}$$

Logged from the first implementation alongside the ECE, Brier and
overconfidence the protocol already computes for every learner. It is a
delta-method approximation and **systematically over-confident**, because it
ignores both the linearisation remainder and model misspecification — so
recording it from day one makes the calibration claim measurable rather than
assertable.

## 8. What is deferred

The diffusion version (phase 5 proper) adds two axes this document does not
cover, and the centralised code is shaped so they drop in rather than require
restructuring:

* **What the combine step sends.** The mean only (eq 45, the standard diffusion
  choice, $O(p)$ per link) or the mean and covariance (eq 46, covariance
  intersection, $O(p^2)$ per link and realistic only for small or structured
  $\bm P$). Crossed with the two state models this gives the four variants.
* **The adapt scope.** Local (no communication) or one-hop (neighbours exchange
  $(\bm B_{u,t},\bm H_{u,t}^{\top}\bm\nu_{u,t})$). Complete-graph exactness —
  the filter's analogue of X0 — holds **only** for the one-hop variant, because
  the EKF gain is data-dependent where SGD's step size is fixed. That is a
  tenfold difference in communication and it decides which row of the
  communication ledger the phase-5 claim can be stated in. See open question Q5.
