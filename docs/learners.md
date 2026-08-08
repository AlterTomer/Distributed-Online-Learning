# The learning methods

Every method here is **adapt, then combine**: use local data with no
communication, then exchange with one-hop neighbours. That split is the whole
architecture. It exists so the phase-5 filter can differ from diffusion SGD *in
the adapt step alone* — with a single `step()` the filter would not fit, and the
interface would have to be rewritten at the point where rewriting is most
expensive.

If this document and the code disagree, the code is right and this is stale.

---

## 1. The four methods

| learner | adapt sees | combine | per link |
|---|---|---|---|
| `centralized_sgd` | the pooled batch $\bigcup_v \mathcal D_t^v$ | nothing | — |
| `diffusion_sgd_atc` | agent $v$'s own batch | average the $\bm\psi$ | $2p$ |
| `diffusion_sgd_atc_plain` | agent $v$'s own batch | average the $\bm\psi$ | $p$ |
| `diffusion_sgd_cta` | agent $v$'s own batch | average *before* the gradient | $2p$ |
| `local_only` | agent $v$'s own batch | nothing | 0 |

Everything else — the gradient, the update rule, the state container — is
shared. A difference in the results is therefore a difference in the *method*,
not in two implementations that drifted apart.

### `centralized_sgd` — the upper reference

Sees the pooled batch every step and takes one optimizer step. Every agent holds
the same $\bm\theta$ by construction, so `combine` is a no-op and
$E_\text{agree}$ is identically zero.

It is an upper reference **for the online setting**, not a deployable
competitor: pooling means shipping every sample to a centre, which is precisely
what the paradigm exists to avoid. On F2 it appears as a horizontal line rather
than a point on the communication axis (design note D30).

### `diffusion_sgd_atc` — the primary method

$$\bm\psi_v = \bm\theta_v - \eta\nabla L(\bm\theta_v;\mathcal D_v), \qquad \bm\theta_v \leftarrow \sum_u a_{vu}\bm\psi_u$$

Each agent steps on its own data, then averages the *result* with its
neighbours. **Parameters are mixed, not gradients** — that is what drives the
agents toward agreement, and Olshevskyi et al. find that running consensus on
gradients instead needs roughly twice the message-passing rounds for the same
error.

ATC is primary because Diff-EKF is ATC. With both ATC, phase 5 differs in the
adapt step alone, so any advantage is attributable to the second-order update
rather than to the ordering.

### `diffusion_sgd_atc_plain` — ATC (payload-matched)

**Not a second algorithm.** The registry maps this name and `diffusion_sgd_atc`
to the *same class*, `DiffusionSGDATC`. Given the same optimizer the two are
bit-identical — measured at exactly 0.0 divergence over 40 steps. Both run

$$\bm\psi_v = \bm\theta_v - \eta\,\bm d_v, \qquad \bm\theta_v \leftarrow \sum_u a_{vu}\bm\psi_u$$

and differ only in what $\bm d_v$ is:

| | $\bm d_v$ | mixed | per link |
|---|---|---|---|
| `diffusion_sgd_atc` | $\bm m_v \leftarrow \beta\bm m_v + \bm g_v$ | $\bm\theta$ and $\bm m$ | $2p$ |
| **ATC (payload-matched)** | $\bm g_v$ | $\bm\theta$ only | $p$ |

So the payload-matched variant is **exactly the $\beta = 0$ case**. At
$\beta = 0$ the mixing choice also becomes vacuous: $\bm m_v = \bm g_v$, so
averaging the buffers cannot influence any later step. The two config knobs
collapse into one.

**Halving the message and dropping momentum are therefore the same decision,
not two.** `comm_scalars_per_step` computes $(1 + |\text{mixed}|)\,p$ per link —
you cannot mix a buffer you never transmitted (design note D29).

**Why it is carried at all.** It is one method at two points on a
communication/performance trade-off, and phase 5 needs the cheaper point:
Diff-EKF sends one $p$-vector per link, so its honest competitor is this variant
at 0.0902, not the momentum one at 0.0768. Comparing the filter against the $2p$
configuration would compare across different bandwidths.

The one thing that genuinely differs is the *tuned* learning rate — 0.20 here
against 0.01 for momentum — and that is empirical, not structural: momentum
multiplies the effective step by $1/(1-\beta) = 10$, so $0.01 \times 10 = 0.1$
sits on the same order as 0.20.

### `diffusion_sgd_cta` — the ordering comparison

$$\bm\theta_v(t{+}1) = \sum_u \bm W_{vu}\bm\theta_u(t) \;-\; \alpha_t\nabla L(\bm\theta_v(t);\mathcal D_v)$$

Eq. (17) of Olshevskyi et al. — the Nedić–Ozdaglar consensus-plus-local-gradient
form. The gradient is evaluated **before** the averaging, which is the entire
difference from ATC. Same cost, different ordering; X1b measures how much it
matters so the ATC choice is reported rather than assumed.

Implemented in the same adapt/combine shape by having `adapt` compute and stash
the gradient while emitting the *un-stepped* parameters as the message, then
applying the stashed gradient after averaging. Deferring the update is what lets
both orderings run through one runner rather than two loops that might differ
elsewhere.

### `local_only` — the lower reference

No communication at all. The gap between this and the diffusion methods **is**
the value of cooperation, and it is the cleanest answer to Q2.

---

## 2. Optimizer state, and what the combine step mixes

`mix_optimizer_state: momentum` averages $\bm m$ with the **same weights** as
$\bm\theta$, so combine is one operator on the whole learner state:

$$\begin{pmatrix}\bm\theta_v \\ \bm m_v\end{pmatrix} \leftarrow \sum_u a_{vu}\begin{pmatrix}\bm\psi_u \\ \bm m_u\end{pmatrix}$$

Every property established for $\bm A$ then covers all of it: row-stochasticity
keeps the result inside the neighbours' convex hull, double stochasticity
preserves the network average. Unmixed, $\bm\theta$ gets those guarantees and
$\bm m$ — the part that diverges — gets none.

**Why it is not optional.** Olshevskyi et al. Fig. 2a: **D-Adam**, which mixes
parameters and keeps moments local, converges then *diverges*; **D-AMSGrad**,
which runs consensus on the moments too, is their best distributed method. The
config rejects a stateful optimizer with `mix_optimizer_state: none` for that
reason.

**Mixing means transmitting.** A neighbour cannot average a buffer it was never
sent, which is why `momentum` costs $2p$ per link and plain SGD costs $p$.

| optimizer | state carried | mixed under `momentum` | per link |
|---|---|---|---|
| `sgd` | — | — | $p$ |
| `sgd_momentum` | $\bm m$ | $\bm m$ | $2p$ |
| `adamw` | $\bm m, \bm v$ | $\bm m$ (or both under `all`) | $2p$–$3p$ |

Implemented directly rather than through `torch.optim`: torch optimizers own
their state internally and expose it only as
`optimizer.state[param]['momentum_buffer']`, which makes mixing across agents
awkward — and the phase-5 filter has no torch optimizer at all, so routing
phase 3 through one would leave the interface differing between phases at
exactly the boundary that must not move.

---

## 3. The exactness check (X0)

On a complete graph with uniform weights and plain SGD, ATC is **algebraically
identical** to centralized SGD:

$$\sum_v \tfrac1N\bigl(\bm\theta - \eta\nabla L_v\bigr) = \bm\theta - \eta\,\tfrac1N\sum_v\nabla L_v$$

given a common $\bm\theta$, which the identity then preserves inductively.

**Measured: 1.7e-15** over 50 steps in float64, against a 1e-12 target. Via the
full pipeline, the two learners' error rates agree to four decimals and
$E_\text{agree} \approx 7\times10^{-31}$.

### The four preconditions, and why they are checked at run start

Each failure produces a *small, plausible, non-zero* residual rather than an
obvious break — which invites loosening the tolerance until it "passes". So
`check_exactness_preconditions` refuses to start and names the offending field.

| precondition | why | residual if broken |
|---|---|---|
| complete graph | one combine reaches full consensus | > 1e-6 |
| uniform weights | the identity needs $a_{vu} = 1/N$ | — |
| $\pi_\text{lab} = 1$ | equal batch sizes: mean-of-means = pooled mean | > 1e-12 |
| float64 | the tolerance is 1e-12 | > 1e-12 |
| plain SGD | canonical configuration — see below | *none* |

### What X0 actually certifies

Broader than plain SGD. The identity survives **any optimizer linear in the
gradients**, because averaging commutes with linear maps:

| optimizer | mixing | residual | |
|---|---|---|---|
| plain SGD | — | 9.99e-16 | exact |
| momentum $\beta{=}0.9$ | mixed | 7.22e-16 | exact |
| momentum $\beta{=}0.9$ | not mixed | 7.77e-16 | exact |
| AdamW | all | 5.24 | **breaks** |

Heavy-ball is linear — $\bm m \leftarrow \beta\bm m + \bm g$ — so
$\frac1N\sum_v\bm m_v$ *is* the centralized momentum recursion. Adam's second
moment carries $\bm g^2$, which is not.

Plain SGD is required as the **canonical** configuration so the check leans on
nothing but the diffusion algebra — not because momentum would break it (design
note D35).

**AdamW breaking is a positive control.** Without a case that fails, a test that
always passes is indistinguishable from one that checks nothing. Three more
controls exist: a ring, unequal batch sizes, and float32.

**A limit worth stating.** All of this holds on a *complete* graph, where every
agent linearises at the same $\bm\theta$. On a ring the agents differ and the
argument collapses — so it says nothing about whether mixing matters in the
experiments actually run.

---

## 4. Communication, and the pairing the phase-5 claim rests on

`WORKPLAN.md` §3.2 says diffusion SGD exchanges "one $p$-vector per link per
step". True of the payload-matched variant. But §3.4 makes the X1–X6 primary *SGD with momentum,
momentum mixed* — which sends $2p$, while Diff-EKF sends $\bm\psi$ alone.

At $N{=}10$ on a ring, $p = 2908$:

| learner | per link | scalars/step | relative |
|---|---|---|---|
| `diffusion_sgd_atc_plain` | $p$ | 58 160 | 1.0× |
| `diffusion_sgd_atc` | $2p$ | 116 320 | 2.0× |
| `diffusion_ekf` (local) | $p$ | 58 160 | 1.0× |
| `diffusion_ekf` (one_hop) | $p(q'{+}1)$ | 581 600 | 10.0× |

So **"at identical communication" is a claim about a particular pairing**. X1
runs both ATC variants, so phase 5 can state the claim against the matched
baseline while still showing the stronger one — and the difference between them
measures what the extra $p$ buys (design note D29).

**Now measured.** With both tuned at $n=4$, `atc_plain` reaches 0.1133 and the
momentum variant 0.0964. The extra $p$ per link buys **0.017** of error. That
0.1133 is the number Diff-EKF has to beat, since its claim is stated at $p$ per
link — the stronger momentum baseline is not its competitor (`results.md` §2.1).

**`one_hop` is worth understanding before phase 5.** It selects *whose*
likelihood information an agent folds into its own adapt step. Under `local` the
agent uses only its own data; under `one_hop` neighbours exchange the raw
information $(\bm B_u, \bm H_u^{\mathsf T}\bm\nu_u)$ first. Proposition 1 —
complete-graph exactness for the filter — holds only for `one_hop`, because the
EKF's gain is *data-dependent* where SGD's step size is fixed: averaging
estimates computed with different gains does not reproduce one joint update. So
the filter's X0 is a diagnostic in a configuration nobody deploys, unlike SGD's.

---

## 5. State, and what phase 5 adds

`LearnerState` is dict-like: `theta` plus whatever the method carries.

```python
state[v] = {
    "theta": Tensor(p),
    "momentum": Tensor(p),      # phase 3
    # "P": Tensor(p, p),        # phase 5 — no runner change
}
```

That is `IMPLEMENTATION.md` §13.6, and it is why the covariance can be added
without touching `simulate.py`.

Two smaller decisions that guard silent failures:

- **`init()` clones $\bm\theta_0$ per agent** rather than sharing one tensor. Sharing would make the first in-place update change every agent at once, and the run would show perfect consensus for a reason unconnected to the combine step.
- **`combine` reads every message before writing any.** The obvious loop would let agent 1 combine agent 0's *already-updated* parameters, making the result depend on node ordering and breaking X0 while still producing a plausible curve.

---

## 6. Running one

```
python scripts/run_experiment.py            # or: ... x2_rotating
```

Serial and in-process, so a breakpoint in `simulate.run` stops where you expect.
Results and a checkpoint are written every `eval_every` steps, and re-running
the same experiment **resumes** from the last completed evaluation.

Resumption is exact, not approximate: the loop consumes no randomness — the
environment is positional and every draw happened at construction — so there is
no RNG state to restore. `test_recording.py` asserts a resumed run matches an
uninterrupted one bit-for-bit rather than assuming it.
