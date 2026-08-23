# The figures

What each one is for, how to read it, and what would count as a surprise.
Produced by `scripts/make_figures.py` from logged results, with no manual steps.

If this document and the code disagree, the code is right and this is stale.

---

## 1. How to run them

```
python scripts/make_figures.py                # collect, cache, and draw all
python scripts/make_figures.py f3             # just one
python scripts/make_figures.py --from-cache   # redraw without re-reading results/
python scripts/make_figures.py --dpi 300      # publication resolution
```

Runnable from the IDE as-is: the module-level `ONLY` selects one figure, so a
breakpoint in a single panel needs no arguments.

**Where they land.** `figures/` at the repository root by default — gitignored,
because a PNG in the history goes stale the moment a run is re-tuned. Publish
somewhere else without editing anything by setting `DEKF_FIGURES_DIR`, absolute
or repo-root-relative:

```
DEKF_FIGURES_DIR="/path/to/a/shared/folder" python scripts/make_figures.py
```

Set it once in the shell profile or the IDE run configuration and every script
that writes or reads a figure follows — `make_preliminary_figures.py` too. See
`src/dekf_bench/utils/paths.py`.

| id | file | source |
|---|---|---|
| F1 | `12_f1_error_vs_time.png` | X1, X2 |
| F2 | `13_f2_error_vs_communication.png` | X1, X2 |
| F3 | `16_f3_price_of_connectivity.png` | X3 |
| F4 | `17_f4_per_agent_spread.png` | X1, X2 |
| F5 | `14_f5_disagreement.png` | X1, X2 |
| F6a | `18_f6a_sparsity_tuned.png` (slides: `18a`, `18b`) | X4 + per-cell tuning sweep |
| F6b | `19_f6b_cost_of_not_retuning.png` (slides: `19a`, `19b`) | X4 (both tunings) |
| F7 | `20_f7_adaptation_transient.png` | X5 |
| F8 | `15_f8_atc_vs_cta.png` | X1b |
| F9 | `21_f9_non_iid.png` | X6 |
| F10 | `22_f10_forgetting.png` | X7 |

### Two stages, and why

**Collect** reduces ~775 000 raw rows per experiment to the few hundred points a
figure actually draws, and writes them to `figure_data/` beside the PNGs.
**Draw** renders from that.

So changing a label, a colour, or the resolution is a second of work against a
small tidy table — `--from-cache` never touches `results/`. That matters for two
reasons: the raw parquet is large and slow to re-aggregate, and the figure data
is what survives into a talk or a paper long after a run has been superseded by
a re-tuned one.

One parquet and one JSON per figure:

| column | meaning |
|---|---|
| `figure` | `f1`, `f2`, `f3`, … |
| `panel` | which subplot — e.g. `Stationary`, or `Stationary\|e_agree` for F5 |
| `series` | the learner, or `reference`, or a topology name for F3 |
| `role` | `line`, `reference`, `point` or `cell` |
| `x`, `y` | the plotted point |
| `lo`, `hi` | the ±1 s.d. band, or `NaN` where there is none |
| `cost` | cumulative scalars transmitted, for F2's x-axis |

The `.meta.json` beside it carries what the caption states — the seed count and
the smoothing window — so a redrawn figure cannot claim a seed count it does not
have.

---

## 2. Reading conventions

These hold across every figure.

**Aggregation is counts-then-divide, never a mean of rates.** Every error rate
is $1 - \sum n_\text{correct} / \sum n_\text{samples}$ over the rows being
pooled. Averaging per-agent *rates* would weight an agent that saw two samples
the same as one that saw eight, quietly reweighting the result toward whichever
agents held the least data.

**Bands are ±1 s.d. across seeds**, computed on the per-seed aggregate — so a
band means "how far would a rerun move this curve", not "how much do the agents
differ". Per-agent spread is F4's job.

**Differences between two learners are paired within seed.** Both run on the
same environment at the same seed, so most run-to-run variation is common and
cancels. Measured on X3: individual error rates carry a seed s.d. of 0.0035
while the paired gap carries 0.0012. Unpaired, the connectivity axis of F3 would
be mostly noise.

**Colour follows the method, never its rank.** A figure that drops a learner
does not repaint the survivors, so two figures can be compared directly.

**Every line is labelled twice** — a legend below the figure and a direct label
at its right edge. Below rather than inside because the convergence tail is the
part worth reading and a legend box lands exactly on it. Direct labels are nudged
apart in typographic points where curves converge, with a leader back to the true
endpoint.

**Log axes drop zeros rather than clipping them.** $E_\text{agree}$ is
identically zero for `centralized_sgd`; clipping to a floor would draw a line
that looks like a small positive disagreement.

**Smoothing is applied only to per-step series.** ⚠ The `prequential` evalset is
scored every step; `current` and `canonical` are scored every `eval_every` (25)
steps. A rolling window of $w$ points therefore spans $25w$ steps on the latter.
The first version of F7 used a 9-point window on `current` — 225 steps — and
**erased the entire transient it exists to show**, producing a plausible smooth
curve with no sign anything was wrong. F1 may smooth; F4, F7 and F9 must not.

---

## 3. F1 — error rate over time

**The headline.** Prequential error against $t$, one panel per drift regime, with
the offline reference $e^\star$ as a dashed line.

*Prequential* means test-then-train: each agent predicts on its incoming batch
**before** learning from it, so the curve is honest online performance and needs
no held-out split. Scoring after the update would leak the label and pull every
method's error down by an amount that grows with the learning rate — which would
look like a result.

**$e^\star$ is a curve, not a constant.** Under rotation the offline reference is
retrained per rotation, so one horizontal line would quote $e^\star$ at a state
most of the run never visits. Under a stationary schedule it flattens by itself.

**What to look for.** The ordering, and the size of the gap to $e^\star$. The
reference is trained offline on the full shard over many epochs, so no online
method should reach it; the gap is the price of one pass, in a stream, with no
fusion centre.

**What would be a surprise.** Any distributed method beating `centralized_sgd`.
That is an upper reference — this exact signal is what exposed the tuning
confound of `results.md` §6.

---

## 4. F2 — the same, against communication

F1 replotted with **cumulative scalars transmitted** on a log x-axis. A different
question: not "best after 1500 steps" but "best per unit of bandwidth" — what
matters when the network, not the clock, binds.

**Two methods are horizontal dashed references, for opposite reasons.**
`centralized_sgd` communicates heavily but *off this axis*: it ships raw samples
to a centre, which is what the paradigm exists to avoid. `local_only` genuinely
never speaks. Plotting either at $x=0$ would read as "free and this good"
(design note D30). Both are labelled in place, because three different things use
dashed horizontals across F1 and F2.

**What to look for: the two payload variants change places.** At equal *time*
momentum ATC wins by 0.013 (~4x the seed noise); at equal *bandwidth* the
payload-matched variant is ahead. This is the axis on which phase 5's claim is
actually stated.

### Why the ordering flips — a step-count identity

Not a subtlety about momentum. **At equal bandwidth the payload-matched variant
has taken twice as many steps.** Momentum ships $2p$ per link ($\bm\theta$ and
its buffer), plain ships $p$ — 116 320 against 58 160 scalars per step on the
ring — so a budget $B$ buys momentum $t$ steps and plain $2t$. The comparison is
therefore $e_\text{plain}(2t)$ against $e_\text{momentum}(t)$, and plain wins
exactly when

$$\underbrace{e_\text{plain}(t) - e_\text{plain}(2t)}_{\text{what doubling the steps buys}}
\;>\;
\underbrace{e_\text{plain}(t) - e_\text{momentum}(t)}_{\text{what momentum buys per step}}$$

| $t$ | doubling gain | momentum advantage | equal-bandwidth winner |
|---|---|---|---|
| 100 | **0.054** | 0.020 | payload-matched |
| 300 | 0.020 | 0.014 | payload-matched |
| 500 | 0.018 | 0.020 | ~tie |
| 700 | 0.012 | 0.010 | ~tie |

**The two terms are substitutes.** Momentum and extra steps do the same job —
reduce gradient noise — but buy it from different places: momentum averages over
*time*, extracting more from gradients already paid for, while extra steps
average over *more data*. So the winner is whichever is cheaper per unit of noise
reduction. Early the curve is steep and far from $e^\star = 0.047$, so fresh data
dominates. Late it flattens and doubling buys 0.012 instead of 0.054; momentum's
edge decays too, but more slowly, so the gap closes.

**How far the claim actually goes.** Paired within seed, the payload-matched
variant is significantly ahead (>2 s.e.) only up to $\approx 1.2\times10^7$
scalars; past that the difference is **not significantly different from zero in
either direction** at any point measured. So "the curves cross" overstates it:
the supported statement is *payload-matched wins early and decisively, then the
two become indistinguishable per scalar sent*, while momentum wins clearly per
step. **Momentum never establishes a significant lead on this axis** within the
horizon.

⚠ The equal-bandwidth comparison can only run to $t = 750$, half the horizon,
because it needs the plain curve at $2t$. A claim about the far end of this axis
is not available from a $T = 1500$ run.

**For phase 5** this is the favourable reading. A Diff-EKF sending a mean and no
covariance is a $p$-per-link method, so it inherits the doubled step count, and
on the axis where its claim is stated the $2p$ method never pulls significantly
ahead.

---

## 5. F3 — the price of connectivity

$$\text{gap} = e(\texttt{diffusion\_sgd\_atc}) - e(\texttt{centralized\_sgd})$$

one point per topology, in two windows against two **different** predictors.

**What the window names claim, and how it is checked.** `settled` asserts the gap
has *stopped changing* — not merely "the last 100 steps". The test is whether a
line fitted over the final 500 steps has a slope exceeding the seed s.d.:

| topology | final gap | seed s.d. | slope / 500 steps |
|---|---|---|---|
| path | 0.0035 | 0.0023 | −0.0021 |
| star | 0.0077 | 0.0037 | −0.0026 |
| ring | 0.0014 | 0.0012 | −0.0008 |
| complete | 0.0000 | 0.0000 | +0.0000 |

No topology exceeds its noise, and the two with meaningful gaps (path, star) stop
moving by $t \approx 1075$ — well before the window opens at 1400. `transient`
$t\in[150,300)$ is a window while the gap is still falling for every topology,
which is where connectivity matters most.

Both labels were originally *asserted*: the windows were picked as "the end" and
"early on", and the names attached afterwards. `tests/test_figures.py` now checks
them against the data — that the settled window is flat, that the transient
precedes it, and that the spread across topologies really is wider in the
transient (otherwise the second column buys nothing). A change to `run.horizon`
that left the window inside the transient would fail those tests rather than
quietly relabel a moving target.

**Why two predictors rather than one.** The spec asks for gap vs spectral gap.
Measured on seven topologies, $1-\rho$ does predict ($\rho_s = -0.786$, exact
$p = 0.048$) but the **mean self-weight $\bar a_{vv}$** predicts better
($+0.964$, $p = 0.0028$) and is the one that ranks `star` correctly — 7th of 7
rather than 3rd. Showing both makes the comparison the figure's content instead
of a claim in the caption.

**The signs are opposite and both expected.** Larger spectral gap = better
connected = smaller price (negative). Larger self-weight = each agent keeps more
of its own estimate = mixes less = larger price (positive).

**Why $\bar a_{vv}$ works.** It is the average fraction of its own estimate an
agent keeps, so $1 - \bar a_{vv}$ is literally mixing-per-round. Star follows
immediately: Metropolis gives each leaf a hub-weight of $1/(1+9) = 0.1$, so every
leaf retains **90%** of its own estimate and the network barely mixes despite a
diameter of 2. The spectral gap describes the *asymptotic* rate of consensus;
over 1500 online steps what matters is mixing per step.

**A caveat that belongs with the figure.** $\bar a_{vv}$ was chosen after seeing
five of the seven topologies. It faced one genuine out-of-sample test
(`erdos_renyi`) and **failed it**. Report it as the better of two descriptive
correlations on seven graphs, not as a law. See `results.md` §7.3.

---

## 6. F4 — per-agent spread

Mean over agents with a **min–max band**, for each method. This is disagreement
in *performance*, which is a different quantity from F5's disagreement in
*parameters* — agents can hold different weights and still score alike.

**What to look for.** `centralized_sgd` has no band at all: every agent holds the
same $\bm\theta$ by construction, so a visible band there would mean the runner is
reading the wrong state. `local_only` has a wide one. ATC sits between, and how
narrow it is measures how well the combine step is holding the network together.

Not smoothed — the `current` evalset is already sparse (§2).

---

## 7. F5 — disagreement and deviation from centralized

$$E_\text{agree} = \tfrac1N\sum_v \lVert\bm\theta_v - \bar{\bm\theta}\rVert^2
\qquad
E_\text{cent} = \lVert\bar{\bm\theta} - \bm\theta^\text{cent}\rVert^2$$

**$E_\text{agree}$ is consensus** — how far the agents are from each other.
Identically zero for `centralized_sgd` by construction, which is why that series
is absent from the top row rather than drawn along the floor.

**$E_\text{cent}$ is fidelity** — how far the network average is from what a
fusion centre would have computed. The two are independent: a network can agree
perfectly on the wrong answer.

**The third row is the second one normalised**, $E_\text{cent}/\lVert\bar{\bm\theta}\rVert^2$,
and it exists because the raw curve is easy to misread.

### Why $E_\text{cent}$ rises, and why that is not "getting worse"

| $t$ | $\lVert\bar{\bm\theta}\rVert^2$ | $E_\text{cent}$ | ratio |
|---|---|---|---|
| 100 | 47.7 | 0.03 | 0.0006 |
| 600 | 64.7 | 0.12 | 0.0019 |
| 1499 | 79.5 | 0.31 | 0.0039 |

$E_\text{cent}$ is an **unnormalised** squared distance, and the weights
themselves grow — $\lVert\bar{\bm\theta}\rVert^2$ nearly doubles over the run. Two
trajectories a fixed *relative* distance apart therefore separate in absolute
terms simply by travelling further from the origin. Roughly half the rise is
that, which the third row removes.

The residual rise is real but **costs almost nothing in error**: at $t = 1499$
centralized is at 0.0749 and ATC at 0.0762. The two models are functionally
near-identical while their parameters separate — the ordinary situation for
neural networks, whose loss surfaces have wide flat directions and permutation
symmetries. **$E_\text{cent}$ measures parameter distance, not disagreement about
predictions**, and those come apart. Read the rise as the network wandering along
a flat direction, not as degradation.

### Why $E_\text{agree}$ plateaus for ATC and diverges for local-only

ATC settles near 0.009 because two forces balance every step: the combine is a
*contraction* toward consensus, while the per-agent gradients push apart because
each agent sees different data. The plateau is where they cancel. The small rise
then settle (0.0110 → 0.0089) is the initial transient, when gradients are
largest and consensus has not caught up.

`local_only` has **no contraction at all**, so disagreement accumulates as a
random walk. Measured, $E_\text{agree}$ grows essentially linearly in $t$
($r = 0.993$) — exactly what a random walk gives in *squared* distance, since
displacement goes as $\sqrt t$. It reaches 17.3 by the end with no sign of
stopping, because nothing stops it.

That contrast is the clearest single picture of what the combine step buys: it
converts an unbounded random walk into a bounded equilibrium.

**A useful diagnostic.** $E_\text{cent}$ rising while $E_\text{agree}$ stays flat
means the agents agree with each other but are drifting away from the centralized
solution *together* — a mixing problem, not a consensus problem.

---

## 8. F6a — the sparsity plane, tuned per cell

Four heatmaps over $(n, \pi_\text{lab})$: ATC's error, the cooperation gap
(`local_only` − ATC), the pooling gap (ATC − `centralized_sgd`), and the payload
cost (ATC (payload-matched) − ATC).

**Every cell uses each method's own best (optimizer, lr) for that cell**, not one
global setting. This is not fussiness. At $\pi_\text{lab} < 1$ most agents are
idle, and an idle agent contributes its *unchanged* $\bm\theta$ to the combine
step — so ATC's effective step is $\eta\, n_\text{active}/N$ while centralized
takes the full $\eta$. At $\pi_\text{lab} = 0.25$ that is a **4×** difference.
Comparing the two at one lr compares step sizes, not methods. (Measured: the two
optima sit at lr 0.0025 and 0.01, exactly the predicted ratio.)

**Tuning the payload-matched variant.** Both names map to one class, and the
sweep sets the optimizer for every learner it runs — so left unconstrained the
variant picks momentum and becomes numerically identical to ATC, making panel 4
exactly 0.000 everywhere. That is a definition being overridden, not a null
result: carrying no optimizer state is precisely what makes its message $p$ per
link rather than $2p$. Its optimum is therefore taken **within the plain-SGD arm
only** (`make_figures.x4_tuned`).

**Colour.** Sequential single-hue for the three magnitudes; **diverging with a
neutral midpoint** for the pooling gap, which is signed. The payload cost stays
sequential deliberately — properly tuned it is positive in all twelve cells, so a
neutral midpoint would imply a sign change that does not occur. Values are
printed in each cell, with ink on light cells and paper on dark ones.

**What to look for — panel 3.** A negative pooling gap — diffusion beating pooled
data — confined to the sparse corner and shrinking as $n$ and $\pi_\text{lab}$
rise. That pattern is the signature of **implicit iterate averaging**: diffusion
maintains $N$ trajectories under continuous averaging, a variance-reduction
device a single centralized trajectory lacks, and it should pay only where the
per-step gradient is noisiest. If instead the negative region were scattered, the
remaining cause would be residual mis-tuning rather than a real mechanism.

**What to look for — panel 4.** The same corner, darkening the same way: the
payload cost runs 0.046 in the sparsest cell down to 0.007 in the densest, about
**2.5×** along each axis in the marginals (0.0118 → 0.0304 in $\pi_\text{lab}$,
0.0125 → 0.0292 in $n$). Both axes control how much signal one step carries; the
extra $p$ scalars buy momentum, whose job is to accumulate a consistent direction
out of noisy gradients, so it is worth most where each gradient is worst. Read
panel 4 against F9's non-IID result, where the same cost is **flat** (0.015 /
0.013 / 0.015 across three decades of skew): skew changes *what* an agent sees,
not how often it updates. **The second half of the message earns its keep under
sparsity, not under heterogeneity** — which bounds what a $p$-per-link Diff-EKF
gives up at ~0.046 worst case.

The trend is clean in the marginals but not cell-by-cell: $n{=}4$ is non-monotone
in $\pi_\text{lab}$ (0.013 / 0.021 / 0.014), which at two sweep seeds is inside
the noise.

**These are measured, not pessimistic — a caveat that was tested and withdrawn.**
An earlier version noted that the variant's tuned optimum sat at lr 0.2, the
largest rate then swept, in 7 of 12 cells, and inferred that its true optimum lay
past the edge and the costs were therefore overstated. The reasoning was
plausible: plain SGD needs roughly 10× momentum's nominal rate for the same
effective step, since $\eta/(1-\beta) = 10\eta$. The grid was extended to 0.5 and
1.0 to check, and **neither wins a single cell** — for any method. Panel 4 is
unchanged to four decimals. The $10\eta$ rule predicts a *band* of 0.05–0.5 from
ATC's 0.005–0.05, and the plain variant's 0.05–0.2 sits inside it, at the low
end; reading the top of that band as "past the edge" was the error.

---

## 9. F6b — the cost of not re-tuning

$$\text{penalty} = e(\text{headline lr}) - e(\text{best lr for this cell})$$

one heatmap per method, four in all.

F6a asks "how do the methods compare when each is used properly?". F6b asks the
practitioner's question: **"what does it cost me that I tuned for the nominal
regime and deployment turned out sparser?"** Nobody re-tunes when a sensor's
label rate drops, so this is the more operational of the two.

**What it shows.** Worst penalty across the plane:

| method | worst | where | median |
|---|---|---|---|
| centralized | **0.183** | $n{=}1,\ \pi{=}0.25$ | 0.007 |
| local only | 0.151 | $n{=}1,\ \pi{=}1.0$ | 0.004 |
| ATC (payload-matched) | 0.110 | $n{=}1,\ \pi{=}0.5$ | 0.000 |
| **ATC** | **0.028** | $n{=}8,\ \pi{=}0.25$ | 0.004 |

ATC's panel is nearly blank — that flatness *is* the figure. It is roughly 6.5×
more forgiving of a step size chosen for a different regime than centralized.

**Both terms come from the tuning sweep, not from the X4 runs.** A penalty
defined as "headline minus the minimum over a grid containing the headline"
cannot be negative, so a negative cell is a bug, not a finding. Two produced them
and both are fixed: lr 0.2 was absent from the grid, and the two terms were drawn
from different estimators (a five-seed X4 number minus a two-seed sweep number),
which gave penalties as low as −0.042. With one estimator on both sides, **0 of
48 cells are negative**. The five-seed X4 runs remain what is quoted elsewhere;
they are simply not what *this* difference can be built from.

### Why — candidate mechanisms, only one of which survives

An earlier version of this text said "the combine step damps an oversized
update". That is loose, and testing it showed it is not the mechanism.

| candidate | verdict |
|---|---|
| ATC's error-vs-lr curve is *flatter* | **not robustly** — the measure reorders when the grid is trimmed (below) |
| ATC's optimum *moves less* across the plane | **yes** — its chosen lr spans 1.0 decades against 1.9 for centralized and 1.7 for local-only |
| the combine *damps* a too-large update | superseded by the sharper statement below |

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
reorders when the grid is trimmed is measuring the grid, not the method — so it
is not the mechanism, even though on the full grid it happens to favour ATC.

**The actual mechanism is automatic step-size scaling.** The optimum's *location*
is stable, and at $n{=}1,\ \pi_\text{lab}{=}0.25$ the momentum-arm profiles show
why:

| lr | centralized | ATC |
|---|---|---|
| 0.0025 | **0.208** ← its optimum | 0.321 |
| 0.005 | 0.259 | 0.207 |
| **0.01** ← headline | 0.374 | **0.170** ← its optimum |
| 0.02 | 0.726 | 0.199 |
| 0.05 | 0.896 | 0.302 |

ATC's optimum is *still the headline value*; centralized's has moved by a factor
of four. With ~2.5 of 10 agents active, an idle agent contributes its unchanged
$\bm\theta$ to the combine, so ATC's effective step is
$\eta \cdot n_\text{active}/N \approx \eta/4$ **automatically**. A smaller batch
needs a ~4× smaller step, and diffusion supplies that reduction by itself, so its
*nominal* rate need not move. Centralized applies the full $\eta$ however many
agents happened to hold labels, so its nominal optimum has to move to compensate.

**The grid ceiling does not bind — checked.** An earlier version warned that the
payload-matched variant's optimum sat at lr 0.2, the largest rate then swept, in
7 of 12 cells, so its stability was "the grid's width, not the method's", and
that centralized's 1.9-decade span was a lower bound. The grid was extended to
**0.5 and 1.0** to settle it: **neither wins a single cell for any method**, and
every number here is unchanged to four decimals. The spans are the methods'.

**The argmin within a cell is still barely determined**, which is a separate
point and survives. The grid reports centralized's
optimum at $n{=}1,\pi{=}0.25$ as plain SGD at lr 0.02 (0.191) while a finer sweep
found momentum at lr 0.0025 (0.193), two arms within 0.002 of each other inside
the ±0.023 seed spread. **The identity of the best lr in any single cell should
not be quoted as a finding.** The penalties themselves (0.183 against 0.028) are
an order of magnitude above that ambiguity and are safe.

### Slide variants

Both figures are also written as two-panel halves for the deck:
`18a`/`18b` for F6a and `19a`/`19b` for F6b. **The documents use the full
four-panel versions; only the slides use the halves.**

Not a duplicate figure — a sizing one. The slide layout gives a wide image the
frame width and whatever height the notes leave, so a 4.5:1 figure is scaled to
0.64x and its cell values land near 5pt projected, which cannot be read from a
room. The halves sit near 1.5:1, which puts them in the layout's *image-left,
notes-right* branch instead: near 1:1 scale, and 10.3pt projected. Same panels,
same data, one code path (`_panel_figure`) — the halves are the same `Panel`
list, sliced.

**F6b's halves carry the whole figure's colour scale**, not their own. The limit
is taken over all four panels before any of them is drawn. A half that rescaled
to its own two would make ATC's near-blank plane look like centralized's and
destroy the only comparison the figure makes — invisibly, since each half would
still look internally sensible. Both half titles say "shared colour scale" for
the reader who sees only one.

Sequential colour on **one shared scale across all four panels**, so they can be
compared to each other rather than only read individually — which is the whole
point of the figure. Per-panel colourbars would autoscale ATC's near-zero plane
up to look like centralized's and destroy the comparison.

---

## 10. F7 — the adaptation transient

Held-out error over $[t^\ast - 50,\ t^\ast + 300]$ around the abrupt shift, where
$t^\ast$ is **read from the recorded drift state** rather than hardcoded — the
figure finds the step where the rotation actually changes.

**Not smoothed, and that is load-bearing** (§2). The shift appears in a *single*
evaluation point ($0.097 \to 0.174 \to 0.136$), so any rolling window flattens it.

**What to look for.** The height of the spike, how many steps recovery takes, and
whether the ordering changes across it. Measured: every method loses ~0.075
immediately and recovers within ~150 steps, and the ordering never changes — so
the shift does not advantage any method, it just costs them all the same.

---

## 11. F8 — ATC vs CTA

Two panels — error rate and $E_\text{agree}$ — for the two orderings at
**identical communication cost**.

$$\text{ATC:}\quad \bm\psi_v = \bm\theta_v - \eta\nabla L_v,\qquad \bm\theta_v \leftarrow \textstyle\sum_u a_{vu}\bm\psi_u$$
$$\text{CTA:}\quad \bm\theta_v \leftarrow \textstyle\sum_u a_{vu}\bm\theta_u - \eta\nabla L(\bm\theta_v)$$

The gradient is evaluated **before** the averaging in CTA and after it in ATC.
That is the entire difference; they cost the same.

**Why it matters.** Diff-EKF is ATC, so phase 5 differs from diffusion SGD in the
adapt step alone. F8 measures what the ordering itself is worth, so the choice is
*reported* rather than assumed.

**What it actually shows.** At tuned settings the two are **indistinguishable**
(0.0768 vs 0.0774, seed s.d. 0.0034). ATC wins 47 of 50 grid cells, so the sign is
real, but it separates only where the step is too large (0.118 vs 0.251 at
momentum lr 0.2). **ATC's advantage is robustness to step size, not accuracy at
the right one** — which fits mechanically, since ATC averages after stepping and
so damps an oversized update.

Both learners select the same optimum (momentum, lr 0.01) at $n \in
\{2,4,6,8\}$, so matched settings *are* each one's best and the comparison needs
no caveat. That was checked, not assumed — see design note D40.

---

## 12. F9 — non-IID

Two panels against Dirichlet $\beta$ on a log axis: each method's error, and the
cooperation gap.

**The clearest result in the benchmark.** `local_only` runs 0.142 → 0.629 as skew
increases while ATC stays nearly flat (0.080 → 0.103), so the cooperation gap goes
**0.062 → 0.527**, an 8.5× increase. Under $\beta = 0.1$ an agent sees three or
four digits and alone lands near chance; the same agent inside a diffusion network
reaches 0.103.

**The payload cost is flat across skew** — 0.015, 0.013, 0.015 at five seeds, so
the payload-matched curve tracks ATC's at a constant offset. Contrast F6a panel
4, where the same quantity grows ~2.5× as the problem gets sparser. Skew changes
*what* an agent sees; sparsity changes *how often* it updates, and momentum — the
thing the second $p$ scalars buy — only compensates for the latter.

**A free correctness check.** `centralized_sgd` is flat across $\beta$ (0.078,
0.077, 0.080) — it pools every agent's samples, so the partition is invisible to
it. A slope there would mean the skew is leaking into the data path rather than
the partition.

Shard *sizes* are held equal across $\beta$; only the label composition varies.
Otherwise skew and shard starvation would be confounded.

---

## 13. F10 — forgetting

Two panels: current versus backward error over time, and their paired gap.

### The three terms this figure depends on

**Held-out error rate.** Fraction misclassified on the MNIST **test** split —
images never used for training by anyone (guarantee G3). "Held out" is what makes
it a measure of *generalisation* rather than of memorisation: an error rate
computed on the training stream would fall simply by the model memorising the
images it just saw. Both curves in the left panel are held-out; they differ only
in **which rotation** the test images are drawn at.

**"A rotation left behind."** The task drifts: at step $t$ every training image
is rotated by $\varphi(t)$, and under X7's sinusoidal schedule $\varphi$ swings
$\pm30°$ and comes back. So "the task" at $t = 200$ and "the task" at $t = 700$
are genuinely different classification problems on the same digits.

The **`backward`** evalset scores the *current* model on the test split rendered
at a rotation the model has already visited and moved away from:

$$t' = \max\{\,s < t \;:\; |\varphi(s) - \varphi(t)| \ge \Delta\varphi\,\},
\qquad \Delta\varphi = 15°$$

— the **most recent earlier step far enough away in rotation**. Two properties
matter and neither is free:

- it is a state the model **actually trained on**, so a failure there is
  forgetting rather than never having learned;
- it is **separated** from the current state, so the two probes are not asking
  the same question.

Anchoring by *rotation* rather than by a fixed step offset $t - \Delta$ is the
reason this works. A step offset degenerates: set $\Delta$ to the sinusoidal
period and the separation is *identically zero* — the schedule chosen to expose
forgetting could not measure it. Where no qualifying $t'$ exists the probe is
**undefined and reported as such**, never as "no forgetting". It is defined for
97 % of steps here, against 67 % under linear drift and 0 % under stationary.

**Forgetting = backward − current.** Both terms are the same model, at the same
step, on the same test images — only the rotation differs. So the subtraction
cancels everything except the effect of the rotation:

- **positive** → worse on the old state than the current one → the model has
  **forgotten** what it knew there;
- **zero** → it handles both equally → no forgetting;
- **negative** → *better* on the old state → the model is still tuned to where
  the task used to be, i.e. it is **lagging** behind the drift.

The sign is the whole content of the second panel. It is computed paired within
seed (same seed's backward minus that same seed's current, then averaged), so
seed-to-seed noise common to both cancels instead of adding.

**⚠ The instantaneous gap is dominated by phase, not by forgetting.** It swings
±0.05 with the drift cycle, and the sign of any average depends on how much of a
period the window covers: $+0.016$ over a fifth of a period, $-0.0035$ over a
whole one. **A scalar summary is only meaningful over a whole number of
periods**, which is why the figure draws the cycle mean as a dashed line — the
peaks would otherwise read as forgetting.

**What to look for.** The dashed lines, not the peaks. All three cooperative
methods sit at 1.1–1.6 σ from zero: no measurable forgetting. `local_only` is
significantly *negative* at 10 σ — better on a state it has left than on the
current one, which is **lag**, not retention: a slow learner's parameters trail
the world.

**Reading the two panels together.** In the left panel `local_only`'s solid
(current) curve sits well above everyone else's *and* above its own dashed
(backward) curve — it is the only method whose past beats its present. The other
three have solid and dashed curves interleaving, crossing as the cycle turns:
that interleaving is what "no forgetting" looks like. The oscillation visible in
both panels is the drift cycle itself, not instability.

**Why the negative result is the useful one.** A benchmark that showed heavy
forgetting would make continual-learning machinery the obvious next step. This
one does not, which says the Diff-EKF's case has to be argued somewhere else —
tracking, uncertainty, or the sparse regime — rather than on retention. It also
makes the *lag* reading concrete: `local_only`'s problem under drift is that it
learns too slowly to keep up, and cooperation is what fixes that.

---

## 14. The drift-benchmark figures (23–25)

These come from their own scripts rather than from `make_figures.py`, because
each needs a *paired control run* and the F1–F10 pipeline is built around one
experiment at a time. They write to the same folder and take the same
`DEKF_FIGURES_DIR`.

| id | file | script | source |
|---|---|---|---|
| 23 | `23_breaks_x9_rate_ramp.png` | `plot_breaks.py` | x9 + x9_control |
| 24 | `24_x11_recovery_diffusion_sgd_atc.png` | `plot_recurring.py` | X11 grid + x11_control |
| 25 | `25_abrupt_vs_smooth.png` | `plot_abrupt_vs_smooth.py` | X11 + X12 + both controls |

### 23 — where tracking breaks

Drift damage against **drift rate**, not against the step. Under a ramp the step
is only an index into a rate sweep, and plotting against it invites reading a
break as "it survived 1100 steps" when the claim is "it survived to 0.038°/step".

Two panels because the frozen baseline ends an order of magnitude above
everything else and flattens the rest on a shared axis. It is drawn purple and
dashed rather than a second grey: on the shared axis it sits beside
`centralized_sgd`, and those two are the one pair a reader must not confuse.

Circles mark the located break, using the **same pooled bar** as
`report_breaks.py`. If the figure used each learner's own noise its markers
would disagree with the table and nothing would say which was right.

### 24 — recovery under repeated shifts

Three heatmaps across the $t' \times J$ grid, and two transient panels showing
the shape the summary numbers come from. A grid alone leaves "recovered 0.2"
ambiguous between a small wound that heals slowly and a large one that heals
fast, and those are different problems.

**Dark is worse in every panel.** `rise` and `standing` are costs so a plain
ramp reads correctly; `recovered` is a good thing, so its ramp is reversed
rather than its numbers negated — the annotated value stays the quantity
`report_recurring.py` prints, and nobody has to reconcile a sign.

Read the `recovered` panel along $t'$ only, not along $J$ — see results §13.4.

### 25 — abrupt against smooth

Two panels because two windows are in play and one axis would conflate them.
Left is the controlled comparison, both regimes over the same steps. Right drops
the smooth side and shows every abrupt cell over its own second half, with the
shaded region marking where **no smooth counterpart can exist**, because
constant drift leaves the 45° band. That gap is the finding, not missing data.

Hollow markers mean the window holds one shift, so the point is a single
transient rather than an average. This was first drawn as marker *size*, which
looked fine until the legend swatch took its size from the first point plotted —
making the key silently disagree with the data.

## 15. Still to come

**F11** *(phase 5)* — Diff-EKF added to F1 and F2. Its competitor on F2 is
`diffusion_sgd_atc_plain`, not the momentum variant, because the filter sends one
$p$-vector per link.
