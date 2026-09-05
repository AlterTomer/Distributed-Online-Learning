# The experiments

Every experiment this benchmark has run, what it asks, and the command that
reproduces it. `WORKPLAN.md` §6 states the *design* of X0–X7 and why each exists;
this file is the operational index — what to type, in what order, and what it
costs.

If this document and the scripts disagree, the scripts are right and this is
stale. Each script's own docstring carries the reasoning: why a grid is
proportioned as it is, why a rate ceiling was chosen, what a null result would
have looked like. They are worth reading before running anything expensive.

---

## Before anything else

```bash
python scripts/check_environment.py     # graph stats, sample streams, smoke test
python scripts/check_data.py            # download and cache MNIST, once
python scripts/train_reference.py       # the offline reference classifier, once
```

`train_reference.py` is a prerequisite for every experiment that reports error
against the reference. It is cached, so it runs once per machine.

Every script below is runnable from an IDE's run button with no arguments:
module-level constants at the top, `argparse` optional and never required.

---

## The experiments

The time column is **measured wall-clock where we have it** and blank where we do
not — a guessed runtime is worse than none for deciding what to start before
lunch. The SGD arms run on a laptop CPU; the filter needs a GPU (design note
D58). Every script prints a per-cell ETA as it goes.

| # | Question | Command | Time |
|---|---|---|---|
| **X0** | Does ATC diffusion reproduce centralized SGD exactly on a complete graph? *(the correctness gate)* | `run_experiment.py x0_exactness` | minutes |
| **X1** | Do all methods learn on stationary data, and what is the cooperation gap? | `run_experiment.py x1_stationary` | |
| **X1b** | ATC or CTA — does the diffusion ordering matter? | `run_experiment.py x1b_atc_vs_cta` | |
| **X2** | Does the gap widen under smooth rotation? Does local-only collapse? | `run_experiment.py x2_rotating` | |
| **X3** | Gap against spectral gap: the price of connectivity, lr re-tuned per topology | `run_topology_sweep.py` | |
| **X4** | Where does the sparse regime hurt? $n\in\{1,2,4,8\}$ × $\pi_{\text{lab}}$ | `run_sparsity_sweep.py x4` | |
| **X5** | The adaptation transient after a single abrupt shift | `run_experiment.py x5_abrupt_shift` | |
| **X6** | Does cooperation survive Dirichlet label skew? | `run_sparsity_sweep.py x6` | |
| **X7** | Does a method forget a rotation it has left? *(sinusoidal; the only schedule that revisits states)* | `run_experiment.py x7_sinusoidal` | |
| **X8** | Does cooperation still pay when agents drift *differently*? | `run_experiment.py x8_global` then `x8_per_node_drift` | |
| **X9** | At what drift **rate** does each method's tracking break? *(accelerating ramp)* | `run_experiment.py x9_rate_ramp` then `x9_control` | |
| **X10** | Does cooperation pay under label shift rather than covariate shift? | `run_experiment.py x10_prior_drift` then `x10_control` | |
| **X11** | Recovery under *repeated* abrupt shifts, $J$ × $t'$ | `run_recurring_sweep.py` | 38 h |
| **X12** | Smooth drift at four constant rates, to bracket X11 | `run_linear_sweep.py` | |
| **X13** | Tuning the centralised EKF: $(\sigma_0^2,\gamma,\bm Q)$ and $(\sigma_0^2,\lambda)$ | `run_ekf_sweep.py`, then `--baselines`, then `--full` | 12 + 2 + 36 h |
| **X14** | Does the filter's advantage generalise? 21 drift conditions, rate crossed with state count | `run_ekf_generalization.py --lr` then `run_ekf_generalization.py` | 0.5 + 32 h |
| **X15** | Is the $\gamma$/$\lambda$ ordering a mechanism or a tuning artefact? Both families re-tuned at one fast condition | `run_ekf_retune.py` | ~6 h |
| **X16** | The filter on X9's ramp, so the break figure has a filter curve | `run_ekf_ramp.py` | 1 h |
| **X17** | The filter under Dirichlet label skew, still and drifting — and does abrupt or smooth motion hurt more alongside skew? | `run_ekf_skew.py` | ~2.5 h |

**Order matters in three places.** X13 needs `--baselines` before `--full`, or it
compares a drift-tuned filter against a stationary-tuned baseline. X14 needs
`--lr` first, for the same reason. X15 and X16 read settings that X13 selected
and pair against runs X14 and X9 produced, so they refuse to start until those
exist rather than silently substituting a default.

**Controls are experiments too.** Every drifting run is paired with a stationary
twin identical in seed, data order and every hyperparameter — `x9_control`,
`x10_control`, `x11_control`, `x14_control_n4_T1500` and so on. Damage is the
difference, so a missing control does not degrade a result, it removes it.

---

## Reading the results back

Runs write one parquet per seed to a gitignored `results/<name>/`. Each sweep has
a reader that turns those into the numbers quoted in `docs/results.md` and the
design notes:

```bash
python scripts/report_breaks.py               # X9: where each method breaks, as a rate
python scripts/report_recurring.py            # X11: recovery per (J, t') cell
python scripts/report_abrupt_vs_smooth.py     # X11 against X12 at matched speed
python scripts/report_cooperation.py          # X8 and X10: does cooperation still pay
python scripts/report_ekf_sweep.py            # X13: the tuning grid, ranked
python scripts/report_ekf_generalization.py   # X14: damage per drift condition
```

Runs are **resumable and exact**: the loop consumes no randomness, so a resumed
run reproduces an uninterrupted one bit-for-bit, and re-running a sweep skips
cells already on disk. `--fresh` discards and redoes.

A cell that diverges is recorded as diverged and the sweep continues. That is a
measurement, not a crash — see design notes D61 and D76.

---

## What the numbers mean

`docs/results.md` reports what was found. `docs/design_notes.md` records *why*
each choice was made and, more usefully, which conclusions were later withdrawn:
a claim that did not survive a harder condition is logged beside the one that
replaced it. `docs/figures.md` describes what each figure shows and how to read
it, though the figure builders themselves are not shipped (D75).

To change any of this rather than re-run it as-is — a parameter, a learner, a
drift schedule, a new experiment — see [`howto.md`](howto.md).
