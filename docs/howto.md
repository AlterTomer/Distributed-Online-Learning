# How do I…?

Task-oriented answers with the code to type. For *what* each experiment asks see
[`experiments.md`](experiments.md); for *every* config field and its legal values
see [`configs.md`](configs.md); for *why* a choice was made see
[`design_notes.md`](design_notes.md). This file is the middle: you know what you
want to do, you want the three lines that do it.

If an answer here and the code disagree, the code is right and this is stale.

---

## Getting started

**Q. I just cloned this. What is the shortest path to a result?**

```bash
pip install -e ".[dev]"
python scripts/check_environment.py            # does anything work?
python scripts/check_data.py                   # cache MNIST, once
python scripts/train_reference.py              # the offline reference, once
python scripts/run_experiment.py x1_stationary # ~20 min
```

`train_reference.py` is a prerequisite for anything reporting error against the
reference, and it is cached, so it runs once per machine. If you skip it you get
a `ReferenceError` naming the script rather than a confusing failure later.

**Q. Do I need a GPU?**

Not for the SGD baselines — they are *faster* on CPU at this model size, because
$p=2908$ cannot amortise a kernel launch (0.69× at batch 4). The filter is
dominated by dense $p\times p$ covariance operations and is a 14× win on CUDA
(D58). `device: auto` picks per run; the shipped configs already do the right
thing.

**Q. How long does X take?**

[`experiments.md`](experiments.md) has measured runtimes where we have them, and
blanks where we do not rather than guesses. Every sweep prints a per-cell ETA as
it goes, averaged over cells that actually ran rather than over all indices —
a cached cell costs nothing and would otherwise make the estimate climb.

---

## Changing parameters

**Q. How do I change a setting for one run without editing anything permanent?**

Every script has module-level constants at the top. Edit those, or build the
config in Python:

```python
from dekf_bench.utils.config import load_config

config = load_config(
    "x1_stationary",                       # the base to inherit from
    overrides={
        "run": {"horizon": 500, "seeds": [0], "device": "cpu"},
        "env": {"samples_per_node_per_step": 8},
        "graph": {"topology": "complete"},
    },
)
```

Overrides merge key by key, so you name only what changes. Composition rules are
in [`configs.md`](configs.md) §1.

**Q. How do I change something permanently?**

Put it in a config file. `configs/base.yaml` holds the defaults every run
inherits; `configs/experiment/*.yaml` compose one entry from each of `env/`,
`graph/`, `model/` plus a learner list. **If it is not a config field it cannot
be swept**, and it will quietly become hard-coded — that is the rule the layout
exists to enforce.

**Q. How do I change the drift?**

Six schedules exist: `stationary`, `linear`, `ramp`, `piecewise`, `recurring`,
`sinusoidal`.

```python
overrides={"env": {"drift": {"schedule": "recurring",
                             "jump_every": 25,
                             "jump_degrees": 15.0,
                             "jump_seed": 0}}}
```

Fields are per-schedule and documented in [`configs.md`](configs.md) §`env.drift`.
Two things that will bite you:

- **Rotation is capped at 45°.** Past it a 6 becomes a 9, so a rising error would
  measure label ambiguity rather than tracking failure.
- **`recurring` reflects at that cap**, so the reachable angles are
  $2\lfloor 45/J\rfloor + 1$. At $J=30$ that is three, and the walk spends half
  its steps unrotated — the nominal rate $J/t'$ then overstates the displacement
  anyone has to follow. Keep $J \le 22.5$ if you want sustained motion (D74).

**Q. How do I change the filter's hyperparameters?**

They are learner config fields, so they can be swept like anything else:

```python
overrides={"learners": [{
    "name": "centralized_ekf_gamma",
    "transition": "scalar",     # "scalar" = F = gamma*I; "identity" = F = I
    "gamma": 0.9995,
    "process_noise_q": 6e-5,
    "prior_scale": 0.01,        # sigma_0^2
    "trust_region_ratio": 1e6,  # the divergence guard
}]}
```

The shipped values are what X13 selected and X14/X15 validated. `prior_scale` is
a **trust region, not just a prior** — too large a value diverges on the first
step rather than converging slowly (D61).

---

## Running things

**Q. How do I run one experiment?**

```bash
python scripts/run_experiment.py x1_stationary
python scripts/run_experiment.py x1_stationary --fresh   # discard and redo
```

The name is a file in `configs/experiment/`. With no arguments the script runs
its module-level default, so the IDE run button works with no setup.

**Q. My run died halfway. Do I lose it?**

No. Runs are **resumable and exact**: the loop consumes no randomness, so a
resumed run reproduces an uninterrupted one bit-for-bit, and a sweep skips cells
already on disk. Just run the same command again.

A run directory with neither a `_complete` nor a `_diverged` marker is debris
from an interrupted run and is deleted rather than resumed — the recorder
checkpoints as it goes, so resuming it would restart the *learner* from
$\bm\theta_0$ while the recorder skipped ahead, and the result would look clean
and be nonsense (D68).

**Q. How do I run only part of a sweep?**

```bash
python scripts/run_ekf_generalization.py --only every25_jump30 every10_jump15
python scripts/run_sparsity_sweep.py x4        # just the sparsity half
python scripts/run_ekf_sweep.py --baselines    # just the baseline re-tune
```

Subsets write into the same directories with the same settings, so a full pass
can be done in instalments and picks up the rest as cached.

**Q. Can I run two sweeps in parallel to save time?**

No. They write a shared status file and would each erase the other's entries, and
several sweeps depend on a `--lr` or `--baselines` pass having finished. Run them
one at a time.

---

## Comparing methods

**Q. How do I compare a new method against the baselines fairly?**

Two rules, both learned the hard way:

1. **Re-tune every baseline under the condition you are reporting.** A fixed
   learning rate once produced a headline that was an optimizer artefact. ATC
   prefers `lr 0.02` under smooth drift and `0.01` under abrupt shifts — worth
   0.0050, which is larger than several results we care about.
2. **Report damage, not error**, whenever drift is involved. Damage is the
   drifting run's error minus a stationary twin identical in seed, data order and
   every hyperparameter. The learner's own convergence cancels, so what survives
   is what the drift cost.

**Q. What counts as a real difference?**

**0.0013** for five-seed means — the standard error of a difference of two
five-seed means. A gap smaller than that is a tie, and reporting the argmin of
tied cells is choosing between things the measurement cannot separate (D71).

The older two-seed floor is 0.0021; use it only where the runs really had two
seeds, and say which you used, because the two give different verdicts on
marginal comparisons.

**Q. How do I tell "better optimiser" from "better tracker"?**

Split the advantage. Every run has a paired twin, so:

```
total    = baseline drift error - yours drift error
fitting  = baseline stationary  - yours stationary      (no drift involved)
tracking = baseline damage      - yours damage          (what drift cost each)
```

They sum exactly, because damage *is* drift minus stationary. A method can lead
on the total while contributing nothing to tracking — that is precisely how the
$\lambda$ family was rejected (D76).

---

## Adding things

**Q. How do I add a learner?**

Implement the `Learner` protocol in `learners/base.py` — `init`, `adapt`,
`combine`, `predict`, `state`, `flat_params`, `comm_scalars_per_step` — then
register it:

```python
# src/dekf_bench/learners/registry.py
BUILDERS["my_method"] = MyMethod
DIFFUSING.add("my_method")   # if its combine step actually transmits
POOLING.add("my_method")     # if it consumes the union of every agent's batch
```

Those sets are not cosmetic. `POOLING` decides whether the runner calls
`adapt_pooled(x, y)` with the union or `adapt(node, obs)` per agent, and
`DIFFUSING` drives the communication ledger. A pooled method in neither set will
silently be fed one agent's data.

Add a `configs/learner/my_method.yaml` and it can be named in any experiment.

**Q. How do I add a drift schedule?**

Subclass `DriftSchedule` in `env/drift.py`. The abstract interface is smaller
than it looks — three members:

```python
class MySchedule(DriftSchedule):
    def progress_at(self, step: int) -> float: ...   # how far along, normalised
    @property
    def degrees_scale(self) -> float: ...            # degrees at progress 1.0
    @property
    def name(self) -> str: ...
```

then add a branch to the builder near the bottom of the file.

**Progress is the primitive, not degrees**, and that is what makes one schedule
drive every channel coherently: rotation multiplies progress by a scale, the
class-prior channel interpolates two distributions by the same number, and adding
a channel touches no schedule. Progress is *not* confined to $[0,1]$ —
`sinusoidal` runs over $[-1,1]$ because it goes backwards.

`rotation_at`, `rate_at`, `peak_rate` and `mean_rate` come free from the base
class. Note `rate_at` is a **difference, not a derivative** — it is what a learner
experiences per update, needs no special case for `piecewise` where the
derivative is a delta, and is the quantity the break threshold is a threshold on.

**Q. How do I add a topology or a model?**

Topology: [`configs.md`](configs.md) §3.4. Model: implement the `Model` protocol
(`models/base.py`) and register it in `models/registry.py`; the filter needs only
$\bm\theta\mapsto\bm h(\bm\theta)$ and its Jacobian, so anything differentiable
works.

**Q. How do I add an experiment?**

For a single run, a YAML in `configs/experiment/` is enough. For a sweep, copy
the nearest `run_*.py`: they share a shape — module-level constants, a `cells()`
function, `run_one` per cell, a status JSON, and a printed ETA. Then add a row to
[`experiments.md`](experiments.md) so it is findable.

**Pair every drifting condition with a stationary twin.** Damage is the
difference, so a missing control does not degrade a result, it removes it.

---

## Reading results

**Q. Where do results go and what is in them?**

`results/<run-name>/seed_<k>.parquet`, one per seed, plus `config.yaml` and
`metadata.json` recording exactly what produced them. The columns that matter:
`learner`, `seed`, `t`, `evalset`, `metric`, `value`, plus provenance
(`git_sha`, `run_id`, topology, spectral gap) and the communication counters.

**Q. How do I turn a run back into the numbers in the docs?**

```bash
python scripts/report_breaks.py               # where each method breaks, as a rate
python scripts/report_ekf_sweep.py            # the tuning grid, ranked
python scripts/report_ekf_generalization.py   # damage per drift condition
python scripts/report_cooperation.py          # does cooperation still pay
python scripts/report_recurring.py            # recovery per (J, t') cell
python scripts/report_abrupt_vs_smooth.py     # abrupt against smooth at matched speed
```

**Q. What is the "settled" error everyone quotes?**

The mean over the **last fifth** of the run, on the `current` evalset. Tuning
reads it because a filter that converges fast and then tracks badly is not the
one to carry forward, and the full curve would let the early advantage hide the
late failure.

**Q. Which evalset should I read?**

`current` scores against the distribution at the present rotation — the tracking
question. `prequential` is test-then-train on arriving data. `backward` probes a
rotation the run has already left and is meaningful **only** under `sinusoidal`,
the one schedule that revisits states; elsewhere it asks about a state the model
will never face again.

---

## Debugging

**Q. Can I put a breakpoint in the training loop?**

Yes, and that is a design constraint rather than an accident. Everything on the
algorithmic path runs **in-process** from the IDE run button: no `torch.compile`,
no dynamically generated functions, sweep parallelism serial by default because
breakpoints do not fire in worker processes. `argparse` exists in places but is
never required.

**Q. My results changed between runs. Why?**

They should not. Seeds are drawn per *concern* — `init`, `graph`, `partition`,
`stream`, `priors` — so changing one leaves the others fixed, and the loop
consumes no randomness. Check: are you on the same `git_sha` (it is in
`metadata.json`), the same `dtype`, and the same `device`? `float32` and
`float64` genuinely differ, and the exactness check needs `float64`.

The practical use of separable seeds: to re-draw the graph while holding the data
order fixed, change the graph seed alone and every other stream is unchanged —
so the difference you measure is the graph, not a new sample of everything.

**Q. A cell diverged. Is my run ruined?**

No — a diverged cell is a **measurement**, recorded and skipped so the rest of
the sweep survives. `prior_scale` is a trust region and grids deliberately
bracket its edge (D61). Read the message: it names which guard fired and reports
the state that caused it.

---

## Errors you will actually hit

| Error | What it means |
|---|---|
| `ReferenceError: ... train_reference.py` | The offline reference is not cached. Run that script once. |
| `ConfigError: learner[x].prior_scale must be > 0` | Zero prior variance is a point mass the filter can never move away from. |
| `FilterError: ... diverged at step N` | The mean went non-finite. Usually `prior_scale` too large. |
| `FilterError: ... left the trust region` | The mean is finite but a millionfold from $\bm\theta_0$ — diverged without overflowing. Tune `trust_region_ratio` only if your model's weights genuinely grow. |
| `FilterError: ... lost positive definiteness` | The covariance collapsed. With $\gamma=1$ and $\bm Q=\zero$ it only ever shrinks; give the filter a way to stay uncertain. |
| `MetricError: probabilities must sum to one` | Almost always a diverged belief reaching the metrics. The guards above should catch it first; if this fires, one did not. |
| `BreakError: no rows to pool a noise estimate from` | The learner name is not in the run, or the filter arguments excluded everything. |

---

## Rules that will bite you

Collected because each cost us a result before it became a rule.

- **Tune under the condition you report.** Comparing a drift-tuned method against
  a stationary-tuned baseline is not a comparison (D39).
- **A control must match the mean of what the treatment does**, not just exist.
- **A caveat attached to a figure does not survive the condition being reused.**
  X11 documented "larger $J$ reaches fewer states" as a footnote about one panel;
  it was never promoted to a property of the conditions, so it did not travel
  when those conditions were reused (D74).
- **Quantities fixed by construction are tests, not observations.** A
  non-negative-by-construction penalty that renders negative is a bug the figure
  will show as an unremarkable cell.
- **Extrapolating from a shortened run is only valid for terms linear in what was
  shortened.** A per-step benchmark that omits the recorder underestimated a
  sweep by 4×.
- **A null result at a mild condition is not a null result.** Two claims have
  been withdrawn for exactly this (D72 → D73 → D76).
