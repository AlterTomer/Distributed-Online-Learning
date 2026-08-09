r"""X4 (sparsity) and X6 (non-IID) — the remaining phase-4 experiments.

Run this file directly.

    python scripts/run_sparsity_sweep.py            # both
    python scripts/run_sparsity_sweep.py x4         # just the sparsity grid
    python scripts/run_sparsity_sweep.py x6 --fresh

## X4 — where does sparsity bite? (Q4)

Grid: $n \in \{1,2,4,8\}$ samples per agent per step, crossed with
$\pi_\text{lab} \in \{0.25, 0.5, 1.0\}$ label availability. Produces F6, a
heatmap of the final gap.

**$T = 750$, not 1500.** The largest cell, $n = 8$, consumes
$10 \times 8 \times 750 = 60000$ -- the entire training split, exactly. Holding
the horizon fixed across cells is what makes the heatmap comparable; letting each
cell run until its own shard is exhausted would confound sparsity with run
length.

**An idle step consumes no data** (design note D24), so $\pi_\text{lab}$ changes
how often an agent updates, never how fast its shard runs out. That is why the
budget above depends on $n$ alone.

This axis matters more than "sparsity sweep" suggests: §2.2 of `results.md`
measured the cooperation gap collapsing sixfold from $n=2$ to $n=10$, so $n$
moves the very effect Q2 exists to measure.

## X6 — does cooperation survive label skew? (Q2)

Dirichlet partition at $\beta \in \{0.1, 1.0, 100\}$: strong skew, mild skew, and
effectively IID. `balance_sizes` stays on, because with it off the draw leaves
some agents far below $60000/N$ and the config-time budget check -- which assumes
equal shards -- does not protect the run.

Non-IID is where distributed learning becomes genuinely hard: an agent that only
ever sees three digits cannot learn the other seven alone, so the value of
communication should be at its largest here.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dekf_bench.data.mnist import is_cached, load_mnist  # noqa: E402
from dekf_bench.env.environment import build_environment  # noqa: E402
from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: E402
from dekf_bench.learners.registry import build_learners  # noqa: E402
from dekf_bench.likelihoods.categorical import Categorical  # noqa: E402
from dekf_bench.models.registry import build_model_from_config  # noqa: E402
from dekf_bench.recording import recorder as rec  # noqa: E402
from dekf_bench.recording.schema import RunContext  # noqa: E402
from dekf_bench.runner import simulate  # noqa: E402
from dekf_bench.utils.config import load_config  # noqa: E402
from dekf_bench.utils.determinism import git_revision  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these, then run the file.
# ---------------------------------------------------------------------------
X4_SAMPLES = [1, 2, 4, 8]
X4_LABEL_AVAILABILITY = [0.25, 0.5, 1.0]
X4_HORIZON = 750

X6_BETAS = [0.1, 1.0, 100.0]
X6_HORIZON = 1500

# atc_plain is included for the same reason as in X1, X2 and X5: it is the
# p-per-link variant, and therefore the baseline phase 5 is actually measured
# against. Non-IID is where the gaps are largest, so leaving it out would omit
# the payload-matched number from exactly the regime that matters most.
LEARNERS = [
    "centralized_sgd",
    "diffusion_sgd_atc",
    "diffusion_sgd_atc_plain",
    "local_only",
]
#: Five, matching the headline runs. Three was a compromise when X4 had 12 cells
#: and three learners; with the payload-matched variant added, the payload costs
#: it reports (0.010-0.019) sit close enough to the seed noise that three seeds
#: could not rank them.
SEEDS = [0, 1, 2, 3, 4]
FRESH = False

DATA_ROOT = ROOT / "data"


def config_for_x4(n: int, label_availability: float):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {
                "name": f"x4_n{n}_pi{label_availability}",
                "horizon": X4_HORIZON,
                "seeds": SEEDS,
            },
            "env": {"samples_per_node_per_step": n, "label_availability": label_availability},
            "learners": [{"name": name} for name in LEARNERS],
        },
    )


def config_for_x6(beta: float):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": f"x6_beta{beta}", "horizon": X6_HORIZON, "seeds": SEEDS},
            "env": {"partition": {"kind": "dirichlet", "beta": beta}},
            "learners": [{"name": name} for name in LEARNERS],
        },
    )


def run_one(config, train, test, fresh: bool) -> None:
    out_dir = ROOT / config.run.out_dir / config.run.name
    if fresh and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    rec.write_metadata(out_dir, config, {"experiment": config.run.name})

    for seed in config.run.seeds:
        environment = build_environment(config, seed, train)
        model = build_model_from_config(config)
        likelihood = Categorical(config.model.output_dim)
        learners = build_learners(config, model, likelihood)
        theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
        sha, _dirty = git_revision(ROOT)
        recorder = rec.Recorder(
            out_dir,
            RunContext.from_config(
                config, seed, environment.graph, run_id=rec.new_run_id(), git_sha=sha
            ),
            config,
        )
        simulate.run(
            config,
            environment,
            learners,
            build_evalsets(config, environment, test),
            likelihood,
            theta0,
            recorder=recorder,
            verify_observations=False,
            progress_every=0,
        )
        recorder.finalize()


def main(which: str = "both", fresh: bool = FRESH) -> int:
    if not is_cached(DATA_ROOT):
        print("MNIST is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_mnist(DATA_ROOT, download=False)
    started = time.perf_counter()

    if which in ("both", "x4"):
        cells = list(itertools.product(X4_SAMPLES, X4_LABEL_AVAILABILITY))
        print(f"X4: {len(cells)} cells, {len(SEEDS)} seeds, T={X4_HORIZON}")
        for index, (n, pi) in enumerate(cells, start=1):
            config = config_for_x4(n, pi)
            print(
                f"  [{index}/{len(cells)}] n={n} pi_lab={pi}  "
                f"(shard budget {10 * n * X4_HORIZON}/60000)  "
                f"{(time.perf_counter() - started) / 60:.0f} min elapsed"
            )
            run_one(config, train, test, fresh)

    if which in ("both", "x6"):
        print(f"\nX6: {len(X6_BETAS)} betas, {len(SEEDS)} seeds, T={X6_HORIZON}")
        for index, beta in enumerate(X6_BETAS, start=1):
            config = config_for_x6(beta)
            print(
                f"  [{index}/{len(X6_BETAS)}] beta={beta}  "
                f"{(time.perf_counter() - started) / 60:.0f} min elapsed"
            )
            run_one(config, train, test, fresh)

    print(f"\ndone in {(time.perf_counter() - started) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(
        main(args[0].lower() if args else "both", fresh=FRESH or "--fresh" in sys.argv)
    )
