r"""D1d -- why the iterated EKF gets worse with K: overshoot, or a mode that over-fits.

    python scripts/run_iterated_probe.py               # MNIST still, seed 0, the CPU
    python scripts/run_iterated_probe.py --steps 100 200 300 --iterations 8

## The question

D1 (D142) found every filter worse at every $K>1$ than at $K=1$, on both tasks,
the decentralised ones diverging at $K=8$ on MNIST. `iterated_update` is exact
Gauss--Newton on the per-step objective

.. math::
    J(\bm m)=\tfrac12(\bm m-\bm m^-)^{\mathsf T}(\bm P^-)^{-1}(\bm m-\bm m^-)
    +s\,\mathrm{NLL}(\bm y\mid\bm m),

$s$ the information rescaling the filter applies ($1$ for the centralised filter,
$(N/1)^{e}$ for local adapt). Two explanations predict different things:

* **overshoot** -- undamped Gauss--Newton steps past the minimum, so $J$ *rises*
  across iterations. A damped variant (a line search, Levenberg--Marquardt) fixes it.
* **over-fitting** -- the steps reach the minimum, $J$ falls, but the minimum of a
  four-sample batch under a prior tuned for one pass is a worse model: held-out
  error rises while $J$ falls. Damping cannot fix that; the prior has to change.

## The probe

The filters run at $K=1$, exactly as D1's $K=1$ cell (P5.4's config, seed 0), so the
trajectory is the one every $K=1$ result was measured on. At each probed step the
prior $(\bm m^-,\bm P^-)$ and the batch are held fixed and `iterations` Gauss--Newton
steps are taken *beside* the run -- the run itself keeps its one-iteration update.
After each: $J$, the batch NLL, $\lVert\bm m_i-\bm m^-\rVert$, and the error on the
held-out current set. The centralised filter on the pooled batch, and local adapt
at agent 0. CPU only, minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_diffusion_skew import CENTRALIZED, FILTER  # noqa: E402
from run_inner_steps import CENTRAL, HORIZON, LOCAL, config_for  # noqa: E402

from dekf_bench.data.registry import dataset_is_cached, load_dataset  # noqa: E402
from dekf_bench.env.environment import build_environment  # noqa: E402
from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: E402
from dekf_bench.evaluation.protocol import _score_evalset  # noqa: E402
from dekf_bench.learners import diffusion_ekf, ekf  # noqa: E402
from dekf_bench.learners.registry import build_learners  # noqa: E402
from dekf_bench.likelihoods.registry import build_likelihood  # noqa: E402
from dekf_bench.models.registry import build_model_from_config  # noqa: E402
from dekf_bench.runner import simulate  # noqa: E402

DATA_ROOT = ROOT / "data"
CELL, SEED = "still", 0


def objective(model, likelihood, mean, prior, factor, x, y, scale):
    """J at ``mean``, its batch NLL, and its distance from the prior mean."""
    offset = mean - prior
    quadratic = 0.5 * float(offset @ torch.cholesky_solve(offset[:, None], factor)[:, 0])
    logits = model.forward(model.unflatten(mean), x)
    nll = float(likelihood.nll(logits, y, reduction="sum"))
    return quadratic + scale * nll, nll, float(offset.norm())


def held_out(model, likelihood, builder, step, mean) -> float:
    def predict(_node, images):
        return model.forward(model.unflatten(mean), images)

    return _score_evalset(0, builder.at("current", step), predict, likelihood,
                          1000, False).error_rate


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=[SEED])
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 200, 300],
                        help="the steps at which to probe")
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args(argv)
    if "--device" not in (argv if argv is not None else sys.argv[1:]):
        args.device = "cpu"
    if not dataset_is_cached(args.dataset, DATA_ROOT):
        print(f"{args.dataset} is not cached. Run scripts/check_data.py once, then retry.")
        return 1
    train, test = load_dataset(args.dataset, DATA_ROOT, download=False)
    entries = [{"name": CENTRAL, **CENTRALIZED, "inner_steps": 1},
               {"name": LOCAL, **FILTER, "inner_steps": 1}]
    config = config_for(args, "ins_iterated_probe", CELL, entries, seeds=[SEED])
    environment = build_environment(config, SEED, train)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    builder = build_evalsets(config, environment, test)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    theta0 = theta0.to(environment.train.images.device)
    probed = set(args.steps)
    rows: list[tuple] = []
    plain = ekf.iterated_update

    def probe(linearise, mean, covariance, iterations, context=""):
        caller = sys._getframe(1)
        learner = caller.f_locals["self"]
        if learner._name == CENTRAL:
            x, y, scale = caller.f_locals["x"], caller.f_locals["y"], 1.0
        else:
            adapt = caller.f_back
            if adapt.f_locals.get("node") != 0:
                return plain(linearise, mean, covariance, iterations, context)
            x, y = adapt.f_locals["x"], adapt.f_locals["y"]
            scale = (learner._n_nodes / 1) ** learner.information_exponent
        # The diffusion filter keeps the simulation's step; the centralised one counts
        # its calls from 1.
        step = learner._steps - (learner._name == CENTRAL)
        if step in probed:
            factor = torch.linalg.cholesky(covariance)
            start = objective(model, likelihood, mean, mean, factor, x, y, scale)
            rows.append((learner._name, step, 0, *start,
                         held_out(model, likelihood, builder, step, mean)))
            for i in range(1, args.iterations + 1):
                iterate, _ = plain(linearise, mean, covariance, i, context)
                rows.append((learner._name, step, i,
                             *objective(model, likelihood, iterate, mean, factor, x, y, scale),
                             held_out(model, likelihood, builder, step, iterate)))
        return plain(linearise, mean, covariance, iterations, context)

    ekf.iterated_update = probe
    diffusion_ekf.iterated_update = probe
    try:
        print(f"D1d probe: MNIST {CELL}, seed {SEED}, K=1 run to step {max(probed)}, "
              f"{args.iterations} Gauss-Newton steps beside it at {sorted(probed)}, "
              f"on {args.device}\n", flush=True)
        simulate.run(config, environment, learners, builder, likelihood, theta0,
                     verify_observations=False, stop_after=max(probed))
    finally:
        ekf.iterated_update = plain
        diffusion_ekf.iterated_update = plain

    print(f"  {'learner':<24}{'step':>5}{'i':>3}{'J':>14}{'dJ':>12}{'batch NLL':>11}"
          f"{'|m-m-|':>9}{'held-out':>10}")
    for index, (name, step, i, value, nll, distance, error) in enumerate(rows):
        change = "" if i == 0 else f"{value - rows[index - 1][3]:+12.4g}"
        print(f"  {name:<24}{step:>5}{i:>3}{value:>14.6g}{change:>12}{nll:>11.4g}"
              f"{distance:>9.4f}{error:>10.4f}")
    print("\n  overshoot: J rises across iterations. over-fitting: J falls, held-out rises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
