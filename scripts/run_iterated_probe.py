r"""D1d -- why the iterated EKF gets worse with K: overshoot, or a mode that over-fits (MG).

    python scripts/run_iterated_probe.py               # MG stationary, seed 0, the CPU
    python scripts/run_iterated_probe.py --steps 100 200 300 --iterations 8
    python scripts/run_iterated_probe.py --trajectory k2_final   # one arm of the second probe

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
  single step's batch under a prior tuned for one pass is a worse model: held-out
  RMSE rises while $J$ falls. Damping cannot fix that; the prior has to change.

## The probe

The filters run at $K=1$, exactly as D1's $K=1$ cell (M6's stationary config, M4/M5's
selection, seed 0), so the
trajectory is the one every $K=1$ result was measured on. At each probed step the
prior $(\bm m^-,\bm P^-)$ and the batch are held fixed and `iterations` Gauss--Newton
steps are taken *beside* the run -- the run itself keeps its one-iteration update.
After each: $J$, the batch NLL, $\lVert\bm m_i-\bm m^-\rVert$, and the RMSE on the
held-out current set. The centralised filter on the pooled batch, and local adapt
at agent 0. CPU only, minutes.

## The trajectory probe (`--trajectory ARM`)

The first probe found every iterated step harmless: $J$ falls, the held-out RMSE
moves by at most $10^{-3}$. Yet the $K=2$ run matches $K=1$ to step 100 and then drifts
(0.155 → 0.740 by step 1499, against $K=1$'s 0.137). So the damage is carried forward,
and the candidates are what the filter carries: the iterated mean, and a covariance
linearised at the second-to-last iterate rather than the last. The centralised
filter alone, run to `--until`, one arm per process:

* `k1` -- the plain filter;
* `k2_last` -- $K=2$ as D1 ran it, the covariance from the last linearisation ($\bm m_1$);
* `k2_final` -- $K=2$, the covariance re-linearised at the final iterate $\bm m_2$ (the
  textbook IEKF, $K+1$ Jacobians);
* `k2_first` -- the $K=2$ mean with the first linearisation's covariance (at $\bm m^-$).

Every `--every` steps: held-out RMSE, $\operatorname{tr}\bm P$ and $\lVert\bm m\rVert$.

`--gamma` replaces the centralised filter's $\gamma$ (MG's selection is 1, a pure random
walk; MNIST's is 0.9995). The second probe found $\lVert\bm m\rVert$ growing steadily
at $K=2$ under every covariance choice; a $\gamma<1$ pulls the mean back each step.
`--q` replaces its process noise likewise: with $\gamma=0.9995$ the drift slowed but the
belief kept tightening, which points at a $q$ selected for one pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from _args import sweep_parser  # noqa: E402
from run_inner_steps import CENTRAL, HORIZON, LOCAL, config_for  # noqa: E402
from run_m6_comparison import load_settings, r_scale_override  # noqa: E402

from dekf_bench.evaluation.protocol import _score_evalset  # noqa: E402
from dekf_bench.learners import diffusion_ekf, ekf  # noqa: E402
from dekf_bench.learners.ekf import woodbury_update  # noqa: E402
from dekf_bench.learners.registry import build_learners  # noqa: E402
from dekf_bench.likelihoods.registry import build_likelihood  # noqa: E402
from dekf_bench.models.registry import build_model_from_config  # noqa: E402
from dekf_bench.runner import simulate  # noqa: E402
from dekf_bench.runner.task import build_task  # noqa: E402

CELL, SEED = "stationary", 0
ARMS = ["k1", "k2_last", "k2_final", "k2_first"]


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
                          1000, False).rmse


def main(argv: list[str] | None = None) -> int:
    parser = sweep_parser(__doc__.split("\n")[0], horizon=HORIZON, seeds=[SEED])
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 200, 300],
                        help="the steps at which to probe")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--trajectory", choices=ARMS, default=None,
                        help="run the centralised filter with this update instead")
    parser.add_argument("--until", type=int, default=600)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=None,
                        help="the centralised filter's gamma, in place of its selection")
    parser.add_argument("--q", type=float, default=None,
                        help="the centralised filter's process noise, in place of its selection")
    args = parser.parse_args(argv)
    if "--device" not in (argv if argv is not None else sys.argv[1:]):
        args.device = "cpu"
    args.smoke = False
    centralised, diffusion, _m3 = load_settings(False)
    name = CENTRAL
    if args.gamma is not None:
        # The walk fixes gamma = 1 by name; a shrinking transition is its own learner (D71).
        name, centralised = "centralized_ekf_gamma", {**centralised, "gamma": args.gamma}
    if args.q is not None:
        centralised = {**centralised, "process_noise_q": args.q}
    entries = [{"name": name, **centralised, "inner_steps": 1}]
    if args.trajectory is None:
        entries.append({"name": LOCAL, **diffusion, "inner_steps": 1})
    config = config_for(args, "ins_iterated_probe", CELL, entries, [SEED],
                        r_scale_override(False))
    environment, builder = build_task(config, SEED, None, None)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    theta0 = theta0.to(environment.device)
    probed = set(args.steps)
    rows: list[tuple] = []
    plain = ekf.iterated_update
    if args.trajectory:
        return trajectory(args, config, environment, learners, builder, model,
                          likelihood, theta0, plain)

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
        print(f"D1d probe: MG {CELL}, seed {SEED}, K=1 run to step {max(probed)}, "
              f"{args.iterations} Gauss-Newton steps beside it at {sorted(probed)}, "
              f"on {args.device}\n", flush=True)
        simulate.run(config, environment, learners, builder, likelihood, theta0,
                     verify_observations=False, stop_after=max(probed))
    finally:
        ekf.iterated_update = plain
        diffusion_ekf.iterated_update = plain

    print(f"  {'learner':<24}{'step':>5}{'i':>3}{'J':>14}{'dJ':>12}{'batch NLL':>11}"
          f"{'|m-m-|':>9}{'RMSE':>10}")
    for index, (name, step, i, value, nll, distance, error) in enumerate(rows):
        change = "" if i == 0 else f"{value - rows[index - 1][3]:+12.4g}"
        print(f"  {name:<24}{step:>5}{i:>3}{value:>14.6g}{change:>12}{nll:>11.4g}"
              f"{distance:>9.4f}{error:>10.4f}")
    print("\n  overshoot: J rises across iterations. over-fitting: J falls, RMSE rises.")
    return 0



def trajectory(args, config, environment, learners, builder, model, likelihood, theta0,
               plain) -> int:
    """One arm of the trajectory probe: the centralised filter, its update replaced."""
    arm, rows = args.trajectory, []

    def update(linearise, mean, covariance, iterations, context=""):
        if arm == "k1":
            new_mean, new_cov = plain(linearise, mean, covariance, 1, context)
        else:
            new_mean, new_cov = plain(linearise, mean, covariance, 2, context)
            if arm == "k2_final":
                stacked, score = linearise(new_mean)
                _ignored, new_cov = woodbury_update(mean, covariance, stacked, score, context)
            elif arm == "k2_first":
                _ignored, new_cov = plain(linearise, mean, covariance, 1, context)
        step = sys._getframe(1).f_locals["self"]._steps - 1
        if step % args.every == 0 or step == args.until:
            rows.append((step, held_out(model, likelihood, builder, step, new_mean),
                         float(new_cov.diagonal().sum()), float(new_mean.norm())))
            print(f"  {arm:<9}{step:>5}{rows[-1][1]:>9.4f}{rows[-1][2]:>12.4g}"
                  f"{rows[-1][3]:>9.3f}", flush=True)
        return new_mean, new_cov

    ekf.iterated_update = update
    try:
        print(f"D1d trajectory probe: MG {CELL}, seed {SEED}, the centralised filter, arm "
              f"{arm}, gamma {args.gamma if args.gamma is not None else 'as selected'}, "
              f"q {args.q if args.q is not None else 'as selected'}, "
              f"to step {args.until}, on {args.device}\n")
        print(f"  {'arm':<9}{'step':>5}{'RMSE':>9}{'tr P':>12}{'|m|':>9}", flush=True)
        simulate.run(config, environment, learners, builder, likelihood, theta0,
                     verify_observations=False, stop_after=args.until)
    finally:
        ekf.iterated_update = plain
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
