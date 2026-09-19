r"""M1 -- the series task's gates, before any M-series number is believed.

Three identities, each the Mackey--Glass counterpart of one MNIST already passes:

1. **ATC equals centralised SGD on a complete graph** (X0), now through a causal
   Transformer and a Gaussian likelihood.
2. **On the linear AR the centralised EKF is the exact Kalman posterior.** $h$ is
   linear in $\bm\theta$, so with $\gamma=1$ and $Q=0$ the filter after $t$ steps
   must equal the batch posterior $\bm P_t^{-1}=\bm P_0^{-1}+\sum\bm J^{\mathsf T}
   \bm R^{-1}\bm J$, $\bm m_t=\bm P_t(\bm P_0^{-1}\bm m_0+\sum\bm J^{\mathsf T}\bm
   R^{-1}\bm y)$ -- computed here by direct inversion, not by the Woodbury path the
   filter uses, so the two are independent. `ex:linear`: no linearisation error
   anywhere to hide behind.
3. **One-hop diffusion equals the centralised EKF on a complete graph**
   (`prop:complete_graph`), at both linearisation points, on both models.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.registry import build_likelihood
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.runner import simulate
from dekf_bench.runner.task import build_task
from dekf_bench.utils.config import load_config

FILTER = {
    "transition": "scalar", "gamma": 1.0, "process_noise_q": 0.0,
    "lambda_forget": 1.0, "prior_scale": 0.01,
}
STEPS = 6


def _config(model: str, learners: list[dict]):
    return load_config(
        "x1_stationary",
        overrides={
            "run": {"name": "m1_gate", "horizon": STEPS, "seeds": [0], "dtype": "float64",
                    "device": "cpu"},
            # Metropolis weights on a complete graph are uniform: 1/N everywhere.
            "graph": {"topology": "complete", "weights": "metropolis", "n_nodes": 5},
            "env": {"dataset": "mackey_glass", "series": {"burn_in": 50.0}},
            "model": {"name": model, "likelihood": "gaussian", "output_dim": 31,
                      "observation_variance": 0.02},
            "learners": learners,
        },
    )


def _drive(config, check):
    environment, _evalsets = build_task(config, 0)
    model = build_model_from_config(config)
    likelihood = build_likelihood(config)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    for learner in learners.values():
        learner.init(theta0)
    nodes = list(range(environment.n_nodes))
    history = []
    for step in range(STEPS):
        observations = environment.step(step)
        history.append(observations)
        pooled_x, pooled_y = environment.pool(observations)
        for name, learner in learners.items():
            simulate._advance(
                learner, name, observations, nodes, environment.graph.weights, pooled_x, pooled_y
            )
        check(learners, nodes)
    return learners, model, likelihood, theta0, history


def test_atc_equals_centralised_sgd_through_the_transformer() -> None:
    # The shared gradient is the Gaussian score summed over 31 positions and scaled
    # by 1/R, about 775x a per-value MSE gradient at R = 0.02: an MSE-scale rate
    # diverges (a first version of this test reached 5e101). 1e-5 here is ~0.008 in
    # MSE units. The identity holds at any rate in exact arithmetic, but a diverging
    # trajectory amplifies rounding until float64 cannot show it.
    sgd = {"optimizer": "sgd", "lr": 1.0e-5, "momentum": 0.0, "mix_optimizer_state": "none"}
    config = _config(
        "causal_transformer",
        [{"name": "centralized_sgd", **sgd}, {"name": "diffusion_sgd_atc_plain", **sgd}],
    )
    worst = [0.0]

    def check(learners, nodes):
        reference = learners["centralized_sgd"].flat_params(0)
        for v in nodes:
            gap = (learners["diffusion_sgd_atc_plain"].flat_params(v) - reference).abs().max()
            worst[0] = max(worst[0], float(gap))

    _drive(config, check)
    assert worst[0] < 1e-12, worst[0]


def test_the_ekf_on_the_linear_ar_is_the_exact_kalman_posterior() -> None:
    config = _config("linear_ar", [{"name": "centralized_ekf_walk", **FILTER}])
    learners, model, likelihood, theta0, history = _drive(config, lambda *_: None)

    p = model.num_params
    precision = torch.eye(p, dtype=torch.float64) / FILTER["prior_scale"]
    information = precision @ theta0
    params = model.unflatten(theta0)
    for observations in history:
        for observation in observations.values():
            jacobian = model.per_sample_jacobian(params, observation.x).reshape(-1, p)
            precision = precision + jacobian.T @ jacobian / 0.02
            information = information + jacobian.T @ observation.y.reshape(-1) / 0.02
    covariance = torch.linalg.inv(precision)
    mean = covariance @ information

    ekf = learners["centralized_ekf_walk"]
    assert torch.allclose(ekf.flat_params(0), mean, atol=1e-9, rtol=1e-9)
    assert torch.allclose(ekf.covariance, covariance, atol=1e-12, rtol=1e-8)


@pytest.mark.parametrize("model", ["linear_ar", "causal_transformer"])
@pytest.mark.parametrize(
    "variant", ["diffusion_ekf_onehop_mean", "diffusion_ekf_onehop_mean_receiver"]
)
def test_one_hop_diffusion_equals_the_centralised_filter(model: str, variant: str) -> None:
    config = _config(model, [{"name": "centralized_ekf_walk", **FILTER}, {"name": variant, **FILTER}])
    worst = [0.0]

    def check(learners, nodes):
        reference = learners["centralized_ekf_walk"].flat_params(0)
        for v in nodes:
            gap = (learners[variant].flat_params(v) - reference).abs().max()
            worst[0] = max(worst[0], float(gap))

    _drive(config, check)
    assert worst[0] < 1e-10, worst[0]
