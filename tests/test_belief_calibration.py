r"""P5.11 / P5.14: scoring the belief, the covariance scale kappa*, and the pre-combine hook.

The metric has to earn trust before it is read, so the first half checks it on
synthetic data whose right answer is known: a predictive whose covariance is too
small must return kappa* > 1, too large kappa* < 1, and a Gaussian with a known
scale must recover it. The second half checks the plumbing: the diffusion filter
keeps exactly the belief combine consumed, the runner asks for it only when it will
be scored, and nothing changes for a run that does not ask.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from dekf_bench.data.mnist import ImageSplit
from dekf_bench.env.environment import build_environment
from dekf_bench.evaluation.evalsets import build_evalsets
from dekf_bench.learners.registry import build_learners
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.metrics.belief import (
    KAPPA_GRID,
    classification_scores,
    gaussian_kappa_scan,
    gaussian_scores,
    kappa_scan,
    temperature_star,
)
from dekf_bench.metrics.classification import MetricError
from dekf_bench.models.registry import build_model_from_config
from dekf_bench.runner import simulate
from dekf_bench.utils.config import ConfigError, load_config

# =========================================================================== #
# the metric, on data whose answer is known
# =========================================================================== #


def test_the_grid_holds_plugin_and_belief_and_is_sorted() -> None:
    assert KAPPA_GRID[0] == 0.0 and 1.0 in KAPPA_GRID
    assert list(KAPPA_GRID) == sorted(KAPPA_GRID)


def test_a_grid_without_zero_and_one_is_refused() -> None:
    with pytest.raises(MetricError, match="must contain 0"):
        kappa_scan(lambda kappa: kappa, grid=(0.5, 2.0))


def test_the_gaussian_scan_recovers_a_known_scale() -> None:
    """Residuals drawn at 3x the reported model variance: kappa* must find ~3."""
    generator = torch.Generator().manual_seed(0)
    model_variance = torch.rand(40_000, generator=generator, dtype=torch.float64) + 0.5
    noise_variance = torch.full_like(model_variance, 0.2)
    truth = 3.0 * model_variance + noise_variance
    residuals = torch.randn(40_000, generator=generator, dtype=torch.float64) * truth.sqrt()
    scan = gaussian_kappa_scan(residuals, model_variance, noise_variance)
    assert scan.kappa_star == pytest.approx(3.0, rel=0.15)
    assert not scan.at_edge
    assert scan.nll_at_star <= scan.nll_at_one


def test_gaussian_scores_agree_with_the_moment_estimate() -> None:
    """Scan and variance matching are two routes to the same scale."""
    generator = torch.Generator().manual_seed(7)
    s = torch.rand(40_000, 31, generator=generator, dtype=torch.float64) + 0.5
    noise = torch.full_like(s, 0.2)
    targets = torch.randn(40_000, 31, generator=generator, dtype=torch.float64) \
        * (2.0 * s + noise).sqrt()
    scores = gaussian_scores(torch.zeros_like(targets), targets, s, noise)
    assert scores["kappa_star"] == pytest.approx(2.0, rel=0.1)
    assert scores["kappa_moment"] == pytest.approx(2.0, rel=0.05)
    assert scores["belief_nll"] < scores["plugin_nll"]


def test_an_overcovering_noise_floors_raw_kappa_but_not_the_tempered_one() -> None:
    """R twice the true noise and a spread that tracks the residuals: kappa* sits at
    0, the moment estimate goes negative, and once R is rescaled the spread counts.

    Not at its true 1: rho* is fitted first on the plug-in predictive, so it takes up
    the spread's *average* and leaves kappa only the per-input part (0.2 here) --
    the reason the tempered reading is exploratory rather than the scale of P.
    """
    generator = torch.Generator().manual_seed(8)
    s = torch.rand(40_000, 31, generator=generator, dtype=torch.float64) * 0.3
    true_noise = torch.full_like(s, 0.2)
    targets = torch.randn(40_000, 31, generator=generator, dtype=torch.float64) \
        * (s + true_noise).sqrt()
    scores = gaussian_scores(torch.zeros_like(targets), targets, s, 2.0 * true_noise)
    assert scores["kappa_star"] == 0.0
    assert scores["kappa_moment"] < 0.0
    assert scores["noise_scale_star"] < 1.0
    assert scores["tempered_kappa_star"] > 0.0


def _synthetic(reported_fraction: float, n: int = 20_000):
    """Labels drawn through true logit noise; the belief reports a fraction of it."""
    generator = torch.Generator().manual_seed(1)
    means = 2.0 * torch.randn(n, 10, generator=generator, dtype=torch.float64)
    true_variance = 4.0
    noisy = means + true_variance**0.5 * torch.randn(n, 10, generator=generator,
                                                     dtype=torch.float64)
    targets = torch.multinomial(torch.softmax(noisy, dim=-1), 1, generator=generator).squeeze(1)
    covariance = torch.diag_embed(torch.full((n, 10), reported_fraction * true_variance,
                                             dtype=torch.float64))
    return means, covariance, targets


def test_an_overconfident_belief_gets_kappa_above_one() -> None:
    means, covariance, targets = _synthetic(reported_fraction=0.25)
    scores = classification_scores(means, covariance, targets, mc_samples=32,
                                   generator=torch.Generator().manual_seed(2))
    assert scores["kappa_star"] > 1.5


def test_a_conservative_belief_gets_kappa_below_one() -> None:
    means, covariance, targets = _synthetic(reported_fraction=4.0)
    scores = classification_scores(means, covariance, targets, mc_samples=32,
                                   generator=torch.Generator().manual_seed(2))
    assert scores["kappa_star"] < 0.7


def test_the_temperature_scan_recovers_a_known_temperature() -> None:
    generator = torch.Generator().manual_seed(4)
    logits = 3.0 * torch.randn(30_000, 10, generator=generator, dtype=torch.float64)
    targets = torch.multinomial(torch.softmax(logits / 2.0, dim=-1), 1,
                                generator=generator).squeeze(1)
    assert temperature_star(logits, targets) == pytest.approx(2.0, rel=0.1)


def test_an_underconfident_mean_floors_raw_kappa_but_not_the_tempered_one() -> None:
    """The case the tempered reading exists for (D120): the mean is too soft, so any
    spread makes the raw predictive worse and kappa* sits at 0 -- yet the spread is
    genuinely informative, and once the mean is tempered the scan finds it.

    Not guaranteed in general: with a larger spread (scale 8 rather than 1) the raw
    kappa* came out at 0.35 despite the soft mean, because softening the
    high-variance inputs paid anyway. So the runner applies its degeneracy rule to
    what is measured rather than assuming the floor.
    """
    generator = torch.Generator().manual_seed(5)
    n = 30_000
    means = torch.randn(n, 10, generator=generator, dtype=torch.float64)
    spread = torch.rand(n, 1, generator=generator, dtype=torch.float64) * 1.0
    noisy = means + spread.sqrt() * torch.randn(n, 10, generator=generator,
                                                 dtype=torch.float64)
    # Labels follow logits three times sharper than the mean the belief reports.
    targets = torch.multinomial(torch.softmax(3.0 * noisy, dim=-1), 1,
                                generator=generator).squeeze(1)
    covariance = torch.diag_embed(spread.expand(n, 10))
    scores = classification_scores(means, covariance, targets, mc_samples=8,
                                   generator=torch.Generator().manual_seed(6))
    assert scores["kappa_star"] == 0.0
    assert scores["temperature_star"] < 1.0
    assert scores["tempered_kappa_star"] > 0.0


def test_a_belief_with_no_spread_scores_exactly_as_the_plugin() -> None:
    """The Sigma -> 0 limit: an uncertainty correction that does not vanish there
    is wrong. The scan is then flat, so its minimum is the grid's first point."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(64, 10, generator=generator, dtype=torch.float64)
    targets = torch.randint(0, 10, (64,), generator=generator)
    scores = classification_scores(logits, torch.zeros(64, 10, 10, dtype=torch.float64),
                                   targets, mc_samples=8, generator=generator)
    assert scores["belief_nll"] == pytest.approx(scores["plugin_nll"], abs=1e-12)
    assert scores["belief_nll_mc"] == pytest.approx(scores["plugin_nll"], abs=1e-12)
    assert scores["kappa_star"] == 0.0 and scores["kappa_star_at_edge"] == 1.0


# =========================================================================== #
# the plumbing
# =========================================================================== #

STEPS = 12
FILTER = {"transition": "scalar", "gamma": 0.9995, "lambda_forget": 1.0,
          "process_noise_q": 6.0e-4, "prior_scale": 1.0e-3}


@pytest.fixture(scope="module")
def data() -> tuple[ImageSplit, ImageSplit]:
    generator = torch.Generator().manual_seed(0)
    return (
        ImageSplit(images=torch.rand(4000, 1, 28, 28, generator=generator),
                   labels=torch.randint(0, 10, (4000,), generator=generator), split="train"),
        ImageSplit(images=torch.rand(400, 1, 28, 28, generator=generator),
                   labels=torch.randint(0, 10, (400,), generator=generator), split="test"),
    )


def _setup(data, belief: bool, **eval_overrides: Any):
    train, test = data
    config = load_config("x1_stationary", overrides={
        "run": {"horizon": STEPS, "eval_every": 4},
        "graph": {"n_nodes": 4},
        "learners": [
            {"name": "centralized_ekf_gamma", **FILTER},
            {"name": "diffusion_ekf", **FILTER},
            {"name": "diffusion_ekf_full", **FILTER},
            {"name": "centralized_sgd", "lr": 0.01},
        ],
        "eval": {"evalsets": ["prequential", "current"], "belief_calibration": belief,
                 "belief_subset": 24, "belief_from": 0.5, "belief_mc_samples": 8,
                 **eval_overrides},
    })
    environment = build_environment(config, 0, train)
    model = build_model_from_config(config)
    likelihood = Categorical(10)
    learners = build_learners(config, model, likelihood)
    theta0 = model.flatten(model.init_params(environment.seeds.torch_generator("init")))
    return config, environment, learners, build_evalsets(config, environment, test), \
        likelihood, theta0


def _rows(records) -> list[dict[str, Any]]:
    return [row for record in records for row in record.rows]


def test_the_combine_keeps_exactly_what_it_consumed(data) -> None:
    """psi is the adapted mean; under mean-only sharing P^psi *is* the kept P,
    while full sharing replaces it with the mixture."""
    _config, environment, learners, _e, _l, theta0 = _setup(data, belief=False)
    observations = environment.step(0)
    for name in ("diffusion_ekf", "diffusion_ekf_full"):
        learner = learners[name]
        learner.init(theta0)
        intermediates = {v: learner.adapt(v, observations[v]) for v in range(4)}
        learner.retain_pre_combine = True
        learner.combine(intermediates, environment.graph.weights)
        for node in range(4):
            psi, cov_psi = learner.belief(node, "pre")
            mean, covariance = learner.belief(node, "post")
            assert psi is intermediates[node].psi
            assert not torch.equal(psi, mean)
            if name == "diffusion_ekf":
                assert cov_psi is covariance
            else:
                assert not torch.equal(cov_psi, covariance)
        learner.release_pre_combine()
        with pytest.raises(Exception, match="no pre-combine belief"):
            learner.belief(0, "pre")


def test_the_centralised_filter_has_no_pre_combine_stage(data) -> None:
    _config, _environment, learners, _e, _l, theta0 = _setup(data, belief=False)
    learner = learners["centralized_ekf_gamma"]
    learner.init(theta0)
    assert learner.belief_stages() == ("post",)
    with pytest.raises(Exception, match="no 'pre' belief"):
        learner.belief(0, "pre")


def test_a_run_that_does_not_ask_logs_no_belief_rows(data) -> None:
    records = simulate.run(*_setup(data, belief=False))
    metrics = {row["metric"] for row in _rows(records)}
    assert not any("kappa" in metric or "belief_" in metric for metric in metrics)


def test_beliefs_are_scored_in_the_window_for_the_filters_only(data) -> None:
    config, environment, learners, evalsets, likelihood, theta0 = _setup(data, belief=True)
    records = simulate.run(config, environment, learners, evalsets, likelihood, theta0)
    rows = [row for row in _rows(records) if row["metric"] in ("kappa_star", "pre_kappa_star")]
    by_learner: dict[str, set[tuple[str, int]]] = {}
    for row in rows:
        by_learner.setdefault(row["learner"], set()).add((row["metric"], row["t"]))
    # Full evaluations at 0, 4, 8, 11; the window opens at 0.5 * 12 = 6.
    window = {8, 11}
    assert set(by_learner) == {"centralized_ekf_gamma", "diffusion_ekf", "diffusion_ekf_full"}
    assert by_learner["centralized_ekf_gamma"] == {("kappa_star", t) for t in window}
    for name in ("diffusion_ekf", "diffusion_ekf_full"):
        assert by_learner[name] == {(m, t) for m in ("kappa_star", "pre_kappa_star")
                                    for t in window}
    # Every agent is scored, and the filters let go of the retained beliefs.
    nodes = {row["node_id"] for row in rows if row["learner"] == "diffusion_ekf"}
    assert nodes == {0, 1, 2, 3}
    assert all(not learner._pre_combine for learner in learners.values()
               if hasattr(learner, "_pre_combine"))


def test_the_monte_carlo_check_is_reproducible(data) -> None:
    def mc(records) -> list[float]:
        return [row["value"] for row in _rows(records) if row["metric"] == "belief_nll_mc"]
    first = mc(simulate.run(*_setup(data, belief=True)))
    second = mc(simulate.run(*_setup(data, belief=True)))
    assert first and first == second


def test_belief_calibration_needs_the_current_set() -> None:
    with pytest.raises(ConfigError, match="must include"):
        load_config("x1_stationary", overrides={
            "eval": {"evalsets": ["prequential", "canonical"], "belief_calibration": True}})
