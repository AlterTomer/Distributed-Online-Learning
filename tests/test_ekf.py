"""The centralised EKF.

The load-bearing test is :func:`test_matches_the_exact_kalman_filter`. Everything
else here checks a property; that one checks the filter against a *different
derivation* of the same recursion, in a setting where the EKF is not an
approximation at all, and it is the filter's analogue of X0.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.learners.ekf import CentralizedEKF, FilterError
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.likelihoods.gaussian import Gaussian
from dekf_bench.models.mlp import MLP

# Small enough that a p-by-p inverse is affordable in a test, large enough that
# the low-rank structure Woodbury exploits is real (rank 28 against p = 209).
SMALL = dict(input_size=6, hidden=(5,), output_dim=4, dtype=torch.float64)


def _filter(model: MLP, likelihood: object, **kwargs: object) -> CentralizedEKF:
    settings: dict = dict(transition="identity", prior_scale=0.01)
    settings.update(kwargs)
    filt = CentralizedEKF("ekf", model, likelihood, n_nodes=3, **settings)
    filt.init(model.flatten(model.init_params(torch.Generator().manual_seed(0))))
    return filt


def _batch(model: MLP, size: int = 7, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    side = model.input_size
    x = torch.rand(size, 1, side, side, dtype=torch.float64, generator=generator)
    y = torch.randint(0, model.output_dim, (size,), generator=generator)
    return x, y


# -- the update is the update it claims to be ----------------------------- #


def test_woodbury_matches_the_direct_information_form() -> None:
    r"""$\bm P - \bm A\bm S^{-1}\bm A^\top = (\bm P^{-1}+\sum\bm H^\top\bm\Lambda\bm H)^{-1}$.

    Woodbury is an identity, so this is a test of the implementation rather than
    of the mathematics: the einsums that build $\bar{\bm B}$ from $\bm H$ and
    $\bm G$ are where a transpose or a reshape would go wrong silently.
    """
    model = MLP(**SMALL)
    likelihood = Categorical(output_dim=4)
    x, y = _batch(model)

    filt = _filter(model, likelihood)
    theta0 = filt.flat_params(0).clone()
    scatter = torch.randn(model.num_params, model.num_params, dtype=torch.float64)
    prior = scatter @ scatter.T / model.num_params + 0.5 * torch.eye(
        model.num_params, dtype=torch.float64
    )
    filt._covariance = prior.clone()
    filt.adapt_pooled(x, y)

    params = model.unflatten(theta0)
    logits = model.forward(params, x)
    jacobians = model.per_sample_jacobian(params, x)
    increment = torch.einsum("nqp,nqs,nsr->pr", jacobians, likelihood.fisher(logits), jacobians)
    expected_cov = torch.linalg.inv(torch.linalg.inv(prior) + increment)
    score = torch.einsum("nqp,nq->p", jacobians, likelihood.score(logits, y))

    assert torch.allclose(filt.covariance, expected_cov, atol=1e-9)
    assert torch.allclose(filt.flat_params(0), theta0 + expected_cov @ score, atol=1e-9)


def test_matches_the_exact_kalman_filter() -> None:
    """A linear probe with Gaussian noise is a linear-Gaussian state-space model.

    The probe has no hidden layer, so $\\bm h(\\bm\\theta)=\\bm H\\bm\\theta$
    exactly and the linearisation has no remainder to neglect. The reference runs
    in the **gain** form, which shares no algebra with the Woodbury update, so
    agreement means both are right rather than that one echoes the other.
    """
    gamma, noise, variance, prior, steps, batch = 0.97, 0.01, 0.35, 0.5, 30, 6
    model = MLP(input_size=5, hidden=(), output_dim=3, dtype=torch.float64)
    assert model.is_linear
    likelihood = Gaussian(output_dim=3, variance=variance)
    p = model.num_params

    filt = _filter(
        model, likelihood, transition="scalar", gamma=gamma,
        process_noise_q=noise, prior_scale=prior,
    )
    mean = filt.flat_params(0).clone()
    covariance = prior * torch.eye(p, dtype=torch.float64)
    identity = torch.eye(p, dtype=torch.float64)

    generator = torch.Generator().manual_seed(3)
    for _ in range(steps):
        x = torch.rand(batch, 1, 5, 5, dtype=torch.float64, generator=generator)
        y = torch.randn(batch, 3, dtype=torch.float64, generator=generator)

        mean = gamma * mean
        covariance = gamma**2 * covariance + noise * identity
        jacobian = model.per_sample_jacobian(model.unflatten(mean), x).reshape(batch * 3, p)
        predicted = model.forward(model.unflatten(mean), x).reshape(-1)
        assert torch.allclose(jacobian @ mean, predicted, atol=1e-12), "probe is not linear"

        gain = (
            covariance
            @ jacobian.T
            @ torch.linalg.inv(
                jacobian @ covariance @ jacobian.T
                + variance * torch.eye(batch * 3, dtype=torch.float64)
            )
        )
        mean = mean + gain @ (y.reshape(-1) - predicted)
        covariance = (identity - gain @ jacobian) @ covariance

        filt.adapt_pooled(x, y)
        assert torch.allclose(filt.flat_params(0), mean, atol=1e-9)
        assert torch.allclose(filt.covariance, covariance, atol=1e-9)


def test_the_update_only_removes_uncertainty() -> None:
    """$\\bm P-\\bm P^+\\succeq\\bm 0$: an observation cannot make the filter less sure.

    Fixed by construction, so it is a test rather than a measurement -- the class
    of quantity worth asserting precisely because no run can talk it out of it.
    """
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4), prior_scale=0.05)
    before = filt.covariance.clone()
    filt.adapt_pooled(*_batch(model))
    assert float(torch.linalg.eigvalsh(before - filt.covariance).min()) > -1e-9


def test_covariance_stays_symmetric_and_positive_definite() -> None:
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4), lambda_forget=0.999)
    for step in range(200):
        filt.adapt_pooled(*_batch(model, seed=step))
        covariance = filt.covariance
        assert torch.equal(covariance, covariance.T), f"asymmetric at step {step}"
        torch.linalg.cholesky(covariance)  # raises if it is not positive definite


# -- the two state models --------------------------------------------------- #


def test_gamma_one_is_exactly_the_random_walk() -> None:
    """$\\gamma=1$ *is* the driftless random walk, not merely close to it.

    The two families overlap at exactly this point, which is why the config
    refuses to let a run express it twice (design note D56).
    """
    model = MLP(**SMALL)
    walk = _filter(model, Categorical(output_dim=4), transition="identity", lambda_forget=1.0)
    scalar = _filter(model, Categorical(output_dim=4), transition="scalar", gamma=1.0)
    for step in range(15):
        batch = _batch(model, seed=step)
        walk.adapt_pooled(*batch)
        scalar.adapt_pooled(*batch)
    assert torch.equal(walk.flat_params(0), scalar.flat_params(0))
    assert torch.equal(walk.covariance, scalar.covariance)


def test_gamma_shrinks_the_mean_toward_the_origin() -> None:
    """$\\bm F=\\gamma\\bm I$ is weight decay in state-space form (design note D26).

    Asserted against the closed form $\\gamma^T$ so the test says *how much*
    rather than only "it got smaller" -- with no data, the prediction step is the
    only thing acting.
    """
    model = MLP(**SMALL)
    gamma, steps = 0.99, 50
    filt = _filter(model, Categorical(output_dim=4), transition="scalar", gamma=gamma)
    start = filt.flat_params(0).norm()
    empty = (torch.empty(0, 1, 6, 6, dtype=torch.float64), torch.empty(0, dtype=torch.int64))
    for _ in range(steps):
        filt.adapt_pooled(*empty)
    assert filt.flat_params(0).norm() == pytest.approx(start * gamma**steps, rel=1e-9)


def test_an_empty_batch_predicts_without_updating() -> None:
    """No labels anywhere still costs certainty: that is what a prediction step is."""
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4), lambda_forget=0.99)
    before_mean, before_var = filt.flat_params(0).clone(), filt.covariance.diagonal().clone()
    filt.adapt_pooled(torch.empty(0, 1, 6, 6, dtype=torch.float64), torch.empty(0, dtype=torch.int64))
    assert torch.equal(filt.flat_params(0), before_mean)
    assert torch.all(filt.covariance.diagonal() > before_var)


# -- failure modes are reported, not absorbed ------------------------------- #


def test_divergence_is_raised_rather_than_returned_as_nan() -> None:
    r"""Too large a $\sigma_0^2$ diverges, and must say so (design note D61).

    The mean update is a Gauss--Newton step whose trust region is $\bm P$, so a
    large prior overshoots on the very first step and never recovers. A NaN
    reaching the metrics would average into a seed mean and look like a bad
    result rather than an absent one.
    """
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4), prior_scale=1e12)
    with pytest.raises(FilterError, match="diverged at step"):
        for step in range(50):
            filt.adapt_pooled(*_batch(model, seed=step))


def test_persistence_tracks_the_evaluation_cadence() -> None:
    """A persistence in *points* changes meaning when the cadence changes.

    Lives here rather than in the break tests because X13 is what surfaced it:
    its tuning pass evaluates every 25 steps and its measurement pass every 5,
    so an unchanged ``persistence=3`` would demand 75 steps in one and 15 in the
    other while looking identical in the call.
    """
    from dekf_bench.metrics.breaks import (
        DEFAULT_PERSISTENCE,
        DEFAULT_PERSISTENCE_STEPS,
        persistence_for,
    )

    assert persistence_for(25) == DEFAULT_PERSISTENCE
    assert persistence_for(5) == 15
    assert persistence_for(1) == 75
    for cadence in (1, 5, 25, 50):
        assert persistence_for(cadence) * cadence == pytest.approx(
            DEFAULT_PERSISTENCE_STEPS, abs=cadence
        )
    # A window shorter than one interval still needs a point to test.
    assert persistence_for(200) == 1


def test_a_singular_innovation_covariance_is_reported_as_divergence() -> None:
    r"""An enormous prior breaks the Woodbury solve *before* the mean goes bad.

    $\bm I+\bar{\bm B}^{\mathsf T}\bm P\bar{\bm B}$ is positive definite by
    construction, so its Cholesky can only fail when $\bm P$ has grown until the
    identity is negligible beside it. That is the same divergence as D61, caught
    a step earlier, and it must surface as a ``FilterError`` -- a raw
    ``LinAlgError`` is not a ``LearnerError`` and would escape a sweep's handler
    and kill the whole run.
    """
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4), prior_scale=1e30)
    with pytest.raises(FilterError, match="singular|diverged"):
        for step in range(30):
            filt.adapt_pooled(*_batch(model, seed=step))


def test_per_agent_adapt_is_refused() -> None:
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4))
    with pytest.raises(FilterError, match="pooled batch"):
        filt.adapt(0, None)


def test_stepping_before_init_is_refused() -> None:
    model = MLP(**SMALL)
    filt = CentralizedEKF("ekf", model, Categorical(output_dim=4), n_nodes=3)
    with pytest.raises(FilterError, match="no belief"):
        filt.adapt_pooled(*_batch(model))


def test_theta0_must_fit_the_model() -> None:
    model = MLP(**SMALL)
    filt = CentralizedEKF("ekf", model, Categorical(output_dim=4), n_nodes=3)
    with pytest.raises(FilterError, match="parameters"):
        filt.init(torch.zeros(model.num_params + 1, dtype=torch.float64))


# -- one belief, shared ----------------------------------------------------- #


def test_every_agent_reports_the_same_belief() -> None:
    """Centralized means one belief, so agreement is identically zero.

    The covariance is handed back rather than cloned: ten copies of a 68 MB
    matrix at the real model size would carry no information.
    """
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4))
    filt.adapt_pooled(*_batch(model))
    for node in range(1, 3):
        assert torch.equal(filt.flat_params(node), filt.flat_params(0))
        assert filt.state(node).extras["P"] is filt.state(0).extras["P"]


def test_unknown_agents_are_refused() -> None:
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4))
    with pytest.raises(FilterError, match="no agent"):
        filt.flat_params(3)


def test_the_filter_transmits_nothing() -> None:
    model = MLP(**SMALL)
    assert _filter(model, Categorical(output_dim=4)).comm_scalars_per_step(20) == 0


def test_logit_covariance_is_positive_semidefinite() -> None:
    r"""$\bm H\bm P\bm H^\top$ is a covariance, whatever else it approximates."""
    model = MLP(**SMALL)
    filt = _filter(model, Categorical(output_dim=4))
    x, y = _batch(model)
    filt.adapt_pooled(x, y)
    spread = filt.logit_covariance(0, x)
    assert spread.shape == (x.shape[0], 4, 4)
    for sample in spread:
        assert torch.allclose(sample, sample.T, atol=1e-12)
        assert float(torch.linalg.eigvalsh(sample).min()) > -1e-10
