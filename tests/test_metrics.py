"""Metrics: accuracy, disagreement, communication, calibration.

The communication tests carry the most weight, because the ledger is what makes
"at identical communication" a checkable statement rather than a claim -- and
because the numbers turned out not to match the plan's wording.
"""

from __future__ import annotations

import math

import pytest
import torch

from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.metrics.calibration import (
    CalibrationScores,
    brier,
    confidence_and_correctness,
    reliability,
    score,
)
from dekf_bench.metrics.classification import (
    AgentScores,
    MetricError,
    accuracy,
    error_rate,
    score_agents,
)
from dekf_bench.metrics.communication import (
    LedgerError,
    centralized_cost,
    diffusion_cost,
    directed_links,
    ledger,
    local_only_cost,
)
from dekf_bench.metrics.disagreement import (
    DisagreementError,
    e_agree,
    e_cent,
    max_pairwise_distance,
    measure,
    network_mean,
)

P = 2908
EDGES = 10  # a 10-agent ring
DTYPE = torch.float64


# =========================================================================== #
# 1. the communication ledger
# =========================================================================== #


def test_a_message_crosses_each_edge_in_both_directions() -> None:
    assert directed_links(EDGES) == 2 * EDGES


@pytest.mark.parametrize(("policy", "vectors"), [("none", 1), ("momentum", 2), ("all", 3)])
def test_mixing_a_moment_means_transmitting_it(policy: str, vectors: int) -> None:
    """A neighbour cannot mix a momentum buffer it was never sent."""
    cost = diffusion_cost("atc", P, EDGES, policy)
    assert cost.vectors_per_link == vectors
    assert cost.scalars_per_step == vectors * P * 2 * EDGES


def test_the_ekf_and_plain_atc_send_exactly_the_same() -> None:
    """The pairing the phase-5 claim rests on: same payload, so any advantage is
    attributable to the second-order update rather than to bandwidth."""
    plain = diffusion_cost("atc", P, EDGES, "none")
    filt = diffusion_cost("ekf", P, EDGES, "none")
    assert plain.scalars_per_step == filt.scalars_per_step == 58_160


def test_the_primary_sgd_baseline_sends_twice_what_the_filter_does() -> None:
    """The finding behind design note D29: X1-X6 run SGD with momentum *mixed*,
    which broadcasts (psi, m) = 2p, while the filter broadcasts psi alone. The
    plan's 'identical communication' wording holds for plain ATC only."""
    primary = diffusion_cost("atc", P, EDGES, "momentum")
    filt = diffusion_cost("ekf", P, EDGES, "none")
    assert primary.scalars_per_step == 2 * filt.scalars_per_step


def test_one_hop_adapt_costs_the_information_pair() -> None:
    """Exactness on a complete graph needs M = V, which exchanges (B, H^T nu) at
    O(p q') instead of O(p)."""
    cost = diffusion_cost("ekf", P, EDGES, "none", adapt_scope="one_hop", fisher_rank=9)
    assert cost.vectors_per_link == 10
    assert cost.scalars_per_step == 10 * P * 2 * EDGES


def test_one_hop_without_a_rank_is_rejected() -> None:
    with pytest.raises(LedgerError, match="positive fisher_rank"):
        diffusion_cost("ekf", P, EDGES, "none", adapt_scope="one_hop")


def test_unknown_mix_policy_is_rejected() -> None:
    with pytest.raises(LedgerError, match="unknown mix policy"):
        diffusion_cost("atc", P, EDGES, "sometimes")


def test_local_only_sends_nothing() -> None:
    """And the gap to it *is* the value of cooperation."""
    cost = local_only_cost("local_only")
    assert cost.scalars_per_step == 0
    assert cost.rounds_per_step == 0
    assert not cost.diffuses


def test_centralized_is_not_on_the_communication_axis() -> None:
    """It is an upper reference, not a competitor: F2 shows it as a horizontal
    line. The notional pooling cost is recorded anyway."""
    cost = centralized_cost("centralized_sgd", n_nodes=10, samples_per_step=2, input_dim=196)
    assert not cost.diffuses
    assert cost.vectors_per_link == 0
    assert cost.scalars_per_step == 10 * 2 * 197


def test_pooling_is_cheaper_than_ring_diffusion() -> None:
    """The uncomfortable fact, recorded rather than hidden: shipping raw samples
    to a centre is far less traffic than diffusing parameters, so the case for
    decentralization has to rest on something other than bandwidth."""
    pooling = centralized_cost("c", 10, 2, 196).scalars_per_step
    diffusing = diffusion_cost("atc", P, EDGES, "none").scalars_per_step
    assert pooling < diffusing / 10


def test_communication_does_not_depend_on_the_data() -> None:
    """An idle agent still takes part in the combine step, so the exchange
    happens regardless of label availability."""
    first = diffusion_cost("atc", P, EDGES, "none")
    second = diffusion_cost("atc", P, EDGES, "none")
    assert first.scalars_per_step == second.scalars_per_step


def test_cost_scales_with_the_graph() -> None:
    sparse = diffusion_cost("atc", P, 9, "none").scalars_per_step  # path
    dense = diffusion_cost("atc", P, 45, "none").scalars_per_step  # complete
    assert dense == 5 * sparse


def test_total_scalars_accumulate_over_the_run() -> None:
    cost = diffusion_cost("atc", P, EDGES, "none")
    assert cost.total_scalars(1500) == cost.scalars_per_step * 1500


def test_the_ledger_reports_cost_relative_to_the_cheapest_diffusing_method() -> None:
    rows = ledger(
        [
            diffusion_cost("plain", P, EDGES, "none"),
            diffusion_cost("momentum", P, EDGES, "momentum"),
            diffusion_cost("adamw", P, EDGES, "all"),
            local_only_cost("local_only"),
        ],
        horizon=1500,
    )
    relative = {row["learner"]: row["relative_to_cheapest_diffusing"] for row in rows}
    assert relative["plain"] == pytest.approx(1.0)
    assert relative["momentum"] == pytest.approx(2.0)
    assert relative["adamw"] == pytest.approx(3.0)
    assert relative["local_only"] is None


def test_every_method_uses_one_exchange_round() -> None:
    """One round per step is the design trade: the consensus alternatives the
    note rejects need C inner rounds each carrying O(p^2)."""
    for cost in (
        diffusion_cost("atc", P, EDGES, "none"),
        diffusion_cost("ekf", P, EDGES, "none", adapt_scope="one_hop", fisher_rank=9),
    ):
        assert cost.rounds_per_step == 1


# =========================================================================== #
# 2. classification
# =========================================================================== #


def test_accuracy_on_a_hand_made_case() -> None:
    predictions = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 9, 9])
    assert accuracy(predictions, targets) == pytest.approx(0.5)
    assert error_rate(predictions, targets) == pytest.approx(0.5)


def test_perfect_and_worthless_predictions() -> None:
    targets = torch.tensor([1, 2, 3])
    assert accuracy(targets, targets) == 1.0
    assert error_rate(targets, targets + 1) == 1.0


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(MetricError, match="do not match targets"):
        accuracy(torch.zeros(3, dtype=torch.int64), torch.zeros(4, dtype=torch.int64))


def test_an_empty_batch_is_rejected() -> None:
    with pytest.raises(MetricError, match="empty batch"):
        accuracy(torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))


def test_agent_scores_aggregate() -> None:
    scores = AgentScores(error_rates=(0.1, 0.2, 0.3))
    assert scores.mean == pytest.approx(0.2)
    assert scores.best == pytest.approx(0.1)
    assert scores.worst == pytest.approx(0.3)
    assert scores.spread == pytest.approx(0.2)
    assert scores.std == pytest.approx(0.1)


def test_a_good_mean_can_hide_a_terrible_spread() -> None:
    """Which is the whole reason spread is reported alongside the mean."""
    tight = AgentScores(error_rates=(0.2, 0.2, 0.2))
    ragged = AgentScores(error_rates=(0.0, 0.0, 0.6))
    assert tight.mean == pytest.approx(ragged.mean)
    assert ragged.spread > tight.spread == 0.0


def test_the_gap_is_measured_against_the_reference() -> None:
    """Q1's headline number: the raw error rate also moves with the drift state
    and the architecture, so the difference is what carries meaning."""
    scores = AgentScores(error_rates=(0.09, 0.11))
    assert scores.gap(reference_error=0.02) == pytest.approx(0.08)


def test_a_single_agent_has_no_spread() -> None:
    scores = AgentScores(error_rates=(0.3,))
    assert scores.spread == 0.0
    assert scores.std == 0.0


def test_error_rates_outside_the_unit_interval_are_rejected() -> None:
    with pytest.raises(MetricError, match=r"must lie in \[0, 1\]"):
        AgentScores(error_rates=(0.5, 1.5))


def test_only_per_agent_rows_are_stored() -> None:
    """Aggregates are derived at plot time, so a new one never needs a re-run."""
    rows = AgentScores(error_rates=(0.1, 0.2)).as_rows()
    assert [row["node_id"] for row in rows] == [0, 1]
    assert all(row["metric"] == "error_rate" for row in rows)


def test_score_agents_keeps_node_order() -> None:
    predictions = {1: torch.tensor([1, 1]), 0: torch.tensor([0, 9])}
    targets = {0: torch.tensor([0, 0]), 1: torch.tensor([1, 1])}
    scores = score_agents(predictions, targets)
    assert scores.error_rates == pytest.approx((0.5, 0.0))


def test_score_agents_rejects_a_missing_agent() -> None:
    with pytest.raises(MetricError, match="predictions cover agents"):
        score_agents({0: torch.tensor([0])}, {0: torch.tensor([0]), 1: torch.tensor([1])})


# =========================================================================== #
# 3. disagreement
# =========================================================================== #


def agents(*vectors: list[float]) -> dict[int, torch.Tensor]:
    return {i: torch.tensor(v, dtype=DTYPE) for i, v in enumerate(vectors)}


def test_agreement_is_exactly_zero_when_the_agents_agree() -> None:
    """What a complete graph produces after one combine step, and the reason X0
    holds inductively."""
    assert e_agree(agents([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])) == 0.0


def test_agreement_matches_a_hand_computation() -> None:
    # mean is (0, 0); each agent is at distance^2 = 1 from it.
    assert e_agree(agents([1.0, 0.0], [-1.0, 0.0])) == pytest.approx(1.0)


def test_network_mean_is_the_average() -> None:
    mean = network_mean(agents([0.0, 0.0], [2.0, 4.0]))
    assert torch.allclose(mean, torch.tensor([1.0, 2.0], dtype=DTYPE))


def test_e_cent_is_zero_when_the_agents_match_centralized() -> None:
    """With a shared theta_0 this holds exactly at t=0, so E_cent measures
    algorithmic divergence and has a meaningful zero (design note D30)."""
    reference = torch.tensor([1.0, 2.0], dtype=DTYPE)
    assert e_cent(agents([1.0, 2.0], [1.0, 2.0]), reference) == 0.0


def test_e_cent_differs_from_e_agree() -> None:
    """Agents can agree with each other while all being far from centralized."""
    parameters = agents([5.0, 5.0], [5.0, 5.0])
    reference = torch.tensor([0.0, 0.0], dtype=DTYPE)
    assert e_agree(parameters) == 0.0
    assert e_cent(parameters, reference) == pytest.approx(50.0)


def test_max_pairwise_catches_one_stray_agent() -> None:
    """E_agree is a mean and can stay small while one agent drifts far off."""
    parameters = agents([0.0], [0.0], [0.0], [0.0], [12.0])
    assert max_pairwise_distance(parameters) == pytest.approx(12.0)


def test_measure_reports_the_norm_for_later_normalisation() -> None:
    """E_agree is absolute and scales with p and the weight magnitudes; logging
    the mean's norm lets any normalisation be derived at plot time."""
    result = measure(agents([3.0, 4.0], [3.0, 4.0]))
    assert result.mean_norm_squared == pytest.approx(25.0)
    assert result.e_cent is None


def test_measure_omits_e_cent_rather_than_faking_it() -> None:
    """local_only and centralized have nothing to compare against; a zero or a
    NaN there would be read as a measurement."""
    rows = measure(agents([1.0], [2.0])).as_rows()
    assert not any(row["metric"] == "e_cent" for row in rows)


def test_mismatched_parameter_shapes_are_rejected() -> None:
    parameters = {0: torch.zeros(3, dtype=DTYPE), 1: torch.zeros(4, dtype=DTYPE)}
    with pytest.raises(DisagreementError, match="different parameter shapes"):
        e_agree(parameters)


def test_non_flat_parameters_are_rejected() -> None:
    with pytest.raises(DisagreementError, match="flat parameter vectors"):
        e_agree({0: torch.zeros(2, 3, dtype=DTYPE)})


def test_a_mismatched_centralized_vector_is_rejected() -> None:
    with pytest.raises(DisagreementError, match="does not match the agents"):
        e_cent(agents([1.0, 2.0]), torch.zeros(3, dtype=DTYPE))


# =========================================================================== #
# 4. calibration
# =========================================================================== #


def probabilities_from(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=DTYPE)


def test_brier_of_a_perfect_prediction_is_zero() -> None:
    probs = probabilities_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert brier(probs, torch.tensor([0, 1])) == pytest.approx(0.0)


def test_brier_of_a_uniform_prediction_matches_the_closed_form() -> None:
    """(q-1)/q for a uniform predictor, whatever the label."""
    q = 4
    probs = torch.full((3, q), 1.0 / q, dtype=DTYPE)
    assert brier(probs, torch.tensor([0, 1, 2])) == pytest.approx((q - 1) / q)


def test_brier_scores_the_whole_distribution_not_just_the_true_class() -> None:
    """A confident wrong second choice must not score the same as a diffuse one."""
    targets = torch.tensor([0])
    concentrated = probabilities_from([[0.5, 0.5, 0.0, 0.0]])
    diffuse = probabilities_from([[0.5, 0.1667, 0.1667, 0.1666]])
    assert brier(concentrated, targets) > brier(diffuse, targets)


def test_probabilities_that_do_not_normalise_are_rejected() -> None:
    with pytest.raises(MetricError, match="must sum to one"):
        brier(probabilities_from([[0.5, 0.2]]), torch.tensor([0]))


def test_reliability_bins_account_for_every_sample() -> None:
    likelihood = Categorical(output_dim=4)
    logits = torch.randn(200, 4, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    curve = reliability(likelihood.mu(logits), torch.randint(0, 4, (200,)), n_bins=10)
    assert curve.total == 200


def test_confidence_of_exactly_one_lands_in_a_bin() -> None:
    """The closed final bin: an edge case that would otherwise drop samples."""
    probs = probabilities_from([[1.0, 0.0]])
    curve = reliability(probs, torch.tensor([0]), n_bins=5)
    assert curve.total == 1
    assert curve.counts[-1] == 1


def test_a_perfectly_calibrated_predictor_has_near_zero_ece() -> None:
    """Half the samples predicted at confidence 1.0 and correct, half at 0.5 and
    right half the time."""
    probs = probabilities_from([[1.0, 0.0]] * 100 + [[0.5, 0.5]] * 100)
    targets = torch.tensor([0] * 100 + [0] * 50 + [1] * 50)
    curve = reliability(probs, targets, n_bins=10)
    assert curve.ece < 0.02


def test_overconfidence_is_signed() -> None:
    """The note predicts the filter's predictive covariance is over-confident, so
    a magnitude-only report could neither confirm nor refute it."""
    probs = probabilities_from([[0.9, 0.1]] * 100)
    half_right = torch.tensor([0] * 50 + [1] * 50)
    assert reliability(probs, half_right, n_bins=10).overconfidence == pytest.approx(0.4)

    under = probabilities_from([[0.6, 0.4]] * 100)
    all_right = torch.zeros(100, dtype=torch.int64)
    assert reliability(under, all_right, n_bins=10).overconfidence == pytest.approx(-0.4)


def test_max_calibration_error_is_at_least_the_ece() -> None:
    """It is unweighted, so a small badly-miscalibrated bin survives averaging."""
    likelihood = Categorical(output_dim=3)
    logits = torch.randn(300, 3, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    curve = reliability(likelihood.mu(logits), torch.randint(0, 3, (300,)), n_bins=15)
    assert curve.max_calibration_error >= curve.ece


def test_empty_bins_do_not_contribute() -> None:
    probs = probabilities_from([[0.99, 0.01]] * 10)
    curve = reliability(probs, torch.zeros(10, dtype=torch.int64), n_bins=20)
    assert sum(1 for count in curve.counts if count == 0) > 0
    assert curve.ece == pytest.approx(0.01, abs=0.02)


def test_confidence_is_the_predicted_classes_probability() -> None:
    probs = probabilities_from([[0.7, 0.3], [0.2, 0.8]])
    confidence, correct = confidence_and_correctness(probs, torch.tensor([0, 0]))
    assert confidence.tolist() == pytest.approx([0.7, 0.8])
    assert correct.tolist() == [1.0, 0.0]


def test_score_bundles_everything_for_one_batch() -> None:
    likelihood = Categorical(output_dim=5)
    logits = torch.randn(64, 5, generator=torch.Generator().manual_seed(2), dtype=DTYPE)
    targets = torch.randint(0, 5, (64,))
    scores = score(logits, targets, likelihood)

    assert isinstance(scores, CalibrationScores)
    assert scores.nll > 0
    assert 0.0 <= scores.brier <= 2.0
    assert 0.0 <= scores.ece <= 1.0
    assert {row["metric"] for row in scores.as_rows()} == {
        "nll",
        "brier",
        "ece",
        "max_calibration_error",
        "overconfidence",
        "mean_confidence",
    }


def test_a_uniform_predictor_scores_log_q() -> None:
    likelihood = Categorical(output_dim=8)
    logits = torch.zeros(16, 8, dtype=DTYPE)
    scores = score(logits, torch.randint(0, 8, (16,)), likelihood)
    assert scores.nll == pytest.approx(math.log(8))
    assert scores.brier == pytest.approx(7 / 8)
