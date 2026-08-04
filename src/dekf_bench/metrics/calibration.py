"""Calibration: NLL, Brier, and reliability.

These come free once the likelihood exists, and they matter for a reason beyond
completeness. Claim (M4) of the research note is that the posterior covariance
supplies usable uncertainty -- and the note is explicit that the delta-method
predictive covariance is *systematically over-confident*, because it ignores both
the linearisation remainder and model misspecification. So calibration has to be
**demonstrated rather than asserted**, and the machinery to demonstrate it should
predate the filter that will be judged by it.

Logging these for the SGD runs also gives phase 5 a baseline: "the filter is
better calibrated" is only meaningful against a number.

**Brier is the multiclass sum**, $\\frac1n\\sum_i\\sum_k(\\pi_{ik}-y_{ik})^2$, which
ranges over $[0,2]$. The variant that scores only the true class, $(1-\\pi_c)^2$,
is common and is *not* what is used here: it ignores how the remaining mass is
distributed, so a confident wrong second choice scores the same as a diffuse one.

**Reliability bins are equal-width and their occupancy is logged.** Equal-mass
bins are more robust but their edges move with the predictions, so two runs are
no longer comparable bin by bin. Equal width keeps the axis fixed and reports the
count per bin, so an ECE resting on a nearly empty bin is visible rather than
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dekf_bench.metrics.classification import MetricError

#: Fifteen equal-width bins over [0, 1], the conventional default.
DEFAULT_BINS = 15


def _check(probabilities: torch.Tensor, targets: torch.Tensor) -> None:
    if probabilities.ndim != 2:
        raise MetricError(f"expected (n, q) probabilities, got shape {tuple(probabilities.shape)}")
    if targets.shape != probabilities.shape[:1]:
        raise MetricError(
            f"targets {tuple(targets.shape)} do not match {tuple(probabilities.shape[:1])}"
        )
    if probabilities.numel() == 0:
        raise MetricError("cannot score an empty batch")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise MetricError("probabilities must sum to one along the class axis")


def brier(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Multiclass Brier score, in $[0, 2]$. Lower is better."""
    _check(probabilities, targets)
    one_hot = torch.zeros_like(probabilities)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    return float(((probabilities - one_hot) ** 2).sum(dim=1).mean())


def confidence_and_correctness(
    probabilities: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The predicted class's probability, and whether it was right."""
    _check(probabilities, targets)
    confidence, predicted = probabilities.max(dim=-1)
    return confidence, (predicted == targets).to(probabilities.dtype)


@dataclass(frozen=True)
class Reliability:
    """A reliability curve, plus the counts that say whether to believe it."""

    bin_lower: tuple[float, ...]
    bin_upper: tuple[float, ...]
    counts: tuple[int, ...]
    mean_confidence: tuple[float, ...]
    mean_accuracy: tuple[float, ...]

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def ece(self) -> float:
        """Expected calibration error: occupancy-weighted $|{\\rm acc}-{\\rm conf}|$."""
        if not self.total:
            return 0.0
        return (
            sum(
                count * abs(accuracy - confidence)
                for count, accuracy, confidence in zip(
                    self.counts, self.mean_accuracy, self.mean_confidence, strict=True
                )
            )
            / self.total
        )

    @property
    def max_calibration_error(self) -> float:
        """The worst *occupied* bin. Unweighted, so a small badly-miscalibrated
        bin shows up here even when the ECE averages it away."""
        gaps = [
            abs(accuracy - confidence)
            for count, accuracy, confidence in zip(
                self.counts, self.mean_accuracy, self.mean_confidence, strict=True
            )
            if count
        ]
        return max(gaps) if gaps else 0.0

    @property
    def overconfidence(self) -> float:
        """Signed: positive means the model claims more than it delivers.

        The direction matters. The note predicts the filter's predictive
        covariance is over-confident, so a calibration report that only gives a
        magnitude cannot confirm or refute it.
        """
        if not self.total:
            return 0.0
        return (
            sum(
                count * (confidence - accuracy)
                for count, accuracy, confidence in zip(
                    self.counts, self.mean_accuracy, self.mean_confidence, strict=True
                )
            )
            / self.total
        )


def reliability(
    probabilities: torch.Tensor, targets: torch.Tensor, n_bins: int = DEFAULT_BINS
) -> Reliability:
    """Bin predictions by confidence and compare accuracy against it."""
    if n_bins < 1:
        raise MetricError(f"n_bins must be >= 1, got {n_bins}")
    confidence, correct = confidence_and_correctness(probabilities, targets)

    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=confidence.dtype)
    lower, upper, counts, mean_conf, mean_acc = [], [], [], [], []
    for index in range(n_bins):
        low, high = float(edges[index]), float(edges[index + 1])
        # Half-open bins, with the last one closed so confidence == 1.0 lands
        # somewhere rather than being silently dropped.
        in_bin = (confidence > low) & (confidence <= high) if index else confidence <= high
        count = int(in_bin.sum())
        lower.append(low)
        upper.append(high)
        counts.append(count)
        mean_conf.append(float(confidence[in_bin].mean()) if count else 0.0)
        mean_acc.append(float(correct[in_bin].mean()) if count else 0.0)

    return Reliability(
        bin_lower=tuple(lower),
        bin_upper=tuple(upper),
        counts=tuple(counts),
        mean_confidence=tuple(mean_conf),
        mean_accuracy=tuple(mean_acc),
    )


@dataclass(frozen=True)
class CalibrationScores:
    """Everything worth logging about how well the probabilities are trusted."""

    nll: float
    brier: float
    ece: float
    max_calibration_error: float
    overconfidence: float
    mean_confidence: float

    def as_rows(self) -> list[dict[str, float | str]]:
        return [
            {"metric": name, "value": value}
            for name, value in (
                ("nll", self.nll),
                ("brier", self.brier),
                ("ece", self.ece),
                ("max_calibration_error", self.max_calibration_error),
                ("overconfidence", self.overconfidence),
                ("mean_confidence", self.mean_confidence),
            )
        ]


def score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    likelihood,
    n_bins: int = DEFAULT_BINS,
) -> CalibrationScores:
    """Every calibration quantity for one batch of predictions."""
    probabilities = likelihood.mu(logits)
    curve = reliability(probabilities, targets, n_bins)
    confidence, _ = confidence_and_correctness(probabilities, targets)
    return CalibrationScores(
        nll=float(likelihood.nll(logits, targets, reduction="mean")),
        brier=brier(probabilities, targets),
        ece=curve.ece,
        max_calibration_error=curve.max_calibration_error,
        overconfidence=curve.overconfidence,
        mean_confidence=float(confidence.mean()),
    )
