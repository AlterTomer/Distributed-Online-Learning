"""The image pipeline: rotate, downsample, normalize.

The ordering claims in the module docstring are asserted here rather than left
as prose, because every one of them fails *silently* if the order is changed:
the images still look like digits and the curves still look like curves.
"""

from __future__ import annotations

import pytest
import torch

from dekf_bench.data.transforms import (
    FILL_VALUE,
    ImageTransform,
    TransformError,
    build_transform,
    build_transform_from_config,
    canonical_statistics,
    downsample,
    rotate,
)
from dekf_bench.env.drift import DriftState, build_drift
from dekf_bench.utils.config import load_config

SIZE = 14


def images(n: int = 8, side: int = 28, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, side, side, generator=generator)


def digit_like(n: int = 4) -> torch.Tensor:
    """A bright blob on a black field -- rotation-sensitive, like a digit."""
    canvas = torch.zeros(n, 1, 28, 28)
    for index in range(n):
        canvas[index, 0, 8 : 20 - index, 12 : 16 + index] = 1.0
    return canvas


@pytest.fixture
def transform() -> ImageTransform:
    return build_transform(digit_like(16), size=SIZE)


# =========================================================================== #
# 1. the pieces
# =========================================================================== #


def test_zero_rotation_is_the_exact_identity() -> None:
    """Not merely close: an `alpha=0` run must be bit-identical to a stationary
    one, or the drift ablation carries an interpolation confound."""
    batch = images()
    assert rotate(batch, 0.0) is batch


def test_rotation_changes_the_image() -> None:
    batch = digit_like()
    assert not torch.allclose(rotate(batch, 30.0), batch)


def test_rotation_preserves_shape_and_dtype() -> None:
    batch = images()
    rotated = rotate(batch, 17.0)
    assert rotated.shape == batch.shape
    assert rotated.dtype == batch.dtype


def test_rotation_fills_exposed_corners_with_zero() -> None:
    """The property normalization ordering depends on."""
    batch = torch.ones(1, 1, 28, 28)
    rotated = rotate(batch, 45.0)
    assert float(rotated[0, 0, 0, 0]) == pytest.approx(FILL_VALUE, abs=1e-6)


def test_a_full_turn_returns_approximately_the_original() -> None:
    batch = digit_like(2)
    assert torch.allclose(rotate(batch, 360.0), batch, atol=1e-5)


def test_rotation_rejects_the_wrong_rank() -> None:
    with pytest.raises(TransformError, match=r"expected \(n, c, h, w\)"):
        rotate(torch.rand(28, 28), 10.0)


def test_downsample_halves_each_side() -> None:
    assert downsample(images(), SIZE).shape == (8, 1, SIZE, SIZE)


def test_downsample_to_the_same_size_is_a_no_op() -> None:
    batch = images()
    assert downsample(batch, 28) is batch


def test_downsample_preserves_the_mean() -> None:
    """Average pooling preserves the mean but shrinks the standard deviation --
    which is exactly why the 28x28 constants cannot be reused."""
    batch = images(64)
    pooled = downsample(batch, SIZE)
    assert float(pooled.mean()) == pytest.approx(float(batch.mean()), abs=1e-6)
    assert float(pooled.std()) < float(batch.std())


def test_downsample_rejects_an_indivisible_size() -> None:
    with pytest.raises(TransformError, match="not divisible"):
        downsample(images(), 5)


# =========================================================================== #
# 2. the ordering, which is the whole point
# =========================================================================== #


def test_rotating_before_pooling_differs_from_pooling_first() -> None:
    """If these agreed, the ordering would not matter and D2 would be vacuous."""
    batch = digit_like()
    rotate_first = downsample(rotate(batch, 30.0), SIZE)
    pool_first = rotate(downsample(batch, SIZE), 30.0)
    assert not torch.allclose(rotate_first, pool_first, atol=1e-3)


def test_rotating_at_full_resolution_preserves_more_signal() -> None:
    """Rotating a 14x14 image directly destroys information that rotating at
    28x28 and then pooling retains."""
    batch = digit_like(8)
    there_and_back_hi = downsample(rotate(rotate(batch, 20.0), -20.0), SIZE)
    there_and_back_lo = rotate(rotate(downsample(batch, SIZE), 20.0), -20.0)
    reference = downsample(batch, SIZE)
    assert (there_and_back_hi - reference).abs().mean() < (
        there_and_back_lo - reference
    ).abs().mean()


def test_exposed_corners_match_the_interior_background(transform: ImageTransform) -> None:
    """The D13 property, stated positively: after the full pipeline a corner
    exposed by rotation is indistinguishable from the ordinary background."""
    blank = torch.zeros(1, 1, 28, 28)
    output = transform.apply(blank, rotation_degrees=30.0)
    assert torch.allclose(output, torch.full_like(output, transform.background), atol=1e-6)


def test_normalizing_before_rotating_brands_every_rotated_image(
    transform: ImageTransform,
) -> None:
    """The failure the ordering prevents.

    Normalize first and the corners rotation exposes take the value 0, which is
    mu/sigma standard deviations away from the true background -- a large,
    constant region present in every rotated image and no unrotated one. A
    classifier learns it instead of the digit, and nothing about the curves
    looks wrong.
    """
    blank = torch.zeros(1, 1, 28, 28)

    correct = transform.apply(blank, rotation_degrees=30.0)
    normalized_first = (blank - transform.mean) / transform.std
    wrong = downsample(rotate(normalized_first, 30.0), SIZE)

    assert float(correct.std()) == pytest.approx(0.0, abs=1e-6)
    assert float(wrong.std()) > 0.05, "the artefact must be visible in the wrong order"

    expected_offset = transform.mean / transform.std
    assert float(wrong.max() - wrong.min()) == pytest.approx(expected_offset, rel=0.05)


def test_the_artefact_is_absent_without_rotation(transform: ImageTransform) -> None:
    """Which is what makes it correlate with the drift state rather than being
    a constant offset the model could absorb into a bias.

    Compared as a ratio, not against an absolute floor: a constant field in
    float32 still carries ~1e-8 of pooling noise, and asserting exact zero would
    be testing the dtype rather than the ordering.
    """
    blank = torch.zeros(1, 1, 28, 28)
    normalized_first = (blank - transform.mean) / transform.std

    unrotated = downsample(rotate(normalized_first, 0.0), SIZE)
    rotated = downsample(rotate(normalized_first, 30.0), SIZE)

    assert float(unrotated.std()) < 1e-6
    assert float(rotated.std()) > 1e4 * float(unrotated.std())


# =========================================================================== #
# 3. normalization constants
# =========================================================================== #


def test_statistics_are_computed_after_pooling() -> None:
    """Reusing the 28x28 standard deviation would over-scale the input."""
    batch = images(64)
    _, pooled_std = canonical_statistics(batch, SIZE)
    assert pooled_std == pytest.approx(float(downsample(batch.double(), SIZE).std()))
    assert pooled_std < float(batch.std())


def test_statistics_are_computed_in_float64() -> None:
    batch = images(64).to(torch.float32)
    mean, std = canonical_statistics(batch, SIZE)
    exact = downsample(batch.to(torch.float64), SIZE)
    assert mean == pytest.approx(float(exact.mean()), abs=1e-12)
    assert std == pytest.approx(float(exact.std()), abs=1e-12)


def test_constants_are_baked_in_not_refitted(transform: ImageTransform) -> None:
    """A transform that could be re-fitted mid-run is one that can drift apart
    between the training path and the evaluation path."""
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(transform, "mean", 0.5)  # noqa: B010


def test_constants_do_not_depend_on_the_drift_state() -> None:
    """Statistics recomputed per step would drift with the rotation and confound
    the very thing being measured."""
    canonical = build_transform(digit_like(16), size=SIZE)
    rotated_fit = build_transform(rotate(digit_like(16), 30.0), size=SIZE)
    assert canonical.mean != pytest.approx(rotated_fit.mean, abs=1e-6)
    # The pipeline must use the canonical constants at every step.
    for degrees in (0.0, 15.0, 45.0):
        output = canonical.apply(digit_like(4), degrees)
        assert output.shape == (4, 1, SIZE, SIZE)


def test_zero_variance_input_is_rejected() -> None:
    with pytest.raises(TransformError, match="zero variance"):
        canonical_statistics(torch.zeros(4, 1, 28, 28), SIZE)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(TransformError, match="empty tensor"):
        canonical_statistics(torch.empty(0, 1, 28, 28), SIZE)


# =========================================================================== #
# 4. the transform object
# =========================================================================== #


def test_output_shape_and_input_dim(transform: ImageTransform) -> None:
    output = transform.apply(images(5), 10.0)
    assert output.shape == (5, 1, SIZE, SIZE)
    assert transform.input_dim == SIZE * SIZE == 196


def test_normalized_output_is_standardised(transform: ImageTransform) -> None:
    output = transform.apply(digit_like(16), 0.0)
    assert float(output.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(output.std()) == pytest.approx(1.0, abs=1e-3)


def test_normalization_can_be_switched_off_for_figures() -> None:
    raw = build_transform(digit_like(16), size=SIZE, normalize=False)
    output = raw.apply(digit_like(4), 0.0)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    assert raw.background == 0.0


def test_background_is_what_an_empty_pixel_reads_as(transform: ImageTransform) -> None:
    blank = torch.zeros(1, 1, 28, 28)
    output = transform.apply(blank, 0.0)
    assert float(output[0, 0, 0, 0]) == pytest.approx(transform.background, abs=1e-6)


def test_float64_flows_through(transform: ImageTransform) -> None:
    """The exactness check runs the whole pipeline in double precision."""
    output = transform.apply(digit_like(2).to(torch.float64), 15.0)
    assert output.dtype == torch.float64


def test_invalid_size_is_rejected() -> None:
    with pytest.raises(TransformError, match="size must be >= 1"):
        ImageTransform(size=0, mean=0.1, std=0.3)


def test_non_positive_std_is_rejected() -> None:
    with pytest.raises(TransformError, match="std must be > 0"):
        ImageTransform(size=SIZE, mean=0.1, std=0.0)


def test_summary_reports_every_field(transform: ImageTransform) -> None:
    assert set(transform.summary()) == {
        "size",
        "input_dim",
        "mean",
        "std",
        "normalize",
        "background",
    }


# =========================================================================== #
# 5. one implementation for train and eval
# =========================================================================== #


def test_at_a_drift_state_matches_applying_the_rotation(transform: ImageTransform) -> None:
    """The call the environment and the evaluation sets both make, so neither
    can pass a rotation the other did not."""
    batch = digit_like(4)
    state = DriftState(step=750, rotation_degrees=22.5)
    assert torch.equal(transform.at(batch, state), transform.apply(batch, 22.5))


def test_train_and_eval_paths_produce_identical_pixels(transform: ImageTransform) -> None:
    """The same transform object, the same drift state, the same output -- which
    is what makes 'train and eval see the same rotation' checkable."""
    drift = build_drift(load_config("x2_rotating"))
    batch = digit_like(4)
    for step in (0, 375, 750, 1500):
        state = drift.state_at(step)
        train_view = transform.at(batch, state)
        eval_view = transform.at(batch, state)
        assert torch.equal(train_view, eval_view)


def test_different_drift_states_give_different_pixels(transform: ImageTransform) -> None:
    """Otherwise the previous test would pass vacuously."""
    drift = build_drift(load_config("x2_rotating"))
    batch = digit_like(4)
    early = transform.at(batch, drift.state_at(0))
    late = transform.at(batch, drift.state_at(1500))
    assert not torch.allclose(early, late)


def test_stationary_runs_produce_one_view_for_every_step(transform: ImageTransform) -> None:
    drift = build_drift(load_config("x1_stationary"))
    batch = digit_like(4)
    first = transform.at(batch, drift.state_at(0))
    for step in (1, 500, 1499):
        assert torch.equal(transform.at(batch, drift.state_at(step)), first)


def test_config_selects_the_input_size() -> None:
    small = build_transform_from_config(load_config("x1_stationary"), digit_like(16))
    assert small.size == 14
    assert small.input_dim == 196

    full = build_transform_from_config(
        load_config("x1_stationary", overrides={"include": {"model": "mlp"}}), digit_like(16)
    )
    assert full.size == 28
    assert full.input_dim == 784


# =========================================================================== #
# 6. against real MNIST
# =========================================================================== #


@pytest.fixture(scope="session")
def mnist_train():
    from dekf_bench.data.mnist import is_cached, load_split

    if not is_cached():
        pytest.skip("MNIST not cached; run scripts/check_data.py once")
    return load_split("train", download=False)


@pytest.mark.needs_data
def test_mnist_pooled_statistics_differ_from_the_published_constants(mnist_train) -> None:
    """The 28x28 constants are 0.1307 / 0.3081. Pooling preserves the mean and
    shrinks the deviation, so reusing them would over-scale the input."""
    from dekf_bench.data.mnist import MNIST_MEAN, MNIST_STD

    mean, std = canonical_statistics(mnist_train.images, SIZE)
    assert mean == pytest.approx(MNIST_MEAN, abs=1e-3)
    assert std < MNIST_STD - 0.01


@pytest.mark.needs_data
def test_mnist_normalized_output_is_standardised(mnist_train) -> None:
    transform = build_transform(mnist_train.images, size=SIZE)
    output = transform.apply(mnist_train.images[:2000], 0.0)
    assert float(output.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(output.std()) == pytest.approx(1.0, abs=0.02)


@pytest.mark.needs_data
def test_mnist_corners_are_clean_under_rotation(mnist_train) -> None:
    """On real digits: the corners a 30-degree rotation exposes read exactly as
    background, with no bright wedge."""
    transform = build_transform(mnist_train.images, size=SIZE)
    output = transform.apply(mnist_train.images[:200], 30.0)
    corners = torch.stack(
        [output[:, 0, 0, 0], output[:, 0, 0, -1], output[:, 0, -1, 0], output[:, 0, -1, -1]]
    )
    assert torch.allclose(corners, torch.full_like(corners, transform.background), atol=1e-5)
