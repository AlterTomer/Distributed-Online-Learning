"""The image pipeline: rotate, downsample, normalize -- in that order.

**This is the single implementation.** Training, all three evaluation sets, and
the offline reference classifier call the same :class:`ImageTransform` instance.
That is not tidiness: under drift the evaluation set must carry the *same*
rotation as the training data at step $t$, and the only way to make that
checkable rather than hoped-for is to have one object that both paths use.

The order is fixed and each step is where it is for a reason.

**Rotate at full resolution, downsample second.** Rotating a 14x14 image
directly destroys far more information than rotating at 28x28 and then pooling,
and it makes the transform resolution-dependent.

**Normalize last.** Rotation fills the corners it exposes with zero. On raw
intensities zero *is* the black background, so the fill is invisible. Normalize
first and zero becomes $\\mu/\\sigma \\approx 0.42$ standard deviations above the
true background, so every rotated image acquires four bright corners that no
unrotated image has -- a signal perfectly correlated with the drift state, which
a classifier will happily learn instead of the digit. Nothing about the
resulting curves would look wrong.

**The normalization constants are computed once, on canonical data.** They come
from the *unrotated, downsampled* training images and are then held fixed. Two
traps avoided: the standard 28x28 constants no longer apply once the image is
pooled (average pooling preserves the mean but shrinks the standard deviation),
and statistics recomputed per step would drift with the rotation, confounding
the very thing being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate as _tv_rotate

#: Bilinear rather than nearest-neighbour. At a few degrees, nearest-neighbour
#: rotation quantises the digit into visible staircase artefacts that vary with
#: the angle -- another signal correlated with the drift state.
INTERPOLATION = InterpolationMode.BILINEAR

#: The value rotation pads exposed corners with. Zero is the black background on
#: raw intensities, which is exactly why normalization has to come afterwards.
FILL_VALUE = 0.0


class TransformError(ValueError):
    """Raised for an unbuildable or inconsistent transform."""


def rotate(images: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate a batch about its centre, filling exposed corners with zero.

    Called at full resolution, before downsampling.
    """
    if images.ndim != 4:
        raise TransformError(f"expected (n, c, h, w), got {tuple(images.shape)}")
    if degrees == 0.0:
        # Not merely an optimisation: an exact identity keeps `alpha=0` runs
        # bit-identical to stationary ones, which is a test in test_drift.py.
        return images
    return _tv_rotate(images, degrees, interpolation=INTERPOLATION, fill=[FILL_VALUE])


def downsample(images: torch.Tensor, size: int) -> torch.Tensor:
    """Average-pool to ``size`` x ``size``."""
    if images.ndim != 4:
        raise TransformError(f"expected (n, c, h, w), got {tuple(images.shape)}")
    height = images.shape[-1]
    if height == size:
        return images
    if height % size != 0:
        raise TransformError(
            f"cannot pool {height}x{height} down to {size}x{size}: "
            f"{height} is not divisible by {size}"
        )
    return F.avg_pool2d(images, height // size)


def canonical_statistics(images: torch.Tensor, size: int) -> tuple[float, float]:
    """Mean and standard deviation of the *unrotated, downsampled* images.

    Computed in float64 regardless of the input dtype: this runs once over
    60 000 images and the constants then sit inside every forward pass, so it is
    the wrong place to accumulate float32 error.
    """
    if images.numel() == 0:
        raise TransformError("cannot compute statistics of an empty tensor")
    pooled = downsample(images.to(torch.float64), size)
    std = float(pooled.std())
    if std <= 0.0:
        raise TransformError("images have zero variance; normalization would divide by zero")
    return float(pooled.mean()), std


@dataclass(frozen=True)
class ImageTransform:
    """The pipeline, with its normalization constants baked in.

    Frozen, and the constants are values rather than a reference to a dataset:
    a transform that could be re-fitted mid-run is a transform that can drift
    apart between the training path and the evaluation path.
    """

    size: int
    mean: float
    std: float
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.size < 1:
            raise TransformError(f"size must be >= 1, got {self.size}")
        if self.std <= 0.0:
            raise TransformError(f"std must be > 0, got {self.std}")

    @property
    def input_dim(self) -> int:
        """Flattened input width the model sees."""
        return self.size * self.size

    @property
    def background(self) -> float:
        """What an empty pixel reads as after the full pipeline.

        The value rotation-exposed corners take. Equal to the interior
        background by construction, which is the property the ordering buys.
        """
        return (0.0 - self.mean) / self.std if self.normalize else 0.0

    def apply(self, images: torch.Tensor, rotation_degrees: float = 0.0) -> torch.Tensor:
        """Run the full pipeline at a given drift state."""
        out = rotate(images, rotation_degrees)
        out = downsample(out, self.size)
        if self.normalize:
            out = (out - self.mean) / self.std
        return out

    def at(self, images: torch.Tensor, state) -> torch.Tensor:
        """Apply at a :class:`~dekf_bench.env.drift.DriftState`.

        The call the environment and the evaluation sets both make, so neither
        can pass a rotation the other did not.
        """
        return self.apply(images, state.rotation_degrees)

    def summary(self) -> dict[str, float | int | bool]:
        return {
            "size": self.size,
            "input_dim": self.input_dim,
            "mean": self.mean,
            "std": self.std,
            "normalize": self.normalize,
            "background": self.background,
        }


def build_transform(
    train_images: torch.Tensor, size: int = 14, normalize: bool = True
) -> ImageTransform:
    """Fit the transform on canonical training images.

    Args:
        train_images: the **unrotated** training split. Fitting on rotated
            images would make the constants a function of the drift state.
        size: output side length. 14 for the primary model, 28 for full
            resolution.
        normalize: off only for figures and diagnostics that want raw
            intensities.
    """
    mean, std = canonical_statistics(train_images, size)
    return ImageTransform(size=size, mean=mean, std=std, normalize=normalize)


def build_transform_from_config(config, train_images: torch.Tensor) -> ImageTransform:
    """The transform a run's config asks for."""
    return build_transform(train_images, size=config.model.input_size)
