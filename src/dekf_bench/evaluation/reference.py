r"""The offline reference classifier: what is achievable if you knew everything.

$e^\star$ is the error rate a standard offline run reaches on the full training
set, at the same architecture. Every online method is reported as a **gap**
against it, because the raw error rate also moves with the drift state, the
architecture and the horizon, and only the difference answers "what does
decentralization cost".

**Recomputed per rotation level.** A single reference would conflate
decentralization cost with drift cost: at $45°$ the task is simply harder, and a
gap measured against $e^\star(0°)$ would charge that difficulty to the
distributed method. The grid covers every rotation the configured schedules
visit -- $[-30°, +45°]$ -- so no run has to extrapolate.

**A fixed asset, not part of any experiment run.** Trained once, cached to
``data/reference/``, and independent of a run's master seed. It has its own seed
so that re-running an experiment never silently retrains it.

Three ways the per-rotation models can relate, all selectable so the choice can
be measured rather than argued (design note D33):

``shared_seed``
    Independent runs from a common $\bm\theta_0$. Default: independent, so
    $e^\star(\varphi)$ is genuinely "best achievable here", and sharing the
    initialisation keeps the curve smooth in $\varphi$ rather than jittering
    from initialisation noise that has nothing to do with rotation.
``independent_seeds``
    Independent runs from different $\bm\theta_0$. Honest about run-to-run
    variance, at the cost of jitter the gap curve then inherits.
``warm_start``
    Each level initialised from the previous. Cheaper, but $e^\star(45°)$ then
    depends on having passed through $40°$ -- contaminated by exactly the
    history the reference exists to be free of.

**Model selection never touches the test split.** Under ``validation`` a slice
is held out of *train*, the best epoch is chosen on it, and the test set is
scored once at the end. Early-stopping on test would leak into the very quantity
everything else is measured against -- and would do so invisibly, biasing every
gap in the same direction.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from dekf_bench.data.mnist import MnistSplit
from dekf_bench.data.transforms import ImageTransform
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.models.mlp import MLP
from dekf_bench.runner.seeding import derive_seed

#: Grid points closer than this count as the same rotation when looking up.
ROTATION_TOLERANCE = 1e-6

CACHE_VERSION = 1


class ReferenceError(RuntimeError):
    """Raised when a reference cannot be trained or loaded."""


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of one reference run."""

    epoch: int
    train_loss: float
    validation_error: float | None
    test_error: float | None


@dataclass(frozen=True)
class ReferenceResult:
    r"""One trained reference: $e^\star$ at one rotation, and how it got there."""

    rotation_degrees: float
    error_rate: float
    selected_epoch: int
    epochs: tuple[EpochRecord, ...]
    n_train: int
    n_validation: int
    seconds: float

    @property
    def accuracy(self) -> float:
        return 1.0 - self.error_rate

    @property
    def converged(self) -> bool:
        """Whether the selected epoch is comfortably inside the budget.

        Selecting the final epoch means the budget, not convergence, decided
        where training stopped -- so "trained to convergence" would be an
        assumption rather than an observation.
        """
        return self.selected_epoch < len(self.epochs) - 1


@dataclass(frozen=True)
class Reference:
    r"""$e^\star$ across the rotation grid."""

    results: tuple[ReferenceResult, ...]
    init_strategy: str
    selection: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rotations(self) -> tuple[float, ...]:
        return tuple(result.rotation_degrees for result in self.results)

    def at(self, rotation: float) -> float:
        r"""$e^\star$ at a rotation, linearly interpolated between grid points.

        Interpolated rather than nearest-neighbour: the grid is $5°$ apart and
        $e^\star$ varies smoothly with rotation, so rounding would introduce a
        sawtooth into the gap curve at a scale comparable to the effect being
        measured.
        """
        points = sorted(zip(self.rotations, [r.error_rate for r in self.results], strict=True))
        lowest, highest = points[0][0], points[-1][0]
        if rotation < lowest - ROTATION_TOLERANCE or rotation > highest + ROTATION_TOLERANCE:
            raise ReferenceError(
                f"rotation {rotation:.3f} lies outside the reference grid "
                f"[{lowest}, {highest}]. Extend reference.rotation_min/max_degrees rather "
                "than extrapolating: e* outside the grid is not measured, and a gap "
                "against an extrapolated reference is not a measurement either."
            )
        for (left, left_error), (right, right_error) in zip(points, points[1:], strict=False):
            if left - ROTATION_TOLERANCE <= rotation <= right + ROTATION_TOLERANCE:
                if right - left < ROTATION_TOLERANCE:
                    return left_error
                weight = (rotation - left) / (right - left)
                return left_error + weight * (right_error - left_error)
        return points[-1][1]  # pragma: no cover - covered by the range check

    def symmetry_error(self) -> float | None:
        r"""How closely $e^\star(-\varphi)$ matches $e^\star(+\varphi)$.

        A cheap check on an assumption the grid deliberately does *not* make:
        rotating $+15°$ and $-15°$ give genuinely different image distributions,
        so their difficulty being equal is plausible but not free. ``None`` when
        the grid is one-sided.
        """
        errors = dict(zip(self.rotations, [r.error_rate for r in self.results], strict=True))
        pairs = [
            abs(errors[rotation] - errors[-rotation])
            for rotation in errors
            if rotation > 0 and -rotation in errors
        ]
        return max(pairs) if pairs else None

    def summary(self) -> dict[str, Any]:
        errors = [result.error_rate for result in self.results]
        return {
            "init_strategy": self.init_strategy,
            "selection": self.selection,
            "n_levels": len(self.results),
            "rotation_range": [min(self.rotations), max(self.rotations)],
            "best_error": min(errors),
            "worst_error": max(errors),
            "all_converged": all(result.converged for result in self.results),
            "symmetry_error": self.symmetry_error(),
            "seconds": sum(result.seconds for result in self.results),
        }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def _split_train(
    train: MnistSplit, validation_size: int, generator: torch.Generator
) -> tuple[MnistSplit, MnistSplit | None]:
    """Hold out a validation slice, or none under a fixed budget."""
    if validation_size <= 0:
        return train, None
    if validation_size >= len(train):
        raise ReferenceError(
            f"validation_size {validation_size} leaves no training data out of {len(train)}"
        )
    permutation = torch.randperm(len(train), generator=generator)
    return train.subset(permutation[validation_size:]), train.subset(permutation[:validation_size])


def _error_rate(
    model: MLP, params: dict, images: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> float:
    wrong = 0
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        logits = model.forward(params, images[start:stop])
        wrong += int((logits.argmax(dim=-1) != labels[start:stop]).sum())
    return wrong / len(labels)


def train_one(
    rotation: float,
    train: MnistSplit,
    test: MnistSplit,
    model: MLP,
    transform: ImageTransform,
    config: Any,
    initial_params: dict | None = None,
    seed: int = 0,
) -> tuple[ReferenceResult, dict]:
    """Train one reference at one rotation.

    Returns the result and the final parameters, the latter so ``warm_start``
    can hand them to the next level.
    """
    likelihood = Categorical(model.output_dim)
    generator = torch.Generator().manual_seed(seed)

    fitting, validation = _split_train(train, config.validation_size, generator)
    train_x = transform.apply(fitting.images, rotation)
    train_y = fitting.labels
    validation_x = transform.apply(validation.images, rotation) if validation is not None else None
    test_x = transform.apply(test.images, rotation)

    params = (
        {name: tensor.clone() for name, tensor in initial_params.items()}
        if initial_params is not None
        else model.init_params(generator)
    )
    for tensor in params.values():
        tensor.requires_grad_(True)
    optimiser = torch.optim.AdamW(list(params.values()), lr=config.lr)

    started = time.perf_counter()
    records: list[EpochRecord] = []
    best = (float("inf"), 0, {name: t.detach().clone() for name, t in params.items()})

    for epoch in range(config.epochs):
        order = torch.randperm(len(train_y), generator=generator)
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = likelihood.nll(model.forward(params, train_x[batch]), train_y[batch], "mean")
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch)

        with torch.no_grad():
            detached = {name: tensor.detach() for name, tensor in params.items()}
            validation_error = (
                _error_rate(model, detached, validation_x, validation.labels, 2048)
                if validation is not None and validation_x is not None
                else None
            )
            # Recorded for inspection only. Selection never reads it -- doing so
            # would leak the test set into the quantity everything is measured
            # against, invisibly and always in the same direction.
            test_error = _error_rate(model, detached, test_x, test.labels, 2048)

        records.append(EpochRecord(epoch, total / len(train_y), validation_error, test_error))
        criterion = validation_error if validation_error is not None else float("inf")
        if criterion < best[0]:
            best = (criterion, epoch, {name: t.detach().clone() for name, t in params.items()})

    if config.selection == "validation":
        selected_epoch, selected = best[1], best[2]
    else:
        selected_epoch = config.epochs - 1
        selected = {name: tensor.detach().clone() for name, tensor in params.items()}

    with torch.no_grad():
        error = _error_rate(model, selected, test_x, test.labels, 2048)

    return (
        ReferenceResult(
            rotation_degrees=rotation,
            error_rate=error,
            selected_epoch=selected_epoch,
            epochs=tuple(records),
            n_train=len(train_y),
            n_validation=0 if validation is None else len(validation),
            seconds=time.perf_counter() - started,
        ),
        selected,
    )


def repeat_seeds(
    rotation: float,
    train: MnistSplit,
    test: MnistSplit,
    model: MLP,
    transform: ImageTransform,
    config: Any,
    n_repeats: int = 5,
    progress: bool = True,
) -> tuple[ReferenceResult, ...]:
    r"""Train the same rotation several times, varying only the seed.

    The grid has **one draw per point** and no stated uncertainty, so a
    difference between two levels cannot be told from run-to-run noise by
    looking at the curve. This measures that noise directly: same rotation, same
    everything, different seed.

    It is the same argument WORKPLAN section 5.4 makes for the experiments --
    five seeds and a band, never a single run -- applied to the quantity every
    one of those experiments is measured against.
    """
    results = []
    for repeat in range(n_repeats):
        result, _ = train_one(
            rotation,
            train,
            test,
            model,
            transform,
            config,
            seed=derive_seed(config.seed, "reference_repeat", repeat),
        )
        results.append(result)
        if progress:
            print(
                f"  seed {repeat}   e* = {result.error_rate:.4f}   "
                f"epoch {result.selected_epoch + 1}/{config.epochs}   {result.seconds:.0f}s"
            )
    return tuple(results)


def seed_spread(results: tuple[ReferenceResult, ...], n_test: int = 10_000) -> dict[str, float]:
    r"""How much $e^\star$ moves when only the seed changes.

    ``binomial_standard_error`` is the floor set by scoring a finite test set:
    $\sqrt{e(1-e)/n}$. If the observed seed spread is comparable to it, the
    measurement is at the resolution limit and differences of that size across
    the rotation grid are not signal.
    """
    import statistics

    errors = [result.error_rate for result in results]
    mean = statistics.fmean(errors)
    return {
        "mean": mean,
        "std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
        "range": max(errors) - min(errors),
        "binomial_standard_error": (mean * (1 - mean) / n_test) ** 0.5,
        "n_repeats": len(errors),
    }


def train_reference(
    config: Any,
    train: MnistSplit,
    test: MnistSplit,
    model: MLP,
    transform: ImageTransform,
    progress: bool = True,
) -> Reference:
    """Train the whole grid, honouring the configured init strategy."""
    settings = config.reference
    results: list[ReferenceResult] = []
    previous: dict | None = None
    shared: dict | None = None

    for index, rotation in enumerate(settings.rotations):
        if settings.init_strategy == "shared_seed":
            seed = derive_seed(settings.seed, "reference")
            initial = shared
        elif settings.init_strategy == "independent_seeds":
            seed = derive_seed(settings.seed, "reference", index)
            initial = None
        else:  # warm_start
            seed = derive_seed(settings.seed, "reference")
            initial = previous

        result, params = train_one(rotation, train, test, model, transform, settings, initial, seed)
        if settings.init_strategy == "shared_seed" and shared is None:
            # Capture theta_0, not the trained result: sharing the *initial*
            # parameters is what keeps the runs independent.
            shared = {
                name: tensor.clone()
                for name, tensor in model.init_params(torch.Generator().manual_seed(seed)).items()
            }
        previous = params
        results.append(result)

        if progress:
            flag = "" if result.converged else "  (stopped at the budget)"
            print(
                f"  {rotation:>6.1f} deg   e* = {result.error_rate:.4f}   "
                f"epoch {result.selected_epoch + 1}/{settings.epochs}   "
                f"{result.seconds:.1f}s{flag}"
            )

    return Reference(
        results=tuple(results),
        init_strategy=settings.init_strategy,
        selection=settings.selection,
        metadata={
            "epochs": settings.epochs,
            "lr": settings.lr,
            "batch_size": settings.batch_size,
            "validation_size": settings.validation_size,
            "model": model.summary(),
        },
    )


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #


def cache_path(root: Path, init_strategy: str, selection: str) -> Path:
    """One file per (strategy, selection), so the variants never overwrite
    each other and can be compared."""
    return root / "reference" / f"e_star_{init_strategy}_{selection}_v{CACHE_VERSION}.json"


def save(reference: Reference, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "init_strategy": reference.init_strategy,
        "selection": reference.selection,
        "metadata": reference.metadata,
        "results": [asdict(result) for result in reference.results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(path: Path) -> Reference:
    if not path.is_file():
        raise ReferenceError(
            f"no cached reference at {path}. Run scripts/train_reference.py once; "
            "e* is a fixed asset and is not rebuilt by an experiment run."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = tuple(
        ReferenceResult(
            rotation_degrees=entry["rotation_degrees"],
            error_rate=entry["error_rate"],
            selected_epoch=entry["selected_epoch"],
            epochs=tuple(EpochRecord(**record) for record in entry["epochs"]),
            n_train=entry["n_train"],
            n_validation=entry["n_validation"],
            seconds=entry["seconds"],
        )
        for entry in payload["results"]
    )
    return Reference(
        results=results,
        init_strategy=payload["init_strategy"],
        selection=payload["selection"],
        metadata=payload.get("metadata", {}),
    )
