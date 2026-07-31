"""Download MNIST if needed, then report what arrived.

This is the phase-0 gate: run it once and the dataset is cached for every run
afterwards. Hit the run (or debug) button with no arguments; it downloads on
first use and reads the cache thereafter.

Set ``SHOW_GRID = True`` to also render a sample of digits, which is worth doing
once to confirm the images are the right way up and the labels line up.
"""

from __future__ import annotations

import time

import torch

from dekf_bench.data.mnist import (
    NUM_CLASSES,
    SPLIT_SIZES,
    DataError,
    channel_statistics,
    class_counts,
    default_data_dir,
    is_cached,
    load_mnist,
)

# ---------------------------------------------------------------------------
# Edit these, then run the file. No command-line arguments required.
# ---------------------------------------------------------------------------
DOWNLOAD = True
SHOW_GRID = True
GRID_ROWS = 3


def summarise(name: str, data) -> str:
    mean, std = channel_statistics(data.images)
    counts = class_counts(data)
    megabytes = data.images.element_size() * data.images.nelement() / 1024**2
    return "\n".join(
        [
            f"{name}",
            f"  shape        {tuple(data.images.shape)}  {data.images.dtype}  ({megabytes:.0f} MB)",
            f"  labels       {tuple(data.labels.shape)}  {data.labels.dtype}  "
            f"range [{int(data.labels.min())}, {int(data.labels.max())}]",
            f"  intensities  [{float(data.images.min()):.3f}, {float(data.images.max()):.3f}]  "
            f"mean {mean:.4f}  std {std:.4f}",
            f"  per class    {counts.tolist()}",
        ]
    )


def show_grid(data, rows: int = 3) -> None:
    """Render `rows` x NUM_CLASSES digits, one column per class."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(rows, NUM_CLASSES, figsize=(NUM_CLASSES, rows))
    for digit in range(NUM_CLASSES):
        matching = (data.labels == digit).nonzero(as_tuple=True)[0][:rows]
        for row, index in enumerate(matching):
            axis = axes[row][digit] if rows > 1 else axes[digit]
            axis.imshow(data.images[index, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(str(digit))
    figure.suptitle("MNIST, raw intensities, before rotation and downsampling")
    figure.tight_layout()
    plt.show()


def main(download: bool = DOWNLOAD, show: bool = SHOW_GRID) -> int:
    root = default_data_dir()
    cached = is_cached(root)
    print(f"data dir     {root}")
    print(f"cache        {'present' if cached else 'absent, will build'}\n")

    started = time.perf_counter()
    try:
        train, test = load_mnist(root, download=download)
    except DataError as error:
        print(f"failed: {error}")
        return 1
    elapsed = time.perf_counter() - started

    print(summarise("train", train))
    print()
    print(summarise("test", test))
    print(f"\nloaded in {elapsed:.2f}s from {'cache' if cached else 'source'}")

    problems = []
    for name, data in (("train", train), ("test", test)):
        if len(data) != SPLIT_SIZES[name]:
            problems.append(f"{name} has {len(data)} samples, expected {SPLIT_SIZES[name]}")
        if int(class_counts(data).min()) == 0:
            problems.append(f"{name} is missing at least one class")
    if not torch.equal(torch.unique(train.labels), torch.arange(NUM_CLASSES, dtype=torch.int64)):
        problems.append("train labels are not exactly 0..9")

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nall checks passed")
    if show:
        show_grid(train, GRID_ROWS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
