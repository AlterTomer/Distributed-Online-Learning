"""One seed's environment and evaluation sets, for whichever task the config names.

The sweep scripts share one entry point (`run_one`), and it should not need to know
which task it is running: a rotated-image task loads its splits once and builds an
image environment over them; a generated series integrates its own data per seed
and needs no splits at all (`docs/mackey_glass_plan.md`, WP7). The choice is the
dataset's ``kind``, read from the registry.
"""

from __future__ import annotations

from typing import Any


def build_task(config: Any, seed: int, train: Any = None, test: Any = None) -> tuple[Any, Any]:
    """``(environment, evalsets)`` for one master seed.

    ``train`` and ``test`` are the image splits, and are ignored -- and may be
    ``None`` -- for a generated series.
    """
    if config.env.is_series:
        from dekf_bench.env.series import build_series_environment  # noqa: PLC0415
        from dekf_bench.evaluation.series_evalsets import build_series_evalsets  # noqa: PLC0415

        environment = build_series_environment(config, seed)
        return environment, build_series_evalsets(config, environment)

    from dekf_bench.env.environment import build_environment  # noqa: PLC0415
    from dekf_bench.evaluation.evalsets import build_evalsets  # noqa: PLC0415

    if train is None or test is None:
        raise ValueError(
            f"env.dataset={config.env.dataset!r} is loaded from splits; pass train and test"
        )
    environment = build_environment(config, seed, train)
    return environment, build_evalsets(config, environment, test)
