r"""Buffered writes, and resumption from where a run stopped.

**Flush on evaluation steps.** Rows accumulate between full evaluations and are
written every $K$ steps. A crash loses at most $K$ steps, the flush points
coincide with where the interesting rows are produced anyway, and -- unlike a
fixed row budget -- a flush never lands mid-step with some learners' rows
written and others' not.

**Resumption is exact, not approximate.** The run loop draws **no random
numbers**: the environment is positional (design note D23), and the stream, the
partition and the graph were all realised at construction from their own seed
streams. So a run resumed from step $t$ produces the same trajectory as an
uninterrupted one, bit for bit -- there is no RNG state to restore because none
is consumed. ``test_recording.py`` asserts that rather than assuming it.

What *does* need saving is the learner state: $\bm\theta$ per agent plus any
optimizer buffers. That is written at each flush, atomically, so a checkpoint is
never half-written.

**A checkpoint is refused if the config changed.** Resuming a different
configuration into an existing run would splice two experiments together and
produce a file that looks like one run. The resolved config is hashed, and a
mismatch is an error naming what differs.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dekf_bench.recording.schema import RunContext, empty_frame, validate
from dekf_bench.utils.atomic import replace_with_retry

CHECKPOINT_VERSION = 1


class RecorderError(RuntimeError):
    """Raised when results cannot be written, read or resumed."""


def config_fingerprint(config: Any) -> str:
    """A stable hash of the resolved config.

    Sorted keys and a fixed separator, so the same configuration always hashes
    the same way regardless of the order it was assembled in.
    """
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class Checkpoint:
    """Everything needed to continue a run.

    Deliberately *not* including any RNG state: nothing in the loop consumes
    randomness, so there is none to restore. Storing one would imply otherwise.
    """

    step: int
    config_hash: str
    learner_states: dict[str, dict[int, dict[str, torch.Tensor]]]
    n_rows: int
    version: int = CHECKPOINT_VERSION

    @property
    def resume_from(self) -> int:
        """The first step *not* yet completed."""
        return self.step + 1


class Recorder:
    """Collects rows for one (experiment, seed) and writes them to parquet.

    One file per seed: a crashed seed loses only itself, seeds can later be run
    in parallel without contention, and a partial sweep is still readable.
    """

    def __init__(
        self,
        out_dir: Path,
        context: RunContext,
        config: Any,
        strict: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.context = context
        self.config = config
        self.strict = strict
        self.config_hash = config_fingerprint(config)
        self._rows: list[dict[str, Any]] = []
        self._buffer: list[dict[str, Any]] = []
        self._started = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------- #

    @property
    def data_path(self) -> Path:
        return self.out_dir / f"seed_{self.context.seed}.parquet"

    @property
    def checkpoint_path(self) -> Path:
        return self.out_dir / f"seed_{self.context.seed}.checkpoint.pt"

    # -- writing ------------------------------------------------------------ #

    def log(self, row: dict[str, Any]) -> None:
        """Buffer one row, stamped with the run's constant fields.

        ``node_id`` is coerced to string here, at the schema boundary. Per-agent
        rows arrive with an integer and aggregate rows with a name like
        ``"mean"``; parquet infers the column type from the first rows it sees
        and then fails on the mismatch, so the coercion has to happen before any
        of them are written rather than at the point they are produced.
        """
        stamped = self.context.stamp(row)
        stamped["node_id"] = str(stamped["node_id"])
        stamped.setdefault("wallclock_s", time.perf_counter() - self._started)
        validate(stamped, strict=self.strict)
        self._buffer.append(stamped)

    def log_many(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.log(row)

    def flush(self, step: int, learners: dict[str, Any] | None = None) -> None:
        """Write everything buffered, and checkpoint if learners are given.

        The parquet file is rewritten whole rather than appended to. Parquet has
        no cheap append, and at ~2 MB per seed rewriting 60 times over a run
        costs less than the machinery to avoid it -- while keeping exactly one
        file per seed, which is what makes a partial result readable.
        """
        self._rows.extend(self._buffer)
        self._buffer.clear()
        if not self._rows:
            return

        self._write_atomic(self.data_path, self._frame())
        if learners is not None:
            self._write_checkpoint(step, learners)

    def finalize(self) -> Path:
        """Final flush, then write the run's metadata alongside."""
        self._rows.extend(self._buffer)
        self._buffer.clear()
        self._write_atomic(self.data_path, self._frame())
        return self.data_path

    @property
    def n_rows(self) -> int:
        return len(self._rows) + len(self._buffer)

    def _frame(self):
        import pandas as pd

        frame = pd.DataFrame(self._rows)
        template = empty_frame()
        for column in template.columns:
            if column not in frame:
                frame[column] = None
        return frame[list(template.columns)]

    def _write_atomic(self, path: Path, frame: Any) -> None:
        """Write via a temporary file and rename, so an interrupted write cannot
        leave a truncated file that looks valid."""
        staging = path.with_suffix(path.suffix + ".tmp")
        try:
            frame.to_parquet(staging, index=False)
        except Exception as error:  # noqa: BLE001 - pyarrow raises various types
            raise RecorderError(f"could not write {path.name}: {error}") from error
        replace_with_retry(staging, path)

    # -- checkpointing ------------------------------------------------------ #

    def _write_checkpoint(self, step: int, learners: dict[str, Any]) -> None:
        states = {
            name: {
                node: {
                    "theta": learner.state(node).theta.clone(),
                    **{key: tensor.clone() for key, tensor in learner.state(node).extras.items()},
                }
                for node in range(learner.n_nodes)
            }
            for name, learner in learners.items()
            if hasattr(learner, "state")
        }
        payload = {
            "step": step,
            "config_hash": self.config_hash,
            "learner_states": states,
            "n_rows": len(self._rows),
            "version": CHECKPOINT_VERSION,
        }
        staging = self.checkpoint_path.with_suffix(".tmp")
        torch.save(payload, staging)
        replace_with_retry(staging, self.checkpoint_path)

    def load_checkpoint(self) -> Checkpoint | None:
        """The saved checkpoint, or ``None`` if there is none or it is unusable.

        A checkpoint from a *different* configuration is an error rather than a
        silent fresh start: resuming one experiment into another's directory
        would produce a file that looks like a single coherent run.
        """
        if not self.checkpoint_path.is_file():
            return None
        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - a damaged checkpoint is recoverable
            return None

        if payload.get("version") != CHECKPOINT_VERSION:
            return None
        if payload.get("config_hash") != self.config_hash:
            raise RecorderError(
                f"the checkpoint in {self.out_dir} was written for a different "
                "configuration. Resuming would splice two experiments into one file, "
                "with nothing in the parquet recording the seam.\n"
                "  To start over:  python scripts/run_experiment.py <name> --fresh\n"
                "  (or set FRESH = True in that script, or point run.out_dir elsewhere)\n"
                "This fires after any config change -- a retuned learning rate is "
                "enough -- so during a tuning pass it is expected rather than a fault."
            )
        return Checkpoint(
            step=payload["step"],
            config_hash=payload["config_hash"],
            learner_states=payload["learner_states"],
            n_rows=payload["n_rows"],
        )

    def resume(self, learners: dict[str, Any]) -> int:
        """Restore learner state and already-written rows; return the next step.

        Returns 0 when there is nothing to resume, so a caller can always write
        ``for step in range(recorder.resume(learners), horizon)``.
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return 0

        for name, states in checkpoint.learner_states.items():
            learner = learners.get(name)
            if learner is None:
                raise RecorderError(
                    f"the checkpoint holds state for learner {name!r}, which this run "
                    f"does not have. Present: {sorted(learners)}"
                )
            for node, entries in states.items():
                state = learner.state(node)
                state.theta = entries["theta"].clone()
                state.extras = {
                    key: tensor.clone() for key, tensor in entries.items() if key != "theta"
                }

        if self.data_path.is_file():
            import pandas as pd

            existing = pd.read_parquet(self.data_path)
            # Drop anything past the checkpoint: those rows were written before a
            # flush that never completed, and replaying their steps would
            # duplicate them.
            existing = existing[existing["t"] <= checkpoint.step]
            self._rows = existing.to_dict("records")

        return checkpoint.resume_from


# --------------------------------------------------------------------------- #
# run-level artefacts
# --------------------------------------------------------------------------- #


def write_metadata(out_dir: Path, config: Any, extra: dict[str, Any] | None = None) -> Path:
    """The resolved config, the environment, and the git revision.

    Written once per experiment so a result is always traceable to the exact
    settings and code that produced it.
    """
    from dekf_bench.utils.config import dump_config
    from dekf_bench.utils.determinism import collect_metadata

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.yaml")

    payload = {
        "config_hash": config_fingerprint(config),
        "environment": collect_metadata().as_dict(),
        **(extra or {}),
    }
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_ledger(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """The communication table, emitted once per experiment.

    Duplicates the per-row ``cum_scalars_tx`` on purpose: the column makes F2 a
    plain groupby with no join, while this table carries the *per-method*
    breakdown and the notes explaining why a payload is what it is -- which have
    nowhere to live in a long-format row.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ledger.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def new_run_id() -> str:
    """A short unique id, so two runs of the same config never collide."""
    return uuid.uuid4().hex[:12]


def read_results(path: Path):
    """Read one seed's file, or every seed in an experiment directory."""
    import pandas as pd

    path = Path(path)
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.glob("seed_*.parquet"))
    if not files:
        raise RecorderError(f"no results under {path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
