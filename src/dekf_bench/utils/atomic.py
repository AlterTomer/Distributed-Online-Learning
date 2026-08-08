r"""Writing a file so a crash cannot leave a half-written one behind.

Everything durable in this project is written to a staging file and then renamed
over the target, so an interrupted write leaves the *previous* version intact
rather than a truncated file that loads without complaint.

The rename is the delicate part, and it is why this module exists rather than a
bare ``staging.replace(target)`` at each call site. See :func:`replace_with_retry`.

Lives in ``utils`` rather than ``recording`` because both the recorder and the
MNIST cache need it, and ``data`` must not depend on ``recording``.
"""

from __future__ import annotations

import time
from pathlib import Path

#: How many times to retry the rename, and the base backoff between attempts.
#: The waits are cumulative (0.1, 0.2, 0.3, ...), so total patience is about
#: 3.6 s -- comfortably longer than a scanner holds a file, and far shorter than
#: the evaluation interval this runs inside.
ATTEMPTS = 8
BACKOFF_S = 0.1


def replace_with_retry(staging: Path, target: Path) -> None:
    """``staging.replace(target)``, retried on a transient Windows lock.

    ``os.replace`` is genuinely atomic on POSIX. On Windows it fails outright
    with ``PermissionError`` (WinError 5) when the target is held open by **any**
    process -- an antivirus scanner, a search indexer, a cloud-sync client, or a
    handle the OS has not finished reaping from a killed run. The write is
    complete and correct; the rename simply cannot land at that instant.

    This showed up as a crash mid-experiment right after a run was interrupted:
    the next run reached its first checkpoint and died, losing a 15-minute
    experiment to a condition that clears in milliseconds (design note D42).

    **The retry is deliberately narrow.** Only ``PermissionError``, only for a
    few seconds, and the original error is re-raised after the last attempt
    rather than swallowed -- a genuinely read-only directory raises the same
    class and must still fail loudly.
    """
    for attempt in range(ATTEMPTS):
        try:
            staging.replace(target)
            return
        except PermissionError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(BACKOFF_S * (attempt + 1))
