"""Append-only request journal for post-mortem analysis.

Writes structured JSONL to ``~/.cache/mlxz/request_journal.jsonl``.  The
journal records admission, completion, cancellation, and rejection events
with minimal overhead.  Prompt content is **never** logged.

File writes are protected with ``fcntl.flock`` for process-level safety.
On clean startup the previous journal is rotated (renamed with a timestamp
suffix) so each server session produces a fresh file.
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

_DEFAULT_JOURNAL_DIR = Path.home() / ".cache" / "mlxz"
_DEFAULT_JOURNAL_NAME = "request_journal.jsonl"


class RequestJournal:
    """Append-only JSONL journal for request lifecycle events."""

    def __init__(
        self,
        journal_dir: Path | None = None,
        *,
        rotate_on_init: bool = True,
    ) -> None:
        self._dir = journal_dir or _DEFAULT_JOURNAL_DIR
        self._path = self._dir / _DEFAULT_JOURNAL_NAME

        self._dir.mkdir(parents=True, exist_ok=True)

        if rotate_on_init:
            self._rotate()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_admitted(
        self,
        request_id: str,
        prompt_tokens: int,
        max_tokens: int,
    ) -> None:
        """Record that a request was admitted for inference."""
        self._append(
            {
                "event": "admitted",
                "request_id": request_id,
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
            }
        )

    def log_completed(
        self,
        request_id: str,
        tokens_generated: int,
    ) -> None:
        """Record that a request completed successfully."""
        self._append(
            {
                "event": "completed",
                "request_id": request_id,
                "tokens_generated": tokens_generated,
            }
        )

    def log_cancelled(
        self,
        request_id: str,
        reason: str,
    ) -> None:
        """Record that a request was cancelled (e.g. client disconnect)."""
        self._append(
            {
                "event": "cancelled",
                "request_id": request_id,
                "reason": reason,
            }
        )

    def log_rejected(
        self,
        request_id: str,
        reason: str,
    ) -> None:
        """Record that a request was rejected by admission control."""
        self._append(
            {
                "event": "rejected",
                "request_id": request_id,
                "reason": reason,
            }
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, record: dict[str, object]) -> None:
        """Serialise *record* as a single JSON line and flush to disk."""
        record["timestamp"] = time.time()
        line = json.dumps(record, separators=(",", ":")) + "\n"

        with open(self._path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _rotate(self) -> None:
        """Rename the existing journal file with a timestamp suffix."""
        if not self._path.exists():
            return
        # Only rotate non-empty files
        if self._path.stat().st_size == 0:
            return
        suffix = time.strftime("%Y%m%dT%H%M%S")
        rotated = self._path.with_suffix(f".{suffix}.jsonl")
        self._path.rename(rotated)

    @property
    def path(self) -> Path:
        """Return the path to the active journal file."""
        return self._path
