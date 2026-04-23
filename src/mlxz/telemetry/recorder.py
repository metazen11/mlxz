"""Fire-and-forget telemetry recorder.

All database writes are dispatched to a background thread via a
:class:`queue.Queue` so that the engine's hot path is never blocked by
I/O.  The background thread drains the queue in a tight loop, batching
inserts into single sessions where possible.

Usage::

    engine = create_engine_from_config()
    recorder = TelemetryRecorder(engine)
    run_id = recorder.start_run(config, hardware="m4_max_64gb", commit_sha="abc123")
    recorder.record_request(run_id, request_id="...", prompt_tokens=100, ...)
    recorder.end_run(run_id)
    recorder.close()  # flush remaining items and join the thread
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Any

from pydantic import SecretStr
from sqlalchemy import Engine

from mlxz.config import RuntimeConfig
from mlxz.telemetry.db import get_session_factory
from mlxz.telemetry.models import Measurement, RequestRow, Run

logger = logging.getLogger(__name__)

# Sentinel value pushed to the queue to signal the writer thread to stop.
_STOP = object()


def _safe_config_json(config: RuntimeConfig) -> str:
    """Serialise *config* to JSON, excluding any ``SecretStr`` fields.

    Walks the Pydantic model tree and replaces ``SecretStr`` values with
    the literal ``"**REDACTED**"`` so that API keys and DSN credentials
    are never persisted in the telemetry database.
    """

    def _redact(obj: Any) -> Any:
        if isinstance(obj, SecretStr):
            return "**REDACTED**"
        if isinstance(obj, dict):
            return {k: _redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_redact(v) for v in obj]
        return obj

    # Use mode="python" so SecretStr instances survive for _redact to catch.
    raw = config.model_dump(mode="python")
    redacted = _redact(raw)
    return json.dumps(redacted, default=str, sort_keys=True)


class TelemetryRecorder:
    """Non-blocking telemetry writer backed by a single daemon thread.

    Parameters
    ----------
    engine:
        A SQLAlchemy ``Engine`` (from :func:`create_engine_from_config`).
    max_queue_size:
        Upper bound on the write queue.  If the queue is full, new items
        are silently dropped and a warning is logged.
    """

    def __init__(self, engine: Engine, *, max_queue_size: int = 4096) -> None:
        self._engine = engine
        self._session_factory = get_session_factory(engine)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="mlxz-telemetry-writer",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        config: RuntimeConfig,
        *,
        hardware: str,
        commit_sha: str,
    ) -> int:
        """Create a new ``Run`` row and return its ``id``.

        This is the one *synchronous* database call so that the caller
        gets the ``run_id`` immediately.  All subsequent writes are
        fire-and-forget.
        """
        run = Run(
            commit_sha=commit_sha,
            hardware=hardware,
            model=config.model,
            draft_model=config.draft_model,
            quant=f"q{config.kv.bits}",
            kv_bits=config.kv.bits,
            wired_limit_mb=config.wired_limit_mb or 0,
            config_json=_safe_config_json(config),
            started_at=datetime.now(timezone.utc),
        )
        with self._session_factory() as session:
            session.add(run)
            session.commit()
            session.refresh(run)
            return run.id  # type: ignore[return-value]

    def record_request(
        self,
        run_id: int,
        *,
        request_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        prefix_cache_hit_tokens: int = 0,
        ttft_ms: float = 0.0,
        decode_tps: float = 0.0,
        acceptance_rate: float | None = None,
        rejected_reason: str | None = None,
    ) -> None:
        """Enqueue a ``RequestRow`` insert (fire-and-forget)."""
        row = RequestRow(
            id=request_id,
            run_id=run_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prefix_cache_hit_tokens=prefix_cache_hit_tokens,
            ttft_ms=ttft_ms,
            decode_tps=decode_tps,
            acceptance_rate=acceptance_rate,
            rejected_reason=rejected_reason,
            created_at=datetime.now(timezone.utc),
        )
        self._enqueue(row)

    def record_measurement(
        self,
        run_id: int,
        *,
        batch_size: int,
        aggregate_decode_tps: float,
        kv_used_bytes: int = 0,
        rss_bytes: int = 0,
        thermal_state: str = "normal",
    ) -> None:
        """Enqueue a ``Measurement`` insert (fire-and-forget)."""
        row = Measurement(
            run_id=run_id,
            sampled_at=datetime.now(timezone.utc),
            batch_size=batch_size,
            aggregate_decode_tps=aggregate_decode_tps,
            kv_used_bytes=kv_used_bytes,
            rss_bytes=rss_bytes,
            thermal_state=thermal_state,
        )
        self._enqueue(row)

    def end_run(self, run_id: int) -> None:  # noqa: ARG002
        """Placeholder for run finalisation logic (e.g. duration, summary).

        Currently a no-op.  Future versions may update the ``Run`` row with
        aggregated statistics or an ``ended_at`` timestamp.
        """

    def close(self, timeout: float = 5.0) -> None:
        """Flush the write queue and join the background thread.

        Parameters
        ----------
        timeout:
            Seconds to wait for the writer thread to drain.  After this
            deadline the thread is abandoned (it is a daemon thread so it
            will not prevent process exit).
        """
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enqueue(self, item: Any) -> None:
        """Best-effort enqueue; drops on overflow to avoid back-pressure."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning(
                "Telemetry write queue full — dropping %s",
                type(item).__name__,
            )

    def _writer_loop(self) -> None:
        """Background thread: drain queue items into the database."""
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                with self._session_factory() as session:
                    session.add(item)
                    session.commit()
            except Exception:
                logger.exception(
                    "Telemetry write failed for %s", type(item).__name__,
                )
