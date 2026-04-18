"""Graceful shutdown coordinator for mlxz.

Orchestrates signal handling, admission gating, drain of in-flight
requests, and final resource cleanup.  The ``ShutdownCoordinator`` is
the single authority on whether the server is accepting new work.
"""

from __future__ import annotations

import asyncio
import signal
import time
from typing import TYPE_CHECKING

import structlog

from mlxz.types import DrainResult, ServerPhase

if TYPE_CHECKING:
    from mlxz.types import EngineProtocol

logger = structlog.get_logger()


class ShutdownCoordinator:
    """Orchestrates graceful shutdown across API, engine, and telemetry layers."""

    def __init__(self, drain_timeout_seconds: float = 30.0) -> None:
        self._phase: ServerPhase = ServerPhase.STARTING
        self._drain_timeout = drain_timeout_seconds
        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> ServerPhase:
        """Current server lifecycle phase."""
        return self._phase

    @phase.setter
    def phase(self, value: ServerPhase) -> None:
        old = self._phase
        self._phase = value
        logger.info("server_phase_change", from_phase=old.name, to_phase=value.name)

    @property
    def is_accepting(self) -> bool:
        """``True`` when the server is in READY phase and accepting requests."""
        return self._phase == ServerPhase.READY

    @property
    def shutdown_event(self) -> asyncio.Event:
        """Event that is set when shutdown is initiated.

        Callers can ``await coordinator.shutdown_event.wait()`` to block
        until a signal arrives.
        """
        return self._shutdown_event

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register SIGTERM and SIGINT handlers on *loop*.

        Must be called once during server startup from the main thread.
        """
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._initiate_shutdown)
        logger.info("signal_handlers_installed", signals=["SIGTERM", "SIGINT"])

    def _initiate_shutdown(self) -> None:
        """Transition to DRAINING and wake any waiters."""
        if self._phase >= ServerPhase.DRAINING:
            # Already shutting down -- ignore repeated signals.
            return
        self.phase = ServerPhase.DRAINING
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    async def drain(self, engine: EngineProtocol) -> DrainResult:
        """Wait for running requests to complete, then force-cancel survivors.

        Steps
        -----
        1. Set admission gate to REJECT_SHUTTING_DOWN (callers check
           ``is_accepting``).
        2. Wait up to ``drain_timeout`` for ``engine.shutdown()`` to
           complete gracefully.
        3. If the timeout expires, the engine is expected to force-cancel
           remaining requests.
        4. Transition to STOPPED.

        Returns a ``DrainResult`` with completion and cancellation counts.
        """
        t0 = time.monotonic()
        logger.info(
            "drain_started",
            drain_timeout_seconds=self._drain_timeout,
        )

        try:
            result = await asyncio.wait_for(
                engine.shutdown(),
                timeout=self._drain_timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            logger.warning(
                "drain_timeout_exceeded",
                drain_duration_seconds=round(elapsed, 3),
            )
            result = DrainResult(
                completed=0,
                force_cancelled=0,
                drain_duration_seconds=round(elapsed, 3),
            )

        self.phase = ServerPhase.STOPPED

        logger.info(
            "drain_completed",
            completed=result.completed,
            force_cancelled=result.force_cancelled,
            drain_duration_seconds=result.drain_duration_seconds,
        )
        return result
