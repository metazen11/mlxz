"""Engine thread supervisor with crash recovery.

Wraps the engine's blocking ``run()`` method in a supervised loop that
catches unhandled exceptions, updates health state, and optionally
restarts the engine up to a configurable maximum.  If restarts are
exhausted the process exits hard to avoid leaving a zombie API server.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Callable

import structlog

from mlxz.types import HealthStatus

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class EngineThreadSupervisor:
    """Wraps the engine's ``run()`` loop with crash recovery.

    Parameters
    ----------
    engine:
        Any object with a blocking ``run()`` method (the engine compute
        loop).  Duck-typed so we don't create a hard dependency on the
        engine module.
    max_restarts:
        Maximum number of crash restarts before hard exit.
    restart_backoff_seconds:
        Base backoff multiplied by the restart ordinal.
    health_callback:
        Called with ``(HealthStatus, reason)`` whenever the supervisor
        needs to update the aggregate health state.
    """

    def __init__(
        self,
        engine: object,
        *,
        max_restarts: int = 3,
        restart_backoff_seconds: float = 2.0,
        health_callback: Callable[[HealthStatus, str], None] | None = None,
    ) -> None:
        self._engine = engine
        self._max_restarts = max_restarts
        self._backoff = restart_backoff_seconds
        self._health_callback = health_callback or self._noop_callback

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_supervised(self) -> None:
        """Target for ``threading.Thread``.

        Runs the engine in a loop, catching exceptions and restarting
        up to ``max_restarts`` times with exponential backoff.  On
        clean exit (no exception), the loop terminates normally.  If
        restarts are exhausted, calls ``os._exit(1)`` to prevent a
        headless API server from continuing to accept traffic.
        """
        restarts = 0

        while restarts <= self._max_restarts:
            try:
                self._get_run_method()()
            except Exception as exc:
                restarts += 1
                reason = f"engine crash: {exc}"
                self._health_callback(HealthStatus.RED, reason)
                logger.error(
                    "engine_crash",
                    exc_info=True,
                    restart_attempt=restarts,
                    max_restarts=self._max_restarts,
                )

                if restarts > self._max_restarts:
                    break

                backoff = self._backoff * restarts
                logger.info(
                    "engine_restart_backoff",
                    backoff_seconds=backoff,
                    restart_attempt=restarts,
                )
                time.sleep(backoff)
            else:
                # Clean exit -- engine shut down without an exception.
                logger.info("engine_clean_exit")
                return

        logger.critical(
            "engine_max_restarts_exceeded",
            max_restarts=self._max_restarts,
        )
        os._exit(1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_run_method(self) -> Callable[[], None]:
        """Resolve the engine's ``run`` method via duck typing."""
        run = getattr(self._engine, "run", None)
        if run is None or not callable(run):
            msg = (
                f"Engine {type(self._engine).__name__!r} does not have "
                f"a callable 'run' method"
            )
            raise TypeError(msg)
        return run  # type: ignore[return-value]

    @staticmethod
    def _noop_callback(_status: HealthStatus, _reason: str) -> None:
        """Default no-op health callback."""
