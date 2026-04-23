"""Health probe endpoints for mlxz.

Implements split Kubernetes-style probes:

- ``/health/live``    -- process liveness (always 200).
- ``/health/ready``   -- readiness gate (200 when READY, 503 otherwise).
- ``/health/startup`` -- startup completion (200 after model load, 503 during).
- ``/health``         -- backwards-compatible alias for ``/health/ready``.

All health endpoints are exempt from bearer-token authentication so that
orchestrators and load balancers can probe without credentials.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from mlxz.types import ServerPhase

# ---------------------------------------------------------------------------
# Shared health state
# ---------------------------------------------------------------------------


class HealthState:
    """Mutable singleton holding the current server phase and engine liveness.

    Updated by the lifespan manager and the engine watchdog.  Read by the
    health probe dependency.
    """

    __slots__ = ("phase", "engine_alive", "load_progress")

    def __init__(self) -> None:
        self.phase: ServerPhase = ServerPhase.STARTING
        self.engine_alive: bool = False
        self.load_progress: float = 0.0


# Module-level instance shared across the application.
_health_state = HealthState()


def get_health_state() -> HealthState:
    """FastAPI dependency that provides the shared :class:`HealthState`."""
    return _health_state


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Liveness probe -- always returns 200 if the process is alive."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(
    state: HealthState = Depends(get_health_state),
) -> Response:
    """Readiness probe -- 200 when READY, 503 otherwise."""
    if state.phase == ServerPhase.READY and state.engine_alive:
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "phase": "READY"},
        )

    phase_name = state.phase.name
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "phase": phase_name},
    )


@router.get("/health/startup")
async def startup_probe(
    state: HealthState = Depends(get_health_state),
) -> Response:
    """Startup probe -- 200 when model loading is complete, 503 during load."""
    if state.phase >= ServerPhase.READY:
        return JSONResponse(
            status_code=200,
            content={"status": "started", "progress": 1.0},
        )

    return JSONResponse(
        status_code=503,
        content={"status": "loading", "progress": state.load_progress},
    )


@router.get("/health")
async def health_alias(
    state: HealthState = Depends(get_health_state),
) -> Response:
    """Backwards-compatible alias for ``/health/ready``."""
    return await readiness(state)
