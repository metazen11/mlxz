"""Circuit breakers for thermal throttling and memory pressure response."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import structlog

from mlxz.types import ThermalState

logger = structlog.get_logger()


class CircuitState(Enum):
    CLOSED = "closed"  # Normal — accepting requests
    OPEN = "open"  # Tripped — rejecting requests
    HALF_OPEN = "half_open"  # Testing — accepting limited requests


@dataclass(slots=True)
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    thermal_critical_threshold: int = 3  # consecutive critical readings to trip
    memory_pressure_threshold: float = 0.90  # KV usage ratio to trip
    cooldown_seconds: float = 30.0  # time in OPEN before testing HALF_OPEN
    half_open_max_requests: int = 2  # max concurrent in HALF_OPEN


class CircuitBreaker:
    """Monitors thermal and memory state, trips when thresholds exceeded.

    States:
    - CLOSED: Normal operation, all requests accepted
    - OPEN: Thresholds exceeded, new requests rejected (429)
    - HALF_OPEN: After cooldown, allows limited requests to test recovery
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._last_trip_time: float = 0
        self._consecutive_thermal_critical: int = 0
        self._half_open_active: int = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    def should_accept(self) -> tuple[bool, str]:
        """Check if a new request should be accepted.

        Returns (accept, reason). Reason is empty string if accepted.
        """
        if self._state == CircuitState.CLOSED:
            return True, ""

        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_trip_time
            if elapsed >= self._config.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_active = 0
                logger.info("circuit_breaker_half_open", elapsed_s=round(elapsed, 1))
                return True, ""
            remaining = self._config.cooldown_seconds - elapsed
            return False, f"Circuit breaker OPEN (cooldown {remaining:.0f}s remaining)"

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_active < self._config.half_open_max_requests:
                self._half_open_active += 1
                return True, ""
            return False, "Circuit breaker HALF_OPEN (at capacity)"

        return True, ""

    def record_success(self) -> None:
        """Record a successful request completion."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_active = max(0, self._half_open_active - 1)
            if self._half_open_active == 0:
                self._state = CircuitState.CLOSED
                self._consecutive_thermal_critical = 0
                logger.info("circuit_breaker_closed", reason="recovery_confirmed")

    def record_thermal(self, state: ThermalState) -> None:
        """Update thermal tracking. May trip the breaker."""
        if state == ThermalState.CRITICAL:
            self._consecutive_thermal_critical += 1
            if self._consecutive_thermal_critical >= self._config.thermal_critical_threshold:
                self._trip("thermal_critical")
        else:
            self._consecutive_thermal_critical = 0

    def record_memory_pressure(self, kv_used: int, kv_budget: int) -> None:
        """Check memory pressure. May trip the breaker."""
        if kv_budget > 0:
            ratio = kv_used / kv_budget
            if ratio >= self._config.memory_pressure_threshold:
                self._trip(f"memory_pressure_{ratio:.0%}")

    def _trip(self, reason: str) -> None:
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._last_trip_time = time.monotonic()
            logger.warning("circuit_breaker_tripped", reason=reason)

    def reset(self) -> None:
        """Force reset to CLOSED."""
        self._state = CircuitState.CLOSED
        self._consecutive_thermal_critical = 0
        self._half_open_active = 0
