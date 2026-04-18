"""Shared types, enums, dataclasses, and protocols for mlxz.

Every public symbol here is imported across multiple modules.  Keep this
file dependency-free (stdlib + typing only) so it never creates circular
imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import asyncio


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SamplingParams:
    """Per-request sampling configuration."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    """``-1`` disables top-k filtering."""
    min_p: float = 0.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    """Stop sequences — generation halts when any is produced."""


# ---------------------------------------------------------------------------
# State enums
# ---------------------------------------------------------------------------


class RequestState(IntEnum):
    """Lifecycle states for an inference request."""

    QUEUED = 0
    ADMITTED = 1
    PREFILLING = 2
    DECODING = 3
    COMPLETED = 4
    CANCELLED = 5
    REJECTED = 6


class AdmissionDecision(IntEnum):
    """Outcome of the admission controller gate."""

    ACCEPT = 0
    REJECT_OVER_BUDGET = 1
    REJECT_QUEUE_FULL = 2
    REJECT_THERMAL = 3
    REJECT_MEMORY_PRESSURE = 4
    REJECT_SHUTTING_DOWN = 5


class ServerPhase(IntEnum):
    """Server lifecycle phase."""

    STARTING = 0
    READY = 1
    DRAINING = 2
    STOPPED = 3


class HealthStatus(Enum):
    """Aggregate health probe result."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class ThermalState(Enum):
    """Thermal throttle classification from powermetrics."""

    NORMAL = "normal"
    WARN = "warn"
    CRITICAL = "critical"


class MemoryPressure(Enum):
    """Unified memory pressure classification."""

    NORMAL = "normal"
    WARN = "warn"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data snapshots
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PrefixCacheStats:
    """Point-in-time prefix cache counters."""

    hits: int = 0
    misses: int = 0
    hit_bytes: int = 0
    evictions: int = 0
    memory_used_bytes: int = 0
    disk_used_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ResidencyBudget:
    """Immutable snapshot of the residency planner's memory layout."""

    wired_limit_bytes: int
    usable_budget_bytes: int
    weight_bytes: int
    activation_scratch_bytes: int
    kv_budget_bytes: int
    prefix_cache_budget_bytes: int


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    """Immutable snapshot consumed by the admission controller."""

    kv_used_bytes: int
    kv_budget_bytes: int
    running_requests: int
    queued_requests: int
    thermal_state: ThermalState
    memory_pressure: MemoryPressure


@dataclass(frozen=True, slots=True)
class DrainResult:
    """Outcome of a graceful shutdown drain."""

    completed: int
    force_cancelled: int
    drain_duration_seconds: float


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EngineProtocol(Protocol):
    """Minimal interface shared by all engine implementations."""

    async def submit(self, request: Any) -> None:
        """Enqueue a request for inference."""
        ...

    def snapshot(self) -> AdmissionSnapshot:
        """Return a point-in-time admission snapshot."""
        ...

    async def shutdown(self) -> DrainResult:
        """Initiate graceful shutdown and return drain statistics."""
        ...


@runtime_checkable
class PrefixCacheProtocol(Protocol):
    """Content-addressed prefix cache interface."""

    async def lookup(
        self, token_hashes: list[bytes]
    ) -> tuple[int, Any | None]:
        """Return ``(n_matched_chunks, cached_kv_reference_or_none)``."""
        ...

    async def store(self, token_hashes: list[bytes], kv: Any) -> None:
        """Persist a prefix KV slice for future reuse."""
        ...

    def stats(self) -> PrefixCacheStats:
        """Return current cache statistics."""
        ...


@runtime_checkable
class KVCacheProtocol(Protocol):
    """Abstract KV cache interface (quantised, paged, streaming, etc.)."""

    def allocate(self, seq_len: int) -> Any:
        """Reserve cache storage for *seq_len* tokens."""
        ...

    def free(self) -> None:
        """Release all storage held by this cache instance."""
        ...

    @property
    def used_bytes(self) -> int:
        """Current memory consumption in bytes."""
        ...
