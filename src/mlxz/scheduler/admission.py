"""Admission controller — pure deterministic gate between API and engine."""
from __future__ import annotations

import structlog

from mlxz.types import (
    AdmissionDecision,
    AdmissionSnapshot,
    MemoryPressure,
    ResidencyBudget,
    ThermalState,
)
from mlxz.config import RuntimeConfig


logger = structlog.get_logger()


class AdmissionController:
    """Projects peak memory for incoming requests; accepts or rejects.

    Decision logic (checked in order):
    1. Thermal critical -> REJECT_THERMAL
    2. Memory pressure critical -> REJECT_MEMORY_PRESSURE
    3. Queue at capacity -> REJECT_QUEUE_FULL
    4. Projected KV exceeds available budget -> REJECT_OVER_BUDGET
    5. Otherwise -> ACCEPT

    Invariant: monotonic — if kv_used increases and budget is unchanged,
    decision never relaxes from REJECT to ACCEPT.
    """

    def __init__(
        self,
        budget: ResidencyBudget,
        config: RuntimeConfig,
        *,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
    ) -> None:
        self._budget = budget
        self._config = config
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._head_dim = head_dim

    def decide(
        self,
        prompt_tokens: int,
        max_new_tokens: int,
        snap: AdmissionSnapshot,
    ) -> tuple[AdmissionDecision, str]:
        """Returns (decision, human-readable reason). Deterministic; no I/O."""
        # 1. Thermal
        if snap.thermal_state == ThermalState.CRITICAL:
            return (AdmissionDecision.REJECT_THERMAL,
                    "System thermal state is critical; throttling new requests")

        # 2. Memory pressure
        if snap.memory_pressure == MemoryPressure.CRITICAL:
            return (AdmissionDecision.REJECT_MEMORY_PRESSURE,
                    "System memory pressure is critical")

        # 3. Queue depth
        max_concurrent = self._config.scheduler.max_concurrent_requests
        if snap.running_requests + snap.queued_requests >= max_concurrent:
            return (AdmissionDecision.REJECT_QUEUE_FULL,
                    f"Queue full: {snap.running_requests} running + {snap.queued_requests} queued >= {max_concurrent}")

        # 4. KV budget projection
        projected_bytes = self._project_peak(prompt_tokens, max_new_tokens)
        headroom_bytes = int(self._budget.kv_budget_bytes * self._config.scheduler.admission_headroom)
        available = self._budget.kv_budget_bytes - snap.kv_used_bytes - headroom_bytes

        if projected_bytes > available:
            return (AdmissionDecision.REJECT_OVER_BUDGET,
                    f"Projected {projected_bytes:,} bytes exceeds available {available:,} bytes "
                    f"(budget={self._budget.kv_budget_bytes:,}, used={snap.kv_used_bytes:,}, headroom={headroom_bytes:,})")

        return (AdmissionDecision.ACCEPT, "")

    def _project_peak(self, prompt_tokens: int, max_new_tokens: int) -> int:
        """Estimate peak KV cache bytes for this request."""
        total_tokens = prompt_tokens + max_new_tokens
        kv_bits = self._config.kv.bits
        # Per-token KV size: 2 (K+V) * n_layers * n_heads * head_dim * (bits/8)
        bytes_per_token = 2 * self._n_layers * self._n_heads * self._head_dim * (kv_bits / 8)
        return int(total_tokens * bytes_per_token)
