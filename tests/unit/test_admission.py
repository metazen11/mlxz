"""Tests for the admission controller."""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlxz.types import (
    AdmissionDecision,
    AdmissionSnapshot,
    MemoryPressure,
    ResidencyBudget,
    ThermalState,
)
from mlxz.config import RuntimeConfig
from mlxz.scheduler.admission import AdmissionController


def _make_budget(kv_budget: int = 10_000_000_000) -> ResidencyBudget:
    return ResidencyBudget(
        wired_limit_bytes=64_000_000_000,
        usable_budget_bytes=56_000_000_000,
        weight_bytes=40_000_000_000,
        activation_scratch_bytes=1_000_000_000,
        kv_budget_bytes=kv_budget,
        prefix_cache_budget_bytes=8_000_000_000,
    )


def _make_snap(
    kv_used: int = 0,
    running: int = 0,
    queued: int = 0,
    thermal: ThermalState = ThermalState.NORMAL,
    pressure: MemoryPressure = MemoryPressure.NORMAL,
) -> AdmissionSnapshot:
    return AdmissionSnapshot(
        kv_used_bytes=kv_used,
        kv_budget_bytes=10_000_000_000,
        running_requests=running,
        queued_requests=queued,
        thermal_state=thermal,
        memory_pressure=pressure,
    )


def _make_controller(kv_budget: int = 10_000_000_000) -> AdmissionController:
    config = RuntimeConfig(model="test")
    return AdmissionController(
        budget=_make_budget(kv_budget),
        config=config,
        n_layers=32, n_heads=32, head_dim=128,
    )


class TestAdmissionBasic:
    def test_accept_within_budget(self):
        ctrl = _make_controller()
        decision, _ = ctrl.decide(100, 100, _make_snap())
        assert decision == AdmissionDecision.ACCEPT

    def test_reject_thermal_critical(self):
        ctrl = _make_controller()
        decision, reason = ctrl.decide(
            100, 100, _make_snap(thermal=ThermalState.CRITICAL)
        )
        assert decision == AdmissionDecision.REJECT_THERMAL
        assert "thermal" in reason.lower()

    def test_reject_memory_pressure(self):
        ctrl = _make_controller()
        decision, _ = ctrl.decide(
            100, 100, _make_snap(pressure=MemoryPressure.CRITICAL)
        )
        assert decision == AdmissionDecision.REJECT_MEMORY_PRESSURE

    def test_reject_queue_full(self):
        ctrl = _make_controller()
        decision, reason = ctrl.decide(
            100, 100, _make_snap(running=8)  # default max is 8
        )
        assert decision == AdmissionDecision.REJECT_QUEUE_FULL

    def test_reject_over_budget(self):
        ctrl = _make_controller(kv_budget=1000)  # tiny budget
        decision, reason = ctrl.decide(
            10000, 10000, _make_snap()  # huge request
        )
        assert decision == AdmissionDecision.REJECT_OVER_BUDGET
        assert "projected" in reason.lower()

    def test_accept_with_existing_usage(self):
        ctrl = _make_controller()
        decision, _ = ctrl.decide(
            100, 100, _make_snap(kv_used=1000, running=1)
        )
        assert decision == AdmissionDecision.ACCEPT


class TestAdmissionMonotonicity:
    """Invariant: increasing kv_used never relaxes REJECT to ACCEPT."""

    @given(
        kv_used=st.integers(min_value=0, max_value=10_000_000_000),
        kv_delta=st.integers(min_value=1, max_value=1_000_000_000),
    )
    @settings(max_examples=50)
    def test_monotonic_reject(self, kv_used, kv_delta):
        ctrl = _make_controller()
        snap_before = _make_snap(kv_used=kv_used)
        snap_after = _make_snap(kv_used=kv_used + kv_delta)
        d1, _ = ctrl.decide(1000, 1000, snap_before)
        d2, _ = ctrl.decide(1000, 1000, snap_after)
        # If d1 was reject, d2 must also be reject (or same)
        if d1 != AdmissionDecision.ACCEPT:
            assert d2 != AdmissionDecision.ACCEPT
