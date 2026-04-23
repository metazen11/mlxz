"""Integration tests for health probes and Prometheus metrics endpoints.

Uses httpx.AsyncClient with ASGITransport to drive the actual FastAPI
application without starting a real server.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import pytest
import httpx
from httpx import ASGITransport

from mlxz.api.app import create_app
from mlxz.api.health import _health_state
from mlxz.api.metrics import create_metrics_app
from mlxz.config import RuntimeConfig
from mlxz.types import ServerPhase

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> RuntimeConfig:
    defaults = {"model": "test-model"}
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


@asynccontextmanager
async def _async_client(app) -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_health_state():
    """Reset the module-level health state before each test."""
    _health_state.phase = ServerPhase.STARTING
    _health_state.engine_alive = False
    _health_state.load_progress = 0.0
    yield
    _health_state.phase = ServerPhase.STARTING
    _health_state.engine_alive = False
    _health_state.load_progress = 0.0


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthLive:
    """GET /health/live -- liveness probe."""

    async def test_returns_200(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            resp = await client.get("/health/live")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    async def test_returns_200_regardless_of_phase(self) -> None:
        """Liveness always succeeds if the process is alive."""
        app = create_app(_make_config())
        async with _async_client(app) as client:
            for phase in ServerPhase:
                _health_state.phase = phase
                resp = await client.get("/health/live")
                assert resp.status_code == 200


class TestHealthReady:
    """GET /health/ready -- readiness probe."""

    async def test_not_ready_when_starting(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.STARTING
            _health_state.engine_alive = False
            resp = await client.get("/health/ready")
            assert resp.status_code == 503
            body = resp.json()
            assert body["status"] == "not_ready"
            assert body["phase"] == "STARTING"

    async def test_ready_when_engine_alive(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.READY
            _health_state.engine_alive = True
            resp = await client.get("/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ready"
            assert body["phase"] == "READY"

    async def test_not_ready_when_engine_dead(self) -> None:
        """Phase is READY but engine is not alive."""
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.READY
            _health_state.engine_alive = False
            resp = await client.get("/health/ready")
            assert resp.status_code == 503

    async def test_not_ready_when_draining(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.DRAINING
            _health_state.engine_alive = True
            resp = await client.get("/health/ready")
            assert resp.status_code == 503
            assert resp.json()["phase"] == "DRAINING"


class TestHealthStartup:
    """GET /health/startup -- startup probe."""

    async def test_loading_during_startup(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.STARTING
            _health_state.load_progress = 0.5
            resp = await client.get("/health/startup")
            assert resp.status_code == 503
            body = resp.json()
            assert body["status"] == "loading"
            assert body["progress"] == 0.5

    async def test_started_when_ready(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.READY
            resp = await client.get("/health/startup")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "started"
            assert body["progress"] == 1.0

    async def test_started_when_draining(self) -> None:
        """Startup probe still returns 200 when phase >= READY (including DRAINING)."""
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.DRAINING
            resp = await client.get("/health/startup")
            assert resp.status_code == 200


class TestHealthAlias:
    """GET /health -- alias for /health/ready."""

    async def test_alias_matches_ready_not_ready(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.STARTING
            resp = await client.get("/health")
            assert resp.status_code == 503

    async def test_alias_matches_ready_ok(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            _health_state.phase = ServerPhase.READY
            _health_state.engine_alive = True
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# Metrics endpoint tests
# ---------------------------------------------------------------------------


class TestMetricsApp:
    """Tests for the standalone Prometheus metrics application."""

    async def test_metrics_returns_200(self) -> None:
        metrics_app = create_metrics_app()
        async with _async_client(metrics_app) as client:
            resp = await client.get("/metrics")
            assert resp.status_code == 200

    async def test_metrics_content_type(self) -> None:
        metrics_app = create_metrics_app()
        async with _async_client(metrics_app) as client:
            resp = await client.get("/metrics")
            ct = resp.headers["content-type"]
            # Prometheus text exposition format
            assert "text/plain" in ct or "openmetrics" in ct.lower()

    async def test_metrics_contains_mlxz_metrics(self) -> None:
        metrics_app = create_metrics_app()
        async with _async_client(metrics_app) as client:
            resp = await client.get("/metrics")
            body = resp.text
            # Verify at least some of our custom metrics appear
            assert "mlxz_requests_total" in body or "mlxz_active_requests" in body


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Verify security headers are present on responses."""

    async def test_nosniff_header(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            resp = await client.get("/health/live")
            assert resp.headers.get("x-content-type-options") == "nosniff"

    async def test_frame_deny_header(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            resp = await client.get("/health/live")
            assert resp.headers.get("x-frame-options") == "DENY"

    async def test_cache_control_header(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            resp = await client.get("/health/live")
            assert resp.headers.get("cache-control") == "no-store"

    async def test_version_header(self) -> None:
        app = create_app(_make_config())
        async with _async_client(app) as client:
            resp = await client.get("/health/live")
            assert "x-mlxz-version" in resp.headers


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    """Verify CORS behaviour based on configuration."""

    async def test_no_cors_when_origins_empty(self) -> None:
        """When cors_origins is empty, no CORS headers should appear."""
        config = _make_config()
        assert config.server.cors_origins == []
        app = create_app(config)
        async with _async_client(app) as client:
            resp = await client.get(
                "/health/live",
                headers={"Origin": "http://evil.example.com"},
            )
            assert resp.status_code == 200
            # No access-control-allow-origin header when CORS is not configured
            assert "access-control-allow-origin" not in resp.headers

    async def test_cors_present_when_origins_configured(self) -> None:
        """When cors_origins is set, CORS headers appear for matching origin."""
        config = _make_config()
        config.server.cors_origins = ["http://example.com"]
        app = create_app(config)
        async with _async_client(app) as client:
            resp = await client.get(
                "/health/live",
                headers={"Origin": "http://example.com"},
            )
            assert resp.status_code == 200
            assert resp.headers.get("access-control-allow-origin") == "http://example.com"
