"""Integration tests for OpenAI-compatible API endpoints.

Exercises the full API stack (FastAPI routers, SSE formatting, schemas,
admission rejection) WITHOUT loading a real model.  A lightweight
MockEngine generates fixed tokens on a background thread, matching the
real engine's contract of putting Token objects on the janus sync_q.
"""
from __future__ import annotations

import json
import threading

import pytest
import httpx
from httpx import ASGITransport

from mlxz.api.health import _health_state
from mlxz.api.health import router as health_router
from mlxz.api.openai import router as openai_router
from mlxz.config import RuntimeConfig
from mlxz.engine.request import Token
from mlxz.engine.thread_boundary import CancellationRegistry, RequestBridge
from mlxz.scheduler.admission import AdmissionController
from mlxz.types import (
    AdmissionSnapshot,
    DrainResult,
    MemoryPressure,
    ResidencyBudget,
    ServerPhase,
    ThermalState,
)

# janus queues require a running asyncio event loop; restrict to asyncio only.
pytestmark = [
    pytest.mark.anyio,
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer that returns deterministic token IDs."""

    eos_token_id = 2

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True
    ):
        return [1, 3, 4, 5, 6]

    def encode(self, text):
        return [1, 3, 4]

    def decode(self, tokens):
        return "mock"


class MockEngine:
    """Engine mock that generates fixed tokens via a background thread.

    The real engine puts ``Token`` objects on ``request.output_channel.sync_q``
    from a dedicated compute thread and terminates with a ``None`` sentinel.
    This mock replicates that contract.
    """

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens or ["Hello", " world", "!"]
        self._model_name = "mock-model"
        self._n_layers = 2
        self._n_heads = 2
        self._head_dim = 64

    @property
    def model_name(self) -> str:
        return self._model_name

    async def submit(self, request) -> None:
        """Kick off token generation in a background thread."""
        t = threading.Thread(
            target=self._generate, args=(request,), daemon=True
        )
        t.start()

    def _generate(self, request) -> None:
        for i, text in enumerate(self._tokens[: request.max_tokens]):
            request.output_channel.sync_q.put(
                Token(token_id=i + 10, text=text)
            )
            request.completion_token_count += 1
        request.finish_reason = "stop"
        request.output_channel.sync_q.put(None)  # EOS sentinel

    def snapshot(self) -> AdmissionSnapshot:
        return AdmissionSnapshot(
            kv_used_bytes=0,
            kv_budget_bytes=10_000_000_000,
            running_requests=0,
            queued_requests=0,
            thermal_state=ThermalState.NORMAL,
            memory_pressure=MemoryPressure.NORMAL,
        )

    async def shutdown(self) -> DrainResult:
        return DrainResult(
            completed=0, force_cancelled=0, drain_duration_seconds=0.0
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_budget() -> ResidencyBudget:
    return ResidencyBudget(
        wired_limit_bytes=64_000_000_000,
        usable_budget_bytes=56_000_000_000,
        weight_bytes=1_000_000,
        activation_scratch_bytes=1_000_000,
        kv_budget_bytes=10_000_000_000,
        prefix_cache_budget_bytes=8_000_000_000,
    )


@pytest.fixture(autouse=True)
def _reset_health_state():
    """Reset module-level health state around each test."""
    _health_state.phase = ServerPhase.READY
    _health_state.engine_alive = True
    _health_state.load_progress = 1.0
    yield
    _health_state.phase = ServerPhase.STARTING
    _health_state.engine_alive = False
    _health_state.load_progress = 0.0


@pytest.fixture
def mock_app():
    """Build a FastAPI app with mock engine -- no real model, no lifespan."""
    from fastapi import FastAPI
    from mlxz.security import SecurityHeadersMiddleware

    config = RuntimeConfig(model="mock-model")

    app = FastAPI(title="mlxz-test")
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(health_router)
    app.include_router(openai_router)

    engine = MockEngine()
    cancellations = CancellationRegistry()
    budget = _make_budget()
    admission = AdmissionController(
        budget, config, n_layers=2, n_heads=2, head_dim=64
    )

    app.state.engine = engine
    app.state.admission = admission
    app.state.tokenizer = MockTokenizer()
    app.state.cancellations = cancellations
    app.state.config = config

    return app


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


class TestChatCompletionsNonStreaming:
    async def test_returns_200_with_valid_body(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"]  # non-empty
        assert data["choices"][0]["finish_reason"] == "stop"

    async def test_usage_tokens_populated(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                },
            )
        data = resp.json()
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert (
            data["usage"]["total_tokens"]
            == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
        )

    async def test_response_id_prefix(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                },
            )
        data = resp.json()
        assert data["id"].startswith("chatcmpl-")

    async def test_generated_text_matches_mock(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                },
            )
        content = resp.json()["choices"][0]["message"]["content"]
        assert content == "Hello world!"


class TestChatCompletionsStreaming:
    async def test_content_type_is_sse(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    async def test_sse_ends_with_done(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                    "stream": True,
                },
            )
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert data_lines[-1] == "data: [DONE]"

    async def test_first_chunk_has_role(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                    "stream": True,
                },
            )
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        first_chunk = json.loads(data_lines[0].removeprefix("data: "))
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["role"] == "assistant"

    async def test_content_chunks_present(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                    "stream": True,
                },
            )
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        # Should have: 1 role chunk + 3 content chunks + 1 finish chunk = 5
        # (role, "Hello", " world", "!", finish)
        assert len(data_lines) >= 4  # role + at least some content + finish

    async def test_final_chunk_has_finish_reason(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                    "stream": True,
                },
            )
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        last_data = json.loads(data_lines[-1].removeprefix("data: "))
        assert last_data["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Text completions (legacy)
# ---------------------------------------------------------------------------


class TestCompletions:
    async def test_returns_200_with_valid_body(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "mock-model",
                    "prompt": "Hello",
                    "max_tokens": 3,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"]  # non-empty

    async def test_response_id_prefix(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "mock-model",
                    "prompt": "Hello",
                    "max_tokens": 1,
                },
            )
        data = resp.json()
        assert data["id"].startswith("cmpl-")

    async def test_usage_populated(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "mock-model",
                    "prompt": "Hello",
                    "max_tokens": 3,
                },
            )
        data = resp.json()
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------


class TestModels:
    async def test_list_models(self, mock_app) -> None:
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "mock-model"
        assert data["data"][0]["object"] == "model"


# ---------------------------------------------------------------------------
# Admission rejection
# ---------------------------------------------------------------------------


class TestAdmissionRejection:
    async def test_thermal_rejection_429(self, mock_app) -> None:
        """CRITICAL thermal state triggers 429."""
        original_snapshot = mock_app.state.engine.snapshot

        def thermal_snapshot():
            return AdmissionSnapshot(
                kv_used_bytes=0,
                kv_budget_bytes=10_000_000_000,
                running_requests=0,
                queued_requests=0,
                thermal_state=ThermalState.CRITICAL,
                memory_pressure=MemoryPressure.NORMAL,
            )

        mock_app.state.engine.snapshot = thermal_snapshot
        try:
            transport = ASGITransport(app=mock_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
            assert resp.status_code == 429
            body = resp.json()
            assert "error" in body["detail"]
            assert body["detail"]["error"]["type"] == "rate_limit_error"
        finally:
            mock_app.state.engine.snapshot = original_snapshot

    async def test_memory_pressure_rejection_429(self, mock_app) -> None:
        """CRITICAL memory pressure triggers 429."""
        original_snapshot = mock_app.state.engine.snapshot

        def pressure_snapshot():
            return AdmissionSnapshot(
                kv_used_bytes=0,
                kv_budget_bytes=10_000_000_000,
                running_requests=0,
                queued_requests=0,
                thermal_state=ThermalState.NORMAL,
                memory_pressure=MemoryPressure.CRITICAL,
            )

        mock_app.state.engine.snapshot = pressure_snapshot
        try:
            transport = ASGITransport(app=mock_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
            assert resp.status_code == 429
        finally:
            mock_app.state.engine.snapshot = original_snapshot

    async def test_completions_also_rejected(self, mock_app) -> None:
        """Admission rejection applies to /v1/completions too."""
        original_snapshot = mock_app.state.engine.snapshot

        def thermal_snapshot():
            return AdmissionSnapshot(
                kv_used_bytes=0,
                kv_budget_bytes=10_000_000_000,
                running_requests=0,
                queued_requests=0,
                thermal_state=ThermalState.CRITICAL,
                memory_pressure=MemoryPressure.NORMAL,
            )

        mock_app.state.engine.snapshot = thermal_snapshot
        try:
            transport = ASGITransport(app=mock_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/completions",
                    json={
                        "model": "mock-model",
                        "prompt": "Hello",
                    },
                )
            assert resp.status_code == 429
        finally:
            mock_app.state.engine.snapshot = original_snapshot


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    async def test_missing_model_field(self, mock_app) -> None:
        """Missing required field triggers 422."""
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert resp.status_code == 422

    async def test_missing_messages_field(self, mock_app) -> None:
        """Missing messages triggers 422."""
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-model"},
            )
        assert resp.status_code == 422

    async def test_invalid_temperature(self, mock_app) -> None:
        """Temperature > 2.0 triggers 422."""
        transport = ASGITransport(app=mock_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "temperature": 3.0,
                },
            )
        assert resp.status_code == 422
