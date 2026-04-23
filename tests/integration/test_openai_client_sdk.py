"""OpenAI SDK smoke tests against an in-process mlxz ASGI app."""
from __future__ import annotations

import threading

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from mlxz.api.health import router as health_router
from mlxz.api.openai import router as openai_router
from mlxz.config import RuntimeConfig
from mlxz.engine.request import Token
from mlxz.engine.thread_boundary import CancellationRegistry
from mlxz.scheduler.admission import AdmissionController
from mlxz.types import (
    AdmissionSnapshot,
    DrainResult,
    MemoryPressure,
    ResidencyBudget,
    ThermalState,
)

openai = pytest.importorskip("openai")
AsyncOpenAI = openai.AsyncOpenAI

pytestmark = pytest.mark.anyio


class _MockTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3]

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens):
        return "x"


class _MockEngine:
    def __init__(self) -> None:
        self._model_name = "mock-model"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def submit(self, request) -> None:
        threading.Thread(target=self._generate, args=(request,), daemon=True).start()

    def _generate(self, request) -> None:
        for idx, text in enumerate(["Hello", " ", "SDK"][: request.max_tokens]):
            request.output_channel.sync_q.put(Token(token_id=idx + 1, text=text))
            request.completion_token_count += 1
        request.finish_reason = "stop"
        request.output_channel.sync_q.put(None)

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
        return DrainResult(completed=0, force_cancelled=0, drain_duration_seconds=0.0)


def _make_budget() -> ResidencyBudget:
    return ResidencyBudget(
        wired_limit_bytes=64_000_000_000,
        usable_budget_bytes=56_000_000_000,
        weight_bytes=1_000_000,
        activation_scratch_bytes=1_000_000,
        kv_budget_bytes=10_000_000_000,
        prefix_cache_budget_bytes=8_000_000_000,
    )


@pytest.fixture
def sdk_app():
    app = FastAPI(title="mlxz-sdk-test")
    app.include_router(health_router)
    app.include_router(openai_router)

    config = RuntimeConfig(model="mock-model")
    app.state.engine = _MockEngine()
    app.state.admission = AdmissionController(_make_budget(), config, n_layers=2, n_heads=2, head_dim=64)
    app.state.tokenizer = _MockTokenizer()
    app.state.cancellations = CancellationRegistry()
    app.state.telemetry = None
    app.state.telemetry_run_id = None


@pytest.fixture
def sdk_transport(sdk_app):
    return ASGITransport(app=sdk_app)


@pytest.mark.anyio
async def test_chat_non_streaming(sdk_transport) -> None:
    async with httpx.AsyncClient(transport=sdk_transport, base_url="http://test") as http_client:
        client = AsyncOpenAI(base_url="http://test/v1", api_key="test", http_client=http_client)
        try:
            resp = await client.chat.completions.create(
                model="mock-model",
                messages=[{"role": "user", "content": "Say hi"}],
                max_tokens=3,
            )
        finally:
            await client.close()
    assert resp.choices[0].message.content == "Hello SDK"
    assert resp.usage.completion_tokens > 0


@pytest.mark.anyio
async def test_chat_streaming(sdk_transport) -> None:
    async with httpx.AsyncClient(transport=sdk_transport, base_url="http://test") as http_client:
        client = AsyncOpenAI(base_url="http://test/v1", api_key="test", http_client=http_client)
        parts: list[str] = []
        try:
            stream = await client.chat.completions.create(
                model="mock-model",
                messages=[{"role": "user", "content": "Say hi"}],
                max_tokens=3,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    parts.append(content)
        finally:
            await client.close()
    assert "".join(parts) == "Hello SDK"


@pytest.mark.anyio
async def test_completions_non_streaming(sdk_transport) -> None:
    async with httpx.AsyncClient(transport=sdk_transport, base_url="http://test") as http_client:
        client = AsyncOpenAI(base_url="http://test/v1", api_key="test", http_client=http_client)
        try:
            resp = await client.completions.create(
                model="mock-model",
                prompt="Hello",
                max_tokens=3,
            )
        finally:
            await client.close()
    assert resp.choices[0].text == "Hello SDK"


@pytest.mark.anyio
async def test_list_models(sdk_transport) -> None:
    async with httpx.AsyncClient(transport=sdk_transport, base_url="http://test") as http_client:
        client = AsyncOpenAI(base_url="http://test/v1", api_key="test", http_client=http_client)
        try:
            models = await client.models.list()
        finally:
            await client.close()
    assert models.data[0].id == "mock-model"
