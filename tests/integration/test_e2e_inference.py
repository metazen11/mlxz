"""End-to-end inference tests against a real mlxz server.

These tests start an actual ``mlxz serve`` process, wait for it to become
ready, run inference requests, and then shut it down.  They require a real
model to be available (downloaded or cached) and Apple Silicon hardware.

Mark: ``@pytest.mark.self_hosted`` -- skipped unless run on a self-hosted
Apple Silicon runner or explicitly selected with ``-m self_hosted``.

Environment variables:
    MLXZ_E2E_MODEL  -- model repo ID (default: mlx-community/Llama-3.1-8B-Instruct-4bit)
    MLXZ_E2E_PORT   -- server port (default: 8399)

Usage:
    pytest tests/integration/test_e2e_inference.py -m self_hosted --timeout=120
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytestmark = [
    pytest.mark.self_hosted,
    pytest.mark.timeout(120),
    pytest.mark.skipif(
        os.environ.get("MLXZ_E2E_ENABLED", "").lower() not in ("1", "true", "yes"),
        reason="Set MLXZ_E2E_ENABLED=1 to run end-to-end inference tests",
    ),
]

MODEL = os.environ.get(
    "MLXZ_E2E_MODEL", "mlx-community/Llama-3.1-8B-Instruct-4bit"
)
PORT = int(os.environ.get("MLXZ_E2E_PORT", "8399"))
BASE_URL = f"http://127.0.0.1:{PORT}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def mlxz_server():
    """Start a real mlxz server for the duration of this test module.

    Yields the base URL once the server is ready.  Kills the process on
    teardown.
    """
    if _port_is_open(PORT):
        # Server already running (e.g. user started it manually)
        yield BASE_URL
        return

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlxz.cli.main",
            "serve",
            MODEL,
            "--port",
            str(PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the server to become ready (poll /health)
    deadline = time.monotonic() + 90  # 90s for model loading
    ready = False
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ready" or data.get("phase") == "ready":
                    ready = True
                    break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1)

    if not ready:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"mlxz server did not become ready within 90s.\n"
            f"stdout: {stdout.decode()[-500:]}\n"
            f"stderr: {stderr.decode()[-500:]}"
        )

    yield BASE_URL

    # Teardown: graceful shutdown then force kill
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Non-streaming chat completion
# ---------------------------------------------------------------------------


class TestChatCompletionNonStreaming:
    def test_returns_valid_response(self, mlxz_server: str) -> None:
        """Non-streaming chat completion returns a well-formed response."""
        resp = httpx.post(
            f"{mlxz_server}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 16,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["choices"][0]["finish_reason"] in ("stop", "length")

    def test_usage_is_populated(self, mlxz_server: str) -> None:
        """Usage block contains non-zero prompt and completion token counts."""
        resp = httpx.post(
            f"{mlxz_server}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count to three."}],
                "max_tokens": 32,
            },
            timeout=30,
        )
        data = resp.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )


# ---------------------------------------------------------------------------
# Streaming chat completion
# ---------------------------------------------------------------------------


class TestChatCompletionStreaming:
    def test_returns_proper_sse_format(self, mlxz_server: str) -> None:
        """Streaming response uses proper SSE format ending with [DONE]."""
        with httpx.Client(timeout=30) as client:
            with client.stream(
                "POST",
                f"{mlxz_server}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "max_tokens": 8,
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                lines = list(resp.iter_lines())

        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 2  # at least role + content + done
        assert data_lines[-1] == "data: [DONE]"

        # First data chunk should have the role
        first = json.loads(data_lines[0][6:])
        assert first["object"] == "chat.completion.chunk"

    def test_streaming_content_is_nonempty(self, mlxz_server: str) -> None:
        """Concatenated streaming content produces non-empty text."""
        content_parts: list[str] = []
        with httpx.Client(timeout=30) as client:
            with client.stream(
                "POST",
                f"{mlxz_server}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 16,
                    "stream": True,
                },
            ) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    text = chunk["choices"][0].get("delta", {}).get("content")
                    if text:
                        content_parts.append(text)

        assert len("".join(content_parts)) > 0


# ---------------------------------------------------------------------------
# Text completions endpoint
# ---------------------------------------------------------------------------


class TestCompletionsEndpoint:
    def test_completions_returns_text(self, mlxz_server: str) -> None:
        """Legacy /v1/completions returns generated text."""
        resp = httpx.post(
            f"{mlxz_server}/v1/completions",
            json={
                "model": MODEL,
                "prompt": "The capital of France is",
                "max_tokens": 16,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"][0]["text"]) > 0


# ---------------------------------------------------------------------------
# Stop sequences
# ---------------------------------------------------------------------------


class TestStopSequences:
    def test_stop_sequence_truncates_output(self, mlxz_server: str) -> None:
        """Stop sequence causes generation to end before max_tokens."""
        resp = httpx.post(
            f"{mlxz_server}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "List the numbers 1 through 20, one per line.",
                    }
                ],
                "max_tokens": 256,
                "stop": ["10"],
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # The output should have stopped before reaching "10"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "15" not in content


# ---------------------------------------------------------------------------
# max_tokens enforcement
# ---------------------------------------------------------------------------


class TestMaxTokens:
    def test_max_tokens_is_respected(self, mlxz_server: str) -> None:
        """Completion should not exceed max_tokens."""
        max_tok = 8
        resp = httpx.post(
            f"{mlxz_server}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a very long essay about history.",
                    }
                ],
                "max_tokens": max_tok,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["completion_tokens"] <= max_tok


# ---------------------------------------------------------------------------
# Deterministic output (temperature=0)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_temperature_zero_is_deterministic(self, mlxz_server: str) -> None:
        """temperature=0 should produce identical output across two calls."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 16,
            "temperature": 0,
            "seed": 42,
        }

        resp1 = httpx.post(
            f"{mlxz_server}/v1/chat/completions", json=payload, timeout=30
        )
        resp2 = httpx.post(
            f"{mlxz_server}/v1/chat/completions", json=payload, timeout=30
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200

        content1 = resp1.json()["choices"][0]["message"]["content"]
        content2 = resp2.json()["choices"][0]["message"]["content"]
        assert content1 == content2, (
            f"Determinism violated: {content1!r} != {content2!r}"
        )


# ---------------------------------------------------------------------------
# OpenAI Python SDK compatibility
# ---------------------------------------------------------------------------


class TestOpenAISDK:
    def test_openai_sdk_chat_completion(self, mlxz_server: str) -> None:
        """The official openai-python SDK can talk to mlxz."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai package not installed")

        client = OpenAI(base_url=f"{mlxz_server}/v1", api_key="not-needed")
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=8,
        )

        assert completion.id.startswith("chatcmpl-")
        assert len(completion.choices) == 1
        assert completion.choices[0].message.role == "assistant"
        assert len(completion.choices[0].message.content) > 0

    def test_openai_sdk_streaming(self, mlxz_server: str) -> None:
        """The openai-python SDK streaming interface works with mlxz."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            pytest.skip("openai package not installed")

        client = OpenAI(base_url=f"{mlxz_server}/v1", api_key="not-needed")
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=8,
            stream=True,
        )

        chunks: list[str] = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        assert len("".join(chunks)) > 0


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_list_models_returns_served_model(self, mlxz_server: str) -> None:
        """GET /v1/models lists the currently served model."""
        resp = httpx.get(f"{mlxz_server}/v1/models", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model_ids = [m["id"] for m in data["data"]]
        assert any(MODEL in mid for mid in model_ids)
