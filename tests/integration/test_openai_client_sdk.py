"""End-to-end tests using the openai-python SDK against a live server.

These tests require a running mlxz server with a real model loaded,
so they are gated behind the ``self_hosted`` marker and skipped in
standard CI runs.

To run locally::

    pytest -m self_hosted tests/integration/test_openai_client_sdk.py

Prerequisites:
    - A model downloaded and accessible (e.g. ``mlx-community/Llama-3.2-1B-Instruct-4bit``)
    - ``pip install openai`` in the test environment
    - A running ``mlxz`` server (``mlxz serve --model <model>``)
"""
from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.self_hosted,
    pytest.mark.skip(reason="Requires a running mlxz server with a real model"),
]


class TestChatCompletionSDK:
    """Verify openai-python SDK works against the mlxz chat completions endpoint."""

    def test_non_streaming(self) -> None:
        """Non-streaming chat completion via the SDK."""
        # TODO: Implement when server_process fixture is available.
        # from openai import OpenAI
        # client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
        # resp = client.chat.completions.create(
        #     model="test-model",
        #     messages=[{"role": "user", "content": "Say hello"}],
        #     max_tokens=16,
        # )
        # assert resp.choices[0].message.content
        # assert resp.usage.completion_tokens > 0

    def test_streaming(self) -> None:
        """Streaming chat completion via the SDK."""
        # TODO: Implement when server_process fixture is available.
        # from openai import OpenAI
        # client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
        # stream = client.chat.completions.create(
        #     model="test-model",
        #     messages=[{"role": "user", "content": "Say hello"}],
        #     max_tokens=16,
        #     stream=True,
        # )
        # chunks = list(stream)
        # assert len(chunks) > 0


class TestCompletionSDK:
    """Verify openai-python SDK works against the mlxz completions endpoint."""

    def test_non_streaming(self) -> None:
        """Non-streaming text completion via the SDK."""
        # TODO: Implement when server_process fixture is available.


class TestModelsSDK:
    """Verify openai-python SDK can list models."""

    def test_list_models(self) -> None:
        """List models via the SDK."""
        # TODO: Implement when server_process fixture is available.
