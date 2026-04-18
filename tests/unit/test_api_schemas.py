"""Unit tests for OpenAI-compatible Pydantic schemas."""
from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from mlxz.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)


# ---------------------------------------------------------------------------
# ChatCompletionRequest
# ---------------------------------------------------------------------------


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest validation and defaults."""

    def test_minimal_fields(self) -> None:
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="hello")],
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].content == "hello"
        # Defaults
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.stream is False
        assert req.max_tokens is None
        assert req.stop is None
        assert req.seed is None
        assert req.logprobs is None
        assert req.top_logprobs is None
        assert req.tools is None
        assert req.tool_choice is None

    def test_all_optional_fields(self) -> None:
        tool_def = {"type": "function", "function": {"name": "get_weather"}}
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hi"),
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            stop=["\n", "###"],
            seed=42,
            logprobs=True,
            top_logprobs=5,
            tools=[tool_def],
            tool_choice="auto",
        )
        assert req.max_tokens == 512
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stream is True
        assert req.stop == ["\n", "###"]
        assert req.seed == 42
        assert req.logprobs is True
        assert req.top_logprobs == 5
        assert req.tools == [tool_def]
        assert req.tool_choice == "auto"

    def test_temperature_bounds(self) -> None:
        # temperature must be 0..2
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                temperature=-0.1,
            )
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                temperature=2.1,
            )
        # Boundary values should be accepted
        req_zero = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=0.0,
        )
        assert req_zero.temperature == 0.0
        req_max = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=2.0,
        )
        assert req_max.temperature == 2.0

    def test_top_p_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                top_p=-0.01,
            )
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                top_p=1.01,
            )

    def test_top_logprobs_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                top_logprobs=-1,
            )
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                top_logprobs=21,
            )

    def test_stop_as_string(self) -> None:
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            stop="<|endoftext|>",
        )
        assert req.stop == "<|endoftext|>"

    def test_tool_choice_as_dict(self) -> None:
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
        assert isinstance(req.tool_choice, dict)


# ---------------------------------------------------------------------------
# ChatCompletionResponse
# ---------------------------------------------------------------------------


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse serialization."""

    def test_serialization(self) -> None:
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hi there!"),
            finish_reason="stop",
        )
        resp = ChatCompletionResponse(
            id="chatcmpl-abc123",
            model="test-model",
            choices=[choice],
            usage=usage,
        )
        data = resp.model_dump()
        assert data["id"] == "chatcmpl-abc123"
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hi there!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["total_tokens"] == 15

    def test_created_default_is_recent(self) -> None:
        usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=""),
            finish_reason="stop",
        )
        before = int(time.time())
        resp = ChatCompletionResponse(
            id="test",
            model="m",
            choices=[choice],
            usage=usage,
        )
        after = int(time.time())
        assert before <= resp.created <= after

    def test_json_round_trip(self) -> None:
        usage = UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="ok"),
            finish_reason="length",
        )
        resp = ChatCompletionResponse(
            id="id-1",
            model="m",
            choices=[choice],
            usage=usage,
            created=1700000000,
        )
        json_str = resp.model_dump_json()
        restored = ChatCompletionResponse.model_validate_json(json_str)
        assert restored.id == resp.id
        assert restored.created == 1700000000
        assert restored.choices[0].finish_reason == "length"


# ---------------------------------------------------------------------------
# CompletionRequest / CompletionResponse
# ---------------------------------------------------------------------------


class TestCompletionRequest:
    """Tests for legacy text completion request."""

    def test_minimal(self) -> None:
        req = CompletionRequest(model="m", prompt="Once upon a time")
        assert req.model == "m"
        assert req.prompt == "Once upon a time"
        assert req.max_tokens == 16  # default
        assert req.stream is False

    def test_prompt_as_list(self) -> None:
        req = CompletionRequest(model="m", prompt=["a", "b"])
        assert req.prompt == ["a", "b"]

    def test_temperature_validation(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="x", temperature=3.0)


class TestCompletionResponse:
    """Tests for legacy text completion response."""

    def test_serialization(self) -> None:
        usage = UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        choice = CompletionChoice(index=0, text="the end", finish_reason="stop")
        resp = CompletionResponse(
            id="cmpl-123",
            model="m",
            choices=[choice],
            usage=usage,
        )
        data = resp.model_dump()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "the end"
        assert data["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# ModelListResponse
# ---------------------------------------------------------------------------


class TestModelListResponse:
    """Tests for GET /v1/models response."""

    def test_structure(self) -> None:
        info = ModelInfo(id="my-model")
        resp = ModelListResponse(data=[info])
        data = resp.model_dump()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "my-model"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "mlxz"

    def test_empty_list(self) -> None:
        resp = ModelListResponse(data=[])
        assert resp.model_dump()["data"] == []

    def test_multiple_models(self) -> None:
        models = [ModelInfo(id=f"model-{i}") for i in range(3)]
        resp = ModelListResponse(data=models)
        assert len(resp.data) == 3


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


class TestErrorResponse:
    """Tests for the error envelope."""

    def test_standard_error(self) -> None:
        err = ErrorResponse(
            error={
                "message": "Model not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        )
        data = err.model_dump()
        assert data["error"]["message"] == "Model not found"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "model_not_found"

    def test_error_with_extra_keys(self) -> None:
        err = ErrorResponse(
            error={
                "message": "Bad request",
                "type": "error",
                "code": "bad_request",
                "param": "temperature",
            }
        )
        assert err.error["param"] == "temperature"


# ---------------------------------------------------------------------------
# UsageInfo with x_mlxz extension
# ---------------------------------------------------------------------------


class TestUsageInfo:
    """Tests for token usage with mlxz extensions."""

    def test_without_extension(self) -> None:
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.x_mlxz is None
        data = usage.model_dump()
        assert data["x_mlxz"] is None

    def test_with_extension(self) -> None:
        ext = {
            "decode_tokens_per_second": 42.5,
            "ttft_seconds": 0.12,
            "prefix_cache_hit": True,
        }
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            x_mlxz=ext,
        )
        assert usage.x_mlxz is not None
        assert usage.x_mlxz["decode_tokens_per_second"] == 42.5
        assert usage.x_mlxz["prefix_cache_hit"] is True

    def test_total_tokens_independent(self) -> None:
        # total_tokens is not auto-computed; it's user-provided
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=99)
        assert usage.total_tokens == 99


# ---------------------------------------------------------------------------
# ChatMessage and field aliases
# ---------------------------------------------------------------------------


class TestChatMessage:
    """Tests for ChatMessage and populate_by_name config."""

    def test_all_roles(self) -> None:
        for role in ("system", "user", "assistant", "tool"):
            msg = ChatMessage(role=role, content="test")  # type: ignore[arg-type]
            assert msg.role == role

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role="admin", content="test")  # type: ignore[arg-type]

    def test_tool_calls_field(self) -> None:
        msg = ChatMessage(
            role="assistant",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_populate_by_name(self) -> None:
        """Verify that populate_by_name=True allows field name access."""
        data = {"role": "user", "content": "hello", "name": "alice"}
        msg = ChatMessage.model_validate(data)
        assert msg.name == "alice"
        assert msg.content == "hello"
