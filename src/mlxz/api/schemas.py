"""OpenAI-compatible Pydantic request/response models for mlxz.

These schemas mirror the JSON shapes produced and consumed by the
``openai-python`` SDK (>= 1.0) so that the SDK is a drop-in test fixture.
Extension fields are namespaced under the ``x_mlxz`` key to avoid
collisions with the upstream specification.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    model_config = ConfigDict(populate_by_name=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class UsageInfo(BaseModel):
    """Token usage statistics for a completion request."""

    model_config = ConfigDict(populate_by_name=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    x_mlxz: dict[str, Any] | None = Field(
        default=None,
        description="Extension fields namespaced under x_mlxz.",
    )


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` request body."""

    model_config = ConfigDict(populate_by_name=True)

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    model_config = ConfigDict(populate_by_name=True)

    index: int
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    """``POST /v1/chat/completions`` response body."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Text completions (legacy)
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """``POST /v1/completions`` request body."""

    model_config = ConfigDict(populate_by_name=True)

    model: str
    prompt: str | list[str]
    max_tokens: int | None = Field(default=16)
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None


class CompletionChoice(BaseModel):
    """A single choice in a text completion response."""

    model_config = ConfigDict(populate_by_name=True)

    index: int
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    """``POST /v1/completions`` response body."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """A single model descriptor returned by ``GET /v1/models``."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlxz"


class ModelListResponse(BaseModel):
    """``GET /v1/models`` response body."""

    model_config = ConfigDict(populate_by_name=True)

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope matching the OpenAI error format."""

    model_config = ConfigDict(populate_by_name=True)

    error: dict[str, Any]
    """Must contain ``message``, ``type``, and ``code`` keys."""


# ---------------------------------------------------------------------------
# Streaming chunks (SSE)
# ---------------------------------------------------------------------------


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """SSE streaming chunk for chat completions."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: UsageInfo | None = None
