"""OpenAI-compatible API router for mlxz.

Provides ``/v1/chat/completions``, ``/v1/completions``, and ``/v1/models``
endpoints that are wire-compatible with the ``openai-python`` SDK (>= 1.0).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import Request as StarletteRequest

from mlxz.api.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)
from mlxz.engine.request import Request as EngineRequest
from mlxz.engine.single_stream import SingleStreamEngine
from mlxz.engine.thread_boundary import CancellationRegistry
from mlxz.scheduler.admission import AdmissionController
from mlxz.types import AdmissionDecision, RequestState, SamplingParams

logger = structlog.get_logger()

router = APIRouter(tags=["openai"])

# Alias used by app.py
openai_router = router

_DEFAULT_TIMEOUT = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_engine(request: StarletteRequest) -> SingleStreamEngine:
    """Retrieve the engine from application state."""
    return request.app.state.engine


def get_admission(request: StarletteRequest) -> AdmissionController:
    """Retrieve the admission controller from application state."""
    return request.app.state.admission


def get_tokenizer(request: StarletteRequest) -> Any:
    """Retrieve the tokenizer from application state."""
    return request.app.state.tokenizer


def get_cancellations(request: StarletteRequest) -> CancellationRegistry:
    """Retrieve the cancellation registry from application state."""
    return request.app.state.cancellations


# ---------------------------------------------------------------------------
# Shared helpers (DRY)
# ---------------------------------------------------------------------------


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    """Normalize stop sequences to a list."""
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _check_admission(
    admission: AdmissionController,
    engine: SingleStreamEngine,
    prompt_token_count: int,
    max_tokens: int,
) -> None:
    """Run admission check; raise HTTP 429 on rejection."""
    snap = engine.snapshot()
    decision, reason = admission.decide(prompt_token_count, max_tokens, snap)
    if decision != AdmissionDecision.ACCEPT:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": reason,
                    "type": "rate_limit_error",
                    "code": decision.name.lower(),
                }
            },
        )


def _tokenize_chat(tokenizer: Any, messages: list[ChatMessage]) -> list[int]:
    """Tokenize chat messages via apply_chat_template."""
    messages_as_dicts = [{"role": m.role, "content": m.content} for m in messages]
    result = tokenizer.apply_chat_template(
        messages_as_dicts, tokenize=True, add_generation_prompt=True
    )
    # apply_chat_template may return a BatchEncoding dict or a plain list
    if isinstance(result, dict):
        return result["input_ids"]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], int):
        return result
    if hasattr(result, "input_ids"):
        return result.input_ids
    return list(result)


def _build_sampling(
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
) -> SamplingParams:
    """Build SamplingParams with defaults."""
    return SamplingParams(
        temperature=temperature if temperature is not None else 1.0,
        top_p=top_p if top_p is not None else 1.0,
        seed=seed,
    )


async def _collect_tokens(
    eng_request: EngineRequest,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[str]:
    """Collect all tokens from the engine request with timeout."""
    parts: list[str] = []
    while True:
        try:
            token = await asyncio.wait_for(
                eng_request.output_channel.async_q.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, detail={"error": {"message": "Request timed out"}})
        if token is None:
            break
        parts.append(token.text)
    return parts


def _make_usage(eng_request: EngineRequest) -> UsageInfo:
    """Build UsageInfo from a completed engine request."""
    return UsageInfo(
        prompt_tokens=eng_request.prompt_token_count,
        completion_tokens=eng_request.completion_token_count,
        total_tokens=eng_request.prompt_token_count + eng_request.completion_token_count,
    )


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------


async def _sse_stream(
    request: EngineRequest,
    model_name: str,
    cancellations: CancellationRegistry,
):
    """Async generator yielding SSE-formatted chat completion chunks."""
    request_id = f"chatcmpl-{request.id}"
    try:
        # First chunk: role
        role_chunk = ChatCompletionChunk(
            id=request_id,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(role="assistant")
                )
            ],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        while True:
            token = await request.output_channel.async_q.get()
            if token is None:
                # Final chunk with finish_reason and usage
                final_chunk = ChatCompletionChunk(
                    id=request_id,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(),
                            finish_reason=request.finish_reason,
                        )
                    ],
                    usage=_make_usage(request),
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            content_chunk = ChatCompletionChunk(
                id=request_id,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(content=token.text)
                    )
                ],
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"
    finally:
        cancellations.cancel(request.id)
        cancellations.unregister(request.id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    engine: SingleStreamEngine = Depends(get_engine),
    admission: AdmissionController = Depends(get_admission),
    tokenizer: Any = Depends(get_tokenizer),
    cancellations: CancellationRegistry = Depends(get_cancellations),
):
    """OpenAI-compatible chat completion endpoint."""
    prompt_tokens = _tokenize_chat(tokenizer, body.messages)
    max_tokens = body.max_tokens or 512
    sampling = _build_sampling(body.temperature, body.top_p, body.seed)
    stop_sequences = _normalize_stop(body.stop)

    _check_admission(admission, engine, len(prompt_tokens), max_tokens)

    eng_request = EngineRequest.create(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        sampling=sampling,
        stop_sequences=stop_sequences,
        channel_depth=64,
    )
    eng_request.transition(RequestState.ADMITTED)
    cancellations.register(eng_request.id)
    await engine.submit(eng_request)

    model_name = body.model

    if body.stream:
        return StreamingResponse(
            _sse_stream(eng_request, model_name, cancellations),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all tokens with timeout
    try:
        generated_text_parts = await _collect_tokens(eng_request)
        return ChatCompletionResponse(
            id=f"chatcmpl-{eng_request.id}",
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="".join(generated_text_parts),
                    ),
                    finish_reason=eng_request.finish_reason,
                )
            ],
            usage=_make_usage(eng_request),
        )
    finally:
        cancellations.cancel(eng_request.id)
        cancellations.unregister(eng_request.id)


@router.post("/v1/completions")
async def completions(
    body: CompletionRequest,
    engine: SingleStreamEngine = Depends(get_engine),
    admission: AdmissionController = Depends(get_admission),
    tokenizer: Any = Depends(get_tokenizer),
    cancellations: CancellationRegistry = Depends(get_cancellations),
):
    """OpenAI-compatible text completion endpoint."""
    prompt_text = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
    prompt_tokens: list[int] = tokenizer.encode(prompt_text)
    max_tokens = body.max_tokens or 16
    sampling = _build_sampling(body.temperature, body.top_p, body.seed)
    stop_sequences = _normalize_stop(body.stop)

    _check_admission(admission, engine, len(prompt_tokens), max_tokens)

    eng_request = EngineRequest.create(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        sampling=sampling,
        stop_sequences=stop_sequences,
        channel_depth=64,
    )
    eng_request.transition(RequestState.ADMITTED)
    cancellations.register(eng_request.id)
    await engine.submit(eng_request)

    try:
        generated_text_parts = await _collect_tokens(eng_request)
        return CompletionResponse(
            id=f"cmpl-{eng_request.id}",
            model=body.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text="".join(generated_text_parts),
                    finish_reason=eng_request.finish_reason,
                )
            ],
            usage=_make_usage(eng_request),
        )
    finally:
        cancellations.cancel(eng_request.id)
        cancellations.unregister(eng_request.id)


@router.get("/v1/models")
async def list_models(
    engine: SingleStreamEngine = Depends(get_engine),
) -> ModelListResponse:
    """List available models."""
    return ModelListResponse(
        data=[
            ModelInfo(id=engine.model_name),
        ]
    )
