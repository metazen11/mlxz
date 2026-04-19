"""OpenAI-compatible API router for mlxz.

Provides ``/v1/chat/completions``, ``/v1/completions``, and ``/v1/models``
endpoints that are wire-compatible with the ``openai-python`` SDK (>= 1.0).
"""

from __future__ import annotations

import asyncio
import time
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
from mlxz.engine.thread_boundary import CancellationRegistry
from mlxz.scheduler.admission import AdmissionController
from mlxz.types import AdmissionDecision, EngineProtocol, RequestState, SamplingParams

logger = structlog.get_logger()

router = APIRouter(tags=["openai"])

# Alias used by app.py
openai_router = router

_DEFAULT_TIMEOUT = 300.0  # 5 minutes
_TOKEN_CHANNEL_DEPTH = 512
_SSE_TOKEN_BATCH = 16


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_engine(request: StarletteRequest) -> EngineProtocol:
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


def get_telemetry(request: StarletteRequest) -> tuple[Any | None, int | None]:
    """Retrieve optional telemetry recorder state."""
    return (
        getattr(request.app.state, "telemetry", None),
        getattr(request.app.state, "telemetry_run_id", None),
    )


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
    engine: EngineProtocol,
    prompt_token_count: int,
    max_tokens: int,
) -> None:
    """Run admission check; raise HTTP 429 on rejection."""
    snap = engine.snapshot()
    decision, reason = admission.decide(prompt_token_count, max_tokens, snap)
    if decision != AdmissionDecision.ACCEPT:
        try:
            from mlxz.api.metrics import admission_rejections_total

            admission_rejections_total.labels(reason=decision.name.lower()).inc()
        except Exception:
            pass
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


def _record_request_telemetry(
    telemetry: Any | None,
    run_id: int | None,
    eng_request: EngineRequest,
) -> None:
    """Best-effort telemetry capture for completed/cancelled requests."""
    if telemetry is None or run_id is None:
        return
    try:
        telemetry.record_request(
            run_id,
            request_id=eng_request.id,
            prompt_tokens=eng_request.prompt_token_count,
            completion_tokens=eng_request.completion_token_count,
            prefix_cache_hit_tokens=eng_request.prefix_cache_hit_tokens,
            ttft_ms=eng_request.ttft_ms,
            decode_tps=eng_request.decode_tps,
        )
    except Exception:
        logger.warning("telemetry_record_request_failed", request_id=eng_request.id, exc_info=True)


def _observe_http_metrics(endpoint: str, status_code: int, duration_seconds: float) -> None:
    """Best-effort Prometheus request metrics."""
    try:
        from mlxz.api.metrics import request_duration_seconds, requests_total

        requests_total.labels(endpoint=endpoint, status=str(status_code)).inc()
        request_duration_seconds.labels(endpoint=endpoint).observe(duration_seconds)
    except Exception:
        pass


def _inc_active_requests() -> None:
    try:
        from mlxz.api.metrics import active_requests

        active_requests.inc()
    except Exception:
        pass


def _dec_active_requests() -> None:
    try:
        from mlxz.api.metrics import active_requests

        active_requests.dec()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------


async def _sse_stream(
    request: EngineRequest,
    model_name: str,
    cancellations: CancellationRegistry,
    started_at: float,
    telemetry: Any | None = None,
    telemetry_run_id: int | None = None,
):
    """Async generator yielding SSE-formatted chat completion chunks."""
    request_id = f"chatcmpl-{request.id}"
    status_code = 200
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

            parts = [token.text]
            saw_eos = False
            for _ in range(_SSE_TOKEN_BATCH - 1):
                try:
                    maybe_token = request.output_channel.async_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if maybe_token is None:
                    saw_eos = True
                    break
                parts.append(maybe_token.text)

            content_chunk = ChatCompletionChunk(
                id=request_id,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(content="".join(parts))
                    )
                ],
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"
            if saw_eos:
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
    finally:
        _observe_http_metrics(
            endpoint="/v1/chat/completions",
            status_code=status_code,
            duration_seconds=time.perf_counter() - started_at,
        )
        _record_request_telemetry(telemetry, telemetry_run_id, request)
        _dec_active_requests()
        cancellations.cancel(request.id)
        cancellations.unregister(request.id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    engine: EngineProtocol = Depends(get_engine),
    admission: AdmissionController = Depends(get_admission),
    tokenizer: Any = Depends(get_tokenizer),
    cancellations: CancellationRegistry = Depends(get_cancellations),
    telemetry_state: tuple[Any | None, int | None] = Depends(get_telemetry),
):
    """OpenAI-compatible chat completion endpoint."""
    t0 = time.perf_counter()
    status_code = 200
    prompt_tokens = _tokenize_chat(tokenizer, body.messages)
    max_tokens = body.max_tokens or 512
    sampling = _build_sampling(body.temperature, body.top_p, body.seed)
    stop_sequences = _normalize_stop(body.stop)

    _check_admission(admission, engine, len(prompt_tokens), max_tokens)

    eng_request = EngineRequest.create(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        sampling=sampling,
        return_logprob=bool(body.logprobs),
        stop_sequences=stop_sequences,
        channel_depth=_TOKEN_CHANNEL_DEPTH,
    )
    eng_request.transition(RequestState.ADMITTED)
    cancellations.register(eng_request.id)
    _inc_active_requests()
    await engine.submit(eng_request)

    model_name = body.model
    telemetry, telemetry_run_id = telemetry_state

    try:
        if body.stream:
            return StreamingResponse(
                _sse_stream(
                    eng_request,
                    model_name,
                    cancellations,
                    t0,
                    telemetry,
                    telemetry_run_id,
                ),
                media_type="text/event-stream",
            )

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
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    except Exception:
        status_code = 500
        raise
    finally:
        if not body.stream:
            _observe_http_metrics(
                endpoint="/v1/chat/completions",
                status_code=status_code,
                duration_seconds=time.perf_counter() - t0,
            )
            _record_request_telemetry(telemetry, telemetry_run_id, eng_request)
            _dec_active_requests()
            cancellations.cancel(eng_request.id)
            cancellations.unregister(eng_request.id)


@router.post("/v1/completions")
async def completions(
    body: CompletionRequest,
    engine: EngineProtocol = Depends(get_engine),
    admission: AdmissionController = Depends(get_admission),
    tokenizer: Any = Depends(get_tokenizer),
    cancellations: CancellationRegistry = Depends(get_cancellations),
    telemetry_state: tuple[Any | None, int | None] = Depends(get_telemetry),
):
    """OpenAI-compatible text completion endpoint."""
    t0 = time.perf_counter()
    status_code = 200
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
        return_logprob=False,
        stop_sequences=stop_sequences,
        channel_depth=_TOKEN_CHANNEL_DEPTH,
    )
    eng_request.transition(RequestState.ADMITTED)
    cancellations.register(eng_request.id)
    _inc_active_requests()
    await engine.submit(eng_request)
    telemetry, telemetry_run_id = telemetry_state

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
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    except Exception:
        status_code = 500
        raise
    finally:
        _observe_http_metrics(
            endpoint="/v1/completions",
            status_code=status_code,
            duration_seconds=time.perf_counter() - t0,
        )
        _record_request_telemetry(telemetry, telemetry_run_id, eng_request)
        _dec_active_requests()
        cancellations.cancel(eng_request.id)
        cancellations.unregister(eng_request.id)


@router.get("/v1/models")
async def list_models(
    engine: EngineProtocol = Depends(get_engine),
) -> ModelListResponse:
    """List available models."""
    model_name = getattr(engine, "model_name", "unknown")
    return ModelListResponse(
        data=[
            ModelInfo(id=model_name),
        ]
    )
