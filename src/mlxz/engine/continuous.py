"""Continuous batching engine — iteration-level batching for concurrent requests."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import structlog

from mlxz.config import RuntimeConfig
from mlxz.engine.decode_compiler import (
    build_compiled_greedy_chunk,
    build_compiled_greedy_step,
)
from mlxz.engine.cache_quant import maybe_quantize_kv_cache
from mlxz.engine.request import Request, Token
from mlxz.engine.sampling import sample
from mlxz.engine.thread_boundary import CancellationRegistry, MxEvalGuard, RequestBridge
from mlxz.types import (
    AdmissionSnapshot,
    DrainResult,
    MemoryPressure,
    RequestState,
    ResidencyBudget,
    ThermalState,
)
from mlx.utils import tree_flatten, tree_map, tree_unflatten

logger = structlog.get_logger()

_SINGLE_REQUEST_GREEDY_CHUNK_SIZE = 16


class ContinuousBatchingEngine:
    """Continuous batching engine for concurrent request processing.

    Implements EngineProtocol. Processes multiple requests per iteration:
    - Prefill requests get priority (one prefill per iteration)
    - Decode requests are batched (up to max_concurrent_requests)
    - Tokens are delivered to per-request janus queues

    Thread safety: single compute thread, same as SingleStreamEngine.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        bridge: RequestBridge,
        cancellations: CancellationRegistry,
    ) -> None:
        self._config = config
        self._bridge = bridge
        self._cancellations = cancellations
        self._model: nn.Module | None = None
        self._tokenizer: Any = None
        self._budget: ResidencyBudget | None = None
        self._model_name: str = config.model
        self._shutdown_requested = threading.Event()
        self._guard: MxEvalGuard | None = None

        # Active request tracking
        self._running: dict[str, _ActiveRequest] = {}  # request_id -> state
        self._kv_used_bytes: int = 0

        # Model arch (set after load)
        self._n_layers: int = 0
        self._n_heads: int = 0
        self._head_dim: int = 0

        # Prefix cache (optional)
        self._prefix_cache_memory: PrefixCacheMemory | None = None
        self._prefix_cache_disk: PrefixCacheDisk | None = None
        self._prefix_hasher: RollingPrefixHasher | None = None

    # -- Properties ---------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_arch(self) -> tuple[int, int, int]:
        """Return (n_layers, n_heads, head_dim) for admission controller."""
        return self._n_layers, self._n_heads, self._head_dim

    @property
    def running_count(self) -> int:
        return len(self._running)

    # -- Setup (same pattern as SingleStreamEngine) -------------------------

    def set_model(self, model: nn.Module, tokenizer: Any) -> None:
        """Set the loaded model and tokenizer. Called from main thread before run()."""
        self._model = model
        self._tokenizer = tokenizer
        if hasattr(model, "args"):
            args = model.args
            self._n_layers = getattr(args, "num_hidden_layers", None) or 32
            self._n_heads = (
                getattr(args, "num_key_value_heads", None)
                or getattr(args, "num_attention_heads", None)
                or 32
            )
            head_dim = getattr(args, "head_dim", None)
            if head_dim is None:
                hidden = getattr(args, "hidden_size", None) or 4096
                n_attn = getattr(args, "num_attention_heads", None) or 32
                head_dim = hidden // n_attn
            self._head_dim = head_dim
        else:
            self._n_layers, self._n_heads, self._head_dim = 32, 32, 128

    def set_budget(self, budget: ResidencyBudget) -> None:
        """Store the computed residency budget."""
        self._budget = budget

    def set_prefix_cache(
        self,
        memory_cache: PrefixCacheMemory | None = None,
        disk_cache: PrefixCacheDisk | None = None,
        block_size: int = 8,
    ) -> None:
        """Configure prefix cache tiers. Called from lifespan after model load."""
        self._prefix_cache_memory = memory_cache
        self._prefix_cache_disk = disk_cache
        if memory_cache is not None or disk_cache is not None:
            self._prefix_hasher = RollingPrefixHasher(block_size=block_size)

    # -- Engine loop --------------------------------------------------------

    def run(self) -> None:
        """Main engine loop. Each iteration:
        1. Drain new requests from the bridge
        2. Process one prefilling request (if any)
        3. Batch decode all decoding requests
        4. Retire completed/cancelled requests
        """
        self._guard = MxEvalGuard()
        logger.info(
            "continuous_engine_started",
            model=self._model_name,
            max_batch=self._config.scheduler.max_concurrent_requests,
        )

        while not self._shutdown_requested.is_set():
            # 1. Admit new requests
            self._admit_requests()

            if not self._running:
                self._shutdown_requested.wait(timeout=0.005)
                continue

            # 2. Process prefilling requests (one at a time)
            self._process_prefills()

            # 3. Batch decode all decoding requests
            self._process_decodes()

            # 4. Retire completed/cancelled
            self._retire_requests()

        # Cleanup on shutdown
        self._cancel_all()
        logger.info("continuous_engine_stopped")

    def _kv_bytes_per_token(self) -> float:
        """Estimate KV-cache bytes consumed per token."""
        return (
            2  # keys + values
            * self._n_layers
            * self._n_heads
            * self._head_dim
            * (self._config.kv.bits / 8)
        )

    def _admit_requests(self) -> None:
        """Drain new requests from the bridge into the running set."""
        max_new = self._config.scheduler.max_concurrent_requests - len(self._running)
        for _ in range(max(max_new, 0)):
            request = self._bridge.get_next_sync()
            if request is None:
                break

            # Create KV cache for this request
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(self._model)

            active = _ActiveRequest(
                request=request,
                cache=cache,
                prefill_done=False,
                rng_key=(
                    mx.random.key(request.sampling.seed)
                    if request.sampling.seed is not None
                    and request.sampling.temperature != 0.0
                    else None
                ),
            )
            self._running[request.id] = active
            request.transition(RequestState.PREFILLING)
            logger.debug(
                "request_admitted",
                request_id=request.id,
                batch_size=len(self._running),
            )

    def _process_prefills(self) -> None:
        """Process one prefilling request per iteration."""
        for req_id, active in self._running.items():
            if active.prefill_done:
                continue
            if self._cancellations.is_cancelled(req_id):
                continue

            request = active.request
            log = logger.bind(request_id=req_id)
            # Prefix cache lookup
            n_prefix_tokens = 0
            token_hashes: tuple[bytes, ...] = ()
            cache_tier = ""

            if self._prefix_hasher is not None:
                token_hashes = self._prefix_hasher.hash_chunks(request.prompt_tokens)

                # Try memory cache first.
                if self._prefix_cache_memory is not None:
                    n_matched, cached_kv = self._prefix_cache_memory.lookup_sync(
                        token_hashes
                    )
                    if cached_kv is not None and n_matched > 0:
                        n_prefix_tokens = n_matched
                        cache_tier = "memory"
                        for layer_cache, cached_state in zip(active.cache, cached_kv):
                            layer_cache.state = cached_state

                # Fall back to disk cache.
                if n_prefix_tokens == 0 and self._prefix_cache_disk is not None:
                    n_matched, cached_kv = self._prefix_cache_disk.lookup_sync(
                        token_hashes
                    )
                    if cached_kv is not None and n_matched > 0:
                        n_prefix_tokens = n_matched
                        cache_tier = "disk"
                        for layer_cache, cached_state in zip(active.cache, cached_kv):
                            layer_cache.state = cached_state

                n_prefix_tokens = min(n_prefix_tokens, len(request.prompt_tokens))
                request.prefix_cache_hit_tokens = n_prefix_tokens

                if n_prefix_tokens > 0:
                    log.info(
                        "prefix_cache_hit",
                        tier=cache_tier,
                        matched_tokens=n_prefix_tokens,
                        total_prompt_tokens=len(request.prompt_tokens),
                    )

            # Determine input_ids for prefill
            if n_prefix_tokens >= len(request.prompt_tokens):
                # Full hit -- re-run last token for fresh logits
                try:
                    from mlx_lm.models.cache import trim_prompt_cache

                    trim_prompt_cache(active.cache, 1)
                except ImportError:
                    for lc in active.cache:
                        if hasattr(lc, "offset"):
                            lc.offset = max(0, lc.offset - 1)
                input_ids = mx.array([request.prompt_tokens[-1:]])
            elif n_prefix_tokens > 0:
                input_ids = mx.array([request.prompt_tokens[n_prefix_tokens:]])
            else:
                input_ids = mx.array([request.prompt_tokens])

            # Forward pass
            t0 = time.perf_counter()
            with self._guard:
                logits = self._model(input_ids, cache=active.cache)
                mx.eval(logits)
            ttft = time.perf_counter() - t0
            request.ttft_ms = ttft * 1000.0

            # Store in prefix cache on miss
            if (
                n_prefix_tokens == 0
                and self._prefix_cache_memory is not None
                and token_hashes
            ):
                try:
                    self._prefix_cache_memory.store_sync(
                        token_hashes,
                        active.cache,
                        n_tokens=len(request.prompt_tokens),
                        block_size=self._prefix_hasher.block_size if self._prefix_hasher else 8,
                    )
                except Exception:
                    log.warning("prefix_cache_store_failed", exc_info=True)

            # Charge KV memory for prompt tokens
            kv_per_token = self._kv_bytes_per_token()
            kv_charged = int(len(request.prompt_tokens) * kv_per_token)
            self._kv_used_bytes += kv_charged
            active.kv_charged = kv_charged

            # First token
            token_id, logprob = sample(
                logits[:, -1, :], request.sampling, active.rng_key
            )
            token_text = self._tokenizer.decode([token_id])
            request.output_channel.sync_q.put(Token(token_id, token_text, logprob))
            request.completion_token_count = 1
            active.last_token_id = token_id
            active.prefill_done = True
            active.decode_started_at = time.perf_counter()
            request.transition(RequestState.DECODING)

            # Track KV for first generated token
            step_kv = int(kv_per_token)
            active.kv_charged += step_kv
            self._kv_used_bytes += step_kv

            log.info(
                "prefill_complete",
                ttft_ms=round(ttft * 1000, 2),
                prefix_cache="hit" if n_prefix_tokens > 0 else "miss",
                prefix_tokens=n_prefix_tokens,
                batch_size=len(self._running),
            )

            # Only process one prefill per iteration to avoid blocking decodes
            break

    def _process_decodes(self) -> None:
        """Decode step for all active requests.

        When only one request is active, runs a tight decode loop
        (multiple tokens per engine iteration) to minimize Python overhead.
        With multiple requests, processes one token per request per iteration.
        """
        decoding = [
            (rid, active)
            for rid, active in self._running.items()
            if active.prefill_done
            and active.request.state == RequestState.DECODING
            and not self._cancellations.is_cancelled(rid)
        ]

        if not decoding:
            return

        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        kv_per_token = self._kv_bytes_per_token()
        step_kv = int(kv_per_token)
        tokenizer_decode = self._tokenizer.decode

        decode_groups: dict[int, list[tuple[str, _ActiveRequest]]] = {}
        for req_id, active in decoding:
            decode_groups.setdefault(_cache_offset(active.cache), []).append(
                (req_id, active)
            )

        # Fast path: single request — compiled greedy decode loop.
        # Uses MLX compile for the hottest decode path when logprobs are not
        # requested. This keeps the common single-request case competitive while
        # leaving multi-request batching untouched.
        if (
            len(decoding) == 1
            and len(self._running) == 1
            and decoding[0][1].request.sampling.temperature == 0.0
            and not decoding[0][1].request.sampling.return_logprob
        ):
            req_id, active = decoding[0]
            request = active.request
            stop_checker = request._stop_checker
            chunked_fast_path = stop_checker is None
            compiled_chunk = None
            compiled_chunk_state = None
            try:
                if chunked_fast_path:
                    compiled_chunk, compiled_chunk_state = build_compiled_greedy_chunk(
                        self._model,
                        active.cache,
                        chunk_size=_SINGLE_REQUEST_GREEDY_CHUNK_SIZE,
                    )
                    compiled_step = None
                    compiled_state = None
                else:
                    compiled_step, compiled_state = build_compiled_greedy_step(
                        self._model, active.cache
                    )
            except Exception:
                logger.warning(
                    "decode_compile_failed", request_id=req_id, exc_info=True
                )
                compiled_step = None
                compiled_state = None
                compiled_chunk = None
                compiled_chunk_state = None

            if (
                compiled_chunk is None
                and compiled_chunk_state is None
                and (compiled_step is None or compiled_state is None)
            ):
                # Fall back to the prior synchronous greedy decode path.
                pass
            else:
                for _ in range(32):
                    if request.completion_token_count >= request.max_tokens:
                        request.finish_reason = "length"
                        request.transition(RequestState.COMPLETED)
                        return
                    if eos_token_id is not None and active.last_token_id == eos_token_id:
                        request.finish_reason = "stop"
                        request.transition(RequestState.COMPLETED)
                        return
                    if self._cancellations.is_cancelled(req_id):
                        return
                    if stop_checker is not None:
                        last_text = tokenizer_decode([active.last_token_id])
                        should_stop, _ = stop_checker.check(last_text)
                        if should_stop:
                            request.finish_reason = "stop"
                            request.transition(RequestState.COMPLETED)
                            return

                    if compiled_chunk is not None and compiled_chunk_state is not None:
                        remaining = request.max_tokens - request.completion_token_count
                        if remaining <= 0:
                            request.finish_reason = "length"
                            request.transition(RequestState.COMPLETED)
                            return
                        chunk_tokens = compiled_chunk(mx.array([[active.last_token_id]]))
                        mx.eval(chunk_tokens)
                        chunk_token_ids = chunk_tokens.tolist()
                        generated_in_chunk = len(chunk_token_ids)
                        active.kv_charged += step_kv * generated_in_chunk
                        self._kv_used_bytes += step_kv * generated_in_chunk
                        for token_id in chunk_token_ids[:remaining]:
                            token_text = tokenizer_decode([token_id])
                            request.output_channel.sync_q.put(
                                Token(token_id, token_text, None)
                            )
                            request.completion_token_count += 1
                            active.last_token_id = token_id
                            if eos_token_id is not None and token_id == eos_token_id:
                                request.finish_reason = "stop"
                                request.transition(RequestState.COMPLETED)
                                return
                        if request.completion_token_count >= request.max_tokens:
                            request.finish_reason = "length"
                            request.transition(RequestState.COMPLETED)
                            return
                        if not self._bridge._submit_queue.sync_q.empty():
                            return
                        continue

                    next_token = compiled_step(mx.array([[active.last_token_id]]))
                    mx.eval(next_token)
                    token_id = next_token.item()
                    token_text = tokenizer_decode([token_id])
                    request.output_channel.sync_q.put(
                        Token(token_id, token_text, None)
                    )
                    request.completion_token_count += 1
                    active.last_token_id = token_id
                    active.kv_charged += step_kv
                    self._kv_used_bytes += step_kv

                    if not self._bridge._submit_queue.sync_q.empty():
                        return
                return

        if len(decoding) == 1 and len(self._running) == 1:
            req_id, active = decoding[0]
            request = active.request
            if request.sampling.temperature == 0.0:
                stop_checker = request._stop_checker
                gen_stream = mx.new_stream(mx.default_device())

                def _fast_step(y_arr):
                    with mx.stream(gen_stream):
                        logits = self._model(y_arr[None], cache=active.cache)
                        logits = logits[:, -1, :]
                        token = mx.argmax(logits, axis=-1)
                        return token

                # Start first prefetch
                y = mx.array([active.last_token_id])
                next_y = _fast_step(y)
                mx.async_eval(next_y)

                for _ in range(32):
                    if request.completion_token_count >= request.max_tokens:
                        mx.eval(next_y)  # drain in-flight before exit
                        request.finish_reason = "length"
                        request.transition(RequestState.COMPLETED)
                        return
                    if eos_token_id is not None and active.last_token_id == eos_token_id:
                        mx.eval(next_y)
                        request.finish_reason = "stop"
                        request.transition(RequestState.COMPLETED)
                        return
                    if self._cancellations.is_cancelled(req_id):
                        mx.eval(next_y)
                        return
                    if stop_checker is not None:
                        last_text = tokenizer_decode([active.last_token_id])
                        should_stop, _ = stop_checker.check(last_text)
                        if should_stop:
                            mx.eval(next_y)
                            request.finish_reason = "stop"
                            request.transition(RequestState.COMPLETED)
                            return

                    # Wait for prefetched result
                    mx.eval(next_y)

                    # Start NEXT prefetch BEFORE delivery (overlap GPU with Python)
                    next_next_y = _fast_step(next_y)
                    mx.async_eval(next_next_y)

                    # Deliver current token
                    token_id = next_y.item()
                    token_text = tokenizer_decode([token_id])
                    request.output_channel.sync_q.put(
                        Token(token_id, token_text, None)
                    )
                    request.completion_token_count += 1
                    active.last_token_id = token_id
                    active.kv_charged += step_kv
                    self._kv_used_bytes += step_kv
                    next_y = next_next_y

                    if not self._bridge._submit_queue.sync_q.empty():
                        mx.eval(next_y)  # drain before yielding
                        return
                mx.eval(next_y)  # drain at loop boundary
                return

        # Batch compatible requests by their current cache offset.
        for _, group in decode_groups.items():
            if len(group) == 1:
                req_id, active = group[0]
                request = active.request

                if request.completion_token_count >= request.max_tokens:
                    request.finish_reason = "length"
                    request.transition(RequestState.COMPLETED)
                    continue

                if eos_token_id is not None and active.last_token_id == eos_token_id:
                    request.finish_reason = "stop"
                    request.transition(RequestState.COMPLETED)
                    continue

                if request._stop_checker is not None:
                    last_text = tokenizer_decode([active.last_token_id])
                    should_stop, _ = request._stop_checker.check(last_text)
                    if should_stop:
                        request.finish_reason = "stop"
                        request.transition(RequestState.COMPLETED)
                        continue

                if active.rng_key is not None:
                    active.rng_key = mx.random.split(active.rng_key)[0]

                with self._guard:
                    logits = self._model(
                        mx.array([[active.last_token_id]]), cache=active.cache
                    )
                    mx.eval(logits)

                token_id, logprob = sample(
                    logits[:, -1, :], request.sampling, active.rng_key
                )
                token_text = tokenizer_decode([token_id])
                request.output_channel.sync_q.put(
                    Token(token_id, token_text, logprob)
                )

                request.completion_token_count += 1
                active.last_token_id = token_id
                active.kv_charged += step_kv
                self._kv_used_bytes += step_kv
                continue

            live_group = []
            for req_id, active in group:
                request = active.request
                if request.completion_token_count >= request.max_tokens:
                    request.finish_reason = "length"
                    request.transition(RequestState.COMPLETED)
                    continue
                if eos_token_id is not None and active.last_token_id == eos_token_id:
                    request.finish_reason = "stop"
                    request.transition(RequestState.COMPLETED)
                    continue
                if request._stop_checker is not None:
                    last_text = tokenizer_decode([active.last_token_id])
                    should_stop, _ = request._stop_checker.check(last_text)
                    if should_stop:
                        request.finish_reason = "stop"
                        request.transition(RequestState.COMPLETED)
                        continue
                live_group.append((req_id, active))

            if not live_group:
                continue

            # Batch only requests whose current cache length is already aligned.
            # MLX's cache API still requires a shared offset across the batch.
            batch_size = len(live_group)
            batched_cache = _build_batched_cache([active.cache for _, active in live_group])
            batch_tokens = mx.array([[active.last_token_id] for _, active in live_group])
            with self._guard:
                logits = self._model(batch_tokens, cache=batched_cache)
                mx.eval(logits)

            _scatter_batched_cache(
                batched_cache, [active.cache for _, active in live_group]
            )

            token_rows = logits[:, -1, :]
            greedy_batch = all(
                active.request.sampling.temperature == 0.0
                and not active.request.sampling.return_logprob
                for _, active in live_group
            )
            if greedy_batch:
                next_token_ids = mx.argmax(token_rows, axis=-1).tolist()
            else:
                next_token_ids = []

            for row_idx, (req_id, active) in enumerate(live_group):
                request = active.request
                if active.rng_key is not None:
                    active.rng_key = mx.random.split(active.rng_key)[0]

                if greedy_batch:
                    token_id = int(next_token_ids[row_idx])
                    logprob = None
                else:
                    token_id, logprob = sample(
                        token_rows[row_idx], request.sampling, active.rng_key
                    )

                token_text = tokenizer_decode([token_id])
                request.output_channel.sync_q.put(
                    Token(token_id, token_text, logprob)
                )
                request.completion_token_count += 1
                active.last_token_id = token_id
                active.kv_charged += step_kv
                self._kv_used_bytes += step_kv
                maybe_quantize_kv_cache(
                    active.cache,
                    self._config.kv.quantized_kv_start,
                    self._config.kv.group_size,
                    self._config.kv.bits,
                )

        return

    def _retire_requests(self) -> None:
        """Remove completed/cancelled requests and free resources."""
        to_remove = []
        for req_id, active in self._running.items():
            request = active.request

            # Check cancellation
            if self._cancellations.is_cancelled(req_id):
                if request.state == RequestState.DECODING:
                    request.finish_reason = "cancelled"
                    request.transition(RequestState.CANCELLED)

            if request.state in (RequestState.COMPLETED, RequestState.CANCELLED):
                # Send EOS sentinel
                try:
                    request.output_channel.sync_q.put(None)
                except Exception:
                    pass

                log = logger.bind(request_id=req_id)
                if active.decode_started_at is not None:
                    decode_duration = max(
                        time.perf_counter() - active.decode_started_at, 1e-9
                    )
                    request.decode_tps = request.completion_token_count / decode_duration
                log.info(
                    "request_retired",
                    completion_tokens=request.completion_token_count,
                    finish_reason=request.finish_reason,
                    decode_tps=round(request.decode_tps, 2),
                    batch_size=len(self._running) - 1,
                )

                to_remove.append(req_id)

                # Release KV memory charged during this request
                self._kv_used_bytes = max(0, self._kv_used_bytes - active.kv_charged)
                del active.cache  # free KV memory

        for req_id in to_remove:
            del self._running[req_id]

    def _cancel_all(self) -> None:
        """Cancel all running requests on shutdown."""
        for req_id, active in list(self._running.items()):
            request = active.request
            if request.state in (RequestState.PREFILLING, RequestState.DECODING):
                request.finish_reason = "cancelled"
                try:
                    request.transition(RequestState.CANCELLED)
                except ValueError:
                    pass
            try:
                request.output_channel.sync_q.put(None)
            except Exception:
                pass
        self._running.clear()

    # -- EngineProtocol implementation --------------------------------------

    async def submit(self, request: Any) -> None:
        """Enqueue request via bridge (async, called from API thread)."""
        await self._bridge.submit_async(request)

    def snapshot(self) -> AdmissionSnapshot:
        """Point-in-time state for admission decisions."""
        return AdmissionSnapshot(
            kv_used_bytes=self._kv_used_bytes,
            kv_budget_bytes=self._budget.kv_budget_bytes if self._budget else 0,
            running_requests=len(self._running),
            queued_requests=self._bridge._submit_queue.sync_q.qsize(),
            thermal_state=ThermalState.NORMAL,
            memory_pressure=MemoryPressure.NORMAL,
        )

    async def shutdown(self) -> DrainResult:
        """Signal the engine loop to stop."""
        t0 = time.monotonic()
        self._shutdown_requested.set()
        while self._running:
            await asyncio.sleep(0.1)
            if time.monotonic() - t0 > 30:
                break
        return DrainResult(
            completed=0,
            force_cancelled=len(self._running),
            drain_duration_seconds=time.monotonic() - t0,
        )


class _ActiveRequest:
    """Internal state for a request being processed."""

    __slots__ = (
        "request",
        "cache",
        "prefill_done",
        "rng_key",
        "last_token_id",
        "kv_charged",
        "decode_started_at",
        "compiled_step",
        "compiled_step_state",
    )

    def __init__(
        self,
        request: Request,
        cache: list[Any],
        prefill_done: bool = False,
        rng_key: mx.array | None = None,
    ) -> None:
        self.request = request
        self.cache = cache
        self.prefill_done = prefill_done
        self.rng_key = rng_key
        self.last_token_id: int = 0
        self.kv_charged: int = 0
        self.decode_started_at: float | None = None
        self.compiled_step = None
        self.compiled_step_state = None


# -- Type imports at module level for type checking only -------------------
# Placed after the class to avoid circular imports at runtime.

from mlxz.prefix_cache.memory import PrefixCacheMemory  # noqa: E402
from mlxz.prefix_cache.disk import PrefixCacheDisk  # noqa: E402
from mlxz.prefix_cache.hasher import RollingPrefixHasher  # noqa: E402


def _cache_offset(cache: list[Any]) -> int:
    if not cache:
        return 0
    first = cache[0]
    return int(getattr(first, "offset", 0))


def _build_batched_cache(caches: list[list[Any]]) -> list[Any]:
    """Stack per-request caches into a single batched cache."""
    batched_cache = caches[0]
    for layer_idx in range(len(batched_cache)):
        layer_states = [cache[layer_idx].state for cache in caches]
        batched_state = tree_map(lambda *xs: mx.concatenate(xs, axis=0), *layer_states)
        batched_cache[layer_idx].state = batched_state
    return batched_cache


def _scatter_batched_cache(batched_cache: list[Any], caches: list[list[Any]]) -> None:
    """Split a batched cache back into per-request caches."""
    batch_size = len(caches)
    per_layer_states = [
        _unbatch_tree(layer_cache.state, batch_size) for layer_cache in batched_cache
    ]
    for request_idx, cache in enumerate(caches):
        for layer_idx, layer_cache in enumerate(cache):
            layer_cache.state = per_layer_states[layer_idx][request_idx]


def _unbatch_tree(tree: Any, batch_size: int) -> list[Any]:
    """Split a batched pytree into one tree per batch element."""
    leaves = tree_flatten(tree)
    per_batch: list[list[tuple[str, Any]]] = [[] for _ in range(batch_size)]
    for path, leaf in leaves:
        for idx in range(batch_size):
            per_batch[idx].append((path, mx.expand_dims(leaf[idx], axis=0)))
    return [tree_unflatten(items) for items in per_batch]
