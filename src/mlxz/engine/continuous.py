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

logger = structlog.get_logger()


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
        block_size: int = 256,
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

                # Try memory cache first
                if self._prefix_cache_memory is not None:
                    n_matched, cached_kv = self._prefix_cache_memory.lookup_sync(
                        token_hashes
                    )
                    if cached_kv is not None and n_matched > 0:
                        n_prefix_tokens = n_matched
                        cache_tier = "memory"
                        for layer_cache, cached_state in zip(active.cache, cached_kv):
                            layer_cache.state = cached_state

                # Fall back to disk cache
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

            # Store in prefix cache on miss
            if n_prefix_tokens == 0 and self._prefix_cache_memory is not None and token_hashes:
                try:
                    self._prefix_cache_memory.store_sync(
                        token_hashes,
                        active.cache,
                        n_tokens=len(request.prompt_tokens),
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
            try:
                request.output_channel.sync_q.put_nowait(
                    Token(token_id, token_text, logprob)
                )
            except Exception:
                pass
            request.completion_token_count = 1
            active.last_token_id = token_id
            active.prefill_done = True
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

        # Fast path: single request — prefetch decode loop (mlx-lm style)
        if len(decoding) == 1:
            req_id, active = decoding[0]
            request = active.request
            stop_checker = request._stop_checker
            gen_stream = mx.new_stream(mx.default_device())

            def _fast_step(y_arr):
                with mx.stream(gen_stream):
                    logits = self._model(y_arr[None], cache=active.cache)
                    logits = logits[:, -1, :]
                    token = mx.argmax(logits, axis=-1)
                    return token, logits

            # Start first prefetch
            y = mx.array([active.last_token_id])
            next_y, next_logits = _fast_step(y)
            mx.async_eval(next_y, next_logits)

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
                    last_text = self._tokenizer.decode([active.last_token_id])
                    should_stop, _ = stop_checker.check(last_text)
                    if should_stop:
                        request.finish_reason = "stop"
                        request.transition(RequestState.COMPLETED)
                        return

                # Wait for prefetched result
                mx.eval(next_y)
                y = next_y

                # Start NEXT prefetch before delivery
                next_y, next_logits = _fast_step(y)
                mx.async_eval(next_y, next_logits)

                # Deliver token
                token_id = y.item()
                token_text = self._tokenizer.decode([token_id])
                try:
                    request.output_channel.sync_q.put_nowait(
                        Token(token_id, token_text, None)
                    )
                except Exception:
                    return
                request.completion_token_count += 1
                active.last_token_id = token_id

                if not self._bridge._submit_queue.sync_q.empty():
                    return
            return

        # Multi-request path: one token per request per iteration
        for req_id, active in decoding:
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
                last_text = self._tokenizer.decode([active.last_token_id])
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
            token_text = self._tokenizer.decode([token_id])

            try:
                request.output_channel.sync_q.put_nowait(
                    Token(token_id, token_text, logprob)
                )
            except Exception:
                continue

            request.completion_token_count += 1
            active.last_token_id = token_id
            active.kv_charged += step_kv
            self._kv_used_bytes += step_kv

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
                log.info(
                    "request_retired",
                    completion_tokens=request.completion_token_count,
                    finish_reason=request.finish_reason,
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


# -- Type imports at module level for type checking only -------------------
# Placed after the class to avoid circular imports at runtime.

from mlxz.prefix_cache.memory import PrefixCacheMemory  # noqa: E402
from mlxz.prefix_cache.disk import PrefixCacheDisk  # noqa: E402
from mlxz.prefix_cache.hasher import RollingPrefixHasher  # noqa: E402
