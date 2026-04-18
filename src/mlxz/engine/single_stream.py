"""Single-stream inference engine (batch=1, synchronous)."""
from __future__ import annotations

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


class SingleStreamEngine:
    """Batch=1 synchronous inference engine.

    Implements EngineProtocol from mlxz.types.
    Runs on a dedicated thread -- all mx.eval calls happen here.
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
        self._running_requests: int = 0
        self._kv_used_bytes: int = 0
        self._guard: MxEvalGuard | None = None
        # Model arch params (set after load)
        self._n_layers: int = 0
        self._n_heads: int = 0
        self._head_dim: int = 0
        # Prefix cache tiers (set via set_prefix_cache)
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

    # -- Setup --------------------------------------------------------------

    def set_model(self, model: nn.Module, tokenizer: Any) -> None:
        """Set the loaded model and tokenizer. Called from main thread before run()."""
        self._model = model
        self._tokenizer = tokenizer
        # Extract architecture params from model config
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
            self._n_layers = 32
            self._n_heads = 32
            self._head_dim = 128

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
        """Blocking engine loop. Target for EngineThreadSupervisor.

        Invariants:
        - Single compute thread: all mx.eval calls happen here.
        - No token loss: every request gets a ``None`` sentinel on its
          output channel.
        - Backpressure: ``channel.sync_q.put()`` blocks if the client
          is slow.
        """
        self._guard = MxEvalGuard()
        logger.info("engine_started", model=self._model_name)

        while not self._shutdown_requested.is_set():
            request = self._bridge.get_next_sync()
            if request is None:
                self._shutdown_requested.wait(timeout=0.005)
                continue
            self._process_request(request)

        logger.info("engine_stopped")

    # -- Request processing -------------------------------------------------

    def _kv_bytes_per_token(self) -> float:
        """Estimate KV-cache bytes consumed per token."""
        return (
            2  # keys + values
            * self._n_layers
            * self._n_heads
            * self._head_dim
            * (self._config.kv.bits / 8)
        )

    def _process_request(self, request: Request) -> None:
        """Process a single inference request: prefill then decode loop."""
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._guard is not None

        log = logger.bind(request_id=request.id)
        log.info(
            "request_processing",
            prompt_tokens=request.prompt_token_count,
            max_tokens=request.max_tokens,
        )

        kv_per_token = self._kv_bytes_per_token()
        kv_charged: int = 0  # bytes charged to self._kv_used_bytes
        cache = None

        try:
            request.transition(RequestState.PREFILLING)
            self._running_requests += 1

            # Create KV cache for this request
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(self._model)

            # --- Prefix cache lookup ---
            n_prefix_tokens = 0
            token_hashes: tuple[bytes, ...] = ()
            cache_tier = ""

            if self._prefix_hasher is not None:
                token_hashes = self._prefix_hasher.hash_chunks(request.prompt_tokens)

                # Try memory cache first
                if self._prefix_cache_memory is not None:
                    n_matched, cached_kv = self._prefix_cache_memory.lookup_sync(token_hashes)
                    if cached_kv is not None and n_matched > 0:
                        n_prefix_tokens = n_matched
                        cache_tier = "memory"
                        # Restore cached KV into the fresh cache
                        for layer_cache, cached_state in zip(cache, cached_kv):
                            layer_cache.state = cached_state

                # Fall back to disk cache
                if n_prefix_tokens == 0 and self._prefix_cache_disk is not None:
                    n_matched, cached_kv = self._prefix_cache_disk.lookup_sync(token_hashes)
                    if cached_kv is not None and n_matched > 0:
                        n_prefix_tokens = n_matched
                        cache_tier = "disk"
                        # Restore cached KV into the fresh cache
                        for layer_cache, cached_state in zip(cache, cached_kv):
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

            # --- Determine input_ids for prefill ---
            if n_prefix_tokens >= len(request.prompt_tokens):
                # Full hit -- re-run last token for fresh logits
                try:
                    from mlx_lm.models.cache import trim_prompt_cache
                    trim_prompt_cache(cache, 1)
                except ImportError:
                    # Fallback: manually adjust offset
                    for lc in cache:
                        if hasattr(lc, "offset"):
                            lc.offset = max(0, lc.offset - 1)
                input_ids = mx.array([request.prompt_tokens[-1:]])
            elif n_prefix_tokens > 0:
                remaining = request.prompt_tokens[n_prefix_tokens:]
                input_ids = mx.array([remaining])
            else:
                input_ids = mx.array([request.prompt_tokens])

            # Prefill forward pass
            t0 = time.perf_counter()

            with self._guard:
                logits = self._model(input_ids, cache=cache)
                mx.eval(logits)

            ttft = time.perf_counter() - t0

            # --- Store in memory cache on miss ---
            if (
                n_prefix_tokens == 0
                and self._prefix_cache_memory is not None
                and token_hashes
            ):
                try:
                    self._prefix_cache_memory.store_sync(
                        token_hashes, cache, n_tokens=len(request.prompt_tokens)
                    )
                except Exception:
                    log.warning("prefix_cache_store_failed", exc_info=True)

            # --- Record TTFT metrics ---
            try:
                from mlxz.api.metrics import ttft_seconds, prefix_cache_hits_total

                cache_label = "hit" if n_prefix_tokens > 0 else "miss"
                ttft_seconds.labels(prefix_cache=cache_label).observe(ttft)
                if n_prefix_tokens > 0:
                    prefix_cache_hits_total.labels(tier=cache_tier).inc()
            except Exception:
                pass  # metrics are best-effort

            log.info(
                "prefill_complete",
                ttft_ms=round(ttft * 1000, 2),
                prefix_cache=("hit" if n_prefix_tokens > 0 else "miss"),
                prefix_tokens=n_prefix_tokens,
            )

            # Charge KV memory for prompt tokens
            kv_charged = int(len(request.prompt_tokens) * kv_per_token)
            self._kv_used_bytes += kv_charged

            # Transition to decoding
            request.transition(RequestState.DECODING)

            # EOS / stop config (hoist out of loop)
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            stop_checker = request._stop_checker
            channel_put = request.output_channel.sync_q.put

            # --- PREFETCH DECODE LOOP (mlx-lm style) ---
            # Key insight: model forward AND sampling run inside mx.stream()
            # so cache writes are serialized. Token materialization (.item())
            # happens outside, after mx.eval(). This overlaps GPU compute
            # with Python token delivery.

            gen_stream = mx.new_stream(mx.default_device())

            def _decode_step(y_arr):
                """One decode step: forward + greedy sample, all in gen_stream."""
                with mx.stream(gen_stream):
                    step_logits = self._model(y_arr[None], cache=cache)
                    step_logits = step_logits[:, -1, :]
                    step_logprobs = step_logits - mx.logsumexp(step_logits, keepdims=True)
                    step_token = mx.argmax(step_logits, axis=-1)
                    return step_token, step_logprobs

            # First token from prefill logits (already computed)
            first_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(-1)
            first_logprobs = logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], keepdims=True)
            mx.eval(first_token)

            token_id = first_token.item()
            token_text = self._tokenizer.decode([token_id])
            channel_put(Token(token_id, token_text, None))
            request.completion_token_count = 1

            # Start prefetch of second token
            y = first_token.reshape(-1)
            next_y, next_logprobs = _decode_step(y)
            mx.async_eval(next_y, next_logprobs)

            decode_start = time.perf_counter()

            for _step in range(request.max_tokens - 1):
                # Check cancellation
                if self._cancellations.is_cancelled(request.id):
                    request.finish_reason = "cancelled"
                    request.transition(RequestState.CANCELLED)
                    break

                # Check EOS on previous token
                if eos_token_id is not None and token_id == eos_token_id:
                    request.finish_reason = "stop"
                    break

                # Check stop sequences on previous token
                if stop_checker is not None:
                    should_stop, _ = stop_checker.check(token_text)
                    if should_stop:
                        request.finish_reason = "stop"
                        break

                # Wait for prefetched result (GPU already computed it)
                mx.eval(next_y)
                y = next_y

                # Start NEXT prefetch before Python delivery work
                if _step < request.max_tokens - 2:
                    next_y, next_logprobs = _decode_step(y)
                    mx.async_eval(next_y, next_logprobs)

                # Materialize and deliver token (overlaps with GPU prefetch)
                token_id = y.item()
                token_text = self._tokenizer.decode([token_id])
                channel_put(Token(token_id, token_text, None))
                request.completion_token_count += 1

            # Set finish reason if not already set
            if request.state == RequestState.DECODING:
                if request.finish_reason is None:
                    request.finish_reason = "length"
                request.transition(RequestState.COMPLETED)

            decode_duration = time.perf_counter() - decode_start
            decode_tps = (
                request.completion_token_count / decode_duration
                if decode_duration > 0
                else 0
            )
            log.info(
                "request_completed",
                completion_tokens=request.completion_token_count,
                finish_reason=request.finish_reason,
                decode_tps=round(decode_tps, 2),
                ttft_ms=round(ttft * 1000, 2),
                prefix_cache_hit_tokens=n_prefix_tokens,
            )

        except Exception:
            log.exception("request_error")
            if request.state in (RequestState.PREFILLING, RequestState.DECODING):
                request.finish_reason = "error"
                request.transition(RequestState.CANCELLED)

        finally:
            # Always send EOS sentinel
            try:
                request.output_channel.sync_q.put(None)
            except Exception:
                pass  # Channel may be closed if client disconnected
            self._running_requests -= 1
            # Release KV memory charged during this request
            self._kv_used_bytes = max(0, self._kv_used_bytes - kv_charged)
            del cache

    # -- EngineProtocol implementation --------------------------------------

    async def submit(self, request: Any) -> None:
        """Enqueue request via bridge (async, called from API thread)."""
        await self._bridge.submit_async(request)

    def snapshot(self) -> AdmissionSnapshot:
        """Point-in-time state for admission decisions."""
        return AdmissionSnapshot(
            kv_used_bytes=self._kv_used_bytes,
            kv_budget_bytes=self._budget.kv_budget_bytes if self._budget else 0,
            running_requests=self._running_requests,
            queued_requests=self._bridge._submit_queue.sync_q.qsize(),
            thermal_state=ThermalState.NORMAL,  # TODO: wire thermal monitor
            memory_pressure=MemoryPressure.NORMAL,  # TODO: wire memory monitor
        )

    async def shutdown(self) -> DrainResult:
        """Signal the engine loop to stop."""
        import asyncio

        t0 = time.monotonic()
        self._shutdown_requested.set()
        # Wait briefly for current request to finish
        while self._running_requests > 0:
            await asyncio.sleep(0.1)
            if time.monotonic() - t0 > 30:
                break
        return DrainResult(
            completed=0,
            force_cancelled=self._running_requests,
            drain_duration_seconds=time.monotonic() - t0,
        )


# -- Type imports at module level for type checking only -------------------
# Placed after the class to avoid circular imports at runtime.
# The actual imports happen lazily in set_prefix_cache / _process_request.

from mlxz.prefix_cache.memory import PrefixCacheMemory  # noqa: E402
from mlxz.prefix_cache.disk import PrefixCacheDisk  # noqa: E402
from mlxz.prefix_cache.hasher import RollingPrefixHasher  # noqa: E402
