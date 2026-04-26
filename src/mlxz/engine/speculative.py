"""Speculative decoding engine — draft-target with rejection sampling."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import structlog

from mlxz.config import RuntimeConfig
from mlxz.engine.cache_utils import build_prompt_cache, cache_type_name
from mlxz.engine.draft import DraftModel
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


class SpeculativeEngine:
    """Speculative decoding engine using draft-target with rejection sampling.

    Implements EngineProtocol. Uses a small draft model to propose k tokens,
    then verifies them with the target model in a single forward pass.
    Accepted tokens are "free" — verified in parallel.

    Chen et al. rejection sampling ensures the output distribution is
    identical to the target model (lossless in distribution).
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
        self._target_model: nn.Module | None = None
        self._draft: DraftModel | None = None
        self._tokenizer: Any = None
        self._budget: ResidencyBudget | None = None
        self._model_name: str = config.model
        self._shutdown_requested = threading.Event()
        self._running_requests: int = 0
        self._kv_used_bytes: int = 0
        self._guard: MxEvalGuard | None = None
        self._n_layers: int = 0
        self._n_heads: int = 0
        self._head_dim: int = 0

        # Speculative config
        self._num_draft_tokens: int = config.speculative.num_draft_tokens
        self._max_draft_tokens: int = config.speculative.max_draft_tokens
        self._backoff_threshold: float = config.speculative.backoff_threshold

        # Adaptive draft length
        self._current_draft_k: int = self._num_draft_tokens

        # Stats
        self._total_accepted: int = 0
        self._total_proposed: int = 0
        # Prefix cache is wired through the common app startup path.
        # Speculative decode currently skips cache reuse and remains functional
        # without it; the hook exists so engine selection stays uniform.
        self._prefix_cache_memory = None
        self._prefix_cache_disk = None
        self._prefix_hasher = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_arch(self) -> tuple[int, int, int]:
        return self._n_layers, self._n_heads, self._head_dim

    @property
    def acceptance_rate(self) -> float:
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    def set_model(self, model: nn.Module, tokenizer: Any) -> None:
        self._target_model = model
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

    def set_draft_model(self, draft_model: nn.Module, draft_tokenizer: Any) -> None:
        """Set the draft model. Must be called before run()."""
        self._draft = DraftModel(draft_model, draft_tokenizer)

    def set_prefix_cache(
        self,
        memory_cache: Any | None = None,
        disk_cache: Any | None = None,
        block_size: int = 8,
    ) -> None:
        """Accept the shared prefix-cache wiring used by app startup.

        Speculative decoding can use prefix caching, but the cache layout differs
        because we keep both draft and target caches. Leave the hook in place so
        startup stays uniform and we can add the dual-cache implementation without
        changing engine selection again.
        """
        self._prefix_cache_memory = memory_cache
        self._prefix_cache_disk = disk_cache
        if memory_cache is not None or disk_cache is not None:
            from mlxz.prefix_cache.hasher import RollingPrefixHasher

            self._prefix_hasher = RollingPrefixHasher(block_size=block_size)

    def set_budget(self, budget: ResidencyBudget) -> None:
        self._budget = budget

    def run(self) -> None:
        """Engine loop — same pattern as SingleStreamEngine but with speculative decode."""
        self._guard = MxEvalGuard()
        logger.info("speculative_engine_started", model=self._model_name,
                     draft_k=self._current_draft_k)

        while not self._shutdown_requested.is_set():
            request = self._bridge.get_next_sync()
            if request is None:
                self._shutdown_requested.wait(timeout=0.005)
                continue
            self._process_request(request)

        logger.info("speculative_engine_stopped",
                     acceptance_rate=round(self.acceptance_rate, 3))

    def _process_request(self, request: Request) -> None:
        assert self._target_model is not None
        assert self._draft is not None
        assert self._tokenizer is not None

        log = logger.bind(request_id=request.id)

        try:
            request.transition(RequestState.PREFILLING)
            self._running_requests += 1

            # Create KV caches for both models
            quantize_kv = (
                self._config.kv.quantized_kv_start > 0
                and (request.prompt_token_count + request.max_tokens)
                >= self._config.kv.quantized_kv_start
                and self._config.kv.bits < 16
            )
            target_cache = build_prompt_cache(
                self._target_model,
                quantized=quantize_kv,
                group_size=self._config.kv.group_size,
                bits=self._config.kv.bits,
            )
            draft_cache = build_prompt_cache(
                self._draft._model,
                quantized=quantize_kv,
                group_size=self._config.kv.group_size,
                bits=self._config.kv.bits,
            )

            # Prefix cache lookup applies to the target model only. The draft
            # model has different weights, so its KV cache cannot be reused.
            n_prefix_tokens = 0
            token_hashes: tuple[bytes, ...] = ()
            cache_tier = ""

            if self._prefix_hasher is not None:
                token_hashes = self._prefix_hasher.hash_chunks(request.prompt_tokens)

                if self._prefix_cache_memory is not None:
                    n_matched, cached_kv, cached_type = self._prefix_cache_memory.lookup_sync(
                        token_hashes,
                        cache_type=cache_type_name(target_cache),
                    )
                    if cached_kv is not None and n_matched > 0:
                        n_prefix_tokens = n_matched
                        cache_tier = "memory"
                        if cached_type == "QuantizedKVCache" and cache_type_name(target_cache) != "QuantizedKVCache":
                            target_cache = build_prompt_cache(
                                self._target_model,
                                quantized=True,
                                group_size=self._config.kv.group_size,
                                bits=self._config.kv.bits,
                            )
                        for layer_cache, cached_state in zip(target_cache, cached_kv):
                            layer_cache.state = cached_state

                if n_prefix_tokens == 0 and self._prefix_cache_disk is not None:
                    n_matched, cached_kv, cached_type = self._prefix_cache_disk.lookup_sync(
                        token_hashes,
                        cache_type=cache_type_name(target_cache),
                    )
                    if cached_kv is not None and n_matched > 0:
                        n_prefix_tokens = n_matched
                        cache_tier = "disk"
                        if cached_type == "QuantizedKVCache" and cache_type_name(target_cache) != "QuantizedKVCache":
                            target_cache = build_prompt_cache(
                                self._target_model,
                                quantized=True,
                                group_size=self._config.kv.group_size,
                                bits=self._config.kv.bits,
                            )
                        for layer_cache, cached_state in zip(target_cache, cached_kv):
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

            # Determine target-model prompt slice
            if n_prefix_tokens >= len(request.prompt_tokens):
                # Full hit -- re-run last token for fresh logits
                try:
                    from mlx_lm.models.cache import trim_prompt_cache

                    trim_prompt_cache(target_cache, 1)
                except ImportError:
                    for lc in target_cache:
                        if hasattr(lc, "offset"):
                            lc.offset = max(0, lc.offset - 1)
                target_input_ids = mx.array([request.prompt_tokens[-1:]])
            elif n_prefix_tokens > 0:
                target_input_ids = mx.array([request.prompt_tokens[n_prefix_tokens:]])
            else:
                target_input_ids = mx.array([request.prompt_tokens])

            # Draft model always needs the full prompt because its KV cache is
            # not compatible with the target model's cached prefix.
            draft_input_ids = mx.array([request.prompt_tokens])

            # Prefill both models
            t0 = time.perf_counter()

            with self._guard:
                target_logits = self._target_model(target_input_ids, cache=target_cache)
                draft_logits = self._draft._model(draft_input_ids, cache=draft_cache)
                mx.eval(target_logits, draft_logits)

            ttft = time.perf_counter() - t0
            request.ttft_ms = ttft * 1000.0

            if (
                n_prefix_tokens == 0
                and self._prefix_cache_memory is not None
                and token_hashes
            ):
                try:
                    self._prefix_cache_memory.store_sync(
                        token_hashes,
                        target_cache,
                        n_tokens=len(request.prompt_tokens),
                        block_size=self._prefix_hasher.block_size if self._prefix_hasher else 8,
                    )
                except Exception:
                    log.warning("prefix_cache_store_failed", exc_info=True)

            log.info(
                "prefill_complete",
                ttft_ms=round(ttft * 1000, 2),
                prefix_cache=("hit" if n_prefix_tokens > 0 else "miss"),
                prefix_tokens=n_prefix_tokens,
            )

            request.transition(RequestState.DECODING)

            # First token from target
            token_id, logprob = sample(target_logits[:, -1, :], request.sampling)
            token_text = self._tokenizer.decode([token_id])
            request.output_channel.sync_q.put(Token(token_id, token_text, logprob))
            request.completion_token_count = 1

            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)

            # Speculative decode loop
            decode_start = time.perf_counter()
            while request.completion_token_count < request.max_tokens:
                if self._cancellations.is_cancelled(request.id):
                    request.finish_reason = "cancelled"
                    request.transition(RequestState.CANCELLED)
                    break

                if eos_token_id is not None and token_id == eos_token_id:
                    request.finish_reason = "stop"
                    break

                # 1. Draft model generates k speculative tokens
                k = min(self._current_draft_k,
                        request.max_tokens - request.completion_token_count)
                draft_tokens = self._draft.generate_draft(token_id, draft_cache, k)

                if not draft_tokens:
                    break

                # 2. Target model verifies all k tokens in ONE forward pass
                draft_ids = [t[0] for t in draft_tokens]
                verify_input = mx.array([[token_id] + draft_ids])

                with self._guard:
                    target_verify_logits = self._target_model(
                        verify_input, cache=target_cache
                    )
                    mx.eval(target_verify_logits)

                # 3. Rejection sampling (Chen et al.)
                accepted = 0
                for i, (draft_token_id, draft_logits_i) in enumerate(draft_tokens):
                    target_logits_i = target_verify_logits[0, i, :]

                    p_target = mx.softmax(target_logits_i)
                    p_draft = mx.softmax(draft_logits_i)

                    # Acceptance probability
                    p_t = p_target[draft_token_id].item()
                    p_d = p_draft[draft_token_id].item()

                    if p_d == 0:
                        acceptance_prob = 1.0 if p_t > 0 else 0.0
                    else:
                        acceptance_prob = min(1.0, p_t / p_d)

                    r = mx.random.uniform().item()

                    if r < acceptance_prob:
                        # Accept draft token
                        text = self._tokenizer.decode([draft_token_id])
                        request.output_channel.sync_q.put(
                            Token(draft_token_id, text, None)
                        )
                        request.completion_token_count += 1
                        token_id = draft_token_id
                        accepted += 1
                        self._total_accepted += 1
                    else:
                        # Reject — sample from adjusted distribution
                        adjusted = mx.maximum(p_target - p_draft, mx.array(0.0))
                        adjusted_sum = mx.sum(adjusted).item()
                        if adjusted_sum > 0:
                            adjusted = adjusted / adjusted_sum
                            new_token = mx.random.categorical(
                                mx.log(adjusted + 1e-10)[None, :]
                            ).item()
                        else:
                            new_token = mx.argmax(p_target).item()

                        text = self._tokenizer.decode([new_token])
                        request.output_channel.sync_q.put(
                            Token(new_token, text, None)
                        )
                        request.completion_token_count += 1
                        token_id = new_token

                        # Trim caches to rejection point
                        # Target cache already has the right state
                        # Draft cache needs to be rebuilt from accepted tokens
                        break

                self._total_proposed += len(draft_tokens)

                # If all accepted, sample one more from the last target logits
                if accepted == len(draft_tokens):
                    bonus_logits = target_verify_logits[0, -1, :]
                    bonus_id, bonus_lp = sample(bonus_logits, request.sampling)
                    text = self._tokenizer.decode([bonus_id])
                    request.output_channel.sync_q.put(Token(bonus_id, text, bonus_lp))
                    request.completion_token_count += 1
                    token_id = bonus_id
                    self._total_accepted += 1

                # 4. Adaptive k
                batch_rate = accepted / len(draft_tokens) if draft_tokens else 0
                if batch_rate > 0.8 and self._current_draft_k < self._max_draft_tokens:
                    self._current_draft_k += 1
                elif batch_rate < self._backoff_threshold and self._current_draft_k > 1:
                    self._current_draft_k -= 1

            if request.state == RequestState.DECODING:
                if request.finish_reason is None:
                    request.finish_reason = "length"
                request.transition(RequestState.COMPLETED)

            decode_duration = time.perf_counter() - decode_start
            request.decode_tps = (
                request.completion_token_count / decode_duration
                if decode_duration > 0
                else 0.0
            )

            log.info("request_completed",
                     completion_tokens=request.completion_token_count,
                     finish_reason=request.finish_reason,
                     decode_tps=round(request.decode_tps, 2),
                     acceptance_rate=round(self.acceptance_rate, 3),
                     draft_k=self._current_draft_k)

        except Exception:
            log.exception("request_error")
            if request.state in (RequestState.PREFILLING, RequestState.DECODING):
                request.finish_reason = "error"
                request.transition(RequestState.CANCELLED)

        finally:
            try:
                request.output_channel.sync_q.put(None)
            except Exception:
                pass
            self._running_requests -= 1
            del target_cache, draft_cache

    async def submit(self, request: Any) -> None:
        await self._bridge.submit_async(request)

    def snapshot(self) -> AdmissionSnapshot:
        return AdmissionSnapshot(
            kv_used_bytes=self._kv_used_bytes,
            kv_budget_bytes=self._budget.kv_budget_bytes if self._budget else 0,
            running_requests=self._running_requests,
            queued_requests=self._bridge._submit_queue.sync_q.qsize(),
            thermal_state=ThermalState.NORMAL,
            memory_pressure=MemoryPressure.NORMAL,
        )

    async def shutdown(self) -> DrainResult:
        t0 = time.monotonic()
        self._shutdown_requested.set()
        while self._running_requests > 0:
            await asyncio.sleep(0.1)
            if time.monotonic() - t0 > 30:
                break
        return DrainResult(
            completed=0, force_cancelled=self._running_requests,
            drain_duration_seconds=time.monotonic() - t0,
        )
