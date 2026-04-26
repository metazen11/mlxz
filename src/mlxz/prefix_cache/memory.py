"""In-memory prefix cache with LRU eviction."""
from __future__ import annotations

import copy
import time
from typing import Any

import mlx.core as mx
import structlog

from mlxz.prefix_cache.base import CachedPrefix, compute_size_bytes
from mlxz.types import PrefixCacheStats

logger = structlog.get_logger()


def _deep_copy_kv_states(kv_states: list[Any]) -> list[Any]:
    """Deep copy KV state tensors so the cached version is independent."""
    result = []
    for layer_state in kv_states:
        if isinstance(layer_state, tuple):
            copied = tuple(
                mx.array(arr) if isinstance(arr, mx.array) else
                tuple(mx.array(a) if isinstance(a, mx.array) else a for a in arr)
                if isinstance(arr, tuple) else arr
                for arr in layer_state
            )
            result.append(copied)
        else:
            result.append(layer_state)
    return result


class PrefixCacheMemory:
    """In-memory prefix cache with LRU eviction and greedy longest-prefix matching.

    Implements PrefixCacheProtocol (async methods) AND provides sync counterparts
    (lookup_sync/store_sync) for use on the engine thread without event loop overhead.

    Thread safety: NOT thread-safe. Designed for single-stream (Phase 1-2) where
    only one request is processed at a time on the engine thread.
    """

    def __init__(self, memory_budget_bytes: int) -> None:
        self._budget = memory_budget_bytes
        self._entries: dict[tuple[bytes, ...], CachedPrefix] = {}
        self._used_bytes: int = 0
        self._stats = PrefixCacheStats()

    # -- Sync interface (used by engine thread) --

    def lookup_sync(
        self,
        token_hashes: tuple[bytes, ...],
        cache_type: str | None = None,
    ) -> tuple[int, list[Any] | None, str | None]:
        """Find the longest matching prefix.

        Tries progressively shorter prefixes: (all N) -> (N-1) -> ... -> (1).
        Returns (n_matched_tokens, kv_states) on hit, or (0, None) on miss.

        On hit, returns a REFERENCE to cached KV states (not a copy).
        The caller must copy into its own cache via state setter.
        """
        for length in range(len(token_hashes), 0, -1):
            prefix_key = token_hashes[:length]
            if prefix_key in self._entries:
                entry = self._entries[prefix_key]
                if cache_type is not None and entry.cache_type != cache_type:
                    continue
                entry.last_access = time.monotonic()
                self._stats.hits += 1
                self._stats.hit_bytes += entry.size_bytes
                logger.debug("prefix_cache_memory_hit",
                             matched_chunks=length,
                             matched_tokens=entry.n_tokens)
                return entry.n_tokens, entry.kv_states, entry.cache_type
        self._stats.misses += 1
        return 0, None, None

    def store_sync(
        self,
        token_hashes: tuple[bytes, ...],
        kv_cache_layers: list[Any],
        n_tokens: int | None = None,
        block_size: int = 8,
    ) -> None:
        """Store prefix KV data. Deep-copies tensors and evicts LRU if over budget.

        Args:
            token_hashes: Hash tuple identifying this prefix.
            kv_cache_layers: List of KVCache objects from the model. Each has a .state
                property returning the (keys, values) tuple.
            n_tokens: Number of tokens this prefix covers. If None, inferred from
                the first layer's key tensor shape.
            block_size: Prefix-cache chunk size used to derive stored prefix keys.
        """
        if token_hashes in self._entries:
            return  # already cached

        # Extract and deep-copy KV states from a mutable cache snapshot.
        cache_type = type(kv_cache_layers[0]).__name__
        cache_copy = copy.deepcopy(kv_cache_layers)

        def _snapshot_state(copy_cache: list[Any]) -> list[Any]:
            return _deep_copy_kv_states([layer_cache.state for layer_cache in copy_cache])

        if n_tokens is None:
            # Infer from the first layer's key tensor
            first_state = cache_copy[0].state
            if isinstance(first_state, tuple) and len(first_state) >= 1:
                keys = first_state[0]
                if isinstance(keys, mx.array):
                    n_tokens = keys.shape[2] if keys.ndim >= 3 else keys.shape[0]
                elif isinstance(keys, tuple):
                    # Quantized: keys is (data, scales, biases)
                    n_tokens = keys[0].shape[2] if keys[0].ndim >= 3 else keys[0].shape[0]
                else:
                    n_tokens = 0
            else:
                n_tokens = 0

        if n_tokens <= 0:
            return

        # Store the full prompt state and each chunk boundary prefix so future
        # requests can hit on shared system-prompt prefixes instead of only
        # exact full-prompt matches.
        prefix_lengths: list[int] = list(range(block_size, n_tokens, block_size))
        if not prefix_lengths or prefix_lengths[-1] != n_tokens:
            prefix_lengths.append(n_tokens)

        current_len = n_tokens
        for prefix_len in reversed(prefix_lengths):
            if current_len > prefix_len:
                from mlx_lm.models.cache import trim_prompt_cache

                trim_prompt_cache(cache_copy, current_len - prefix_len)
                current_len = prefix_len

            prefix_chunks = prefix_len // block_size
            if prefix_len % block_size:
                prefix_chunks += 1
            prefix_key = token_hashes[:prefix_chunks]
            if not prefix_key or prefix_key in self._entries:
                continue

            kv_states = _snapshot_state(cache_copy)
            size = compute_size_bytes(kv_states)

            self._evict_until_room(size)
            if size > self._budget:
                logger.warning(
                    "prefix_cache_entry_too_large",
                    entry_bytes=size,
                    budget_bytes=self._budget,
                )
                continue

            entry = CachedPrefix(
                token_hashes=prefix_key,
                kv_states=kv_states,
                n_tokens=prefix_len,
                size_bytes=size,
                cache_type=cache_type,
            )
            self._entries[prefix_key] = entry
            self._used_bytes += size
            self._stats.memory_used_bytes = self._used_bytes
            logger.debug(
                "prefix_cache_memory_stored",
                chunks=len(prefix_key),
                n_tokens=prefix_len,
                size_bytes=size,
            )

    def _evict_until_room(self, needed_bytes: int) -> None:
        """Evict LRU entries until there's room for needed_bytes."""
        while self._used_bytes + needed_bytes > self._budget and self._entries:
            oldest_key = min(self._entries, key=lambda k: self._entries[k].last_access)
            evicted = self._entries.pop(oldest_key)
            self._used_bytes -= evicted.size_bytes
            self._stats.evictions += 1
            logger.debug("prefix_cache_memory_evicted",
                         n_tokens=evicted.n_tokens, size_bytes=evicted.size_bytes)

    # -- Async interface (satisfies PrefixCacheProtocol) --

    async def lookup(self, token_hashes: list[bytes]) -> tuple[int, Any | None, str | None]:
        return self.lookup_sync(tuple(token_hashes))

    async def store(self, token_hashes: list[bytes], kv: Any) -> None:
        self.store_sync(tuple(token_hashes), kv)

    def stats(self) -> PrefixCacheStats:
        """Return current cache statistics snapshot."""
        return PrefixCacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            hit_bytes=self._stats.hit_bytes,
            evictions=self._stats.evictions,
            memory_used_bytes=self._used_bytes,
            disk_used_bytes=0,
        )
