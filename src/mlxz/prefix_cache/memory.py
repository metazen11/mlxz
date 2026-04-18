"""In-memory prefix cache with LRU eviction."""
from __future__ import annotations

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

    def lookup_sync(self, token_hashes: tuple[bytes, ...]) -> tuple[int, list[Any] | None]:
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
                entry.last_access = time.monotonic()
                self._stats.hits += 1
                self._stats.hit_bytes += entry.size_bytes
                logger.debug("prefix_cache_memory_hit",
                             matched_chunks=length,
                             matched_tokens=entry.n_tokens)
                return entry.n_tokens, entry.kv_states
        self._stats.misses += 1
        return 0, None

    def store_sync(
        self,
        token_hashes: tuple[bytes, ...],
        kv_cache_layers: list[Any],
        n_tokens: int | None = None,
    ) -> None:
        """Store prefix KV data. Deep-copies tensors and evicts LRU if over budget.

        Args:
            token_hashes: Hash tuple identifying this prefix.
            kv_cache_layers: List of KVCache objects from the model. Each has a .state
                property returning the (keys, values) tuple.
            n_tokens: Number of tokens this prefix covers. If None, inferred from
                the first layer's key tensor shape.
        """
        if token_hashes in self._entries:
            return  # already cached

        # Extract and deep-copy KV states
        kv_states = []
        cache_type = type(kv_cache_layers[0]).__name__
        for layer_cache in kv_cache_layers:
            state = layer_cache.state  # (keys, values) sliced to offset
            copied = _deep_copy_kv_states([state])[0]
            kv_states.append(copied)

        if n_tokens is None:
            # Infer from the first layer's key tensor
            first_state = kv_states[0]
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

        size = compute_size_bytes(kv_states)

        # Evict until we have room
        self._evict_until_room(size)

        if size > self._budget:
            logger.warning("prefix_cache_entry_too_large",
                           entry_bytes=size, budget_bytes=self._budget)
            return  # single entry exceeds entire budget

        entry = CachedPrefix(
            token_hashes=token_hashes,
            kv_states=kv_states,
            n_tokens=n_tokens,
            size_bytes=size,
            cache_type=cache_type,
        )
        self._entries[token_hashes] = entry
        self._used_bytes += size
        self._stats.memory_used_bytes = self._used_bytes
        logger.debug("prefix_cache_memory_stored",
                     chunks=len(token_hashes), n_tokens=n_tokens, size_bytes=size)

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

    async def lookup(self, token_hashes: list[bytes]) -> tuple[int, Any | None]:
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
