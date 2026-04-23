"""Shared types for the prefix cache module."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CachedPrefix:
    """A single cached prefix entry.

    kv_states is a list of per-layer KV state tuples. Each element
    corresponds to one transformer layer. The exact structure depends
    on the cache type:
    - KVCache: (keys, values) where each is mx.array
    - QuantizedKVCache: ((qk_data, qk_scales, qk_biases), (qv_data, qv_scales, qv_biases))
    """
    token_hashes: tuple[bytes, ...]
    kv_states: list[Any]  # per-layer state from KVCache.state
    n_tokens: int
    size_bytes: int
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    cache_type: str = "KVCache"


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """Result of a successful prefix cache lookup."""
    n_matched_tokens: int
    kv_states: list[Any]  # reference to cached KV (not a copy)
    cache_tier: str  # "memory" or "disk"


def compute_size_bytes(kv_states: list[Any]) -> int:
    """Compute total memory footprint of KV states.

    Walks the nested structure and sums nbytes of all mx.array leaves.
    """
    import mlx.core as mx

    total = 0

    def _walk(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, mx.array):
            total += obj.nbytes
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    _walk(kv_states)
    return total
