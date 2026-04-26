"""Cache construction helpers for engine request paths."""
from __future__ import annotations

from typing import Any

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache


def cache_type_name(cache: list[Any]) -> str:
    """Return the concrete cache type name for a cache list."""
    return type(cache[0]).__name__ if cache else "KVCache"


def build_prompt_cache(
    model: Any,
    *,
    quantized: bool = False,
    group_size: int = 64,
    bits: int = 8,
    max_kv_size: int | None = None,
) -> list[Any]:
    """Build a prompt KV cache compatible with the loaded model."""
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if quantized:
        return [QuantizedKVCache(group_size=group_size, bits=bits) for _ in range(num_layers)]
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)]
    return [KVCache() for _ in range(num_layers)]
