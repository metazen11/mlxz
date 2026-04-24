"""Helpers for constructing and inspecting MLX KV caches."""
from __future__ import annotations

from typing import Any

from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache


def build_prompt_cache(
    model: Any,
    *,
    quantized: bool = False,
    group_size: int = 64,
    bits: int = 8,
) -> list[Any]:
    """Build a prompt cache for a request.

    When ``quantized`` is true, the cache is created in quantized form from the
    start so repeated long-context workloads can stay on the lower-bandwidth
    cache path.
    """
    cache = make_prompt_cache(model)
    if not quantized:
        return cache

    quantized_cache: list[Any] = []
    for layer_cache in cache:
        if hasattr(layer_cache, "to_quantized"):
            quantized_cache.append(
                layer_cache.to_quantized(group_size=group_size, bits=bits)
            )
        else:
            quantized_cache.append(QuantizedKVCache(group_size=group_size, bits=bits))
    return quantized_cache


def cache_type_name(cache: list[Any]) -> str:
    """Return the cache class name for prefix-cache compatibility checks."""
    if not cache:
        return ""
    return type(cache[0]).__name__
