"""Runtime KV-cache quantization helpers."""
from __future__ import annotations

from typing import Any

from mlx_lm.models import cache as mlx_cache


def maybe_quantize_kv_cache(
    prompt_cache: list[Any],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int,
) -> None:
    """Convert KV caches to quantized form once a sequence grows large enough.

    Mirrors mlx-lm's generation helper so mlxz can use the same cache
    quantization path when the runtime config enables it.
    """
    if not prompt_cache:
        return
    if kv_bits not in (4, 8):
        return
    if isinstance(prompt_cache[0], mlx_cache.QuantizedKVCache):
        return
    if getattr(prompt_cache[0], "offset", 0) <= quantized_kv_start:
        return

    for idx, layer_cache in enumerate(prompt_cache):
        if isinstance(layer_cache, mlx_cache.KVCache):
            prompt_cache[idx] = layer_cache.to_quantized(
                group_size=kv_group_size,
                bits=kv_bits,
            )
