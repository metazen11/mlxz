"""Cache construction helpers for request-aware KV policy selection."""

from __future__ import annotations

from typing import Any

from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

from mlxz.config import KVConfig


class RestoringQuantizedKVCache(QuantizedKVCache):
    """Quantized cache variant that keeps ``offset`` aligned with restored state."""

    @QuantizedKVCache.state.setter
    def state(self, v: tuple[Any, Any]) -> None:  # type: ignore[override]
        self.keys, self.values = v
        self.offset = _state_token_count(v)


def should_quantize_cache(
    kv_config: KVConfig,
    total_requested_tokens: int,
) -> bool:
    """Return ``True`` when the request should start on quantized KV."""
    return total_requested_tokens >= kv_config.quantized_kv_start


def build_prompt_cache(
    model: Any,
    *,
    quantized: bool = False,
    group_size: int = 64,
    bits: int = 8,
    max_kv_size: int | None = None,
) -> list[Any]:
    """Construct a prompt cache, quantizing it for long requests when requested."""
    cache = make_prompt_cache(model, max_kv_size=max_kv_size)
    if not quantized:
        return cache
    return [_quantize_layer(layer, group_size=group_size, bits=bits) for layer in cache]


def cache_type_name(cache_layers: list[Any]) -> str:
    """Return the cache type name for the first layer, or an empty string."""
    if not cache_layers:
        return ""
    first = cache_layers[0]
    if isinstance(first, QuantizedKVCache):
        return "QuantizedKVCache"
    return type(first).__name__


def _quantize_layer(
    layer_cache: Any,
    *,
    group_size: int,
    bits: int,
) -> Any:
    """Convert a cache layer to a quantized cache while preserving its state."""
    if isinstance(layer_cache, RestoringQuantizedKVCache):
        return layer_cache

    if isinstance(layer_cache, QuantizedKVCache):
        quantized = RestoringQuantizedKVCache(
            group_size=group_size,
            bits=bits,
        )
        quantized.keys = layer_cache.keys
        quantized.values = layer_cache.values
        quantized.offset = getattr(layer_cache, "offset", 0)
        quantized.step = getattr(layer_cache, "step", quantized.step)
        return quantized

    if isinstance(layer_cache, KVCache):
        quantized = layer_cache.to_quantized(
            group_size=group_size,
            bits=bits,
        )
        if isinstance(quantized, QuantizedKVCache):
            wrapped = RestoringQuantizedKVCache(
                group_size=quantized.group_size,
                bits=quantized.bits,
            )
            wrapped.keys = quantized.keys
            wrapped.values = quantized.values
            wrapped.offset = getattr(quantized, "offset", 0)
            wrapped.step = getattr(quantized, "step", wrapped.step)
            return wrapped
        return quantized

    return layer_cache


def _state_token_count(state: tuple[Any, Any]) -> int:
    """Infer token count from a cache state tuple."""
    if not state:
        return 0
    first = state[0]
    if hasattr(first, "shape"):
        shape = first.shape
        if len(shape) >= 3:
            return int(shape[2])
        if len(shape) >= 1:
            return int(shape[0])
        return 0
    if isinstance(first, tuple) and first:
        inner = first[0]
        if hasattr(inner, "shape"):
            shape = inner.shape
            if len(shape) >= 3:
                return int(shape[2])
            if len(shape) >= 1:
                return int(shape[0])
    return 0
