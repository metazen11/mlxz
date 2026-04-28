"""Unit tests for cache construction helpers."""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from mlxz.config import KVConfig
from mlxz.engine.cache_utils import (
    RestoringQuantizedKVCache,
    build_prompt_cache,
    cache_type_name,
    should_quantize_cache,
)


class _DummyModel:
    def __init__(self, n_layers: int = 2) -> None:
        self.layers = [object() for _ in range(n_layers)]


def test_should_quantize_cache_uses_total_requested_tokens() -> None:
    cfg = KVConfig(quantized_kv_start=256)
    assert not should_quantize_cache(cfg, 255)
    assert should_quantize_cache(cfg, 256)


def test_build_prompt_cache_keeps_short_requests_on_standard_kv() -> None:
    cache = build_prompt_cache(
        _DummyModel(),
        quantized=False,
    )
    assert len(cache) == 2
    assert isinstance(cache[0], KVCache)
    assert not isinstance(cache[0], QuantizedKVCache)


def test_build_prompt_cache_quantizes_long_requests() -> None:
    cache = build_prompt_cache(
        _DummyModel(),
        quantized=True,
        group_size=64,
        bits=8,
    )
    assert len(cache) == 2
    assert isinstance(cache[0], RestoringQuantizedKVCache)
    assert cache_type_name(cache) == "QuantizedKVCache"


def test_restoring_quantized_cache_state_restores_offset() -> None:
    cache = RestoringQuantizedKVCache(group_size=64, bits=8)
    keys = mx.zeros((1, 2, 3, 4), dtype=mx.float16)
    values = mx.zeros((1, 2, 3, 4), dtype=mx.float16)

    cache.state = (keys, values)

    assert cache.offset == 3
    assert cache.state[0].shape[2] == 3
