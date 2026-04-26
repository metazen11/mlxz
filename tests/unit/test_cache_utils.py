"""Tests for cache construction helpers."""
import mlx.core as mx

from mlxz.engine.cache_utils import QuantizedKVCache


def test_quantized_cache_state_restores_offset() -> None:
    cache = QuantizedKVCache(group_size=64, bits=8)

    keys = (
        mx.zeros((1, 1, 8, 1), dtype=mx.uint32),
        mx.zeros((1, 1, 8, 2), dtype=mx.float32),
        mx.zeros((1, 1, 8, 2), dtype=mx.float32),
    )
    values = (
        mx.zeros((1, 1, 8, 1), dtype=mx.uint32),
        mx.zeros((1, 1, 8, 2), dtype=mx.float32),
        mx.zeros((1, 1, 8, 2), dtype=mx.float32),
    )

    cache.state = (keys, values)

    assert cache.offset == 8
    restored_keys, restored_values = cache.state
    assert restored_keys[0].shape[2] == 8
    assert restored_values[0].shape[2] == 8
