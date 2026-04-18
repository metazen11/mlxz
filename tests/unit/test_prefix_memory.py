"""Tests for the in-memory prefix cache."""
import time
import pytest
import mlx.core as mx

from mlxz.prefix_cache.memory import PrefixCacheMemory


def _make_mock_kv_cache(n_layers: int = 2, seq_len: int = 4, head_dim: int = 8):
    """Create mock KVCache-like objects with .state property."""
    class MockKVCache:
        def __init__(self, seq_len, head_dim):
            self._keys = mx.zeros((1, 1, seq_len, head_dim))
            self._values = mx.ones((1, 1, seq_len, head_dim))
        @property
        def state(self):
            return (self._keys, self._values)
    return [MockKVCache(seq_len, head_dim) for _ in range(n_layers)]


def _make_hashes(n: int) -> tuple[bytes, ...]:
    """Create n distinct hash tuples."""
    return tuple(bytes([i] * 32) for i in range(n))


@pytest.mark.metal
class TestPrefixCacheMemory:
    def test_store_and_lookup(self):
        cache = PrefixCacheMemory(memory_budget_bytes=10_000_000)
        hashes = _make_hashes(2)
        kv = _make_mock_kv_cache()
        cache.store_sync(hashes, kv)

        n_matched, kv_states = cache.lookup_sync(hashes)
        assert n_matched > 0
        assert kv_states is not None

    def test_miss_returns_zero_none(self):
        cache = PrefixCacheMemory(memory_budget_bytes=10_000_000)
        n_matched, kv_states = cache.lookup_sync(_make_hashes(3))
        assert n_matched == 0
        assert kv_states is None

    def test_longest_prefix_matching(self):
        cache = PrefixCacheMemory(memory_budget_bytes=10_000_000)
        # Store a 2-chunk prefix
        short_hashes = _make_hashes(2)
        cache.store_sync(short_hashes, _make_mock_kv_cache())

        # Query with a 4-chunk key that starts with the same 2 chunks
        extra = tuple(bytes([i + 10] * 32) for i in range(2))
        long_hashes = short_hashes + extra

        n_matched, kv_states = cache.lookup_sync(long_hashes)
        assert n_matched > 0  # found the 2-chunk prefix
        assert kv_states is not None

    def test_lru_eviction(self):
        # Budget fits ~1 entry
        kv = _make_mock_kv_cache(n_layers=2, seq_len=4, head_dim=8)
        # Estimate single entry size
        entry_size = sum(
            arr.nbytes for layer in kv
            for arr in layer.state if isinstance(arr, mx.array)
        )

        cache = PrefixCacheMemory(memory_budget_bytes=entry_size + 100)

        h1 = _make_hashes(1)
        h2 = (bytes([99] * 32),)

        cache.store_sync(h1, _make_mock_kv_cache())
        cache.store_sync(h2, _make_mock_kv_cache())

        # h1 should have been evicted (LRU)
        n, _ = cache.lookup_sync(h1)
        assert n == 0
        # h2 should still exist
        n, _ = cache.lookup_sync(h2)
        assert n > 0

    def test_stats_tracking(self):
        cache = PrefixCacheMemory(memory_budget_bytes=10_000_000)

        # Miss
        cache.lookup_sync(_make_hashes(1))
        stats = cache.stats()
        assert stats.misses == 1
        assert stats.hits == 0

        # Store and hit
        hashes = _make_hashes(2)
        cache.store_sync(hashes, _make_mock_kv_cache())
        cache.lookup_sync(hashes)
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_deep_copy_independence(self):
        """Stored KV states should be independent of the original."""
        cache = PrefixCacheMemory(memory_budget_bytes=10_000_000)
        kv = _make_mock_kv_cache(n_layers=1, seq_len=2, head_dim=4)
        hashes = _make_hashes(1)
        cache.store_sync(hashes, kv)

        # Mutate original
        kv[0]._keys = mx.ones_like(kv[0]._keys) * 999

        # Cached version should be unchanged
        _, cached = cache.lookup_sync(hashes)
        assert cached is not None
        # Original was zeros, should still be zeros in cache
        assert mx.allclose(cached[0][0], mx.zeros((1, 1, 2, 4))).item()

    def test_duplicate_store_is_noop(self):
        cache = PrefixCacheMemory(memory_budget_bytes=10_000_000)
        hashes = _make_hashes(1)
        cache.store_sync(hashes, _make_mock_kv_cache())
        cache.store_sync(hashes, _make_mock_kv_cache())  # should not double-count
        stats = cache.stats()
        assert stats.memory_used_bytes > 0
        # Only one entry
        assert len(cache._entries) == 1

    def test_entry_too_large_for_budget(self):
        cache = PrefixCacheMemory(memory_budget_bytes=1)  # tiny budget
        hashes = _make_hashes(1)
        cache.store_sync(hashes, _make_mock_kv_cache())
        # Should not crash, just not store
        assert len(cache._entries) == 0
