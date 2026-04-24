"""Tests for the disk-tier prefix cache."""
import json
import time
from pathlib import Path

import pytest
import mlx.core as mx

from mlxz.prefix_cache.disk import PrefixCacheDisk
from mlxz.exceptions import PrefixCacheCorruption


def _make_mock_kv_cache(n_layers=2, seq_len=4, head_dim=8):
    class MockKVCache:
        def __init__(self):
            self._keys = mx.zeros((1, 1, seq_len, head_dim))
            self._values = mx.ones((1, 1, seq_len, head_dim))

        @property
        def state(self):
            return (self._keys, self._values)

    return [MockKVCache() for _ in range(n_layers)]


def _make_hashes(n):
    return tuple(bytes([i] * 32) for i in range(n))


@pytest.mark.metal
class TestPrefixCacheDisk:
    def test_store_and_lookup(self, tmp_path):
        cache = PrefixCacheDisk(tmp_path, disk_budget_bytes=100_000_000, model_hash="test")
        hashes = _make_hashes(2)
        kv = _make_mock_kv_cache()
        cache.store_sync(hashes, kv, n_tokens=4)

        n_matched, kv_states, cache_type = cache.lookup_sync(hashes)
        assert n_matched == 4
        assert kv_states is not None
        assert cache_type is not None
        assert len(kv_states) == 2  # 2 layers

    def test_miss_returns_zero_none(self, tmp_path):
        cache = PrefixCacheDisk(tmp_path, disk_budget_bytes=100_000_000, model_hash="test")
        n, kv, cache_type = cache.lookup_sync(_make_hashes(3))
        assert n == 0
        assert kv is None
        assert cache_type is None

    def test_checksum_validation(self, tmp_path):
        cache = PrefixCacheDisk(tmp_path, disk_budget_bytes=100_000_000, model_hash="test")
        hashes = _make_hashes(1)
        cache.store_sync(hashes, _make_mock_kv_cache(), n_tokens=4)

        # Corrupt the metadata checksum
        entry_path = cache._entry_path(hashes)
        meta_path = cache._meta_path(entry_path)
        meta = json.loads(meta_path.read_text())
        meta["checksum"] = "0" * 64
        meta_path.write_text(json.dumps(meta))

        # Lookup should detect corruption and return miss
        n, kv, cache_type = cache.lookup_sync(hashes)
        assert n == 0
        assert kv is None
        assert cache_type is None
        # Corrupt file should be removed
        assert not entry_path.exists()

    def test_graceful_io_failure(self, tmp_path):
        # Non-writable path
        cache = PrefixCacheDisk(Path("/nonexistent/path"), disk_budget_bytes=100_000_000, model_hash="test")
        # Should not crash
        cache.store_sync(_make_hashes(1), _make_mock_kv_cache(), n_tokens=4)
        n, kv, cache_type = cache.lookup_sync(_make_hashes(1))
        assert n == 0
        assert cache_type is None

    def test_duplicate_store_is_noop(self, tmp_path):
        cache = PrefixCacheDisk(tmp_path, disk_budget_bytes=100_000_000, model_hash="test")
        hashes = _make_hashes(1)
        cache.store_sync(hashes, _make_mock_kv_cache(), n_tokens=4)
        cache.store_sync(hashes, _make_mock_kv_cache(), n_tokens=4)  # noop
        # Only one file
        files = list((tmp_path / "test").glob("*.safetensors"))
        assert len(files) == 1

    def test_lru_eviction_by_mtime(self, tmp_path):
        kv = _make_mock_kv_cache()
        entry_size = sum(arr.nbytes for layer in kv for arr in layer.state if isinstance(arr, mx.array))

        cache = PrefixCacheDisk(tmp_path, disk_budget_bytes=entry_size * 2 + 1000, model_hash="test")

        h1 = (bytes([1] * 32),)
        h2 = (bytes([2] * 32),)
        h3 = (bytes([3] * 32),)

        cache.store_sync(h1, _make_mock_kv_cache(), n_tokens=4)
        time.sleep(0.1)
        cache.store_sync(h2, _make_mock_kv_cache(), n_tokens=4)
        time.sleep(0.1)
        cache.store_sync(h3, _make_mock_kv_cache(), n_tokens=4)

        # h1 should have been evicted (oldest mtime)
        files = list((tmp_path / "test").glob("*.safetensors"))
        assert len(files) <= 2

    def test_model_hash_isolation(self, tmp_path):
        cache_a = PrefixCacheDisk(tmp_path, 100_000_000, model_hash="model_a")
        cache_b = PrefixCacheDisk(tmp_path, 100_000_000, model_hash="model_b")

        hashes = _make_hashes(1)
        cache_a.store_sync(hashes, _make_mock_kv_cache(), n_tokens=4)

        # model_b should not find model_a's cache
        n, _, _ = cache_b.lookup_sync(hashes)
        assert n == 0

        # model_a should find it
        n, _, _ = cache_a.lookup_sync(hashes)
        assert n == 4
