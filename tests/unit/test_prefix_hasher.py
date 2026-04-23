"""Tests for the rolling prefix hasher."""
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlxz.prefix_cache.hasher import RollingPrefixHasher


class TestRollingPrefixHasher:
    def test_empty_tokens_returns_empty(self):
        h = RollingPrefixHasher(block_size=256)
        assert h.hash_chunks([]) == ()

    def test_single_token_returns_one_hash(self):
        h = RollingPrefixHasher(block_size=256)
        result = h.hash_chunks([42])
        assert len(result) == 1
        assert isinstance(result[0], bytes)
        assert len(result[0]) == 32  # SHA-256 = 32 bytes

    def test_exact_block_boundary(self):
        h = RollingPrefixHasher(block_size=4)
        result = h.hash_chunks([1, 2, 3, 4])
        assert len(result) == 1

    def test_one_over_block_boundary(self):
        h = RollingPrefixHasher(block_size=4)
        result = h.hash_chunks([1, 2, 3, 4, 5])
        assert len(result) == 2

    def test_two_full_blocks(self):
        h = RollingPrefixHasher(block_size=4)
        result = h.hash_chunks([1, 2, 3, 4, 5, 6, 7, 8])
        assert len(result) == 2

    def test_determinism(self):
        """Same tokens always produce same hashes."""
        h = RollingPrefixHasher(block_size=4)
        tokens = [10, 20, 30, 40, 50]
        r1 = h.hash_chunks(tokens)
        r2 = h.hash_chunks(tokens)
        assert r1 == r2

    def test_distinct_tokens_produce_distinct_hashes(self):
        h = RollingPrefixHasher(block_size=4)
        r1 = h.hash_chunks([1, 2, 3, 4])
        r2 = h.hash_chunks([5, 6, 7, 8])
        assert r1 != r2

    def test_shared_prefix_produces_shared_hash_prefix(self):
        """If two token streams share a prefix, their hash tuples share a prefix."""
        h = RollingPrefixHasher(block_size=4)
        r1 = h.hash_chunks([1, 2, 3, 4, 10, 20, 30, 40])
        r2 = h.hash_chunks([1, 2, 3, 4, 50, 60, 70, 80])
        # First chunk should be identical (same tokens)
        assert r1[0] == r2[0]
        # Second chunk should differ
        assert r1[1] != r2[1]

    def test_result_is_tuple(self):
        """Tuples are hashable -- needed for dict keys in memory cache."""
        h = RollingPrefixHasher(block_size=4)
        result = h.hash_chunks([1, 2, 3])
        assert isinstance(result, tuple)
        # Should be usable as a dict key
        d = {result: "value"}
        assert d[result] == "value"

    def test_block_size_property(self):
        h = RollingPrefixHasher(block_size=128)
        assert h.block_size == 128

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError):
            RollingPrefixHasher(block_size=0)
        with pytest.raises(ValueError):
            RollingPrefixHasher(block_size=-1)

    @given(tokens=st.lists(st.integers(min_value=0, max_value=100000), min_size=0, max_size=2000))
    @settings(max_examples=50)
    def test_hash_count_equals_ceil_division(self, tokens):
        """Property: number of hashes = ceil(len(tokens) / block_size)."""
        bs = 64
        h = RollingPrefixHasher(block_size=bs)
        result = h.hash_chunks(tokens)
        expected = math.ceil(len(tokens) / bs) if tokens else 0
        assert len(result) == expected

    @given(tokens=st.lists(st.integers(min_value=0, max_value=100000), min_size=1, max_size=500))
    @settings(max_examples=30)
    def test_all_hashes_are_32_bytes(self, tokens):
        """Property: every hash is exactly 32 bytes (SHA-256)."""
        h = RollingPrefixHasher(block_size=64)
        for digest in h.hash_chunks(tokens):
            assert len(digest) == 32
