"""Prefix cache module — content-hashed KV caching for repeated prompts."""

from mlxz.prefix_cache.base import CachedPrefix, PrefixMatch, compute_size_bytes
from mlxz.prefix_cache.disk import PrefixCacheDisk
from mlxz.prefix_cache.hasher import RollingPrefixHasher
from mlxz.prefix_cache.memory import PrefixCacheMemory

__all__ = [
    "CachedPrefix",
    "PrefixCacheDisk",
    "PrefixMatch",
    "PrefixCacheMemory",
    "RollingPrefixHasher",
    "compute_size_bytes",
]
