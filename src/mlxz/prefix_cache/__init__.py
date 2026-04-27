"""Prefix cache module — content-hashed KV caching for repeated prompts."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def __getattr__(name: str) -> Any:
    if name == "CachedPrefix":
        from mlxz.prefix_cache.base import CachedPrefix

        return CachedPrefix
    if name == "PrefixMatch":
        from mlxz.prefix_cache.base import PrefixMatch

        return PrefixMatch
    if name == "compute_size_bytes":
        from mlxz.prefix_cache.base import compute_size_bytes

        return compute_size_bytes
    if name == "PrefixCacheDisk":
        from mlxz.prefix_cache.disk import PrefixCacheDisk

        return PrefixCacheDisk
    if name == "RollingPrefixHasher":
        from mlxz.prefix_cache.hasher import RollingPrefixHasher

        return RollingPrefixHasher
    if name == "PrefixCacheMemory":
        from mlxz.prefix_cache.memory import PrefixCacheMemory

        return PrefixCacheMemory
    raise AttributeError(name)
