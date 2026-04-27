"""Disk-tier prefix cache with safetensors serialization and SHA-256 integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import structlog

from mlxz.exceptions import PrefixCacheCorruption
from mlxz.types import PrefixCacheStats

logger = structlog.get_logger()


class PrefixCacheDisk:
    """Disk-backed prefix cache with integrity validation.

    Stores KV tensors using mx.save_safetensors with SHA-256 checksums
    in metadata. LRU eviction by file modification time. Model-hash
    directory isolation prevents cross-model contamination.

    Gracefully degrades on I/O errors — returns miss, does not crash.
    """

    def __init__(
        self,
        disk_path: Path,
        disk_budget_bytes: int,
        model_hash: str,
    ) -> None:
        self._root = disk_path / model_hash
        self._budget = disk_budget_bytes
        self._stats = PrefixCacheStats()
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("prefix_cache_disk_init_failed", error=str(e))

    def _entry_path(self, token_hashes: tuple[bytes, ...]) -> Path:
        """Derive a stable filename from a hash tuple."""
        key_hash = hashlib.sha256(b"".join(token_hashes)).hexdigest()[:32]
        return self._root / f"{key_hash}.safetensors"

    def _meta_path(self, entry_path: Path) -> Path:
        """Sidecar metadata file."""
        return entry_path.with_suffix(".meta.json")

    # -- Sync interface --

    def lookup_sync(
        self,
        token_hashes: tuple[bytes, ...],
        cache_type: str | None = None,
    ) -> tuple[int, list[Any] | None, str | None]:
        """Check disk for longest matching prefix."""
        for length in range(len(token_hashes), 0, -1):
            prefix_key = token_hashes[:length]
            entry_path = self._entry_path(prefix_key)
            meta_path = self._meta_path(entry_path)
            if entry_path.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if (
                        cache_type is not None
                        and str(meta.get("cache_type", "KVCache")) != cache_type
                    ):
                        continue
                    kv_states, n_tokens = self._load_entry(entry_path, meta_path)
                    entry_path.touch()  # update mtime for LRU
                    meta_path.touch()
                    self._stats.hits += 1
                    self._stats.hit_bytes += entry_path.stat().st_size
                    logger.debug("prefix_cache_disk_hit", matched_tokens=n_tokens)
                    return n_tokens, kv_states, str(meta.get("cache_type", "KVCache"))
                except (PrefixCacheCorruption, OSError, Exception) as e:
                    logger.warning(
                        "prefix_cache_disk_load_failed", path=str(entry_path), error=str(e)
                    )
                    # Remove corrupt entry
                    entry_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
        self._stats.misses += 1
        return 0, None, None

    def store_sync(
        self,
        token_hashes: tuple[bytes, ...],
        kv_cache_layers: list[Any],
        n_tokens: int | None = None,
    ) -> None:
        """Serialize KV state to disk with checksum validation."""
        entry_path = self._entry_path(token_hashes)
        if entry_path.exists():
            return  # already on disk

        try:
            # Extract states
            kv_states = []
            cache_type = type(kv_cache_layers[0]).__name__
            for layer_cache in kv_cache_layers:
                state = layer_cache.state
                kv_states.append(state)

            if n_tokens is None:
                first_state = kv_states[0]
                if isinstance(first_state, tuple) and isinstance(first_state[0], mx.array):
                    n_tokens = first_state[0].shape[2] if first_state[0].ndim >= 3 else 0
                else:
                    n_tokens = 0

            # Flatten tensors for safetensors
            tensors = {}
            for layer_idx, state in enumerate(kv_states):
                if isinstance(state, tuple):
                    for t_idx, arr in enumerate(state):
                        if isinstance(arr, mx.array):
                            tensors[f"l{layer_idx}_t{t_idx}"] = arr
                        elif isinstance(arr, tuple):
                            for q_idx, qarr in enumerate(arr):
                                if isinstance(qarr, mx.array):
                                    tensors[f"l{layer_idx}_t{t_idx}_q{q_idx}"] = qarr

            # Compute checksum
            checksum = hashlib.sha256()
            for name in sorted(tensors):
                checksum.update(np.array(tensors[name], copy=False).tobytes())

            # Evict if needed
            needed = sum(t.nbytes for t in tensors.values())
            self._evict_until_room(needed)

            # Save tensors
            mx.save_safetensors(str(entry_path), tensors)

            # Save metadata sidecar
            meta = {
                "n_tokens": n_tokens,
                "n_layers": len(kv_states),
                "cache_type": cache_type,
                "checksum": checksum.hexdigest(),
                "format_version": "1",
                "tensor_keys": list(sorted(tensors.keys())),
            }
            self._meta_path(entry_path).write_text(json.dumps(meta))
            self._stats.disk_used_bytes += needed

            logger.debug("prefix_cache_disk_stored", n_tokens=n_tokens, path=str(entry_path))

        except (OSError, RuntimeError) as e:
            logger.warning("prefix_cache_disk_store_failed", error=str(e))

    def _load_entry(self, entry_path: Path, meta_path: Path) -> tuple[list[Any], int]:
        """Load and validate a cached prefix entry."""
        meta = json.loads(meta_path.read_text())

        # Load tensors
        tensors = dict(mx.load(str(entry_path)))

        # Validate checksum
        checksum = hashlib.sha256()
        for name in sorted(tensors):
            checksum.update(np.array(tensors[name], copy=False).tobytes())
        if checksum.hexdigest() != meta.get("checksum", ""):
            raise PrefixCacheCorruption(f"Checksum mismatch for {entry_path}")

        # Reconstruct kv_states
        n_layers = int(meta["n_layers"])
        kv_states: list[Any] = []
        for layer_idx in range(n_layers):
            keys_name = f"l{layer_idx}_t0"
            values_name = f"l{layer_idx}_t1"
            if keys_name in tensors and values_name in tensors:
                kv_states.append((tensors[keys_name], tensors[values_name]))
            else:
                # Quantized format
                layer_tensors = {k: v for k, v in tensors.items() if k.startswith(f"l{layer_idx}_")}
                kv_states.append(tuple(layer_tensors.values()))

        return kv_states, int(meta["n_tokens"])

    def _evict_until_room(self, needed_bytes: int) -> None:
        """Evict oldest entries by mtime until budget allows needed_bytes."""
        try:
            entries = sorted(self._root.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
            current_usage = sum(p.stat().st_size for p in entries)
            for entry in entries:
                if current_usage + needed_bytes <= self._budget:
                    break
                size = entry.stat().st_size
                entry.unlink(missing_ok=True)
                self._meta_path(entry).unlink(missing_ok=True)
                current_usage -= size
                self._stats.evictions += 1
        except OSError:
            pass  # graceful degradation

    # -- Async interface --

    async def lookup(self, token_hashes: list[bytes]) -> tuple[int, Any | None, str | None]:
        return self.lookup_sync(tuple(token_hashes))

    async def store(self, token_hashes: list[bytes], kv: Any) -> None:
        self.store_sync(tuple(token_hashes), kv)

    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            hit_bytes=self._stats.hit_bytes,
            evictions=self._stats.evictions,
            memory_used_bytes=0,
            disk_used_bytes=self._stats.disk_used_bytes,
        )
