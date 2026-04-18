"""Block-table-backed KV cache for paged attention."""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxz.paged_attention.block_manager import BlockManager


class PagedKVCache:
    """KV cache backed by a block table.

    Maps logical sequence positions to physical blocks managed by BlockManager.
    Provides an interface compatible with mlx-lm's KVCache for drop-in replacement.

    Each block stores block_size tokens worth of K and V tensors.
    Shape per block: (1, n_kv_heads, block_size, head_dim)
    """

    def __init__(
        self,
        block_manager: BlockManager,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        self._bm = block_manager
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._dtype = dtype
        self._block_size = block_manager.block_size

        # Physical storage: pre-allocate tensor pool
        # Shape: (total_blocks, n_kv_heads, block_size, head_dim)
        self._key_pool = mx.zeros(
            (block_manager.total_blocks, n_kv_heads, self._block_size, head_dim),
            dtype=dtype,
        )
        self._value_pool = mx.zeros(
            (block_manager.total_blocks, n_kv_heads, self._block_size, head_dim),
            dtype=dtype,
        )

        # Per-sequence block tables: sequence_id -> list of block indices
        self._block_tables: dict[str, list[int]] = {}
        # Per-sequence offsets: how many tokens stored
        self._offsets: dict[str, int] = {}

    def allocate_sequence(self, seq_id: str, initial_blocks: int = 1) -> None:
        """Allocate initial blocks for a new sequence."""
        blocks = self._bm.allocate(initial_blocks)
        self._block_tables[seq_id] = blocks
        self._offsets[seq_id] = 0

    def free_sequence(self, seq_id: str) -> None:
        """Free all blocks for a sequence."""
        if seq_id in self._block_tables:
            self._bm.free(self._block_tables.pop(seq_id))
            self._offsets.pop(seq_id, None)

    def append_kv(
        self,
        seq_id: str,
        keys: mx.array,  # (1, n_kv_heads, n_new_tokens, head_dim)
        values: mx.array,  # same shape
    ) -> None:
        """Append new KV entries for a sequence.

        Automatically allocates new blocks as needed.
        """
        n_new = keys.shape[2]
        offset = self._offsets[seq_id]
        blocks = self._block_tables[seq_id]

        for i in range(n_new):
            token_pos = offset + i
            block_idx_in_table = token_pos // self._block_size
            pos_in_block = token_pos % self._block_size

            # Allocate new block if needed
            while block_idx_in_table >= len(blocks):
                new_blocks = self._bm.allocate(1)
                blocks.extend(new_blocks)

            phys_block = blocks[block_idx_in_table]

            # Write K,V into pool at the physical block position
            self._key_pool[phys_block, :, pos_in_block, :] = keys[0, :, i, :]
            self._value_pool[phys_block, :, pos_in_block, :] = values[0, :, i, :]

        self._offsets[seq_id] = offset + n_new

    def get_kv(self, seq_id: str) -> tuple[mx.array, mx.array]:
        """Gather KV for a sequence into contiguous tensors.

        Returns (keys, values) each of shape (1, n_kv_heads, seq_len, head_dim).
        """
        offset = self._offsets[seq_id]
        blocks = self._block_tables[seq_id]

        if offset == 0:
            return (
                mx.zeros(
                    (1, self._n_kv_heads, 0, self._head_dim), dtype=self._dtype
                ),
                mx.zeros(
                    (1, self._n_kv_heads, 0, self._head_dim), dtype=self._dtype
                ),
            )

        # Gather from block pool
        gathered_keys = []
        gathered_values = []

        remaining = offset
        for block_idx in blocks:
            n_tokens_in_block = min(remaining, self._block_size)
            gathered_keys.append(
                self._key_pool[block_idx, :, :n_tokens_in_block, :]
            )
            gathered_values.append(
                self._value_pool[block_idx, :, :n_tokens_in_block, :]
            )
            remaining -= n_tokens_in_block
            if remaining <= 0:
                break

        keys = mx.concatenate(gathered_keys, axis=1)  # along seq dim
        values = mx.concatenate(gathered_values, axis=1)

        # Reshape to (1, n_kv_heads, seq_len, head_dim)
        keys = keys[None, ...]  # add batch dim
        values = values[None, ...]

        return keys, values

    def get_block_table(self, seq_id: str) -> list[int]:
        """Return the block table for a sequence (for attention kernel)."""
        return list(self._block_tables.get(seq_id, []))

    def get_offset(self, seq_id: str) -> int:
        """Return the current offset for a sequence."""
        return self._offsets.get(seq_id, 0)

    @property
    def stats(self) -> dict:
        """Return pool utilization stats."""
        return {
            "total_blocks": self._bm.total_blocks,
            "free_blocks": self._bm.free_blocks,
            "active_sequences": len(self._block_tables),
        }
