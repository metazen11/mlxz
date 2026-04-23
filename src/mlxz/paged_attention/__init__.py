"""Paged attention — block-managed KV cache for memory-efficient serving."""
from mlxz.paged_attention.attention import paged_attention_forward
from mlxz.paged_attention.block_manager import (
    BlockManager,
    DoubleFreeError,
    OutOfBlocksError,
)
from mlxz.paged_attention.paged_kv import PagedKVCache

__all__ = [
    "BlockManager",
    "DoubleFreeError",
    "OutOfBlocksError",
    "PagedKVCache",
    "paged_attention_forward",
]
