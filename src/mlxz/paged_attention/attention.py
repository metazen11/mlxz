"""Paged attention: gather from block table, call SDPA."""
from __future__ import annotations

import mlx.core as mx

from mlxz.paged_attention.paged_kv import PagedKVCache


def paged_attention_forward(
    query: mx.array,  # (1, n_heads, n_new_tokens, head_dim)
    paged_cache: PagedKVCache,
    seq_id: str,
    scale: float | None = None,
) -> mx.array:
    """Compute attention with paged KV cache.

    1. Gather KV from block table into contiguous tensors
    2. Call mx.fast.scaled_dot_product_attention
    3. Return attention output

    This is the Phase 3 approach: gather-then-SDPA.
    Phase 6 stretch goal replaces this with a fused paged-attention kernel.
    """
    keys, values = paged_cache.get_kv(seq_id)

    if scale is None:
        head_dim = query.shape[-1]
        scale = head_dim**-0.5

    # Handle GQA: if n_kv_heads < n_heads, repeat KV
    n_heads = query.shape[1]
    n_kv_heads = keys.shape[1]
    if n_kv_heads < n_heads:
        repeat_factor = n_heads // n_kv_heads
        keys = mx.repeat(keys, repeat_factor, axis=1)
        values = mx.repeat(values, repeat_factor, axis=1)

    # Use mx.fast.scaled_dot_product_attention if available
    try:
        output = mx.fast.scaled_dot_product_attention(
            query, keys, values, scale=scale
        )
    except AttributeError:
        # Fallback: manual attention
        attn_weights = (query @ keys.transpose(0, 1, 3, 2)) * scale
        attn_weights = mx.softmax(attn_weights, axis=-1)
        output = attn_weights @ values

    return output
