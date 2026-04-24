"""MLX attention kernel tuning hooks."""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import mlx.core as mx

_PATCHED_THRESHOLD: int | None = None


def patch_attention_memory_efficient_threshold(threshold: int | None) -> None:
    """Force MLX SDPA to use a configured memory-efficient threshold.

    The model code in ``mlx_lm`` currently calls ``mx.fast.scaled_dot_product_attention``
    without passing the optional threshold. This hook lets the engine
    experiment with a global default at startup without forking model code.
    """
    global _PATCHED_THRESHOLD
    if threshold == _PATCHED_THRESHOLD:
        return

    original = mx.fast.scaled_dot_product_attention

    @wraps(original)
    def wrapped(
        queries: Any,
        keys: Any,
        values: Any,
        scale: float,
        mask: Any | None = None,
        memory_efficient_threshold: int | None = None,
        s: Any = {},
    ) -> Any:
        effective_threshold = (
            threshold
            if memory_efficient_threshold is None
            else memory_efficient_threshold
        )
        return original(
            queries,
            keys,
            values,
            scale,
            mask=mask,
            memory_efficient_threshold=effective_threshold,
            s=s,
        )

    mx.fast.scaled_dot_product_attention = wrapped
    _PATCHED_THRESHOLD = threshold
