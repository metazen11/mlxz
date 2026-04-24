"""MLX attention kernel tuning hooks."""
from __future__ import annotations

import structlog

logger = structlog.get_logger()

def patch_attention_memory_efficient_threshold(threshold: int | None) -> None:
    """Force MLX SDPA to use a configured memory-efficient threshold.

    The model code in ``mlx_lm`` currently calls ``mx.fast.scaled_dot_product_attention``
    without passing the optional threshold. This hook lets the engine
    experiment with a global default at startup without forking model code.
    """
    if threshold is None:
        return

    # MLX's fast SDPA primitive does not expose a memory_efficient_threshold
    # parameter in this runtime. Keep the hook as a no-op so config wiring
    # remains stable, but do not patch the kernel call.
    logger.warning(
        "attention_threshold_unsupported",
        threshold=threshold,
        kernel="mx.fast.scaled_dot_product_attention",
    )
