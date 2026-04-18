"""Model loading via mlx-lm with stable interface isolation."""
from __future__ import annotations

import structlog

import mlx.core as mx
import mlx.nn as nn
import mlx_lm

logger = structlog.get_logger()


class ModelStore:
    """Loads models via mlx_lm.load(), isolating the mlx-lm API surface.

    If mlx-lm changes its load API, only this file needs updating.
    """

    def load(
        self,
        model_path: str,
        tokenizer_config: dict | None = None,
    ) -> tuple[nn.Module, object, int]:
        """Load a model and tokenizer from HuggingFace or local path.

        Args:
            model_path: HuggingFace repo ID or local directory path.
            tokenizer_config: Optional tokenizer configuration overrides.

        Returns:
            Tuple of (model, tokenizer, weight_bytes).
            weight_bytes is the total size of all model parameters.
        """
        logger.info("loading_model", path=model_path)

        model, tokenizer = mlx_lm.load(
            model_path,
            tokenizer_config=tokenizer_config or {},
        )

        # Calculate total weight size
        from mlx.utils import tree_flatten
        weight_bytes = sum(
            p.nbytes for _, p in tree_flatten(model.parameters())
        )

        logger.info(
            "model_loaded",
            path=model_path,
            weight_bytes=weight_bytes,
            weight_gb=round(weight_bytes / (1024**3), 2),
        )

        return model, tokenizer, weight_bytes
