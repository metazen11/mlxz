"""Draft model for speculative token generation."""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import structlog

logger = structlog.get_logger()


class DraftModel:
    """Small draft model that generates speculative tokens quickly.

    Used by SpeculativeEngine to propose tokens that the target model
    then verifies in a single forward pass.
    """

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate_draft(
        self,
        last_token_id: int,
        cache: list,
        num_tokens: int,
    ) -> list[tuple[int, mx.array]]:
        """Generate num_tokens speculative tokens.

        Returns list of (token_id, logits) tuples.
        Each logits array is the full vocabulary distribution from the draft model.
        """
        drafts: list[tuple[int, mx.array]] = []
        token_id = last_token_id

        for _ in range(num_tokens):
            logits = self._model(mx.array([[token_id]]), cache=cache)
            mx.eval(logits)
            logits_1d = logits[0, -1, :]  # (vocab_size,)

            # Greedy draft for simplicity (could use sampling)
            draft_token = mx.argmax(logits_1d).item()
            drafts.append((draft_token, logits_1d))
            token_id = draft_token

        return drafts

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer
