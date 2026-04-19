"""Compiled decode-step helpers for greedy MLX generation."""
from __future__ import annotations

from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def build_compiled_greedy_step(
    model: nn.Module,
    cache: list[Any],
):
    """Compile a greedy decode step that returns the next token id.

    The cache state is captured as compiled outputs so KV mutation stays in
    sync with the model call. This is intended for temperature=0 requests
    where logprobs are not required.
    """
    state = [layer.state for layer in cache]

    @partial(mx.compile, outputs=state)
    def _step(token_ids: mx.array) -> mx.array:
        logits = model(token_ids, cache=cache)
        return mx.argmax(logits[:, -1, :], axis=-1)

    return _step, state


def build_compiled_greedy_chunk(
    model: nn.Module,
    cache: list[Any],
    *,
    chunk_size: int = 16,
):
    """Compile a fixed-length greedy decode chunk.

    This unrolls several greedy steps into one compiled call so the model-side
    decode work amortizes the Python boundary overhead.
    """
    state = [layer.state for layer in cache]

    @partial(mx.compile, outputs=state)
    def _step(token_ids: mx.array) -> mx.array:
        tokens = []
        current = token_ids
        for _ in range(chunk_size):
            logits = model(current, cache=cache)
            current = mx.argmax(logits[:, -1, :], axis=-1)
            tokens.append(current)
            current = current[:, None]
        return mx.concatenate(tokens, axis=0)

    return _step, state
