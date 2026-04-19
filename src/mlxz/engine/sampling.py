"""Sampling pipeline for logit-to-token conversion."""
from __future__ import annotations

import mlx.core as mx

from mlxz.types import SamplingParams


def sample(
    logits: mx.array,
    params: SamplingParams,
    key: mx.array | None = None,
) -> tuple[int, float | None]:
    """Full sampling pipeline. Returns (token_id, logprob_or_none).

    Pipeline order (when temperature > 0):
      1. Temperature scaling
      2. Top-k filtering
      3. Min-p filtering
      4. Top-p (nucleus) filtering
      5. Categorical sampling

    When temperature == 0, returns argmax (greedy).
    """
    # Ensure 1D
    if logits.ndim == 2:
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits[0, -1]

    # Greedy
    if params.temperature == 0.0:
        token_id = mx.argmax(logits).item()
        logprob = _logprob_for_token(logits, token_id) if params.return_logprob else None
        return token_id, logprob

    # Save pre-filter logits for accurate logprob computation
    raw_logits = logits / params.temperature

    # Apply filters to a working copy
    filtered = raw_logits

    # Top-k
    if params.top_k > 0:
        filtered = _apply_top_k(filtered, params.top_k)

    # Min-p
    if params.min_p > 0.0:
        filtered = _apply_min_p(filtered, params.min_p)

    # Top-p
    if params.top_p < 1.0:
        filtered = _apply_top_p(filtered, params.top_p)

    # Sample
    if key is not None:
        token_id = mx.random.categorical(filtered[None, :], key=key).item()
    else:
        token_id = mx.random.categorical(filtered[None, :]).item()

    # Logprob from pre-filter distribution (OpenAI-compatible)
    logprob = _logprob_for_token(raw_logits, token_id) if params.return_logprob else None
    return token_id, logprob


def _apply_top_k(logits: mx.array, k: int) -> mx.array:
    """Keep only the top-k logits, set rest to -inf."""
    if k >= logits.shape[0]:
        return logits
    # Get the k-th largest value as threshold
    # mx.topk returns k largest values in ascending order, so [0] is the k-th largest
    top_k_values = mx.topk(logits, k=k)
    threshold = top_k_values[0]
    return mx.where(logits >= threshold, logits, mx.array(float("-inf")))


def _apply_min_p(logits: mx.array, min_p: float) -> mx.array:
    """Filter tokens with probability < min_p * max_probability."""
    probs = mx.softmax(logits)
    max_prob = mx.max(probs)
    threshold = min_p * max_prob
    return mx.where(probs >= threshold, logits, mx.array(float("-inf")))


def _apply_top_p(logits: mx.array, p: float) -> mx.array:
    """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p."""
    probs = mx.softmax(logits)
    sorted_indices = mx.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = mx.cumsum(sorted_probs)
    # Keep tokens where cumulative prob hasn't exceeded p yet (plus the crossing token)
    sorted_mask = (cumulative_probs - sorted_probs) < p
    # Create inverse permutation to map mask back to original order
    inv_indices = mx.argsort(sorted_indices)
    original_mask = sorted_mask[inv_indices]
    return mx.where(original_mask, logits, mx.array(float("-inf")))


def _logprob_for_token(logits: mx.array, token_id: int) -> float:
    """Compute log probability for a specific token."""
    log_probs = logits - mx.logsumexp(logits)  # log_softmax
    return log_probs[token_id].item()
