"""Tests for the sampling pipeline."""
import pytest
import mlx.core as mx

from mlxz.types import SamplingParams
from mlxz.engine.sampling import sample, _apply_top_k, _apply_min_p, _apply_top_p


@pytest.mark.metal
class TestGreedySampling:
    def test_greedy_returns_argmax(self):
        logits = mx.array([1.0, 5.0, 3.0, 2.0])
        token_id, logprob = sample(logits, SamplingParams(temperature=0.0))
        assert token_id == 1  # index of max value

    def test_greedy_with_2d_logits(self):
        logits = mx.array([[1.0, 5.0, 3.0]])
        token_id, _ = sample(logits, SamplingParams(temperature=0.0))
        assert token_id == 1

    def test_greedy_returns_logprob(self):
        logits = mx.array([1.0, 5.0, 3.0])
        _, logprob = sample(logits, SamplingParams(temperature=0.0))
        assert logprob is not None
        assert logprob < 0  # log probabilities are negative


@pytest.mark.metal
class TestTopK:
    def test_top_k_filters(self):
        logits = mx.array([1.0, 5.0, 3.0, 2.0, 4.0])
        filtered = _apply_top_k(logits, k=2)
        # Only indices 1 (5.0) and 4 (4.0) should be kept
        assert filtered[0].item() == float("-inf")
        assert filtered[2].item() == float("-inf")
        assert filtered[3].item() == float("-inf")
        assert filtered[1].item() == 5.0
        assert filtered[4].item() == 4.0

    def test_top_k_1_is_greedy(self):
        logits = mx.array([1.0, 5.0, 3.0])
        token_id, _ = sample(logits, SamplingParams(temperature=1.0, top_k=1))
        assert token_id == 1  # forced to pick the max


@pytest.mark.metal
class TestTopP:
    def test_top_p_concentrates(self):
        # One dominant token — top_p=0.1 should keep only it
        logits = mx.array([0.0, 10.0, 0.0, 0.0])
        filtered = _apply_top_p(logits, p=0.1)
        # Index 1 should be kept, others -inf
        assert filtered[1].item() == 10.0
        # Most others should be -inf
        ninf_count = sum(1 for i in range(4) if i != 1 and filtered[i].item() == float("-inf"))
        assert ninf_count >= 2


@pytest.mark.metal
class TestMinP:
    def test_min_p_filters_low_prob(self):
        # Token 0 dominates, token 3 is very low
        logits = mx.array([10.0, 5.0, 3.0, -5.0])
        filtered = _apply_min_p(logits, min_p=0.1)
        # Token 3 with prob << 0.1 * max_prob should be filtered
        assert filtered[3].item() == float("-inf")
        assert filtered[0].item() == 10.0


@pytest.mark.metal
class TestDeterminism:
    def test_same_seed_same_result(self):
        logits = mx.array([1.0, 1.0, 1.0, 1.0])  # uniform
        params = SamplingParams(temperature=1.0, seed=42)
        key1 = mx.random.key(42)
        key2 = mx.random.key(42)
        t1, _ = sample(logits, params, key=key1)
        t2, _ = sample(logits, params, key=key2)
        assert t1 == t2

    def test_different_seed_different_result(self):
        """With enough trials, different seeds should produce different tokens."""
        logits = mx.array([1.0, 1.0, 1.0, 1.0])
        params = SamplingParams(temperature=1.0)
        results = set()
        for seed in range(20):
            key = mx.random.key(seed)
            t, _ = sample(logits, params, key=key)
            results.add(t)
        assert len(results) > 1  # at least 2 different tokens
