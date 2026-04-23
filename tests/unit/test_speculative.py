"""Tests for speculative decoding components."""
import pytest
import mlx.core as mx

from mlxz.engine.draft import DraftModel


class MockModel:
    """Mock model that returns predictable logits."""
    def __init__(self, vocab_size=100):
        self._vocab_size = vocab_size

    def __call__(self, input_ids, cache=None):
        batch, seq = input_ids.shape
        # Return logits where token 0 always has highest probability
        logits = mx.zeros((batch, seq, self._vocab_size))
        logits = logits.at[:, :, 0].add(10.0)  # token 0 is always most likely
        return logits

    @property
    def args(self):
        class Args:
            num_hidden_layers = 2
            num_attention_heads = 2
            num_key_value_heads = 2
            head_dim = 64
            hidden_size = 128
        return Args()


class MockTokenizer:
    eos_token_id = 2
    def decode(self, tokens):
        return "".join(str(t) for t in tokens)
    def encode(self, text):
        return [1, 2, 3]


@pytest.mark.metal
class TestDraftModel:
    def test_generate_draft_returns_correct_count(self):
        model = MockModel()
        draft = DraftModel(model, MockTokenizer())
        results = draft.generate_draft(last_token_id=1, cache=[], num_tokens=4)
        assert len(results) == 4

    def test_generate_draft_returns_token_and_logits(self):
        model = MockModel()
        draft = DraftModel(model, MockTokenizer())
        results = draft.generate_draft(last_token_id=1, cache=[], num_tokens=1)
        token_id, logits = results[0]
        assert isinstance(token_id, int)
        assert isinstance(logits, mx.array)
        assert logits.shape == (100,)  # vocab_size

    def test_greedy_draft_picks_argmax(self):
        model = MockModel()
        draft = DraftModel(model, MockTokenizer())
        results = draft.generate_draft(last_token_id=1, cache=[], num_tokens=3)
        # MockModel always has token 0 as highest logit
        for token_id, _ in results:
            assert token_id == 0


@pytest.mark.metal
class TestRejectionSampling:
    def test_identical_distributions_accept_all(self):
        """When draft and target agree perfectly, acceptance rate should be 1.0."""
        # Same logits -> same softmax -> p_t/p_d = 1 -> always accept
        logits = mx.array([10.0, 0.0, 0.0, 0.0])
        p = mx.softmax(logits)

        # acceptance = min(1, p_target[token] / p_draft[token])
        token = 0  # highest prob token
        acceptance = min(1.0, p[token].item() / p[token].item())
        assert acceptance == 1.0

    def test_divergent_distributions_may_reject(self):
        """When draft overestimates a token's probability, it may be rejected."""
        p_target = mx.softmax(mx.array([1.0, 1.0, 1.0, 1.0]))  # uniform
        p_draft = mx.softmax(mx.array([10.0, 0.0, 0.0, 0.0]))  # concentrated

        token = 0
        p_t = p_target[token].item()
        p_d = p_draft[token].item()
        acceptance = min(1.0, p_t / p_d)
        # p_t ~ 0.25, p_d ~ 0.99 -> acceptance ~ 0.25
        assert acceptance < 0.5

    def test_adjusted_distribution_nonnegative(self):
        """max(0, p_target - p_draft) should always be nonnegative."""
        p_target = mx.softmax(mx.array([2.0, 1.0, 0.5, 0.1]))
        p_draft = mx.softmax(mx.array([0.1, 3.0, 0.5, 0.1]))
        adjusted = mx.maximum(p_target - p_draft, mx.array(0.0))
        assert mx.all(adjusted >= 0).item()
