"""Tests for engine request lifecycle and stop checking."""
import pytest
from mlxz.types import RequestState, SamplingParams
from mlxz.engine.request import Request, Token, StopChecker


class TestToken:
    def test_token_fields(self):
        t = Token(token_id=42, text="hello", logprob=-0.5)
        assert t.token_id == 42
        assert t.text == "hello"
        assert t.logprob == -0.5

    def test_token_optional_logprob(self):
        t = Token(token_id=1, text="a")
        assert t.logprob is None


class TestStopChecker:
    def test_single_token_match(self):
        sc = StopChecker(["stop"])
        hit, seq = sc.check("this should stop here")
        assert hit is True
        assert seq == "stop"

    def test_no_match(self):
        sc = StopChecker(["xyz"])
        hit, _ = sc.check("nothing matches")
        assert hit is False

    def test_boundary_crossing(self):
        """Stop sequence split across multiple tokens."""
        sc = StopChecker(["hello"])
        hit, _ = sc.check("hel")
        assert hit is False
        hit, seq = sc.check("lo world")
        assert hit is True
        assert seq == "hello"

    def test_multiple_sequences(self):
        sc = StopChecker(["<|end|>", "<|stop|>"])
        hit, _ = sc.check("some text")
        assert hit is False
        hit, seq = sc.check(" more <|stop|> here")
        assert hit is True
        assert seq == "<|stop|>"

    def test_empty_sequences(self):
        sc = StopChecker([])
        hit, _ = sc.check("anything")
        assert hit is False

    def test_reset(self):
        sc = StopChecker(["stop"])
        sc.check("sto")
        sc.reset()
        hit, _ = sc.check("p")  # "p" alone shouldn't match "stop"
        assert hit is False


class TestRequestTransitions:
    def _make_request(self, state=RequestState.QUEUED):
        req = Request.create(
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
            sampling=SamplingParams(),
        )
        req.state = state
        return req

    def test_queued_to_admitted(self):
        req = self._make_request(RequestState.QUEUED)
        req.transition(RequestState.ADMITTED)
        assert req.state == RequestState.ADMITTED

    def test_queued_to_rejected(self):
        req = self._make_request(RequestState.QUEUED)
        req.transition(RequestState.REJECTED)
        assert req.state == RequestState.REJECTED

    def test_admitted_to_prefilling(self):
        req = self._make_request(RequestState.ADMITTED)
        req.transition(RequestState.PREFILLING)
        assert req.state == RequestState.PREFILLING

    def test_decoding_to_completed(self):
        req = self._make_request(RequestState.DECODING)
        req.transition(RequestState.COMPLETED)
        assert req.state == RequestState.COMPLETED

    def test_invalid_transition_raises(self):
        req = self._make_request(RequestState.QUEUED)
        with pytest.raises(ValueError, match="Invalid state transition"):
            req.transition(RequestState.COMPLETED)

    def test_completed_is_terminal(self):
        req = self._make_request(RequestState.COMPLETED)
        with pytest.raises(ValueError):
            req.transition(RequestState.DECODING)

    def test_cancellation_from_any_active_state(self):
        for state in [RequestState.ADMITTED, RequestState.PREFILLING, RequestState.DECODING]:
            req = self._make_request(state)
            req.transition(RequestState.CANCELLED)
            assert req.state == RequestState.CANCELLED


class TestRequestFactory:
    def test_create_sets_defaults(self):
        req = Request.create(
            prompt_tokens=[1, 2, 3, 4, 5],
            max_tokens=100,
            sampling=SamplingParams(temperature=0.7),
        )
        assert req.state == RequestState.QUEUED
        assert req.prompt_token_count == 5
        assert req.completion_token_count == 0
        assert req.finish_reason is None
        assert req.id  # non-empty UUID
        assert req.output_channel is not None

    def test_create_with_stop_sequences(self):
        req = Request.create(
            prompt_tokens=[1],
            max_tokens=10,
            sampling=SamplingParams(),
            stop_sequences=["<end>"],
        )
        assert req.stop_sequences == ["<end>"]
        assert req._stop_checker is not None
