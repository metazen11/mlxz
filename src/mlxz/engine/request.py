"""Request lifecycle and token delivery types."""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field, replace
from typing import ClassVar

import janus

from mlxz.types import SamplingParams, RequestState
from mlxz.engine.thread_boundary import RequestBridge


@dataclass(frozen=True, slots=True)
class Token:
    """Single decoded token delivered through the output channel."""

    token_id: int
    text: str
    logprob: float | None = None


class StopChecker:
    """Checks if generated text contains any stop sequence.

    Handles sequences that span token boundaries by maintaining
    a sliding window of recent text.
    """

    def __init__(self, stop_sequences: list[str]) -> None:
        self._sequences = stop_sequences
        self._max_len = max(len(s) for s in stop_sequences) if stop_sequences else 0
        self._buffer = ""

    def check(self, new_text: str) -> tuple[bool, str | None]:
        """Append new_text and check for stop sequences.

        Returns (should_stop, matched_sequence_or_none).
        """
        self._buffer += new_text
        for seq in self._sequences:
            if seq in self._buffer:
                return True, seq
        # Keep only the tail needed for boundary detection
        if len(self._buffer) > self._max_len * 2:
            self._buffer = self._buffer[-self._max_len:]
        return False, None

    def reset(self) -> None:
        self._buffer = ""


@dataclass(slots=True)
class Request:
    """Inference request with lifecycle state machine and token delivery channel."""

    id: str
    prompt_tokens: list[int]
    max_tokens: int
    sampling: SamplingParams
    state: RequestState
    output_channel: janus.Queue[Token | None]  # None sentinel = EOS
    created_at: float = field(default_factory=time.monotonic)
    prompt_token_count: int = 0
    completion_token_count: int = 0
    prefix_cache_hit_tokens: int = 0
    ttft_ms: float = 0.0
    decode_tps: float = 0.0
    finish_reason: str | None = None  # "stop" | "length"
    stop_sequences: list[str] = field(default_factory=list)
    _stop_checker: StopChecker | None = field(default=None, repr=False, init=False)

    # Valid state transitions
    _TRANSITIONS: ClassVar[dict[RequestState, set[RequestState]]] = {
        RequestState.QUEUED: {RequestState.ADMITTED, RequestState.REJECTED},
        RequestState.ADMITTED: {RequestState.PREFILLING, RequestState.CANCELLED},
        RequestState.PREFILLING: {RequestState.DECODING, RequestState.CANCELLED},
        RequestState.DECODING: {RequestState.COMPLETED, RequestState.CANCELLED},
        RequestState.COMPLETED: set(),
        RequestState.CANCELLED: set(),
        RequestState.REJECTED: set(),
    }

    def __post_init__(self) -> None:
        self.prompt_token_count = len(self.prompt_tokens)
        if self.stop_sequences:
            self._stop_checker = StopChecker(self.stop_sequences)

    def transition(self, new_state: RequestState) -> None:
        """Validate and apply state transition.

        Raises ValueError on invalid transition.
        """
        if new_state == self.state:
            return
        valid = self._TRANSITIONS.get(self.state, set())
        if new_state not in valid:
            raise ValueError(
                f"Invalid state transition: {self.state.name} -> {new_state.name}. "
                f"Valid targets: {', '.join(s.name for s in valid) or 'none'}"
            )
        self.state = new_state

    @staticmethod
    def create(
        prompt_tokens: list[int],
        max_tokens: int,
        sampling: SamplingParams,
        return_logprob: bool = True,
        stop_sequences: list[str] | None = None,
        channel_depth: int = 64,
    ) -> Request:
        """Factory that creates a Request with a fresh janus channel."""
        sampling = replace(sampling, return_logprob=return_logprob)
        return Request(
            id=str(uuid.uuid4()),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            sampling=sampling,
            state=RequestState.QUEUED,
            output_channel=RequestBridge.create_token_channel(channel_depth),
            stop_sequences=stop_sequences or [],
        )
