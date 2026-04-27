"""Chunked prefill scheduler (Sarathi-Serve style).

Splits long prefill requests into chunks of configurable size,
allowing decode steps to interleave. This prevents head-of-line
blocking where a long prefill starves decoding requests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrefillChunk:
    """A chunk of tokens from a prefill request."""

    request_id: str
    tokens: list[int]  # token IDs for this chunk
    start_pos: int  # position in the full prompt
    is_last: bool  # True if this is the final chunk
    total_prompt_tokens: int  # total prompt length for reference


class ChunkedPrefillScheduler:
    """Splits prefill requests into chunks for interleaved execution.

    When a request has more prompt tokens than chunk_size, it is split
    into multiple chunks. Each iteration processes one chunk per request,
    allowing decode steps to run between chunks.

    This prevents a 32K-token prefill from blocking all decoding for
    several seconds.
    """

    def __init__(self, chunk_size: int = 128) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._chunk_size = chunk_size
        # Track per-request prefill progress
        self._progress: dict[str, int] = {}  # request_id -> tokens_processed

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def needs_chunking(self, prompt_tokens: list[int]) -> bool:
        """Check if a prompt is long enough to require chunking."""
        return len(prompt_tokens) > self._chunk_size

    def get_next_chunk(
        self,
        request_id: str,
        prompt_tokens: list[int],
    ) -> PrefillChunk:
        """Get the next chunk for a prefill request.

        Call repeatedly until is_last is True. Tracks progress internally.
        """
        start = self._progress.get(request_id, 0)
        end = min(start + self._chunk_size, len(prompt_tokens))
        is_last = end >= len(prompt_tokens)

        chunk = PrefillChunk(
            request_id=request_id,
            tokens=prompt_tokens[start:end],
            start_pos=start,
            is_last=is_last,
            total_prompt_tokens=len(prompt_tokens),
        )

        self._progress[request_id] = end
        return chunk

    def reset(self, request_id: str) -> None:
        """Clear progress tracking for a request (on completion/cancellation)."""
        self._progress.pop(request_id, None)

    def get_progress(self, request_id: str) -> int:
        """Return number of tokens processed so far for a request."""
        return self._progress.get(request_id, 0)
