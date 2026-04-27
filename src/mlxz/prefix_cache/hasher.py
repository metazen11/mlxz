"""Content-addressed prefix hashing using SHA-256 at block boundaries."""

from __future__ import annotations

import hashlib
import struct


class RollingPrefixHasher:
    """Emits SHA-256 hashes at every block_size boundary in a token stream.

    Each chunk covers tokens[i*block_size : (i+1)*block_size]. The final
    chunk may be partial (fewer than block_size tokens). Returns a tuple
    of bytes (hashable, usable as dict keys).

    Smaller block sizes increase the chance of reusing short shared prefixes
    (useful for agent-style workloads). Larger block sizes reduce hash churn
    and metadata overhead but require longer identical prefixes to hit.
    """

    def __init__(self, block_size: int = 8) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self._block_size = block_size

    @property
    def block_size(self) -> int:
        return self._block_size

    def hash_chunks(self, tokens: list[int]) -> tuple[bytes, ...]:
        """Hash tokens into block-aligned chunks.

        Returns a tuple of SHA-256 digests, one per chunk.
        Empty token list returns empty tuple.

        The byte representation uses struct.pack with little-endian int32
        for stable, platform-independent hashing.
        """
        if not tokens:
            return ()

        chunks: list[bytes] = []
        bs = self._block_size

        for start in range(0, len(tokens), bs):
            end = min(start + bs, len(tokens))
            chunk_tokens = tokens[start:end]
            # Convert to stable byte representation
            token_bytes = struct.pack(f"<{len(chunk_tokens)}i", *chunk_tokens)
            digest = hashlib.sha256(token_bytes).digest()
            chunks.append(digest)

        return tuple(chunks)
