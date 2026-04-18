"""Fixed-size block pool with reference counting for paged KV cache."""
from __future__ import annotations

from dataclasses import dataclass, field


class OutOfBlocksError(Exception):
    """Raised when block allocation cannot be satisfied."""
    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(f"Cannot allocate {requested} blocks; only {available} free")


class DoubleFreeError(Exception):
    """Raised when freeing a block with refcount 0."""
    def __init__(self, block_idx: int) -> None:
        self.block_idx = block_idx
        super().__init__(f"Double free on block {block_idx} (refcount already 0)")


@dataclass(slots=True)
class PhysicalBlock:
    """A single physical block in the pool."""
    idx: int
    refcount: int = 0


class BlockManager:
    """Fixed-size block pool with reference counting.

    Manages a pool of physical blocks for paged KV cache. Supports:
    - Allocation from a free list
    - Reference counting for shared blocks (prefix cache)
    - Copy-on-write for mutation of shared blocks

    Invariants (verified by Hypothesis):
    - No leaks: free_blocks + sum(block.refcount > 0) == total_blocks
    - No double-free: freeing refcount==0 raises DoubleFreeError
    - Monotonic refcount: refcount never goes negative
    - COW correctness: copy_on_write returns independent block
    """

    def __init__(self, total_blocks: int, block_size: int) -> None:
        if total_blocks < 1:
            raise ValueError(f"total_blocks must be >= 1, got {total_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        self._total_blocks = total_blocks
        self._block_size = block_size
        self._blocks = [PhysicalBlock(idx=i, refcount=0) for i in range(total_blocks)]
        self._free_list: list[int] = list(range(total_blocks))  # stack: pop from end

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def free_blocks(self) -> int:
        return len(self._free_list)

    def allocate(self, n_blocks: int) -> list[int]:
        """Allocate n blocks from the free list.

        Returns list of block indices. Raises OutOfBlocksError if insufficient.
        All allocated blocks start with refcount=1.
        """
        if n_blocks < 0:
            raise ValueError(f"Cannot allocate negative blocks: {n_blocks}")
        if n_blocks == 0:
            return []
        if n_blocks > len(self._free_list):
            raise OutOfBlocksError(n_blocks, len(self._free_list))

        allocated = []
        for _ in range(n_blocks):
            idx = self._free_list.pop()
            self._blocks[idx].refcount = 1
            allocated.append(idx)
        return allocated

    def free(self, block_indices: list[int]) -> None:
        """Decrement refcount for each block. Return to pool if refcount reaches 0.

        Raises DoubleFreeError if any block has refcount 0.
        """
        for idx in block_indices:
            block = self._blocks[idx]
            if block.refcount <= 0:
                raise DoubleFreeError(idx)
            block.refcount -= 1
            if block.refcount == 0:
                self._free_list.append(idx)

    def incref(self, block_indices: list[int]) -> None:
        """Increment refcount for each block (used for shared prefix cache entries).

        Raises ValueError if any block is not allocated (refcount == 0).
        """
        for idx in block_indices:
            block = self._blocks[idx]
            if block.refcount <= 0:
                raise ValueError(f"Cannot incref freed block {idx}")
            block.refcount += 1

    def copy_on_write(self, block_idx: int) -> int:
        """Allocate a new block as a copy of block_idx.

        - Original block's refcount decreases by 1
        - New block starts with refcount 1
        - Returns the new block's index

        The caller is responsible for copying the actual data content.
        """
        original = self._blocks[block_idx]
        if original.refcount <= 0:
            raise ValueError(f"Cannot COW freed block {block_idx}")
        if original.refcount == 1:
            # No sharing — no copy needed, return the same block
            return block_idx

        # Allocate new block
        new_indices = self.allocate(1)
        new_idx = new_indices[0]

        # Decrement original refcount
        original.refcount -= 1

        return new_idx

    def get_refcount(self, block_idx: int) -> int:
        """Get the current refcount of a block."""
        return self._blocks[block_idx].refcount

    def _debug_state(self) -> dict:
        """For testing: return internal state summary."""
        allocated = [(b.idx, b.refcount) for b in self._blocks if b.refcount > 0]
        return {
            "total": self._total_blocks,
            "free": len(self._free_list),
            "allocated": allocated,
        }
