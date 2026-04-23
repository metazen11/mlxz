"""Tests for the paged attention block manager."""
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition

from mlxz.paged_attention.block_manager import (
    BlockManager,
    DoubleFreeError,
    OutOfBlocksError,
    PhysicalBlock,
)


class TestBlockManagerBasic:
    def test_initial_state(self):
        bm = BlockManager(total_blocks=64, block_size=16)
        assert bm.total_blocks == 64
        assert bm.free_blocks == 64
        assert bm.block_size == 16

    def test_allocate_and_free(self):
        bm = BlockManager(total_blocks=10, block_size=16)
        blocks = bm.allocate(3)
        assert len(blocks) == 3
        assert bm.free_blocks == 7
        bm.free(blocks)
        assert bm.free_blocks == 10

    def test_allocate_zero(self):
        bm = BlockManager(total_blocks=10, block_size=16)
        assert bm.allocate(0) == []
        assert bm.free_blocks == 10

    def test_allocate_all(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        blocks = bm.allocate(4)
        assert len(blocks) == 4
        assert bm.free_blocks == 0

    def test_allocate_too_many_raises(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        with pytest.raises(OutOfBlocksError) as exc_info:
            bm.allocate(5)
        assert exc_info.value.requested == 5
        assert exc_info.value.available == 4

    def test_double_free_raises(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        blocks = bm.allocate(1)
        bm.free(blocks)
        with pytest.raises(DoubleFreeError):
            bm.free(blocks)

    def test_incref_and_shared_free(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        blocks = bm.allocate(1)
        bm.incref(blocks)  # refcount = 2
        assert bm.get_refcount(blocks[0]) == 2
        bm.free(blocks)    # refcount = 1, NOT returned to pool
        assert bm.free_blocks == 3  # still allocated
        bm.free(blocks)    # refcount = 0, returned to pool
        assert bm.free_blocks == 4

    def test_incref_freed_block_raises(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        blocks = bm.allocate(1)
        bm.free(blocks)
        with pytest.raises(ValueError, match="Cannot incref freed block"):
            bm.incref(blocks)

    def test_copy_on_write_shared(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        blocks = bm.allocate(1)
        bm.incref(blocks)  # refcount = 2
        new_idx = bm.copy_on_write(blocks[0])
        assert new_idx != blocks[0]
        assert bm.get_refcount(blocks[0]) == 1  # decremented
        assert bm.get_refcount(new_idx) == 1     # new block

    def test_copy_on_write_unshared_returns_same(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        blocks = bm.allocate(1)  # refcount = 1
        new_idx = bm.copy_on_write(blocks[0])
        assert new_idx == blocks[0]  # no copy needed

    def test_copy_on_write_freed_raises(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        blocks = bm.allocate(1)
        bm.free(blocks)
        with pytest.raises(ValueError, match="Cannot COW freed block"):
            bm.copy_on_write(blocks[0])

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            BlockManager(total_blocks=0, block_size=16)
        with pytest.raises(ValueError):
            BlockManager(total_blocks=4, block_size=0)

    def test_negative_allocate_raises(self):
        bm = BlockManager(total_blocks=4, block_size=16)
        with pytest.raises(ValueError):
            bm.allocate(-1)


class BlockManagerMachine(RuleBasedStateMachine):
    """Hypothesis state machine for exhaustive BlockManager testing.

    Tracks expected state alongside the real BlockManager and verifies
    invariants after every operation.
    """

    def __init__(self):
        super().__init__()
        self.bm = BlockManager(total_blocks=32, block_size=16)
        self.owned: dict[str, list[int]] = {}  # seq_id -> block indices
        self._seq_counter = 0

    def _new_seq_id(self) -> str:
        self._seq_counter += 1
        return f"seq_{self._seq_counter}"

    @rule(n=st.integers(min_value=1, max_value=8))
    def allocate(self, n):
        if self.bm.free_blocks >= n:
            seq_id = self._new_seq_id()
            blocks = self.bm.allocate(n)
            assert len(blocks) == n
            self.owned[seq_id] = blocks

    @rule(data=st.data())
    def free_sequence(self, data):
        if not self.owned:
            return
        seq_id = data.draw(st.sampled_from(sorted(self.owned.keys())))
        self.bm.free(self.owned.pop(seq_id))

    @rule(data=st.data())
    def fork_sequence(self, data):
        """Simulate prefix sharing: incref all blocks of an existing sequence."""
        if not self.owned:
            return
        parent_id = data.draw(st.sampled_from(sorted(self.owned.keys())))
        new_id = self._new_seq_id()
        self.bm.incref(self.owned[parent_id])
        self.owned[new_id] = list(self.owned[parent_id])

    @rule(data=st.data())
    def cow_block(self, data):
        """Copy-on-write a random block from a random sequence."""
        if not self.owned:
            return
        seq_id = data.draw(st.sampled_from(sorted(self.owned.keys())))
        blocks = self.owned[seq_id]
        if not blocks:
            return
        idx_pos = data.draw(st.integers(min_value=0, max_value=len(blocks) - 1))
        old_idx = blocks[idx_pos]
        if self.bm.get_refcount(old_idx) > 1 and self.bm.free_blocks == 0:
            return  # can't COW if no free blocks
        new_idx = self.bm.copy_on_write(old_idx)
        blocks[idx_pos] = new_idx

    @invariant()
    def no_leaks(self):
        """free_blocks + allocated_unique_blocks == total_blocks"""
        # Count unique allocated blocks with refcount > 0
        allocated_count = sum(1 for i in range(self.bm.total_blocks)
                              if self.bm.get_refcount(i) > 0)
        assert self.bm.free_blocks + allocated_count == self.bm.total_blocks, \
            f"Leak: free={self.bm.free_blocks}, allocated={allocated_count}, total={self.bm.total_blocks}"

    @invariant()
    def no_negative_refcount(self):
        """No block should ever have negative refcount."""
        for i in range(self.bm.total_blocks):
            rc = self.bm.get_refcount(i)
            assert rc >= 0, f"Block {i} has negative refcount {rc}"

    @invariant()
    def owned_blocks_are_allocated(self):
        """Every block we think we own should have refcount > 0."""
        for seq_id, blocks in self.owned.items():
            for idx in blocks:
                rc = self.bm.get_refcount(idx)
                assert rc > 0, f"Seq {seq_id} owns block {idx} but refcount is {rc}"


TestBlockManagerStateMachine = BlockManagerMachine.TestCase
TestBlockManagerStateMachine.settings = settings(max_examples=500, stateful_step_count=30)
