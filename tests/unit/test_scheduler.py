"""Tests for priority scheduler and chunked prefill."""
import pytest
from mlxz.scheduler.priority import PriorityScheduler, Priority
from mlxz.scheduler.chunker import ChunkedPrefillScheduler


class MockRequest:
    def __init__(self, id: str):
        self.id = id


class TestPriorityScheduler:
    def test_fifo_order(self):
        s = PriorityScheduler()
        s.enqueue(MockRequest("a"))
        s.enqueue(MockRequest("b"))
        s.enqueue(MockRequest("c"))
        assert s.dequeue().id == "a"
        assert s.dequeue().id == "b"
        assert s.dequeue().id == "c"

    def test_priority_ordering(self):
        s = PriorityScheduler()
        s.enqueue(MockRequest("low"), Priority.LOW)
        s.enqueue(MockRequest("high"), Priority.HIGH)
        s.enqueue(MockRequest("normal"), Priority.NORMAL)
        assert s.dequeue().id == "high"
        assert s.dequeue().id == "normal"
        assert s.dequeue().id == "low"

    def test_empty_returns_none(self):
        s = PriorityScheduler()
        assert s.dequeue() is None
        assert s.peek() is None

    def test_queue_full(self):
        s = PriorityScheduler(max_queue_size=2)
        assert s.enqueue(MockRequest("a")) is True
        assert s.enqueue(MockRequest("b")) is True
        assert s.enqueue(MockRequest("c")) is False
        assert s.size == 2

    def test_dequeue_batch(self):
        s = PriorityScheduler()
        for i in range(5):
            s.enqueue(MockRequest(str(i)))
        batch = s.dequeue_batch(3)
        assert len(batch) == 3
        assert s.size == 2

    def test_remove(self):
        s = PriorityScheduler()
        s.enqueue(MockRequest("a"))
        s.enqueue(MockRequest("b"))
        s.enqueue(MockRequest("c"))
        assert s.remove("b") is True
        assert s.size == 2
        assert s.dequeue().id == "a"
        assert s.dequeue().id == "c"

    def test_remove_nonexistent(self):
        s = PriorityScheduler()
        assert s.remove("x") is False


class TestChunkedPrefillScheduler:
    def test_short_prompt_single_chunk(self):
        c = ChunkedPrefillScheduler(chunk_size=256)
        tokens = list(range(100))
        assert not c.needs_chunking(tokens)
        chunk = c.get_next_chunk("r1", tokens)
        assert chunk.is_last is True
        assert len(chunk.tokens) == 100

    def test_long_prompt_multiple_chunks(self):
        c = ChunkedPrefillScheduler(chunk_size=4)
        tokens = list(range(10))
        assert c.needs_chunking(tokens)

        chunk1 = c.get_next_chunk("r1", tokens)
        assert chunk1.tokens == [0, 1, 2, 3]
        assert chunk1.start_pos == 0
        assert chunk1.is_last is False

        chunk2 = c.get_next_chunk("r1", tokens)
        assert chunk2.tokens == [4, 5, 6, 7]
        assert chunk2.start_pos == 4
        assert chunk2.is_last is False

        chunk3 = c.get_next_chunk("r1", tokens)
        assert chunk3.tokens == [8, 9]
        assert chunk3.start_pos == 8
        assert chunk3.is_last is True

    def test_exact_boundary(self):
        c = ChunkedPrefillScheduler(chunk_size=4)
        tokens = list(range(8))
        c.get_next_chunk("r1", tokens)
        chunk2 = c.get_next_chunk("r1", tokens)
        assert chunk2.is_last is True
        assert len(chunk2.tokens) == 4

    def test_reset_clears_progress(self):
        c = ChunkedPrefillScheduler(chunk_size=4)
        tokens = list(range(10))
        c.get_next_chunk("r1", tokens)
        assert c.get_progress("r1") == 4
        c.reset("r1")
        assert c.get_progress("r1") == 0

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            ChunkedPrefillScheduler(chunk_size=0)
