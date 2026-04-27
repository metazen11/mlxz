"""FCFS priority queue for request scheduling."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Priority(IntEnum):
    """Request priority classes. Lower value = higher priority."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(slots=True, order=True)
class QueueEntry:
    """Priority queue entry with FCFS tiebreaking."""

    priority: Priority
    arrival_time: float = field(compare=True)
    request: Any = field(compare=False)


class PriorityScheduler:
    """FCFS scheduler with optional priority classes.

    Within the same priority, requests are served in arrival order.
    Higher priority requests (lower enum value) are served first.
    """

    def __init__(self, max_queue_size: int = 256) -> None:
        self._max_queue_size = max_queue_size
        self._queues: dict[Priority, deque[QueueEntry]] = {p: deque() for p in Priority}
        self._size: int = 0

    def enqueue(self, request: Any, priority: Priority = Priority.NORMAL) -> bool:
        """Add request to queue. Returns False if queue is full."""
        if self._size >= self._max_queue_size:
            return False
        entry = QueueEntry(
            priority=priority,
            arrival_time=time.monotonic(),
            request=request,
        )
        self._queues[priority].append(entry)
        self._size += 1
        return True

    def dequeue(self) -> Any | None:
        """Remove and return highest-priority, oldest request. Returns None if empty."""
        for priority in Priority:
            queue = self._queues[priority]
            if queue:
                entry = queue.popleft()
                self._size -= 1
                return entry.request
        return None

    def peek(self) -> Any | None:
        """Return highest-priority, oldest request without removing. None if empty."""
        for priority in Priority:
            queue = self._queues[priority]
            if queue:
                return queue[0].request
        return None

    def dequeue_batch(self, max_count: int) -> list[Any]:
        """Dequeue up to max_count requests in priority order."""
        batch = []
        for _ in range(max_count):
            req = self.dequeue()
            if req is None:
                break
            batch.append(req)
        return batch

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    def remove(self, request_id: str) -> bool:
        """Remove a specific request by ID. Returns True if found."""
        for priority in Priority:
            queue = self._queues[priority]
            for i, entry in enumerate(queue):
                if hasattr(entry.request, "id") and entry.request.id == request_id:
                    del queue[i]  # O(n) but queue is small
                    self._size -= 1
                    return True
        return False
