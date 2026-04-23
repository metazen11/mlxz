"""Async/sync thread-boundary primitives for the mlxz engine.

The API layer is async (FastAPI + uvicorn on the main event loop).  The engine
is synchronous (dedicated compute thread).  Crossing this boundary incorrectly
causes data corruption, deadlocks, or silent token loss.

This module provides three utilities:

* **RequestBridge** — ``janus.Queue``-based bridge for request submission
  (API -> engine) and per-request token delivery (engine -> API).
* **CancellationRegistry** — per-request ``asyncio.Event`` tracking so the
  engine thread can poll for client disconnects each decode iteration.
* **MxEvalGuard** — lightweight thread-affinity assertion that ensures
  MLX evaluation calls only happen on the dedicated engine thread.
"""

from __future__ import annotations

import asyncio
import functools
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import janus

if TYPE_CHECKING:
    from collections.abc import Generator

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# RequestBridge
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RequestBridge:
    """Thread-safe bridge between the async API layer and the sync engine.

    Uses a shared ``janus.Queue`` for the submission channel (API -> engine)
    and per-request ``janus.Queue`` instances for token delivery
    (engine -> API).

    Parameters
    ----------
    _submit_queue
        Internal janus queue sized to provide back-pressure when the engine
        cannot keep up.  Defaults to ``maxsize=256``.
    """

    _submit_queue: janus.Queue[Any] = field(
        default_factory=lambda: janus.Queue(maxsize=256),
    )

    # -- per-request token channel ------------------------------------------

    @staticmethod
    def create_token_channel(max_depth: int = 64) -> janus.Queue[int | None]:
        """Create a per-request token delivery channel.

        The engine thread puts token IDs on the **sync** side; the API layer
        reads them from the **async** side.  A ``None`` sentinel signals
        end-of-sequence (EOS).

        Parameters
        ----------
        max_depth
            Maximum number of un-consumed tokens before the engine
            experiences back-pressure and pauses decode for this request.
            This prevents unbounded memory growth when a client stops
            reading SSE events.

        Returns
        -------
        janus.Queue[int | None]
            A bidirectional queue whose ``.sync_q`` and ``.async_q``
            facades are safe for their respective threads.
        """
        return janus.Queue(maxsize=max_depth)

    # -- submission ---------------------------------------------------------

    async def submit_async(self, request: Any) -> None:
        """Enqueue a request from the async API layer.

        Blocks (awaits) if the submission queue is full, providing
        natural back-pressure to the HTTP layer.

        Parameters
        ----------
        request
            An inference ``Request`` object to be picked up by the engine
            thread.
        """
        await self._submit_queue.async_q.put(request)

    # -- engine-side polling ------------------------------------------------

    def get_next_sync(self, timeout: float = 0.01) -> Any | None:
        """Non-blocking poll from the engine (sync) thread.

        Parameters
        ----------
        timeout
            Ignored in the current implementation (we use ``get_nowait``),
            but reserved for a future blocking-with-timeout strategy.

        Returns
        -------
        Request | None
            The next queued request, or ``None`` if the queue is empty.
        """
        try:
            return self._submit_queue.sync_q.get_nowait()
        except janus.SyncQueueEmpty:
            return None

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying janus queue and release resources.

        Must be called when the engine is shutting down.  After this call
        any further ``put`` or ``get`` will raise.
        """
        self._submit_queue.close()


# ---------------------------------------------------------------------------
# CancellationRegistry
# ---------------------------------------------------------------------------


class CancellationRegistry:
    """Tracks per-request cancellation events.

    The API layer sets an event on client disconnect; the engine thread
    checks ``is_cancelled`` at the top of each decode iteration so it can
    abort early and free the KV cache.

    Thread-safety note: ``asyncio.Event.set()`` is thread-safe (it
    schedules the wakeup on the event loop), and ``Event.is_set()`` is a
    simple boolean read.  The ``_events`` dict is only mutated from the
    async side (register / unregister happen in the SSE handler's
    ``finally`` block), so no additional locking is required.
    """

    __slots__ = ("_events",)

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    # -- public API ---------------------------------------------------------

    def register(self, request_id: str) -> asyncio.Event:
        """Register a cancellation event for *request_id*.

        Returns the event so the SSE handler can ``await`` it or pass it
        to a disconnect callback.

        Raises
        ------
        ValueError
            If *request_id* is already registered (indicates a UUID
            collision or double-registration bug).
        """
        if request_id in self._events:
            raise ValueError(
                f"Request {request_id!r} is already registered in the "
                "cancellation registry"
            )
        event = asyncio.Event()
        self._events[request_id] = event
        return event

    def cancel(self, request_id: str) -> None:
        """Signal cancellation for *request_id*.

        Safe to call even if the request has already been unregistered
        (e.g. race between completion and disconnect).
        """
        if event := self._events.get(request_id):
            event.set()

    def is_cancelled(self, request_id: str) -> bool:
        """Check whether *request_id* has been cancelled.

        Returns ``False`` if the request is unknown (already unregistered
        or never registered).
        """
        if event := self._events.get(request_id):
            return event.is_set()
        return False

    def unregister(self, request_id: str) -> None:
        """Remove *request_id* from the registry.

        Always called in a ``finally`` block so orphaned events cannot
        accumulate.  Safe to call multiple times.
        """
        self._events.pop(request_id, None)

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        """Number of currently tracked requests."""
        return len(self._events)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._events


# ---------------------------------------------------------------------------
# MxEvalGuard
# ---------------------------------------------------------------------------


class MxEvalGuard:
    """Thread-affinity guard for MLX evaluation calls.

    Records the engine thread ID at construction time and provides
    ``assert_engine_thread()`` to verify that the caller is on the
    correct thread.  Can be used as:

    * An explicit assertion::

        guard = MxEvalGuard()
        guard.assert_engine_thread()

    * A context manager::

        with guard:
            mx.eval(...)

    * A decorator::

        @guard
        def decode_step(...):
            ...
    """

    __slots__ = ("_thread_id", "_thread_name")

    def __init__(self) -> None:
        self._thread_id: int = threading.get_ident()
        self._thread_name: str = threading.current_thread().name

    # -- assertion ----------------------------------------------------------

    def assert_engine_thread(self) -> None:
        """Raise ``RuntimeError`` if not called from the engine thread.

        The error message includes both the expected and actual thread
        identifiers to aid debugging.
        """
        current = threading.get_ident()
        if current != self._thread_id:
            raise RuntimeError(
                f"MLX evaluation must run on the engine thread "
                f"({self._thread_name!r}, id={self._thread_id}), "
                f"but was called from thread id={current} "
                f"({threading.current_thread().name!r})"
            )

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> MxEvalGuard:
        self.assert_engine_thread()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        # No cleanup needed; the guard is purely assertive.
        pass

    # -- decorator ----------------------------------------------------------

    def __call__(self, func: F) -> F:  # type: ignore[override]
        """Wrap *func* so that ``assert_engine_thread()`` runs on entry."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.assert_engine_thread()
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]
