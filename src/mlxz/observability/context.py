"""Request-scoped context propagation for mlxz.

Each incoming inference request gets a ``RequestContext`` that carries
its correlation ID, model name, and token budget.  The context is
stored in a ``ContextVar`` so it propagates automatically through
async call chains and is available to any structlog processor.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

import structlog

# ---------------------------------------------------------------------------
# Context variable — one per async task / thread
# ---------------------------------------------------------------------------

_request_ctx: ContextVar[RequestContext | None] = ContextVar(
    "request_ctx", default=None
)


# ---------------------------------------------------------------------------
# Frozen request context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Immutable snapshot of request metadata, created at admission time."""

    request_id: str
    model: str
    prompt_tokens: int
    max_tokens: int
    prefix_cache_hit: bool = False
    created_at: float = field(default_factory=time.monotonic)

    def bind_logger(self) -> structlog.BoundLogger:
        """Return a bound logger carrying this request's correlation fields."""
        return structlog.get_logger().bind(
            request_id=self.request_id,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def new_request_context(**kwargs: object) -> RequestContext:
    """Create a ``RequestContext``, assign a UUID, and store it in the ContextVar.

    All keyword arguments are forwarded to ``RequestContext`` except
    ``request_id`` which is generated automatically.
    """
    ctx = RequestContext(request_id=str(uuid.uuid4()), **kwargs)  # type: ignore[arg-type]
    _request_ctx.set(ctx)
    return ctx


def get_request_context() -> RequestContext | None:
    """Retrieve the current request context, or ``None`` if unset."""
    return _request_ctx.get()
