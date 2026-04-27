"""Request body size enforcement middleware.

Rejects incoming requests whose ``Content-Length`` (or streamed body)
exceeds the configured maximum *before* any application code runs.
This is the first layer in the security middleware stack.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

__all__ = ["ContentSizeLimitMiddleware"]


class ContentSizeLimitMiddleware:
    """Pure ASGI middleware that enforces a maximum request body size.

    If the declared ``Content-Length`` exceeds *max_bytes*, the request
    is rejected immediately with HTTP 413 Payload Too Large.  For
    chunked-transfer requests (no ``Content-Length``), the middleware
    accumulates received bytes and aborts once the limit is breached.

    Parameters
    ----------
    app:
        The wrapped ASGI application.
    max_bytes:
        Maximum allowed request body in bytes.  Sourced from
        :pyattr:`RequestLimits.max_request_body_bytes`.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope)
        content_length = request.headers.get("content-length")

        # Fast-path: declared Content-Length already exceeds the limit.
        if content_length is not None:
            try:
                if int(content_length) > self._max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "error": "payload_too_large",
                            "max_bytes": self._max_bytes,
                        },
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass  # Malformed header; let downstream handle it.

        # Slow-path: wrap the receive callable to count streamed bytes.
        bytes_received = 0

        async def _counting_receive() -> dict:
            nonlocal bytes_received
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                bytes_received += len(body)
                if bytes_received > self._max_bytes:
                    raise _PayloadTooLarge
            return message

        try:
            await self._app(scope, _counting_receive, send)
        except _PayloadTooLarge:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "max_bytes": self._max_bytes,
                },
            )
            await response(scope, receive, send)


class _PayloadTooLarge(Exception):
    """Internal signal — never escapes the middleware."""
