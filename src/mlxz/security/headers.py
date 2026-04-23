"""Security response headers middleware.

Adds hardening headers to every HTTP response.  This is the outermost
response-modifying layer in the security middleware stack.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from mlxz import __version__

__all__ = ["SecurityHeadersMiddleware"]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects security headers into every response.

    Headers applied:

    - ``X-Content-Type-Options: nosniff`` -- prevents MIME-type sniffing.
    - ``X-Frame-Options: DENY`` -- blocks framing (clickjacking defense).
    - ``Cache-Control: no-store`` -- disables caching of responses.
    - ``X-MLXz-Version`` -- exposes the server version for diagnostics.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Process the request and attach security headers to the response."""
        response: Response = await call_next(request)  # type: ignore[misc]

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-MLXz-Version"] = __version__

        return response
