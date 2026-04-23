"""Bearer token authentication middleware.

Validates ``Authorization: Bearer <token>`` headers using constant-time
comparison to prevent timing side-channels.  Designed as the second layer
in the security middleware stack, after body-size enforcement.
"""

from __future__ import annotations

import hmac
import logging

from pydantic import SecretStr
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

__all__ = ["BearerAuthMiddleware"]

_log = logging.getLogger(__name__)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Constant-time bearer token validation middleware.

    Parameters
    ----------
    app:
        The wrapped ASGI application.
    api_key:
        Expected bearer token wrapped in :class:`~pydantic.SecretStr`.
        When ``None``, the middleware is a transparent pass-through
        (useful during local development).
    exempt_paths:
        URL paths that bypass authentication entirely, e.g.
        ``{"/health/live", "/health/ready"}``.
    """

    def __init__(
        self,
        app: object,
        *,
        api_key: SecretStr | None,
        exempt_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._api_key = api_key
        self._exempt_paths: set[str] = exempt_paths or set()

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Validate the bearer token or reject with 401."""
        # No key configured -- open access.
        if self._api_key is None:
            return await call_next(request)  # type: ignore[misc]

        # Exempt paths skip auth (health probes, metrics, etc.).
        if request.url.path in self._exempt_paths:
            return await call_next(request)  # type: ignore[misc]

        # Extract token from "Bearer <token>" header.
        raw_header = request.headers.get("authorization", "")
        token = raw_header.removeprefix("Bearer ")

        # Constant-time comparison to prevent timing attacks.
        if not hmac.compare_digest(token, self._api_key.get_secret_value()):
            # Log the failure with source IP -- never the attempted key value.
            _log.warning(
                "auth_failure: invalid bearer token from %s for %s",
                request.client.host if request.client else "unknown",
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={"error": "invalid_api_key"},
            )

        return await call_next(request)  # type: ignore[misc]
