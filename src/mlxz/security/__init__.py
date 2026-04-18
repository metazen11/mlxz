"""Security middleware and validation for mlxz.

This package provides the layered security stack applied to every
incoming HTTP request:

1. :class:`ContentSizeLimitMiddleware` -- body size enforcement (HTTP 413).
2. :class:`BearerAuthMiddleware` -- constant-time bearer token auth (HTTP 401).
3. :class:`SecurityHeadersMiddleware` -- hardening response headers.
4. :class:`GGUFValidator` -- pre-parse validation for untrusted model files.
"""

from mlxz.security.auth import BearerAuthMiddleware
from mlxz.security.gguf_validator import GGUFValidator
from mlxz.security.headers import SecurityHeadersMiddleware
from mlxz.security.limits import ContentSizeLimitMiddleware

__all__ = [
    "BearerAuthMiddleware",
    "ContentSizeLimitMiddleware",
    "GGUFValidator",
    "SecurityHeadersMiddleware",
]
