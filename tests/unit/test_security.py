"""Unit tests for security middleware and GGUF validator."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from mlxz import __version__
from mlxz.exceptions import GGUFValidationError
from mlxz.security.auth import BearerAuthMiddleware
from mlxz.security.gguf_validator import GGUFValidator
from mlxz.security.headers import SecurityHeadersMiddleware
from mlxz.security.limits import ContentSizeLimitMiddleware

# ---------------------------------------------------------------------------
# Helpers – minimal Starlette apps with middleware applied
# ---------------------------------------------------------------------------

_SECRET = "test-secret-token-42"


async def _ok_endpoint(request: Request) -> PlainTextResponse:
    """Simple endpoint that returns 200 OK."""
    body = await request.body()
    return PlainTextResponse("ok")


async def _health_endpoint(request: Request) -> PlainTextResponse:
    return PlainTextResponse("healthy")


def _make_auth_app(
    api_key: SecretStr | None = SecretStr(_SECRET),
    exempt_paths: set[str] | None = None,
) -> Starlette:
    app = Starlette(
        routes=[
            Route("/test", _ok_endpoint, methods=["GET", "POST"]),
            Route("/health/live", _health_endpoint, methods=["GET"]),
        ],
    )
    app.add_middleware(
        BearerAuthMiddleware,
        api_key=api_key,
        exempt_paths=exempt_paths,
    )
    return app


def _make_limit_app(max_bytes: int = 100) -> Starlette:
    app = Starlette(
        routes=[Route("/upload", _ok_endpoint, methods=["POST"])],
    )
    app.add_middleware(ContentSizeLimitMiddleware, max_bytes=max_bytes)
    return app


def _make_headers_app() -> Starlette:
    app = Starlette(
        routes=[Route("/test", _ok_endpoint, methods=["GET"])],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    return app


# ===================================================================
# 1. BearerAuthMiddleware tests
# ===================================================================


class TestBearerAuthMiddleware:
    """Tests for bearer-token authentication middleware."""

    @pytest.mark.anyio
    async def test_no_auth_mode_passes(self) -> None:
        """When api_key is None the middleware is a pass-through."""
        app = _make_auth_app(api_key=None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/test")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_correct_bearer_token_passes(self) -> None:
        """A valid Bearer token should yield 200."""
        app = _make_auth_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/test", headers={"Authorization": f"Bearer {_SECRET}"}
            )
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_wrong_token_returns_401(self) -> None:
        """An incorrect token must produce a 401."""
        app = _make_auth_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/test", headers={"Authorization": "Bearer wrong-token"}
            )
        assert resp.status_code == 401
        assert resp.json()["error"] == "invalid_api_key"

    @pytest.mark.anyio
    async def test_missing_authorization_header_returns_401(self) -> None:
        """No Authorization header at all must produce a 401."""
        app = _make_auth_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/test")
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_exempt_paths_skip_auth(self) -> None:
        """Paths listed in exempt_paths should not require a token."""
        app = _make_auth_app(exempt_paths={"/health/live"})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health/live")
        assert resp.status_code == 200
        assert resp.text == "healthy"

    @pytest.mark.anyio
    async def test_constant_time_comparison_used(self) -> None:
        """Verify hmac.compare_digest is invoked for token comparison."""
        app = _make_auth_app()
        with patch("mlxz.security.auth.hmac.compare_digest", return_value=True) as mock_cmp:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    "/test", headers={"Authorization": f"Bearer {_SECRET}"}
                )
            assert resp.status_code == 200
            mock_cmp.assert_called_once_with(_SECRET, _SECRET)


# ===================================================================
# 2. ContentSizeLimitMiddleware tests
# ===================================================================


class TestContentSizeLimitMiddleware:
    """Tests for request body size enforcement."""

    @pytest.mark.anyio
    async def test_request_within_limit_passes(self) -> None:
        """A body smaller than max_bytes should be accepted."""
        app = _make_limit_app(max_bytes=200)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/upload", content=b"x" * 50)
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_request_exceeding_limit_returns_413(self) -> None:
        """A body exceeding max_bytes must produce 413."""
        app = _make_limit_app(max_bytes=50)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/upload", content=b"x" * 100)
        assert resp.status_code == 413
        body = resp.json()
        assert body["error"] == "payload_too_large"
        assert body["max_bytes"] == 50

    @pytest.mark.anyio
    async def test_content_length_header_check(self) -> None:
        """A Content-Length header declaring a size over the limit triggers 413
        before the body is consumed."""
        app = _make_limit_app(max_bytes=50)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload",
                content=b"small",
                headers={"Content-Length": "99999"},
            )
        assert resp.status_code == 413


# ===================================================================
# 3. SecurityHeadersMiddleware tests
# ===================================================================


class TestSecurityHeadersMiddleware:
    """Tests for security response headers."""

    @pytest.mark.anyio
    async def test_all_four_headers_present(self) -> None:
        """Every response must include the four hardening headers."""
        app = _make_headers_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/test")
        assert resp.status_code == 200
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["cache-control"] == "no-store"
        assert "x-mlxz-version" in resp.headers

    @pytest.mark.anyio
    async def test_version_header_matches_package(self) -> None:
        """X-MLXz-Version must equal the importable __version__."""
        app = _make_headers_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/test")
        assert resp.headers["x-mlxz-version"] == __version__


# ===================================================================
# 4. GGUFValidator tests
# ===================================================================

# GGUF header layout (little-endian):
#   magic:              uint32  (4 bytes)  – 0x46475547 ("GGUF")
#   version:            uint32  (4 bytes)
#   tensor_count:       int64   (8 bytes)
#   metadata_kv_count:  int64   (8 bytes)

_GGUF_MAGIC = 0x46475547


def _write_gguf(path: Path, *, magic: int = _GGUF_MAGIC, version: int = 3,
                tensor_count: int = 1, metadata_kv_count: int = 0) -> Path:
    """Write a minimal GGUF file with the given header values."""
    header = struct.pack("<IIqq", magic, version, tensor_count, metadata_kv_count)
    path.write_bytes(header)
    return path


class TestGGUFValidator:
    """Tests for pre-parse GGUF file validation."""

    def test_valid_gguf_passes(self, tmp_path: Path) -> None:
        """A file with correct magic, version, and sane tensor count passes."""
        gguf_file = _write_gguf(tmp_path / "model.gguf")
        validator = GGUFValidator()
        # Should not raise.
        validator.validate(gguf_file, max_total_bytes=1024)

    def test_invalid_magic_raises(self, tmp_path: Path) -> None:
        """Wrong magic bytes must raise GGUFValidationError with reason
        'invalid_magic'."""
        bad_file = _write_gguf(tmp_path / "bad_magic.gguf", magic=0xDEADBEEF)
        validator = GGUFValidator()
        with pytest.raises(GGUFValidationError, match="Invalid GGUF magic") as exc_info:
            validator.validate(bad_file, max_total_bytes=1024)
        assert exc_info.value.reason == "invalid_magic"

    def test_file_too_small_raises(self, tmp_path: Path) -> None:
        """A file smaller than 24 bytes cannot be valid GGUF."""
        tiny_file = tmp_path / "tiny.gguf"
        tiny_file.write_bytes(b"\x00" * 10)
        validator = GGUFValidator()
        with pytest.raises(GGUFValidationError, match="too small") as exc_info:
            validator.validate(tiny_file, max_total_bytes=1024)
        assert exc_info.value.reason == "file_too_small"

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """A nonexistent path must raise GGUFValidationError with reason
        'file_not_found'."""
        missing = tmp_path / "does_not_exist.gguf"
        validator = GGUFValidator()
        with pytest.raises(GGUFValidationError, match="not found") as exc_info:
            validator.validate(missing)
        assert exc_info.value.reason == "file_not_found"
