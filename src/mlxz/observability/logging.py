"""Structured logging configuration for mlxz.

Sets up ``structlog`` with JSON rendering, automatic timestamping, log-level
filtering, and a secret-redaction processor that prevents accidental leakage
of API keys, passwords, tokens, and DSN strings.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_KEY_RE = re.compile(r".*(key|secret|password|token|dsn).*", re.IGNORECASE)
_REDACTED = "[REDACTED]"


class SecretRedactionProcessor:
    """Structlog processor that replaces values for sensitive keys.

    Any event-dict key whose name matches ``*key*``, ``*secret*``,
    ``*password*``, ``*token*``, or ``*dsn*`` (case-insensitive) has
    its value replaced with ``"[REDACTED]"``.
    """

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        for key in list(event_dict):
            if _SECRET_KEY_RE.match(key):
                event_dict[key] = _REDACTED
        return event_dict


# ---------------------------------------------------------------------------
# Content-mode filter
# ---------------------------------------------------------------------------


class _ContentModeFilter:
    """Drops prompt/completion content fields unless content_mode allows them."""

    _CONTENT_KEYS = frozenset({"prompt", "completion", "messages", "content"})
    _METADATA_KEYS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})

    def __init__(self, mode: str) -> None:
        if mode not in ("none", "metadata", "full"):
            msg = f"Invalid content_mode: {mode!r} (expected 'none', 'metadata', or 'full')"
            raise ValueError(msg)
        self._mode = mode

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        if self._mode == "full":
            return event_dict

        for key in list(event_dict):
            if key in self._CONTENT_KEYS:
                del event_dict[key]

        if self._mode == "none":
            for key in list(event_dict):
                if key in self._METADATA_KEYS:
                    del event_dict[key]

        return event_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    content_mode: str = "none",
) -> None:
    """Initialise structured logging for the entire process.

    Parameters
    ----------
    level:
        Root log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
        ``CRITICAL``).
    content_mode:
        Controls whether prompt content appears in logs.

        * ``"none"`` (default) -- no content, no token-count metadata.
        * ``"metadata"`` -- token counts and latency, but no raw text.
        * ``"full"`` -- everything (development only).
    """
    if content_mode == "full":
        structlog.get_logger().warning(
            "content_logging_full_enabled",
            msg="Prompt and completion content will appear in logs. "
            "Do NOT use this setting in production.",
        )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        SecretRedactionProcessor(),
        _ContentModeFilter(content_mode),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
