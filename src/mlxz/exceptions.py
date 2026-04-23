"""Domain error hierarchy for mlxz.

Every exception carries structured context so that error handlers can
produce actionable HTTP responses or CLI diagnostics without parsing
message strings.
"""

from __future__ import annotations

from pathlib import Path

from mlxz.types import AdmissionDecision


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class MlxzError(Exception):
    """Base exception for all mlxz domain errors."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(MlxzError):
    """Raised when the runtime configuration is invalid or contradictory."""


# ---------------------------------------------------------------------------
# Memory / residency
# ---------------------------------------------------------------------------


class ResidencyOverflow(MlxzError):
    """The requested wired-memory limit exceeds what the OS allows.

    Carries a *remediation* string containing the exact ``sysctl`` command
    the operator should run to increase the limit.
    """

    def __init__(self, message: str, *, remediation: str) -> None:
        super().__init__(message)
        self.remediation: str = remediation


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


class AdmissionRejected(MlxzError):
    """The admission controller refused a request.

    Includes projected vs. available resource numbers so the API layer
    can populate a 429 response body.
    """

    def __init__(
        self,
        message: str,
        *,
        decision: AdmissionDecision,
        reason: str,
        projected_bytes: int,
        available_bytes: int,
    ) -> None:
        super().__init__(message)
        self.decision: AdmissionDecision = decision
        self.reason: str = reason
        self.projected_bytes: int = projected_bytes
        self.available_bytes: int = available_bytes


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class GGUFValidationError(MlxzError):
    """Pre-parse validation of a GGUF file failed.

    Reports the offending *path* and a human-readable *reason*.
    """

    def __init__(self, message: str, *, path: Path, reason: str) -> None:
        super().__init__(message)
        self.path: Path = path
        self.reason: str = reason


class DraftIncompatible(MlxzError):
    """Target and draft models are not compatible for speculative decoding.

    Typically raised when vocabulary sizes or tokeniser configs diverge.
    """

    def __init__(
        self,
        message: str,
        *,
        target_model: str,
        draft_model: str,
        reason: str,
    ) -> None:
        super().__init__(message)
        self.target_model: str = target_model
        self.draft_model: str = draft_model
        self.reason: str = reason


# ---------------------------------------------------------------------------
# Engine runtime
# ---------------------------------------------------------------------------


class EngineError(MlxzError):
    """Unrecoverable failure on the engine compute thread."""


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------


class PrefixCacheCorruption(MlxzError):
    """On-disk or in-memory prefix cache data failed a checksum check."""


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class ShutdownTimeout(MlxzError):
    """Graceful drain did not complete within the configured deadline."""
