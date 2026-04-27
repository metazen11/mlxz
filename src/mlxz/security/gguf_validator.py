"""Pre-parse validation layer for untrusted GGUF files.

Performs lightweight structural checks *before* the ``gguf`` parser
allocates any tensors.  This prevents resource exhaustion attacks from
crafted GGUF files that declare enormous tensor counts or sizes.

This module intentionally avoids importing the ``gguf`` package; all
checks are performed via :mod:`struct` on the raw file header.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from mlxz.exceptions import GGUFValidationError

__all__ = ["GGUFValidator"]

# GGUF magic: bytes "GGUF" in little-endian u32 = 0x46475547.
_GGUF_MAGIC = 0x46475547

# Sanity cap on declared tensor count.
_MAX_TENSOR_COUNT = 10_000


class GGUFValidator:
    """Validates GGUF files before handing them to the full parser.

    All checks are O(1) in file size (only the header is read).
    """

    def validate(self, path: Path, max_total_bytes: int | None = None) -> None:
        """Run structural validation on a GGUF file.

        Parameters
        ----------
        path:
            Filesystem path to the GGUF file.
        max_total_bytes:
            Upper bound on the file size.  Defaults to twice the
            physical RAM reported by the OS.

        Raises
        ------
        GGUFValidationError
            With a human-readable *reason* when any check fails.
        """
        # -- Existence check --------------------------------------------------
        if not path.exists():
            raise GGUFValidationError(
                f"GGUF file not found: {path}",
                path=path,
                reason="file_not_found",
            )

        if not path.is_file():
            raise GGUFValidationError(
                f"Path is not a regular file: {path}",
                path=path,
                reason="not_a_file",
            )

        # -- File size check ---------------------------------------------------
        file_size = path.stat().st_size

        if file_size < 24:
            # GGUF header is at minimum: magic(4) + version(4) + tensor_count(8)
            # + metadata_kv_count(8) = 24 bytes.
            raise GGUFValidationError(
                f"File too small to be valid GGUF ({file_size} bytes): {path}",
                path=path,
                reason="file_too_small",
            )

        if max_total_bytes is None:
            max_total_bytes = self._default_max_bytes()

        if file_size > max_total_bytes:
            raise GGUFValidationError(
                f"File size ({file_size:,} bytes) exceeds limit "
                f"({max_total_bytes:,} bytes): {path}",
                path=path,
                reason="file_too_large",
            )

        # -- Header parsing ----------------------------------------------------
        with path.open("rb") as f:
            header_bytes = f.read(24)

        if len(header_bytes) < 24:
            raise GGUFValidationError(
                f"Could not read GGUF header from: {path}",
                path=path,
                reason="read_error",
            )

        # GGUF header layout (little-endian):
        #   magic:              uint32  (4 bytes)
        #   version:            uint32  (4 bytes)
        #   tensor_count:       uint64  (8 bytes)
        #   metadata_kv_count:  uint64  (8 bytes)
        magic, version, tensor_count, _metadata_kv_count = struct.unpack("<IIqq", header_bytes)

        # -- Magic check -------------------------------------------------------
        if magic != _GGUF_MAGIC:
            raise GGUFValidationError(
                f"Invalid GGUF magic bytes (got 0x{magic:08X}, "
                f"expected 0x{_GGUF_MAGIC:08X}): {path}",
                path=path,
                reason="invalid_magic",
            )

        # -- Version check -----------------------------------------------------
        if version not in (1, 2, 3):
            raise GGUFValidationError(
                f"Unsupported GGUF version {version}: {path}",
                path=path,
                reason="unsupported_version",
            )

        # -- Tensor count sanity -----------------------------------------------
        if tensor_count < 0 or tensor_count > _MAX_TENSOR_COUNT:
            raise GGUFValidationError(
                f"Unreasonable tensor count ({tensor_count}) in GGUF header: {path}",
                path=path,
                reason="unreasonable_tensor_count",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_max_bytes() -> int:
        """Return 2x physical RAM as the default file-size ceiling."""
        try:
            # Works on macOS and Linux.
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            physical_ram = page_size * page_count
            return physical_ram * 2
        except (ValueError, OSError):
            # Fallback: 128 GB (generous upper bound for Apple Silicon).
            return 128 * (1024**3)
