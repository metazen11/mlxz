"""Hardware detection for Apple Silicon.

Probes chip identity, core counts, memory capacity, and MLX availability
using stdlib introspection and macOS ``sysctl`` queries.  Every field is
best-effort -- the module never raises on detection failure; it returns
sensible defaults instead.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Immutable snapshot of the host hardware profile."""

    chip_name: str
    cpu_cores: int
    gpu_cores: int
    memory_gb: float
    memory_bandwidth_gbs: float
    mlx_version: str
    os_version: str


def _run_sysctl(key: str) -> str | None:
    """Return the value of a sysctl key, or ``None`` on any failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_chip_name() -> str:
    """Best-effort chip name detection (e.g. 'Apple M4 Max')."""
    brand = _run_sysctl("machdep.cpu.brand_string")
    if brand:
        return brand
    proc = platform.processor()
    return proc if proc else "unknown"


def _detect_gpu_cores() -> int:
    """Try to read GPU core count from IOKit via system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                lower = line.lower().strip()
                if "total number of cores" in lower or "gpu cores" in lower:
                    # Extract the numeric part
                    parts = lower.split(":")
                    if len(parts) == 2:
                        digits = "".join(c for c in parts[1] if c.isdigit())
                        if digits:
                            return int(digits)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return 0


def _detect_memory_gb() -> float:
    """Read physical RAM in GB via sysctl."""
    raw = _run_sysctl("hw.memsize")
    if raw:
        try:
            return int(raw) / (1024**3)
        except ValueError:
            pass
    return 0.0


def _detect_mlx_version() -> str:
    """Return the installed MLX version, or 'not installed'."""
    try:
        import mlx  # type: ignore[import-untyped]  # noqa: F401

        # MLX does not always expose __version__ directly; use
        # importlib.metadata as the reliable source.
        ver = getattr(mlx, "__version__", None)
        if ver:
            return str(ver)
        from importlib.metadata import version as pkg_version

        return pkg_version("mlx")
    except ImportError:
        return "not installed"
    except Exception:
        return "unknown"


# Approximate memory bandwidth table for known Apple Silicon chips (GB/s).
# Used as a fallback when runtime measurement is not available.
_BANDWIDTH_TABLE: dict[str, float] = {
    "M1": 68.25,
    "M1 Pro": 200.0,
    "M1 Max": 400.0,
    "M1 Ultra": 800.0,
    "M2": 100.0,
    "M2 Pro": 200.0,
    "M2 Max": 400.0,
    "M2 Ultra": 800.0,
    "M3": 100.0,
    "M3 Pro": 150.0,
    "M3 Max": 400.0,
    "M3 Ultra": 800.0,
    "M4": 120.0,
    "M4 Pro": 273.0,
    "M4 Max": 546.0,
    "M4 Ultra": 819.0,
}


def _estimate_bandwidth(chip_name: str) -> float:
    """Estimate memory bandwidth from chip name using a lookup table."""
    for suffix, bw in sorted(
        _BANDWIDTH_TABLE.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if suffix in chip_name:
            return bw
    return 0.0


def detect_hardware() -> HardwareInfo:
    """Probe the host and return a best-effort :class:`HardwareInfo`."""
    chip_name = _detect_chip_name()
    cpu_cores = os.cpu_count() or 0
    gpu_cores = _detect_gpu_cores()
    memory_gb = _detect_memory_gb()
    mlx_version = _detect_mlx_version()

    mac_ver = platform.mac_ver()
    os_version = mac_ver[0] if mac_ver[0] else platform.platform()

    memory_bandwidth_gbs = _estimate_bandwidth(chip_name)

    return HardwareInfo(
        chip_name=chip_name,
        cpu_cores=cpu_cores,
        gpu_cores=gpu_cores,
        memory_gb=round(memory_gb, 2),
        memory_bandwidth_gbs=memory_bandwidth_gbs,
        mlx_version=mlx_version,
        os_version=os_version,
    )
