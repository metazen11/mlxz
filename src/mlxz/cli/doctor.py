"""``mlxz doctor`` -- environment diagnostics.

Runs a battery of checks (Python version, platform, MLX, chip, memory,
wired limit, thermal state) and prints a colour-coded PASS/FAIL/WARN
report.  Designed to *never* crash -- every check is wrapped in its own
error boundary so that a single failure does not prevent the remaining
checks from running.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Literal

import typer

from mlxz.profile.hardware import detect_hardware
from mlxz.profile.thermal import ThermalMonitor
from mlxz.types import ThermalState

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

_Status = Literal["PASS", "FAIL", "WARN"]


def _label(status: _Status) -> str:
    """Return a coloured status label for terminal output."""
    colours = {
        "PASS": typer.colors.GREEN,
        "FAIL": typer.colors.RED,
        "WARN": typer.colors.YELLOW,
    }
    return typer.style(f"[{status}]", fg=colours[status], bold=True)


def _print_check(name: str, status: _Status, detail: str) -> None:
    typer.echo(f"  {_label(status)}  {name}: {detail}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_python_version() -> tuple[_Status, str]:
    vi = sys.version_info
    version_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if (vi.major, vi.minor) >= (3, 11):
        return "PASS", version_str
    return "FAIL", f"{version_str} (requires >= 3.11)"


def _check_platform() -> tuple[_Status, str]:
    if sys.platform == "darwin":
        mac_ver = platform.mac_ver()[0]
        label = f"macOS {mac_ver}" if mac_ver else "macOS"
        return "PASS", label
    return "FAIL", f"{sys.platform} (macOS required)"


def _check_mlx(hw_mlx_version: str) -> tuple[_Status, str]:
    if hw_mlx_version == "not installed":
        return "FAIL", "MLX not installed (pip install mlx)"
    return "PASS", f"mlx {hw_mlx_version}"


def _check_chip(chip_name: str) -> tuple[_Status, str]:
    if chip_name == "unknown":
        return "WARN", "could not detect chip"
    if "Apple" in chip_name or "apple" in chip_name:
        return "PASS", chip_name
    # Non-Apple silicon -- MLX may still work on x86 Mac but perf will be bad.
    return "WARN", f"{chip_name} (Apple Silicon recommended)"


def _check_memory(memory_gb: float) -> tuple[_Status, str]:
    if memory_gb <= 0:
        return "WARN", "could not detect memory"
    detail = f"{memory_gb:.1f} GB"
    if memory_gb >= 16:
        return "PASS", detail
    if memory_gb >= 8:
        return "WARN", f"{detail} (16 GB+ recommended for most models)"
    return "FAIL", f"{detail} (insufficient for most models)"


def _check_wired_limit() -> tuple[_Status, str]:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip())
            if mb > 0:
                gb = mb / 1024
                return "PASS", f"{mb} MB ({gb:.1f} GB)"
            return "WARN", "not set (auto-detect from hw.memsize)"
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return "WARN", "not set (auto-detect from hw.memsize)"


def _check_thermal() -> tuple[_Status, str]:
    try:
        monitor = ThermalMonitor()
        state = monitor.sample()
        if state == ThermalState.NORMAL:
            return "PASS", "normal"
        if state == ThermalState.WARN:
            return "WARN", "thermal throttling detected"
        return "FAIL", "critical thermal throttling"
    except Exception:
        return "WARN", "could not read thermal state"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def doctor(
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat WARNs as FAILs.",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Run a smoke test with a small model (Phase 1 placeholder).",
    ),
) -> None:
    """Run environment diagnostics and report PASS/FAIL/WARN per check."""
    typer.echo()
    typer.echo(
        typer.style("mlxz doctor", bold=True) + " -- environment diagnostics"
    )
    typer.echo()

    # Detect hardware once; individual checks reuse the result.
    try:
        hw = detect_hardware()
    except Exception:
        typer.echo(
            typer.style("  [FAIL]", fg=typer.colors.RED, bold=True)
            + "  hardware detection crashed"
        )
        raise typer.Exit(code=1)

    # Ordered list of (name, status, detail).
    checks: list[tuple[str, _Status, str]] = []

    check_fns: list[tuple[str, tuple[_Status, str]]] = [
        ("Python version", _check_python_version()),
        ("Platform", _check_platform()),
        ("MLX", _check_mlx(hw.mlx_version)),
        ("Chip", _check_chip(hw.chip_name)),
        ("Memory", _check_memory(hw.memory_gb)),
        ("Wired limit", _check_wired_limit()),
        ("Thermal", _check_thermal()),
    ]

    for name, (status, detail) in check_fns:
        checks.append((name, status, detail))

    # Print results.
    has_fail = False
    for name, status, detail in checks:
        effective = status
        if strict and status == "WARN":
            effective = "FAIL"
        if effective == "FAIL":
            has_fail = True
        _print_check(name, effective, detail)

    typer.echo()

    # Smoke test placeholder.
    if smoke:
        typer.echo(
            typer.style("  [INFO]", fg=typer.colors.BLUE, bold=True)
            + "  Smoke test requires a model -- not yet implemented (Phase 1)."
        )
        typer.echo()

    if has_fail:
        raise typer.Exit(code=1)
