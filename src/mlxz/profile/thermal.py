"""Thermal monitoring for Apple Silicon.

Samples macOS thermal state via ``pmset`` and provides a best-effort CPU
die temperature reading.  Falls back gracefully when IOKit or
``powermetrics`` access is unavailable (e.g. running without ``sudo``).
"""

from __future__ import annotations

import subprocess

from mlxz.types import ThermalState


class ThermalMonitor:
    """Non-privileged thermal state sampler.

    Uses ``pmset -g therm`` (no root required) for throttle detection.
    Temperature reading requires ``sudo powermetrics`` and is therefore
    best-effort -- returns ``None`` when unavailable.
    """

    def sample(self) -> ThermalState:
        """Return the current thermal throttle classification.

        Parses ``pmset -g therm`` output.  The command reports the
        ``CPU_Speed_Limit`` as a percentage (100 = no throttle).

        Returns :attr:`ThermalState.NORMAL` on any detection failure.
        """
        try:
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return ThermalState.NORMAL

            for line in result.stdout.splitlines():
                line_lower = line.strip().lower()
                if "cpu_speed_limit" in line_lower:
                    # Format: "CPU_Speed_Limit  = 100"
                    parts = line.split("=")
                    if len(parts) == 2:
                        try:
                            speed_limit = int(parts[1].strip())
                        except ValueError:
                            continue
                        if speed_limit >= 90:
                            return ThermalState.NORMAL
                        if speed_limit >= 60:
                            return ThermalState.WARN
                        return ThermalState.CRITICAL

        except (OSError, subprocess.TimeoutExpired):
            pass

        return ThermalState.NORMAL

    def get_temperature(self) -> float | None:
        """Best-effort CPU die temperature in degrees Celsius.

        Attempts to read from ``powermetrics`` (requires root).  Returns
        ``None`` when the reading is unavailable or the command fails.
        """
        try:
            result = subprocess.run(
                [
                    "sudo", "-n",  # non-interactive; fail instead of prompting
                    "powermetrics",
                    "--samplers", "smc",
                    "-n", "1",
                    "-i", "100",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            for line in result.stdout.splitlines():
                lower = line.strip().lower()
                # powermetrics outputs lines like:
                #   CPU die temperature: 45.32 C
                if "cpu die temperature" in lower or "die temp" in lower:
                    parts = line.split(":")
                    if len(parts) == 2:
                        temp_str = parts[1].strip().rstrip(" C").rstrip(" c")
                        try:
                            return float(temp_str)
                        except ValueError:
                            pass

        except (OSError, subprocess.TimeoutExpired):
            pass

        return None
