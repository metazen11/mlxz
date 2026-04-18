"""Hardware profiling, thermal monitoring, and residency planning."""

from mlxz.profile.hardware import HardwareInfo, detect_hardware
from mlxz.profile.residency import ResidencyPlanner
from mlxz.profile.thermal import ThermalMonitor

__all__ = [
    "HardwareInfo",
    "ResidencyPlanner",
    "ThermalMonitor",
    "detect_hardware",
]
