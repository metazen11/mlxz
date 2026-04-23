"""Telemetry subsystem — structured run/request/measurement recording.

Re-exports the public API so callers can write::

    from mlxz.telemetry import create_engine_from_config, TelemetryRecorder
"""

from mlxz.telemetry.db import create_engine_from_config, get_session_factory
from mlxz.telemetry.models import Base, Measurement, RequestRow, Run
from mlxz.telemetry.recorder import TelemetryRecorder

__all__ = [
    "Base",
    "Measurement",
    "RequestRow",
    "Run",
    "TelemetryRecorder",
    "create_engine_from_config",
    "get_session_factory",
]
