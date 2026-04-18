"""Shared test fixtures for mlxz test suite."""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "mlxz.toml"
    config_path.write_text(
        '[project]\nmodel = "test-model"\n'
    )
    return config_path
