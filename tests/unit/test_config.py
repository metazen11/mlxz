"""Unit tests for mlxz.config — defaults, validation, and env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from mlxz.config import (
    KVConfig,
    PagedConfig,
    PrefixCacheConfig,
    RequestLimits,
    RuntimeConfig,
    SchedulerConfig,
    ServerConfig,
    SpeculativeConfig,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _runtime(*, model: str = "mlx-community/test-model", **overrides) -> RuntimeConfig:
    """Build a RuntimeConfig with only ``model`` required, no TOML/env."""
    return RuntimeConfig(model=model, **overrides)


# ===================================================================
# 1. RuntimeConfig loads with all defaults
# ===================================================================


class TestRuntimeConfigDefaults:
    """RuntimeConfig should be constructable with only ``model``."""

    def test_minimal_construction(self):
        cfg = _runtime()
        assert cfg.model == "mlx-community/test-model"
        assert cfg.draft_model is None
        assert cfg.wired_limit_mb is None

    def test_all_nested_sections_present(self):
        cfg = _runtime()
        assert isinstance(cfg.kv, KVConfig)
        assert isinstance(cfg.paged, PagedConfig)
        assert isinstance(cfg.prefix_cache, PrefixCacheConfig)
        assert isinstance(cfg.speculative, SpeculativeConfig)
        assert isinstance(cfg.scheduler, SchedulerConfig)
        assert isinstance(cfg.server, ServerConfig)


# ===================================================================
# 2. Each nested config has correct defaults
# ===================================================================


class TestKVConfigDefaults:
    def test_defaults(self):
        kv = KVConfig()
        assert kv.bits == 8
        assert kv.group_size == 64
        assert kv.quantized_kv_start == 512
        assert kv.streaming_sink_size == 4


class TestPagedConfigDefaults:
    def test_defaults(self):
        p = PagedConfig()
        assert p.block_size == 16
        assert p.enabled is False


class TestPrefixCacheConfigDefaults:
    def test_defaults(self):
        pc = PrefixCacheConfig()
        assert pc.memory_budget_gb == 8.0
        assert pc.disk_budget_gb == 50.0
        assert pc.disk_path == Path.home() / ".cache/mlxz/prefix"
        assert pc.disk_tier_enabled is True
        assert pc.block_size == 8


class TestSpeculativeConfigDefaults:
    def test_defaults(self):
        sp = SpeculativeConfig()
        assert sp.enabled is False
        assert sp.draft_model is None
        assert sp.num_draft_tokens == 4
        assert sp.max_draft_tokens == 8
        assert sp.backoff_threshold == 0.5


class TestSchedulerConfigDefaults:
    def test_defaults(self):
        sc = SchedulerConfig()
        assert sc.max_concurrent_requests == 8
        assert sc.chunked_prefill_chunk_tokens == 128
        assert sc.admission_headroom == 0.10


class TestServerConfigDefaults:
    def test_defaults(self):
        srv = ServerConfig()
        assert srv.host == "127.0.0.1"
        assert srv.port == 8000
        assert srv.api_key is None
        assert srv.ssl_certfile is None
        assert srv.ssl_keyfile is None
        assert srv.metrics_bind == "127.0.0.1:9090"
        assert srv.cors_origins == []
        assert srv.request_timeout_seconds == 300.0
        assert isinstance(srv.request_limits, RequestLimits)


class TestRequestLimitsDefaults:
    def test_defaults(self):
        rl = RequestLimits()
        assert rl.max_prompt_tokens == 32768
        assert rl.max_completion_tokens == 4096
        assert rl.max_request_body_bytes == 10_485_760
        assert rl.max_concurrent_per_client == 16
        assert rl.request_timeout_seconds == 300.0


# ===================================================================
# 3. Field validators reject invalid values
# ===================================================================


class TestKVConfigValidation:
    def test_group_size_zero_rejected(self):
        with pytest.raises(ValidationError, match="group_size"):
            KVConfig(group_size=0)

    def test_group_size_negative_rejected(self):
        with pytest.raises(ValidationError, match="group_size"):
            KVConfig(group_size=-1)

    def test_group_size_over_max_rejected(self):
        with pytest.raises(ValidationError, match="group_size"):
            KVConfig(group_size=257)

    def test_invalid_bits_rejected(self):
        with pytest.raises(ValidationError):
            KVConfig(bits=3)  # type: ignore[arg-type]

    def test_quantized_kv_start_negative_rejected(self):
        with pytest.raises(ValidationError, match="quantized_kv_start"):
            KVConfig(quantized_kv_start=-1)

    def test_streaming_sink_size_zero_rejected(self):
        with pytest.raises(ValidationError, match="streaming_sink_size"):
            KVConfig(streaming_sink_size=0)


class TestSchedulerConfigValidation:
    def test_admission_headroom_too_high_rejected(self):
        with pytest.raises(ValidationError, match="admission_headroom"):
            SchedulerConfig(admission_headroom=2.0)

    def test_admission_headroom_zero_rejected(self):
        with pytest.raises(ValidationError, match="admission_headroom"):
            SchedulerConfig(admission_headroom=0.0)

    def test_admission_headroom_negative_rejected(self):
        with pytest.raises(ValidationError, match="admission_headroom"):
            SchedulerConfig(admission_headroom=-0.1)

    def test_max_concurrent_zero_rejected(self):
        with pytest.raises(ValidationError, match="max_concurrent_requests"):
            SchedulerConfig(max_concurrent_requests=0)

    def test_max_concurrent_over_limit_rejected(self):
        with pytest.raises(ValidationError, match="max_concurrent_requests"):
            SchedulerConfig(max_concurrent_requests=200)


class TestServerConfigValidation:
    def test_port_zero_rejected(self):
        with pytest.raises(ValidationError, match="port"):
            ServerConfig(port=0)

    def test_port_negative_rejected(self):
        with pytest.raises(ValidationError, match="port"):
            ServerConfig(port=-1)

    def test_port_over_max_rejected(self):
        with pytest.raises(ValidationError, match="port"):
            ServerConfig(port=70000)

    def test_request_timeout_below_one_rejected(self):
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            ServerConfig(request_timeout_seconds=0.5)


class TestSpeculativeConfigValidation:
    def test_num_draft_exceeds_max_draft_rejected(self):
        """num_draft_tokens > max_draft_tokens should fail via model_validator."""
        with pytest.raises(ValidationError, match="num_draft_tokens.*<=.*max_draft_tokens"):
            SpeculativeConfig(num_draft_tokens=10, max_draft_tokens=5)

    def test_num_draft_equals_max_draft_accepted(self):
        sp = SpeculativeConfig(num_draft_tokens=8, max_draft_tokens=8)
        assert sp.num_draft_tokens == sp.max_draft_tokens

    def test_backoff_threshold_over_one_rejected(self):
        with pytest.raises(ValidationError, match="backoff_threshold"):
            SpeculativeConfig(backoff_threshold=1.5)

    def test_backoff_threshold_zero_rejected(self):
        with pytest.raises(ValidationError, match="backoff_threshold"):
            SpeculativeConfig(backoff_threshold=0.0)

    def test_num_draft_tokens_zero_rejected(self):
        with pytest.raises(ValidationError, match="num_draft_tokens"):
            SpeculativeConfig(num_draft_tokens=0)


class TestPagedConfigValidation:
    def test_block_size_zero_rejected(self):
        with pytest.raises(ValidationError, match="block_size"):
            PagedConfig(block_size=0)

    def test_block_size_over_max_rejected(self):
        with pytest.raises(ValidationError, match="block_size"):
            PagedConfig(block_size=300)


class TestPrefixCacheConfigValidation:
    def test_memory_budget_zero_rejected(self):
        with pytest.raises(ValidationError, match="memory_budget_gb"):
            PrefixCacheConfig(memory_budget_gb=0)

    def test_disk_budget_negative_rejected(self):
        with pytest.raises(ValidationError, match="disk_budget_gb"):
            PrefixCacheConfig(disk_budget_gb=-1.0)

    def test_block_size_zero_rejected(self):
        with pytest.raises(ValidationError, match="block_size"):
            PrefixCacheConfig(block_size=0)

    def test_block_size_over_max_rejected(self):
        with pytest.raises(ValidationError, match="block_size"):
            PrefixCacheConfig(block_size=300)


class TestRequestLimitsValidation:
    def test_max_prompt_tokens_zero_rejected(self):
        with pytest.raises(ValidationError, match="max_prompt_tokens"):
            RequestLimits(max_prompt_tokens=0)

    def test_max_prompt_tokens_over_limit_rejected(self):
        with pytest.raises(ValidationError, match="max_prompt_tokens"):
            RequestLimits(max_prompt_tokens=200_000)

    def test_max_completion_tokens_zero_rejected(self):
        with pytest.raises(ValidationError, match="max_completion_tokens"):
            RequestLimits(max_completion_tokens=0)

    def test_request_timeout_below_one_rejected(self):
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            RequestLimits(request_timeout_seconds=0.5)


# ===================================================================
# 4. SecretStr api_key is not exposed in string representation
# ===================================================================


class TestSecretStrApiKey:
    def test_api_key_hidden_in_repr(self):
        srv = ServerConfig(api_key="sk-super-secret-key")
        repr_str = repr(srv)
        assert "sk-super-secret-key" not in repr_str

    def test_api_key_hidden_in_str(self):
        srv = ServerConfig(api_key="sk-super-secret-key")
        str_str = str(srv)
        assert "sk-super-secret-key" not in str_str

    def test_api_key_hidden_in_model_dump_json(self):
        srv = ServerConfig(api_key="sk-super-secret-key")
        json_str = srv.model_dump_json()
        assert "sk-super-secret-key" not in json_str

    def test_api_key_retrievable_via_get_secret_value(self):
        srv = ServerConfig(api_key="sk-super-secret-key")
        assert srv.api_key is not None
        assert srv.api_key.get_secret_value() == "sk-super-secret-key"

    def test_api_key_shows_masked_in_repr(self):
        srv = ServerConfig(api_key="sk-super-secret-key")
        repr_str = repr(srv)
        assert "**********" in repr_str

    def test_api_key_none_by_default(self):
        srv = ServerConfig()
        assert srv.api_key is None


# ===================================================================
# 5. RequestLimits defaults and validation (covered above, extra edge cases)
# ===================================================================


class TestRequestLimitsEdgeCases:
    def test_boundary_max_prompt_tokens(self):
        rl = RequestLimits(max_prompt_tokens=1)
        assert rl.max_prompt_tokens == 1

    def test_boundary_max_prompt_tokens_upper(self):
        rl = RequestLimits(max_prompt_tokens=131072)
        assert rl.max_prompt_tokens == 131072

    def test_boundary_max_completion_tokens(self):
        rl = RequestLimits(max_completion_tokens=1)
        assert rl.max_completion_tokens == 1

    def test_boundary_max_completion_tokens_upper(self):
        rl = RequestLimits(max_completion_tokens=32768)
        assert rl.max_completion_tokens == 32768


# ===================================================================
# 6. Environment variable overrides (MLXZ_ prefix)
# ===================================================================


class TestEnvVarOverrides:
    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("MLXZ_MODEL", "env-model/test")
        cfg = RuntimeConfig()  # type: ignore[call-arg]
        assert cfg.model == "env-model/test"

    def test_nested_server_port_from_env(self, monkeypatch):
        monkeypatch.setenv("MLXZ_MODEL", "test-model")
        monkeypatch.setenv("MLXZ_SERVER__PORT", "9999")
        cfg = RuntimeConfig()  # type: ignore[call-arg]
        assert cfg.server.port == 9999

    def test_nested_kv_group_size_from_env(self, monkeypatch):
        monkeypatch.setenv("MLXZ_MODEL", "test-model")
        monkeypatch.setenv("MLXZ_KV__GROUP_SIZE", "128")
        cfg = RuntimeConfig()  # type: ignore[call-arg]
        assert cfg.kv.group_size == 128

    def test_nested_scheduler_max_concurrent_from_env(self, monkeypatch):
        monkeypatch.setenv("MLXZ_MODEL", "test-model")
        monkeypatch.setenv("MLXZ_SCHEDULER__MAX_CONCURRENT_REQUESTS", "32")
        cfg = RuntimeConfig()  # type: ignore[call-arg]
        assert cfg.scheduler.max_concurrent_requests == 32

    def test_wired_limit_from_env(self, monkeypatch):
        monkeypatch.setenv("MLXZ_MODEL", "test-model")
        monkeypatch.setenv("MLXZ_WIRED_LIMIT_MB", "16384")
        cfg = RuntimeConfig()  # type: ignore[call-arg]
        assert cfg.wired_limit_mb == 16384

    def test_explicit_kwarg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MLXZ_MODEL", "env-model")
        cfg = RuntimeConfig(model="explicit-model")
        assert cfg.model == "explicit-model"


# ===================================================================
# 7. Config constructable with minimal required fields (model only)
# ===================================================================


class TestMinimalConstruction:
    def test_model_only(self):
        cfg = RuntimeConfig(model="my-org/my-model")
        assert cfg.model == "my-org/my-model"
        # All nested sections exist with defaults
        assert cfg.kv.bits == 8
        assert cfg.server.port == 8000
        assert cfg.scheduler.max_concurrent_requests == 8
        assert cfg.speculative.enabled is False
        assert cfg.paged.enabled is False
        assert cfg.prefix_cache.disk_tier_enabled is True

    def test_model_required(self):
        """Omitting model without env should raise."""
        with pytest.raises(ValidationError, match="model"):
            RuntimeConfig()  # type: ignore[call-arg]

    def test_model_with_one_override(self):
        cfg = RuntimeConfig(
            model="my-model",
            server=ServerConfig(port=3000),
        )
        assert cfg.server.port == 3000
        # Everything else still default
        assert cfg.kv.bits == 8
        assert cfg.scheduler.admission_headroom == 0.10
