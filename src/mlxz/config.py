"""Runtime configuration for mlxz.

Loads from TOML file, environment variables (``MLXZ_`` prefix), and CLI
overrides.  All nested models are immutable after construction; the
``RuntimeConfig`` instance is the single source of truth threaded through
the application via dependency injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------


class KVConfig(BaseModel):
    """Quantised KV-cache parameters."""

    bits: Literal[4, 8, 16] = 8
    group_size: int = Field(default=64, ge=1, le=256)
    quantized_kv_start: int = Field(default=256, ge=0)
    """Number of initial tokens kept at FP16 before quantisation kicks in."""
    streaming_sink_size: int = Field(default=4, ge=1)
    """Attention-sink tokens pinned at the start of the window (Xiao et al. 2023)."""


class PagedConfig(BaseModel):
    """Paged-attention block layout (Phase 3)."""

    block_size: int = Field(default=16, ge=1, le=256)
    """Tokens per physical block."""
    enabled: bool = False


class PrefixCacheConfig(BaseModel):
    """Two-tier prefix cache budget and paths."""

    memory_budget_gb: float = Field(default=8.0, gt=0)
    disk_budget_gb: float = Field(default=50.0, gt=0)
    disk_path: Path = Path.home() / ".cache/mlxz/prefix"
    disk_tier_enabled: bool = True
    block_size: int = Field(default=8, ge=1, le=256)

    # NOTE: At runtime the engine appends a model-name hash to ``disk_path``
    # so that different models never share on-disk prefix data.

    @model_validator(mode="after")
    def _include_model_hash_in_path(self) -> Self:
        """Placeholder — actual path rewriting happens at engine init."""
        return self


class SpeculativeConfig(BaseModel):
    """Draft-target speculative decoding knobs."""

    enabled: bool = False
    draft_model: str | None = None
    num_draft_tokens: int = Field(default=4, ge=1, le=16)
    max_draft_tokens: int = Field(default=8, ge=1, le=32)
    backoff_threshold: float = Field(default=0.5, gt=0, le=1.0)

    @model_validator(mode="after")
    def _draft_tokens_order(self) -> Self:
        """Ensure ``num_draft_tokens`` never exceeds ``max_draft_tokens``."""
        if self.num_draft_tokens > self.max_draft_tokens:
            msg = (
                f"num_draft_tokens ({self.num_draft_tokens}) must be "
                f"<= max_draft_tokens ({self.max_draft_tokens})"
            )
            raise ValueError(msg)
        return self


class SchedulerConfig(BaseModel):
    """Continuous-batching scheduler limits."""

    max_concurrent_requests: int = Field(default=8, ge=1, le=128)
    chunked_prefill_chunk_tokens: int = Field(default=128, ge=1)
    admission_headroom: float = Field(default=0.10, gt=0, le=0.5)


class AttentionConfig(BaseModel):
    """MLX attention kernel tuning knobs."""

    memory_efficient_threshold: int | None = None


class RequestLimits(BaseModel):
    """Hard caps enforced in Pydantic request schemas before tokenisation."""

    max_prompt_tokens: int = Field(default=32768, ge=1, le=131072)
    max_completion_tokens: int = Field(default=4096, ge=1, le=32768)
    max_request_body_bytes: int = Field(default=10_485_760, ge=1)
    """10 MB default."""
    max_concurrent_per_client: int = Field(default=16, ge=1)
    request_timeout_seconds: float = Field(default=300.0, ge=1.0)


class ServerConfig(BaseModel):
    """HTTP server, TLS, CORS, and auth settings."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: SecretStr | None = None
    """Pydantic ``SecretStr`` — never serialised to logs or telemetry."""
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None
    metrics_bind: str = "127.0.0.1:9090"
    """Separate bind address for ``/metrics``; never exposed publicly."""
    cors_origins: list[str] = Field(default_factory=list)
    """Empty list means CORS is disabled."""
    request_timeout_seconds: float = Field(default=300.0, ge=1.0)
    request_limits: RequestLimits = Field(default_factory=RequestLimits)


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseSettings):
    """Root configuration object.

    Precedence: CLI flags > environment variables > TOML file > defaults.

    Environment variables use the ``MLXZ_`` prefix with ``__`` as the nested
    delimiter (e.g. ``MLXZ_SERVER__PORT=9000``).
    """

    model_config = SettingsConfigDict(
        env_prefix="MLXZ_",
        env_nested_delimiter="__",
        toml_file="mlxz.toml",
    )

    model: str
    """HuggingFace repo ID or local path to the model directory."""
    draft_model: str | None = None
    wired_limit_mb: int | None = None
    """``None`` means auto-probe via ``iogpu.wired_limit_mb``."""

    kv: KVConfig = Field(default_factory=KVConfig)
    paged: PagedConfig = Field(default_factory=PagedConfig)
    prefix_cache: PrefixCacheConfig = Field(default_factory=PrefixCacheConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Enable TOML loading in addition to init kwargs and environment."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls),
        )
