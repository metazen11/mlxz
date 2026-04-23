"""SQLAlchemy 2.x ORM models for telemetry storage.

Tables
------
- ``runs`` -- one row per benchmark invocation or server session.
- ``requests`` -- one row per OpenAI-compatible API call.
- ``measurements`` -- fine-grained system samples taken during a run.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all telemetry tables."""


class Run(Base):
    """A single benchmark invocation or long-running server session."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    commit_sha: Mapped[str] = mapped_column(String(40))
    hardware: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(256))
    draft_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    quant: Mapped[str] = mapped_column(String(32))
    kv_bits: Mapped[int]
    wired_limit_mb: Mapped[int]
    config_json: Mapped[str] = mapped_column(Text)
    """Serialised RuntimeConfig JSON — SecretStr fields are excluded."""
    started_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
    )

    requests: Mapped[list[RequestRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )
    measurements: Mapped[list[Measurement]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )


class RequestRow(Base):
    """Per-request telemetry captured by the engine."""

    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    """UUID string."""
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]
    prefix_cache_hit_tokens: Mapped[int]
    ttft_ms: Mapped[float]
    decode_tps: Mapped[float]
    acceptance_rate: Mapped[float | None] = mapped_column(nullable=True)
    rejected_reason: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
    )

    run: Mapped[Run] = relationship(back_populates="requests")


class Measurement(Base):
    """Fine-grained system sample taken periodically during a run."""

    __tablename__ = "measurements"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    sampled_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
    )
    batch_size: Mapped[int]
    aggregate_decode_tps: Mapped[float]
    kv_used_bytes: Mapped[int]
    rss_bytes: Mapped[int]
    thermal_state: Mapped[str] = mapped_column(String(16))

    run: Mapped[Run] = relationship(back_populates="measurements")
