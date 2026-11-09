"""SQLAlchemy ORM models for the Investment Tool API.

Tables
------
users          User accounts (email + hashed password).
runs           Execution / comparison runs with status lifecycle.
run_results    JSON result payload for completed runs (1-to-1 with runs).
ohlcv_cache    Cached OHLCV market data bars (symbol + date + frequency).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Use JSON on SQLite (tests) and JSONB on PostgreSQL (production).
_JSONColumn = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class UserRow(Base):
    """A registered user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # relationships
    runs: Mapped[list[RunRow]] = relationship(back_populates="user", lazy="selectin")


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class CampaignRow(Base):
    """User-owned research context; never controls engine execution."""

    __tablename__ = "campaigns"
    __table_args__ = (Index("ix_campaigns_user_updated", "user_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, default="", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class RunRow(Base):
    """A single execution or comparison run."""

    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_user_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    run_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    request_payload: Mapped[dict] = mapped_column(
        _JSONColumn, nullable=False, default=dict
    )  # type: ignore[type-arg]
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("campaigns.id", ondelete="SET NULL"), nullable=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    progress_fraction: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    progress_message: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # relationships
    user: Mapped[UserRow] = relationship(back_populates="runs")
    result: Mapped[RunResultRow | None] = relationship(
        back_populates="run",
        uselist=False,
        lazy="joined",
        cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# Run Results
# ---------------------------------------------------------------------------


class RunResultRow(Base):
    """JSON result payload for a completed run (kept separate to keep the
    ``runs`` table lightweight for listing queries)."""

    __tablename__ = "run_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    result: Mapped[dict] = mapped_column(_JSONColumn, nullable=False, default=dict)  # type: ignore[type-arg]

    # relationships
    run: Mapped[RunRow] = relationship(back_populates="result")


# ---------------------------------------------------------------------------
# OHLCV Market Data Cache
# ---------------------------------------------------------------------------


class OHLCVCacheRow(Base):
    """Cached OHLCV bar for a single symbol on a single date.

    The combination of ``(symbol, bar_date, frequency, source)`` is unique
    so that repeated fetches for the same data are de-duplicated.
    """

    __tablename__ = "ohlcv_cache"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "bar_date",
            "frequency",
            "source",
            name="uq_ohlcv_symbol_date_freq_source",
        ),
        Index("ix_ohlcv_symbol_freq", "symbol", "frequency"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    bar_date: Mapped[datetime] = mapped_column(Date, nullable=False)  # type: ignore[assignment]
    frequency: Mapped[str] = mapped_column(String(10), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    trade_count: Mapped[float | None] = mapped_column(Float, nullable=True)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    timing: Mapped[dict[str, Any] | None] = mapped_column(_JSONColumn, nullable=True)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class PresetRow(Base):
    """A user's reusable execution or comparison configuration."""

    __tablename__ = "presets"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "run_type", "name", name="uq_preset_owner_type_name"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    run_type: Mapped[str] = mapped_column(String(30), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    parameters: Mapped[dict] = mapped_column(_JSONColumn, nullable=False)
