"""Add ohlcv_cache table for market data caching.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-05
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ohlcv_cache",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("bar_date", sa.Date, nullable=False),
        sa.Column("frequency", sa.String(10), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=False),
        sa.Column("trade_count", sa.Float, nullable=True),
        sa.Column("vwap", sa.Float, nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "symbol",
            "bar_date",
            "frequency",
            "source",
            name="uq_ohlcv_symbol_date_freq_source",
        ),
    )
    op.create_index(
        "ix_ohlcv_symbol_freq",
        "ohlcv_cache",
        ["symbol", "frequency"],
    )


def downgrade() -> None:
    op.drop_index("ix_ohlcv_symbol_freq", table_name="ohlcv_cache")
    op.drop_table("ohlcv_cache")
