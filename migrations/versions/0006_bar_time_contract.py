"""Preserve canonical bar timing and fetched coverage in the OHLCV cache."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ohlcv_cache",
        sa.Column("timing", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ohlcv_cache", "timing")
