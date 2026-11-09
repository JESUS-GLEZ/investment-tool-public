"""Persist worker heartbeat and latest progress."""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "runs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "runs",
        sa.Column("progress_fraction", sa.Float, server_default="0", nullable=False),
    )
    op.add_column(
        "runs",
        sa.Column("progress_message", sa.Text, server_default="", nullable=False),
    )


def downgrade():
    for column in ("progress_message", "progress_fraction", "heartbeat_at"):
        op.drop_column("runs", column)
