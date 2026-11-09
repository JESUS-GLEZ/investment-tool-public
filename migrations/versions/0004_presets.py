"""Add per-user run presets.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "presets",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("run_type", sa.String(30), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column(
            "parameters", sa.JSON().with_variant(JSONB, "postgresql"), nullable=False
        ),
        sa.UniqueConstraint(
            "user_id", "run_type", "name", name="uq_preset_owner_type_name"
        ),
    )


def downgrade():
    op.drop_table("presets")
