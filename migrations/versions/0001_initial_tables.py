"""Initial migration: create users, runs, and run_results tables.

Revision ID: 0001
Revises: None
Create Date: 2026-04-05
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(320), unique=True, nullable=False),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # --- runs ---
    op.create_table(
        "runs",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer,
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("run_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column(
            "request_payload",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default="{}",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("celery_task_id", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_runs_user_created",
        "runs",
        ["user_id", "created_at"],
    )

    # --- run_results ---
    op.create_table(
        "run_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(12),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column(
            "result",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_table("run_results")
    op.drop_index("ix_runs_user_created", table_name="runs")
    op.drop_table("runs")
    op.drop_table("users")
