"""Add 'cancelled' as a valid run status value.

The ``runs.status`` column is a plain VARCHAR with no CHECK constraint,
so no DDL change is required.  This migration records the intent and
keeps the revision chain intact.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-03
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op  # noqa: F401

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # No DDL needed — status is VARCHAR, not a DB enum.
    pass


def downgrade() -> None:
    pass
