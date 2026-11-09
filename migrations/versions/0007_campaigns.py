"""Optional user-owned campaigns; existing runs remain ungrouped."""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "campaigns",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_campaigns_user_updated", "campaigns", ["user_id", "updated_at"])
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("campaign_id", sa.String(12), nullable=True))
        batch.create_foreign_key(
            "fk_runs_campaign_id",
            "campaigns",
            ["campaign_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index("ix_runs_campaign_id", ["campaign_id"])


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_index("ix_runs_campaign_id")
        batch.drop_constraint("fk_runs_campaign_id", type_="foreignkey")
        batch.drop_column("campaign_id")
    op.drop_index("ix_campaigns_user_updated", table_name="campaigns")
    op.drop_table("campaigns")
