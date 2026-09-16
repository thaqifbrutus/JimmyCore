"""baseline: current schema as it exists in production today

Revision ID: 0001_baseline
Revises:
Create Date: 2026-09-13

IMPORTANT — this migration is never meant to be run with `alembic upgrade`
against your real database. Your tables already exist (created via
Base.metadata.create_all()); running this would fail with "table already
exists" errors. Instead, mark it as already-satisfied without executing it:

    alembic stamp 0001_baseline

That tells Alembic "history starts here, nothing to actually do" — then
the real schema change (0002) runs normally with `alembic upgrade head`.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("original_name", sa.String(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("column_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("uploaded_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "quality_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id"),
            nullable=False,
        ),
        sa.Column("profile_data", postgresql.JSONB(), nullable=True),
        sa.Column("ai_summary", sa.Text(), nullable=True),
        sa.Column("overall_status", sa.String(), server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id"),
            nullable=True,
        ),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("quality_reports.id"),
            nullable=True,
        ),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("quality_reports")
    op.drop_table("datasets")
