"""add catalog_datasets, make quality_reports source-agnostic

Revision ID: 0002_catalog_and_shared_reports
Revises: 0001_baseline
Create Date: 2026-09-13

This is the migration that actually runs: `alembic upgrade head` after
stamping 0001_baseline (see that file's docstring for why it's stamped,
not run).

Two things happen here:
1. New catalog_datasets table — the local search index synced from
   data.gov.my's official dataset list (see app/services/catalog_sync.py).
2. quality_reports becomes source-agnostic: dataset_id becomes nullable,
   a new catalog_dataset_id column is added, and a CHECK constraint
   enforces exactly one of the two is set on every row. Existing rows all
   have dataset_id set already, so they satisfy the constraint
   automatically — no data migration needed for existing reports.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_catalog_and_shared_reports"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "catalog_datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("date_created", sa.DateTime(), nullable=True),
        sa.Column("title_en", sa.String(), nullable=False),
        sa.Column("category_en", sa.String(), nullable=True),
        sa.Column("subcategory_en", sa.String(), nullable=True),
        sa.Column("title_bm", sa.String(), nullable=True),
        sa.Column("category_bm", sa.String(), nullable=True),
        sa.Column("subcategory_bm", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("frequency", sa.String(), nullable=True),
        sa.Column("geography", sa.String(), nullable=True),
        sa.Column("demography", sa.String(), nullable=True),
        sa.Column("dataset_begin", sa.Integer(), nullable=True),
        sa.Column("dataset_end", sa.Integer(), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(), nullable=True),
    )

    # dataset_id was NOT NULL — relax it before adding the constraint that
    # depends on it being nullable in the first place.
    op.alter_column(
        "quality_reports",
        "dataset_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )

    op.add_column(
        "quality_reports",
        sa.Column("catalog_dataset_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_quality_reports_catalog_dataset_id",
        "quality_reports",
        "catalog_datasets",
        ["catalog_dataset_id"],
        ["id"],
    )

    op.create_check_constraint(
        "ck_report_exactly_one_source",
        "quality_reports",
        "(dataset_id IS NOT NULL) != (catalog_dataset_id IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_report_exactly_one_source", "quality_reports", type_="check")
    op.drop_constraint("fk_quality_reports_catalog_dataset_id", "quality_reports", type_="foreignkey")
    op.drop_column("quality_reports", "catalog_dataset_id")
    op.alter_column(
        "quality_reports",
        "dataset_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.drop_table("catalog_datasets")
