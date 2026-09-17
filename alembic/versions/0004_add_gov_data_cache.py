"""add gov_data_cache table

Revision ID: 0004_add_gov_data_cache
Revises: 0003_add_catalog_embedding
Create Date: 2026-09-16

Adds the cache table used by app/services/gov_data_client.py to avoid
re-fetching the same dataset from api.data.gov.my's rate-limited (4
requests/minute) query API on every search hit.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004_add_gov_data_cache"
down_revision: Union[str, None] = "0003_add_catalog_embedding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gov_data_cache",
        sa.Column("catalog_dataset_id", sa.String(), sa.ForeignKey("catalog_datasets.id"), primary_key=True),
        sa.Column("raw_data", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("gov_data_cache")
