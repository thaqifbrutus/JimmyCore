"""add embedding column to catalog_datasets

Revision ID: 0003_add_catalog_embedding
Revises: 0002_catalog_and_shared_reports
Create Date: 2026-09-13

Adds the JSON-serialized embedding vector column used by
app/services/catalog_search.py for semantic search. Nullable — a fresh
sync can add new catalog rows with no embedding yet, and
generate_catalog_embeddings() is a separate step that fills this in
afterward, not something that happens automatically during sync.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003_add_catalog_embedding"
down_revision: Union[str, None] = "0002_catalog_and_shared_reports"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "catalog_datasets",
        sa.Column("embedding", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("catalog_datasets", "embedding")
