"""add chat_messages to quality_reports

Revision ID: 0005_add_chat_messages
Revises: 0004_add_gov_data_cache
Create Date: 2026-09-30

Adds a JSONB column for persisting the chat history of an analysis
session. One session per report; the report_id is the session id.
Shape: list of {"role", "content", "tool_calls"|None, "error_type"|None,
"timestamp"} dicts. Nullable — existing rows get NULL, which is treated
as an empty list by the API layer.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005_add_chat_messages"
down_revision: Union[str, None] = "0004_add_gov_data_cache"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "quality_reports",
        sa.Column("chat_messages", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("quality_reports", "chat_messages")