import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from db.database import Base


class QualityReport(Base):
    """
    An analysis session over some tabular data. Both flows — uploaded CSV
    and government-catalog dataset — share this one table so get_report,
    ask, and every downstream /reports/{id} endpoint work identically
    regardless of where the data came from.

    Exactly one of dataset_id / catalog_dataset_id is set per row, never
    both, never neither — enforced at the database level via the
    CheckConstraint below. Postgres supports `!=` on two boolean
    expressions as XOR, so this reads as "exactly one of these two IS NOT
    NULL checks is true."
    """
    __tablename__ = "quality_reports"
    __table_args__ = (
        CheckConstraint(
            "(dataset_id IS NOT NULL) != (catalog_dataset_id IS NOT NULL)",
            name="ck_report_exactly_one_source",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Uploaded-file flow (existing). Nullable now — a catalog-search report
    # won't have an uploaded Dataset row behind it.
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True)

    # Government-catalog-search flow (new). References CatalogDataset.id,
    # which is a plain string (the government's own dataset id), not a UUID.
    catalog_dataset_id = Column(String, ForeignKey("catalog_datasets.id"), nullable=True)

    profile_data = Column(JSONB, nullable=True)
    ai_summary = Column(Text, nullable=True)
    overall_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Persisted chat history for this session. Shape: list of dicts, each
    # with keys {"role", "content", "tool_calls"|None, "error_type"|None,
    # "timestamp"}. Nullable — pre-migration rows have NULL, treated as
    # an empty list by the API layer.
    chat_messages = Column(JSONB, nullable=True)