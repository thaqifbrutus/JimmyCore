import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from db.database import Base


class QualityReport(Base):
    """
    A report is an analysis of some tabular data — it doesn't matter
    whether that data came from a user's uploaded CSV or a dataset found
    via a government-catalog search. Both flows share this one table
    rather than duplicating get_report / technical-context / ask logic
    across two near-identical tables.

    Exactly one of dataset_id / catalog_dataset_id is set per row, never
    both, never neither — enforced at the database level (not just trusted
    to application code) via the CheckConstraint below. Postgres supports
    `!=` on two boolean expressions as XOR, so this reads as "exactly one
    of these two IS NOT NULL checks is true."
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
