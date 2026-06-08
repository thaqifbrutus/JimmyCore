import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from db.database import Base

class QualityReport(Base):
    __tablename__ = "quality_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    profile_data = Column(JSONB, nullable=True)
    ai_summary = Column(Text, nullable=True)
    overall_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)