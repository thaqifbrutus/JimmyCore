from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey
from db.database import Base


class GovDataCache(Base):
    """
    Cached raw data fetched from api.data.gov.my for a given catalog
    dataset. One row per dataset (catalog_dataset_id is the primary key,
    not a generated id) — this MVP caches one "full fetch" per dataset,
    not per filter/query-param combination.

    Exists because api.data.gov.my's actual query API (not the parquet
    catalog list) is rate-limited to 4 requests/minute — caching means a
    popular dataset being searched by multiple users doesn't re-fetch on
    every single request.
    """
    __tablename__ = "gov_data_cache"

    catalog_dataset_id = Column(String, ForeignKey("catalog_datasets.id"), primary_key=True)
    raw_data = Column(Text, nullable=False)  # JSON-serialized list[dict] of rows
    row_count = Column(Integer, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
