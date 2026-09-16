from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer
from db.database import Base


class CatalogDataset(Base):
    """
    Local index of data.gov.my's official dataset catalog, synced from
    https://storage.data.gov.my/metrics/dataset_list.parquet (see
    catalog_sync.py). This exists because data.gov.my's actual API has no
    search endpoint — you can only fetch a dataset if you already know its
    exact id. This table IS the search index that's otherwise missing.

    id is the government's own dataset id (e.g. "fuelprice") — a natural
    string key, not a generated UUID, since we want upserts to line up
    with their catalog exactly on every sync.
    """
    __tablename__ = "catalog_datasets"

    id = Column(String, primary_key=True)
    date_created = Column(DateTime, nullable=True)
    title_en = Column(String, nullable=False)
    category_en = Column(String, nullable=True)
    subcategory_en = Column(String, nullable=True)
    title_bm = Column(String, nullable=True)
    category_bm = Column(String, nullable=True)
    subcategory_bm = Column(String, nullable=True)
    source = Column(String, nullable=True)
    frequency = Column(String, nullable=True)
    geography = Column(String, nullable=True)
    demography = Column(String, nullable=True)
    dataset_begin = Column(Integer, nullable=True)
    dataset_end = Column(Integer, nullable=True)

    # Ours, not theirs — when our local copy of this row was last refreshed
    # from the official parquet file. Lets us answer "how stale is our
    # index" without needing a separate sync-run log table for the MVP.
    last_synced_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
