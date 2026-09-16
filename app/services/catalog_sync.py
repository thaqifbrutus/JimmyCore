"""
Syncs our local CatalogDataset table against data.gov.my's official
dataset list. This is the "search index" data.gov.my's actual API doesn't
provide — see CatalogDataset's docstring for why this table exists at all.

Design note: fetch_catalog_dataframe() is the ONLY function in this module
that touches the network. Every other function takes a DataFrame as input.
This isn't just for testability (though it is very testable this way,
see tests/test_catalog_sync.py) — it also means sync_catalog can be called
with a DataFrame from anywhere: a live fetch, a cached copy, or a manually
downloaded file, without caring how it got there.
"""
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.models.catalog_dataset import CatalogDataset

CATALOG_PARQUET_URL = "https://storage.data.gov.my/metrics/dataset_list.parquet"

# Columns as documented at https://data.gov.my/data-catalogue/datasets —
# mapped 1:1 to CatalogDataset's fields, so no renaming needed here.
_STRING_COLUMNS = [
    "id", "title_en", "category_en", "subcategory_en",
    "title_bm", "category_bm", "subcategory_bm",
    "source", "frequency", "geography", "demography",
]
_DATE_COLUMNS = ["date_created"]
_INT_COLUMNS = ["dataset_begin", "dataset_end"]


def fetch_catalog_dataframe() -> pd.DataFrame:
    """
    The one network call in this module. Pulls the full, official dataset
    list — not the rate-limited query API (this is a static file dump, so
    it doesn't count against the 4-requests-a-minute budget that
    api.data.gov.my enforces for actual data queries).
    """
    return pd.read_parquet(CATALOG_PARQUET_URL)


def _clean_row(row: pd.Series) -> dict:
    """
    Converts one parquet row into CatalogDataset constructor kwargs,
    normalizing pandas' NaN/NaT into plain None and coercing types so
    SQLAlchemy gets exactly what the column types expect.
    """
    cleaned = {}

    for col in _STRING_COLUMNS:
        value = row.get(col)
        cleaned[col] = None if pd.isna(value) else str(value)

    for col in _DATE_COLUMNS:
        value = row.get(col)
        if pd.isna(value):
            cleaned[col] = None
        else:
            parsed = pd.to_datetime(value)
            cleaned[col] = parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed

    for col in _INT_COLUMNS:
        value = row.get(col)
        cleaned[col] = None if pd.isna(value) else int(value)

    return cleaned


def sync_catalog(db: Session, df: pd.DataFrame | None = None) -> dict:
    """
    Upserts every row in df (or a freshly fetched one, if df is omitted)
    into CatalogDataset, keyed on the government's own dataset id.

    Returns a summary dict: {"inserted": int, "updated": int, "total": int}
    so callers (a manual sync endpoint, a scheduled job, a CLI script) can
    report what happened without needing to inspect the DB themselves.
    """
    if df is None:
        df = fetch_catalog_dataframe()

    if "id" not in df.columns:
        raise ValueError(
            "Catalog dataframe is missing an 'id' column — got columns: "
            f"{list(df.columns)}. Refusing to sync against unexpected schema."
        )

    incoming_ids = set(df["id"].dropna().astype(str))
    existing_ids = {
        row.id for row in db.query(CatalogDataset.id).filter(CatalogDataset.id.in_(incoming_ids)).all()
    }

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        kwargs = _clean_row(row)
        if kwargs["id"] is None:
            continue  # Refuse to index a row with no id — nothing to key it on.

        kwargs["last_synced_at"] = datetime.utcnow()

        db.merge(CatalogDataset(**kwargs))

        if kwargs["id"] in existing_ids:
            updated += 1
        else:
            inserted += 1

    db.commit()

    return {
        "inserted": inserted,
        "updated": updated,
        "total": inserted + updated,
    }
