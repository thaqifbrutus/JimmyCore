"""
Tests for app.services.catalog_sync.

Unlike QualityReport (which needs real Postgres for its JSONB/UUID
columns), CatalogDataset uses only plain String/Integer/DateTime columns
— so these tests run against a fast in-memory SQLite DB. fetch_catalog_dataframe
(the one network-touching function) is never called here; every test
passes a DataFrame directly to sync_catalog, exactly as designed.
"""
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from app.models.catalog_dataset import CatalogDataset
from app.services.catalog_sync import sync_catalog, _clean_row


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def _sample_df(rows):
    """rows: list of dicts using the real parquet column names."""
    columns = [
        "id", "date_created", "title_en", "category_en", "subcategory_en",
        "title_bm", "category_bm", "subcategory_bm", "source", "frequency",
        "geography", "demography", "dataset_begin", "dataset_end",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_sync_inserts_new_rows(db_session):
    df = _sample_df([
        {
            "id": "fuelprice", "date_created": "2023-01-01",
            "title_en": "Fuel Prices", "category_en": "Economy",
            "subcategory_en": "Prices", "title_bm": "Harga Bahan Api",
            "category_bm": "Ekonomi", "subcategory_bm": "Harga",
            "source": "Ministry of Finance", "frequency": "Daily",
            "geography": "National", "demography": None,
            "dataset_begin": 2015, "dataset_end": 2026,
        },
    ])

    result = sync_catalog(db_session, df=df)

    assert result == {"inserted": 1, "updated": 0, "total": 1}
    row = db_session.query(CatalogDataset).filter_by(id="fuelprice").first()
    assert row.title_en == "Fuel Prices"
    assert row.dataset_begin == 2015
    assert row.dataset_end == 2026


def test_sync_updates_existing_rows_without_duplicating(db_session):
    df_v1 = _sample_df([
        {
            "id": "roadaccidents", "date_created": "2022-01-01",
            "title_en": "Road Accidents", "category_en": "Transport",
            "subcategory_en": "Safety", "title_bm": None, "category_bm": None,
            "subcategory_bm": None, "source": "PDRM", "frequency": "Yearly",
            "geography": "National", "demography": None,
            "dataset_begin": 2010, "dataset_end": 2023,
        },
    ])
    sync_catalog(db_session, df=df_v1)

    # Same id, updated title and extended end year — simulates a real
    # government catalog update between sync runs.
    df_v2 = _sample_df([
        {
            "id": "roadaccidents", "date_created": "2022-01-01",
            "title_en": "Road Accidents by State", "category_en": "Transport",
            "subcategory_en": "Safety", "title_bm": None, "category_bm": None,
            "subcategory_bm": None, "source": "PDRM", "frequency": "Yearly",
            "geography": "National", "demography": None,
            "dataset_begin": 2010, "dataset_end": 2024,
        },
    ])
    result = sync_catalog(db_session, df=df_v2)

    assert result == {"inserted": 0, "updated": 1, "total": 1}
    matching = db_session.query(CatalogDataset).filter_by(id="roadaccidents").all()
    assert len(matching) == 1  # not duplicated
    assert matching[0].title_en == "Road Accidents by State"
    assert matching[0].dataset_end == 2024


def test_sync_handles_mixed_insert_and_update_in_one_call(db_session):
    sync_catalog(db_session, df=_sample_df([
        {
            "id": "existing_one", "date_created": None, "title_en": "Old title",
            "category_en": None, "subcategory_en": None, "title_bm": None,
            "category_bm": None, "subcategory_bm": None, "source": None,
            "frequency": None, "geography": None, "demography": None,
            "dataset_begin": None, "dataset_end": None,
        },
    ]))

    result = sync_catalog(db_session, df=_sample_df([
        {
            "id": "existing_one", "date_created": None, "title_en": "New title",
            "category_en": None, "subcategory_en": None, "title_bm": None,
            "category_bm": None, "subcategory_bm": None, "source": None,
            "frequency": None, "geography": None, "demography": None,
            "dataset_begin": None, "dataset_end": None,
        },
        {
            "id": "brand_new", "date_created": None, "title_en": "Brand new dataset",
            "category_en": None, "subcategory_en": None, "title_bm": None,
            "category_bm": None, "subcategory_bm": None, "source": None,
            "frequency": None, "geography": None, "demography": None,
            "dataset_begin": None, "dataset_end": None,
        },
    ]))

    assert result == {"inserted": 1, "updated": 1, "total": 2}
    assert db_session.query(CatalogDataset).count() == 2


def test_sync_skips_rows_with_missing_id(db_session):
    df = _sample_df([
        {
            "id": None, "date_created": None, "title_en": "No id, should skip",
            "category_en": None, "subcategory_en": None, "title_bm": None,
            "category_bm": None, "subcategory_bm": None, "source": None,
            "frequency": None, "geography": None, "demography": None,
            "dataset_begin": None, "dataset_end": None,
        },
    ])

    result = sync_catalog(db_session, df=df)

    assert result == {"inserted": 0, "updated": 0, "total": 0}
    assert db_session.query(CatalogDataset).count() == 0


def test_sync_raises_on_missing_id_column(db_session):
    df = pd.DataFrame([{"title_en": "No id column at all"}])

    with pytest.raises(ValueError, match="missing an 'id' column"):
        sync_catalog(db_session, df=df)


def test_clean_row_converts_nan_to_none_and_coerces_types():
    row = pd.Series({
        "id": "test", "title_en": "Test", "category_en": None,
        "subcategory_en": float("nan"), "title_bm": None, "category_bm": None,
        "subcategory_bm": None, "source": None, "frequency": None,
        "geography": None, "demography": None,
        "dataset_begin": 2020.0, "dataset_end": float("nan"),
        "date_created": "2021-05-01",
    })

    cleaned = _clean_row(row)

    assert cleaned["id"] == "test"
    assert cleaned["subcategory_en"] is None
    assert cleaned["dataset_begin"] == 2020
    assert isinstance(cleaned["dataset_begin"], int)
    assert cleaned["dataset_end"] is None
    assert cleaned["date_created"].year == 2021
