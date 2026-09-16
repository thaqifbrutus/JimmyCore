"""
Tests for POST /catalog/sync.

This endpoint only touches CatalogDataset (no Postgres-specific types),
so — unlike the report/dataset routes, which need real Postgres for their
UUID/JSONB columns — this one runs on fast in-memory SQLite. The one
network call (fetch_catalog_dataframe) is monkeypatched; everything else
runs for real: the FastAPI route, the dependency-injected DB session, and
the actual upsert logic in sync_catalog.
"""
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.database import Base, get_db
from app.models.catalog_dataset import CatalogDataset
from app.routers import catalog as catalog_router
from app.services import catalog_sync


@pytest.fixture()
def db_session():
    # StaticPool is what actually matters here, not just check_same_thread:
    # without it, SQLAlchemy's default SingletonThreadPool hands out a
    # SEPARATE (and separately-empty) :memory: database to each thread,
    # and TestClient runs requests in a different thread than this fixture.
    # create_all() would succeed in this thread while the request handler
    # sees "no such table" in its own thread's database. StaticPool forces
    # every thread onto the one real connection/database.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture()
def client(db_session):
    app = FastAPI()
    app.include_router(catalog_router.router, prefix="/catalog", tags=["Catalog"])

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _fake_catalog_df():
    return pd.DataFrame([
        {
            "id": "fuelprice", "date_created": "2023-01-01", "title_en": "Fuel Prices",
            "category_en": "Economy", "subcategory_en": "Prices", "title_bm": None,
            "category_bm": None, "subcategory_bm": None, "source": "MOF",
            "frequency": "Daily", "geography": "National", "demography": None,
            "dataset_begin": 2015, "dataset_end": 2026,
        },
        {
            "id": "roadaccidents", "date_created": "2022-01-01", "title_en": "Road Accidents",
            "category_en": "Transport", "subcategory_en": "Safety", "title_bm": None,
            "category_bm": None, "subcategory_bm": None, "source": "PDRM",
            "frequency": "Yearly", "geography": "National", "demography": None,
            "dataset_begin": 2010, "dataset_end": 2024,
        },
    ])


def test_sync_endpoint_returns_counts_and_persists_rows(client, db_session, monkeypatch):
    monkeypatch.setattr(catalog_sync, "fetch_catalog_dataframe", lambda: _fake_catalog_df())

    response = client.post("/catalog/sync")

    assert response.status_code == 200
    body = response.json()
    assert body["inserted"] == 2
    assert body["updated"] == 0
    assert body["total"] == 2

    assert db_session.query(CatalogDataset).count() == 2
    fuel = db_session.query(CatalogDataset).filter_by(id="fuelprice").first()
    assert fuel.title_en == "Fuel Prices"


def test_sync_endpoint_is_idempotent_on_repeat_calls(client, db_session, monkeypatch):
    monkeypatch.setattr(catalog_sync, "fetch_catalog_dataframe", lambda: _fake_catalog_df())

    client.post("/catalog/sync")
    second_response = client.post("/catalog/sync")

    body = second_response.json()
    assert body["inserted"] == 0
    assert body["updated"] == 2
    assert db_session.query(CatalogDataset).count() == 2  # not duplicated


def test_sync_endpoint_returns_502_on_fetch_failure(client, monkeypatch):
    def _boom():
        raise ConnectionError("could not reach data.gov.my")

    monkeypatch.setattr(catalog_sync, "fetch_catalog_dataframe", _boom)

    response = client.post("/catalog/sync")

    assert response.status_code == 502
    assert "Catalog sync failed" in response.json()["detail"]
