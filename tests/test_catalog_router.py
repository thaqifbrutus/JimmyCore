import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.models.catalog_dataset import CatalogDataset

from db.database import Base, get_db
from app.models.catalog_dataset import CatalogDataset
from app.routers import catalog as catalog_router
from app.services import catalog_search


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    #Base.metadata.create_all(bind=engine)
    CatalogDataset.__table__.create(bind=engine, checkfirst=True)
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


def test_generate_embeddings_endpoint(client, db_session, monkeypatch):
    db_session.add(CatalogDataset(id="fuelprice", title_en="Fuel Prices"))
    db_session.commit()

    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [[0.1, 0.2] for _ in texts])

    response = client.post("/catalog/embeddings/generate")

    assert response.status_code == 200
    body = response.json()
    assert body["embedded"] == 1
    assert body["skipped"] == 0

    row = db_session.query(CatalogDataset).filter_by(id="fuelprice").first()
    assert json.loads(row.embedding) == [0.1, 0.2]


def test_search_endpoint_returns_ranked_results(client, db_session, monkeypatch):
    db_session.add_all([
        CatalogDataset(id="close", title_en="Road Accidents", embedding=json.dumps([1.0, 0.0])),
        CatalogDataset(id="far", title_en="Weather Data", embedding=json.dumps([0.0, 1.0])),
    ])
    db_session.commit()

    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [[1.0, 0.0]])

    response = client.get("/catalog/search", params={"q": "car crash statistics"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "car crash statistics"
    assert body["results"][0]["id"] == "close"
    assert body["results"][0]["score"] > body["results"][1]["score"]


def test_search_endpoint_requires_query_param(client):
    response = client.get("/catalog/search")
    assert response.status_code == 422  # FastAPI validation error, missing required q


def test_search_endpoint_respects_top_k_bounds(client):
    response = client.get("/catalog/search", params={"q": "test", "top_k": 999})
    assert response.status_code == 422  # top_k has a le=20 bound
