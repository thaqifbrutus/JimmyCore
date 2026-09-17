"""
Tests for app.services.catalog_search.

embed_texts is the only function mocked here — it's the one network call
in the module. Everything else (build_searchable_text, the batching logic
in generate_catalog_embeddings, cosine similarity ranking in
search_catalog) runs for real against a real in-memory SQLite DB. Valid on
SQLite because CatalogDataset still uses no Postgres-specific types, even
with the new embedding column.
"""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from app.models.catalog_dataset import CatalogDataset
from app.services import catalog_search
from app.services.catalog_search import (
    build_searchable_text,
    generate_catalog_embeddings,
    search_catalog,
    _cosine_similarity,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def _add_dataset(db, id, title_en, category_en=None, subcategory_en=None, source=None, embedding=None):
    row = CatalogDataset(
        id=id, title_en=title_en, category_en=category_en,
        subcategory_en=subcategory_en, source=source,
        embedding=json.dumps(embedding) if embedding is not None else None,
    )
    db.add(row)
    db.commit()
    return row


# ---------------------------------------------------------------------------
# build_searchable_text
# ---------------------------------------------------------------------------

def test_build_searchable_text_combines_available_fields():
    row = CatalogDataset(
        id="x", title_en="Fuel Prices", category_en="Economy",
        subcategory_en="Prices", source="MOF",
    )
    text = build_searchable_text(row)
    assert "Fuel Prices" in text
    assert "Economy" in text
    assert "Prices" in text
    assert "MOF" in text


def test_build_searchable_text_skips_missing_fields():
    row = CatalogDataset(id="x", title_en="Fuel Prices", category_en=None, subcategory_en=None, source=None)
    text = build_searchable_text(row)
    assert text == "Fuel Prices"


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero_not_nan():
    # A zero vector would divide by zero norm — must not crash or return NaN.
    result = _cosine_similarity([0.0, 0.0], [1.0, 1.0])
    assert result == 0.0


# ---------------------------------------------------------------------------
# generate_catalog_embeddings
# ---------------------------------------------------------------------------

def test_generate_embeddings_only_processes_rows_missing_embedding(db_session, monkeypatch):
    _add_dataset(db_session, "already_done", "Old One", embedding=[0.1, 0.2])
    _add_dataset(db_session, "needs_embedding", "New One")

    calls = []

    def fake_embed_texts(texts):
        calls.append(texts)
        return [[0.5, 0.5] for _ in texts]

    monkeypatch.setattr(catalog_search, "embed_texts", fake_embed_texts)

    result = generate_catalog_embeddings(db_session)

    assert result == {"embedded": 1, "skipped": 1, "total": 2}
    assert calls == [["New One"]]  # only the un-embedded row was sent

    refreshed = db_session.query(CatalogDataset).filter_by(id="needs_embedding").first()
    assert json.loads(refreshed.embedding) == [0.5, 0.5]

    # The already-embedded row's vector is untouched.
    untouched = db_session.query(CatalogDataset).filter_by(id="already_done").first()
    assert json.loads(untouched.embedding) == [0.1, 0.2]


def test_generate_embeddings_force_reprocesses_everything(db_session, monkeypatch):
    _add_dataset(db_session, "a", "Dataset A", embedding=[0.1, 0.1])
    _add_dataset(db_session, "b", "Dataset B", embedding=[0.2, 0.2])

    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [[0.9, 0.9] for _ in texts])

    result = generate_catalog_embeddings(db_session, force=True)

    assert result == {"embedded": 2, "skipped": 0, "total": 2}
    for row in db_session.query(CatalogDataset).all():
        assert json.loads(row.embedding) == [0.9, 0.9]


def test_generate_embeddings_batches_calls(db_session, monkeypatch):
    for i in range(5):
        _add_dataset(db_session, f"id_{i}", f"Dataset {i}")

    calls = []

    def fake_embed_texts(texts):
        calls.append(len(texts))
        return [[0.0] for _ in texts]

    monkeypatch.setattr(catalog_search, "embed_texts", fake_embed_texts)

    result = generate_catalog_embeddings(db_session, batch_size=2)

    assert result["embedded"] == 5
    # 5 rows, batch_size=2 -> batches of [2, 2, 1]
    assert calls == [2, 2, 1]


def test_generate_embeddings_handles_empty_catalog(db_session, monkeypatch):
    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [])
    result = generate_catalog_embeddings(db_session)
    assert result == {"embedded": 0, "skipped": 0, "total": 0}


# ---------------------------------------------------------------------------
# search_catalog
# ---------------------------------------------------------------------------

def test_search_ranks_by_similarity_descending(db_session, monkeypatch):
    _add_dataset(db_session, "close_match", "Road Accidents", embedding=[1.0, 0.0])
    _add_dataset(db_session, "far_match", "Weather Data", embedding=[0.0, 1.0])
    _add_dataset(db_session, "medium_match", "Traffic Volume", embedding=[0.7, 0.7])

    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [[1.0, 0.0]])

    results = search_catalog(db_session, "drunk driving accidents", top_k=5)

    ids_in_order = [r["dataset"].id for r in results]
    assert ids_in_order[0] == "close_match"
    assert ids_in_order[-1] == "far_match"
    assert results[0]["score"] > results[-1]["score"]


def test_search_excludes_rows_with_no_embedding(db_session, monkeypatch):
    _add_dataset(db_session, "embedded", "Has Embedding", embedding=[1.0, 0.0])
    _add_dataset(db_session, "not_embedded", "No Embedding Yet")  # embedding=None

    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [[1.0, 0.0]])

    results = search_catalog(db_session, "test query")

    ids = [r["dataset"].id for r in results]
    assert "embedded" in ids
    assert "not_embedded" not in ids


def test_search_respects_top_k(db_session, monkeypatch):
    for i in range(10):
        _add_dataset(db_session, f"id_{i}", f"Dataset {i}", embedding=[float(i), 0.0])

    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [[5.0, 0.0]])

    results = search_catalog(db_session, "query", top_k=3)

    assert len(results) == 3


def test_search_returns_empty_list_when_catalog_has_no_embeddings(db_session, monkeypatch):
    _add_dataset(db_session, "a", "Dataset A")  # no embedding

    monkeypatch.setattr(catalog_search, "embed_texts", lambda texts: [[1.0, 0.0]])

    results = search_catalog(db_session, "query")

    assert results == []
