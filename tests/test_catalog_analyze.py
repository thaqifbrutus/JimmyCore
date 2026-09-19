"""
Integration tests for POST /catalog/{catalog_dataset_id}/analyze.

VERIFICATION STATUS — same situation as tests/test_routes.py from Round 2:
QualityReport uses Postgres-specific JSONB/UUID columns, which don't
compile on SQLite (confirmed back in Round 2). These tests therefore need
a real Postgres to actually run — the same docker-compose.test.yml setup
from Round 2 applies here too. They are NOT executed in this sandbox.
Unlike test_profiler.py and test_reports_source_resolution.py (both fully
run, all passing, in this same round), these are written against the
same well-established FastAPI + SQLAlchemy patterns used everywhere else
in this codebase, but not proven against a live database.

All external calls (gov_data_client's live fetch, ai_service's LLM call)
are mocked — this suite tests whether the endpoint wires the DB, the
fetch/cache client, the profiler, and the AI summary together correctly,
not the quality of any of those pieces individually (each has its own
dedicated, already-passing test suite).
"""
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _mock_ai_success(content):
    return {"status": "ok", "reason": None, "content": content}


def _seed_catalog_dataset(db, id="fuelprice", title_en="Fuel Prices"):
    from app.models.catalog_dataset import CatalogDataset

    existing = db.query(CatalogDataset).filter_by(id=id).first()
    if existing:
        db.delete(existing)

    row = CatalogDataset(id=id, title_en=title_en)
    db.add(row)
    db.commit()
    return row


def test_analyze_404s_for_unknown_catalog_dataset(client):
    response = client.post("/catalog/nonexistent/analyze")
    assert response.status_code == 404


def test_analyze_end_to_end_with_mocked_fetch_and_ai(client, db_session, monkeypatch):
    from app.routers import catalog as catalog_router

    _seed_catalog_dataset(db_session)

    fake_df = pd.DataFrame([
        {"date": "2024-01-01", "price": 2.05},
        {"date": "2024-01-02", "price": 2.10},
    ])
    monkeypatch.setattr(catalog_router, "get_dataset_dataframe", lambda db, id, force_refresh=False: fake_df)
    monkeypatch.setattr(
        catalog_router, "generate_dataset_summary",
        lambda profile_data, original_filename: _mock_ai_success("Fuel prices look stable."),
    )

    response = client.post("/catalog/fuelprice/analyze")

    assert response.status_code == 200
    body = response.json()
    assert body["catalog_dataset_id"] == "fuelprice"
    assert body["overview"]["row_count"] == 2
    assert body["ai_summary"]["content"] == "Fuel prices look stable."
    assert "report_id" in body


def test_analyze_returns_422_for_empty_dataset(client, db_session, monkeypatch):
    from app.routers import catalog as catalog_router

    _seed_catalog_dataset(db_session, id="empty_ds")

    monkeypatch.setattr(catalog_router, "get_dataset_dataframe", lambda db, id, force_refresh=False: pd.DataFrame())

    response = client.post("/catalog/empty_ds/analyze")

    assert response.status_code == 422
    assert "no rows" in response.json()["detail"]


def test_analyze_returns_502_when_gov_api_fetch_fails(client, db_session, monkeypatch):
    from app.routers import catalog as catalog_router
    from app.services.gov_data_client import GovAPIError

    _seed_catalog_dataset(db_session, id="unreachable")

    def _boom(db, id, force_refresh=False):
        raise GovAPIError("api.data.gov.my returned 500")

    monkeypatch.setattr(catalog_router, "get_dataset_dataframe", _boom)

    response = client.post("/catalog/unreachable/analyze")

    assert response.status_code == 502
    assert "Could not fetch government data" in response.json()["detail"]


def test_analyze_report_is_readable_via_existing_get_report_endpoint(client, db_session, monkeypatch):
    """
    The real point of the shared QualityReport table: a report created by
    the NEW analyze endpoint should be fully readable by the EXISTING
    /reports/{id} endpoint with zero changes needed there.
    """
    from app.routers import catalog as catalog_router

    _seed_catalog_dataset(db_session, id="roadaccidents", title_en="Road Accidents")

    fake_df = pd.DataFrame([{"state": "Selangor", "count": 120}])
    monkeypatch.setattr(catalog_router, "get_dataset_dataframe", lambda db, id, force_refresh=False: fake_df)
    monkeypatch.setattr(
        catalog_router, "generate_dataset_summary",
        lambda profile_data, original_filename: _mock_ai_success("Selangor has the most incidents."),
    )

    analyze_response = client.post("/catalog/roadaccidents/analyze")
    report_id = analyze_response.json()["report_id"]

    get_response = client.get(f"/reports/{report_id}")

    assert get_response.status_code == 200
    body = get_response.json()
    assert body["catalog_dataset_id"] == "roadaccidents"
    assert body["dataset_id"] is None
    assert body["ai_summary"]["content"] == "Selangor has the most incidents."
