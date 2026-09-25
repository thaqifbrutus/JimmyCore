"""
Integration tests for POST /catalog/{catalog_dataset_id}/analyze.

STATUS: skipped. These tests cannot run in the current sandbox for two
reasons:
  1. They require a `client` fixture that has never been defined — a
     tests/conftest.py would need to provide a FastAPI TestClient with
     a database session override.
  2. They require a real Postgres. QualityReport uses JSONB and UUID
     columns which SQLite cannot compile.

They are kept on disk as a specification of the endpoint's expected
behavior. Restoring them requires: (a) adding tests/conftest.py with
shared `client` and `db_session` fixtures backed by Postgres, and
(b) removing the pytestmark skip below.
"""
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason="requires Postgres and a shared `client`/`db_session` conftest fixture — "
           "not runnable in the current sandbox. See module docstring."
)

@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _mock_overview_success(overview_text, questions=None):
    return {
        "status": "ok",
        "reason": None,
        "content": {
            "overview": overview_text,
            "suggested_questions": questions or [
                "What's the coverage period?",
                "Which category appears most often?",
                "Are there missing values?",
            ],
        },
    }


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


def test_analyze_end_to_end_with_mocked_fetch_and_overview(client, db_session, monkeypatch):
    from app.routers import catalog as catalog_router

    _seed_catalog_dataset(db_session)

    fake_df = pd.DataFrame([
        {"date": "2024-01-01", "price": 2.05},
        {"date": "2024-01-02", "price": 2.10},
    ])
    monkeypatch.setattr(
        catalog_router, "get_dataset_dataframe",
        lambda db, id, force_refresh=False: fake_df,
    )
    monkeypatch.setattr(
        catalog_router, "generate_dataset_overview",
        lambda profile_data, original_filename: _mock_overview_success(
            "Fuel prices look stable across the period."
        ),
    )

    response = client.post("/catalog/fuelprice/analyze")

    assert response.status_code == 200
    body = response.json()

    assert body["catalog_dataset_id"] == "fuelprice"
    assert "report_id" in body

    # New shape: dataset_stats + columns + overview + source
    assert body["dataset_stats"]["row_count"] == 2
    assert body["dataset_stats"]["column_count"] == 2
    assert isinstance(body["columns"], list)
    assert len(body["columns"]) == 2

    assert body["overview"]["status"] == "ok"
    assert body["overview"]["content"]["overview"] == (
        "Fuel prices look stable across the period."
    )
    assert len(body["overview"]["content"]["suggested_questions"]) == 3

    assert "source" in body

    # Fields dropped from the old shape — must NOT be present.
    assert "overall_status" not in body
    assert "issues" not in body
    assert "ai_summary" not in body
    assert "message" not in body


def test_analyze_returns_422_for_empty_dataset(client, db_session, monkeypatch):
    from app.routers import catalog as catalog_router

    _seed_catalog_dataset(db_session, id="empty_ds")

    monkeypatch.setattr(
        catalog_router, "get_dataset_dataframe",
        lambda db, id, force_refresh=False: pd.DataFrame(),
    )

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


def test_analyze_report_is_readable_via_existing_get_report_endpoint(
    client, db_session, monkeypatch
):
    """
    The real point of the shared QualityReport table: a report created by
    the NEW analyze endpoint should be fully readable by the EXISTING
    /reports/{id} endpoint with zero changes needed there.
    """
    from app.routers import catalog as catalog_router

    _seed_catalog_dataset(
        db_session, id="roadaccidents", title_en="Road Accidents"
    )

    fake_df = pd.DataFrame([{"state": "Selangor", "count": 120}])
    monkeypatch.setattr(
        catalog_router, "get_dataset_dataframe",
        lambda db, id, force_refresh=False: fake_df,
    )
    monkeypatch.setattr(
        catalog_router, "generate_dataset_overview",
        lambda profile_data, original_filename: _mock_overview_success(
            "Selangor has the most incidents."
        ),
    )

    analyze_response = client.post("/catalog/roadaccidents/analyze")
    assert analyze_response.status_code == 200
    report_id = analyze_response.json()["report_id"]

    get_response = client.get(f"/reports/{report_id}")
    assert get_response.status_code == 200
    body = get_response.json()

    assert body["catalog_dataset_id"] == "roadaccidents"
    assert body["dataset_id"] is None
    # ai_summary column stores the overview dict now.
    assert body["ai_summary"]["content"]["overview"] == (
        "Selangor has the most incidents."
    )
    # overall_status was dropped from this response shape.
    assert "overall_status" not in body