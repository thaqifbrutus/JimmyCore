"""
Pure unit tests for build_source_context in ai_service.py. No DB, no
network — takes any object with the right attributes, so a SimpleNamespace
stand-in exercises the real logic exactly like a real CatalogDataset row
would, without needing Postgres.
"""
from types import SimpleNamespace

from app.services.ai_service import build_source_context


def _catalog_dataset(**overrides):
    defaults = dict(
        source="PDRM", category_en="Transport", subcategory_en="Road Safety",
        dataset_begin=2010, dataset_end=2024,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_builds_full_source_context():
    result = build_source_context(_catalog_dataset())

    assert result["type"] == "official_government_dataset"
    assert result["agency"] == "PDRM"
    assert result["category"] == "Transport"
    assert result["subcategory"] == "Road Safety"
    assert result["coverage"] == "2010\u20132024"


def test_coverage_is_none_when_years_missing():
    result = build_source_context(_catalog_dataset(dataset_begin=None, dataset_end=None))
    assert result["coverage"] is None


def test_coverage_is_none_when_only_one_year_present():
    result = build_source_context(_catalog_dataset(dataset_begin=2010, dataset_end=None))
    assert result["coverage"] is None


def test_handles_missing_agency_and_category_gracefully():
    result = build_source_context(_catalog_dataset(source=None, category_en=None, subcategory_en=None))
    assert result["agency"] is None
    assert result["category"] is None
    assert result["subcategory"] is None
    assert result["type"] == "official_government_dataset"  # always present regardless
