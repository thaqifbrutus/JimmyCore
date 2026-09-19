"""
Pure unit tests for _resolve_source_name in reports.py — no database
involved, since the function takes plain objects and does no queries
itself. This directly targets the bug that was fixed here: the old code
unconditionally assumed a report came from an uploaded Dataset and would
raise AttributeError on a catalog-sourced report (where dataset is None).

_get_report_source (the function that actually queries the DB to decide
which object to pass in) needs a real Postgres connection to test for
real, since Dataset uses Postgres-specific UUID columns — same limitation
as every other Postgres-touching test in this codebase. That one isn't
covered here; _resolve_source_name is the part that's fully testable in
isolation, and it's the part that actually had the bug.
"""
import pytest
from types import SimpleNamespace

from app.routers.reports import _resolve_source_name


def test_resolves_uploaded_dataset_name():
    dataset = SimpleNamespace(original_name="sales_q3.csv")
    result = _resolve_source_name(dataset, None)
    assert result == "sales_q3.csv"


def test_resolves_catalog_dataset_title():
    catalog_dataset = SimpleNamespace(title_en="Fuel Prices")
    result = _resolve_source_name(None, catalog_dataset)
    assert result == "Fuel Prices"


def test_prefers_dataset_over_catalog_dataset_if_somehow_both_given():
    # Shouldn't happen given the CHECK constraint, but if it ever did,
    # the function should pick one deterministically rather than error.
    dataset = SimpleNamespace(original_name="upload.csv")
    catalog_dataset = SimpleNamespace(title_en="Should not be used")
    result = _resolve_source_name(dataset, catalog_dataset)
    assert result == "upload.csv"


def test_raises_clearly_when_neither_source_is_given():
    # This is the exact case that used to crash with a bare
    # AttributeError ("NoneType has no attribute original_name") instead
    # of a clear, diagnosable error.
    with pytest.raises(ValueError, match="neither an uploaded dataset nor a catalog dataset"):
        _resolve_source_name(None, None)
