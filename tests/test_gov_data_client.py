"""
Tests for app.services.gov_data_client.

Two things are deliberately never mocked here:
1. RateLimiter's actual timing math — tested with an injected fake clock
   (a simple counter, not real time.monotonic) and a fake sleep function
   that just records how long it was asked to sleep instead of actually
   sleeping. This proves the rate-limiting logic itself is correct, not
   just that some function got called.
2. The cache freshness/staleness decision and the DataFrame conversion —
   real SQLite DB, real datetime comparisons.

Only `requests.get` (via fetch_dataset_from_api) is mocked — the one
actual network call.
"""
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from app.models.gov_data_cache import GovDataCache
from app.models.catalog_dataset import CatalogDataset
from app.services import gov_data_client
from app.services.gov_data_client import RateLimiter, GovAPIError, get_dataset_dataframe


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    session.add_all([
    CatalogDataset(id="fuelprice", title_en="Fuel Prices"),
    CatalogDataset(id="empty_dataset", title_en="Empty Dataset"),
])
    session.commit()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# RateLimiter — real timing logic, fake clock/sleep
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_rate_limiter_allows_calls_under_the_limit_without_sleeping():
    clock = FakeClock()
    sleeps = []
    limiter = RateLimiter(max_calls=4, window_seconds=60, time_func=clock.time, sleep_func=sleeps.append)

    for _ in range(4):
        waited = limiter.wait_if_needed()
        assert waited == 0.0
        clock.advance(1)  # small gap between calls, still well under the window

    assert sleeps == []


def test_rate_limiter_blocks_the_fifth_call_within_the_window():
    clock = FakeClock()
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)  # simulate time actually passing during the sleep

    limiter = RateLimiter(max_calls=4, window_seconds=60, time_func=clock.time, sleep_func=fake_sleep)

    for _ in range(4):
        limiter.wait_if_needed()
        clock.advance(1)  # calls at t=0,1,2,3

    # 5th call at t=4 — oldest call (t=0) exits the 60s window at t=60,
    # so we should be told to wait ~56s (60 - (4 - 0)).
    waited = limiter.wait_if_needed()

    assert len(sleeps) == 1
    assert waited == pytest.approx(56.0, abs=0.01)


def test_rate_limiter_allows_a_new_call_once_the_window_has_passed():
    clock = FakeClock()
    limiter = RateLimiter(max_calls=4, window_seconds=60, time_func=clock.time, sleep_func=lambda s: clock.advance(s))

    for _ in range(4):
        limiter.wait_if_needed()

    clock.advance(61)  # fully past the window, no calls should still be "active"

    waited = limiter.wait_if_needed()
    assert waited == 0.0


# ---------------------------------------------------------------------------
# fetch_dataset_from_api — error handling, network mocked
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text

    def json(self):
        return self._json_body


def test_fetch_raises_on_429(monkeypatch):
    monkeypatch.setattr(gov_data_client, "_rate_limiter", RateLimiter(sleep_func=lambda s: None))
    monkeypatch.setattr(
        gov_data_client.requests, "get",
        lambda url, params, **kwargs: _FakeResponse(429, text="rate limited"),
    )

    with pytest.raises(GovAPIError, match="Rate limited"):
        gov_data_client.fetch_dataset_from_api("fuelprice")


def test_fetch_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(gov_data_client, "_rate_limiter", RateLimiter(sleep_func=lambda s: None))
    monkeypatch.setattr(
        gov_data_client.requests, "get",
        lambda url, params, **kwargs: _FakeResponse(404, text="not found"),
    )

    with pytest.raises(GovAPIError, match="404"):
        gov_data_client.fetch_dataset_from_api("nonexistent")


def test_fetch_raises_on_error_shaped_body(monkeypatch):
    monkeypatch.setattr(gov_data_client, "_rate_limiter", RateLimiter(sleep_func=lambda s: None))
    monkeypatch.setattr(
        gov_data_client.requests, "get",
        lambda url, params, **kwargs: _FakeResponse(200, json_body={"status": 400, "errors": ["bad filter"]}),
    )

    with pytest.raises(GovAPIError, match="bad filter"):
        gov_data_client.fetch_dataset_from_api("fuelprice")


def test_fetch_returns_row_list_on_success(monkeypatch):
    monkeypatch.setattr(gov_data_client, "_rate_limiter", RateLimiter(sleep_func=lambda s: None))
    rows = [{"date": "2024-01-01", "price": 2.05}]
    monkeypatch.setattr(
        gov_data_client.requests, "get",
        lambda url, params, **kwargs: _FakeResponse(200, json_body=rows),
    )

    result = gov_data_client.fetch_dataset_from_api("fuelprice")
    assert result == rows


# ---------------------------------------------------------------------------
# get_dataset_dataframe — cache hit / miss / stale, real DB
# ---------------------------------------------------------------------------

def test_returns_fresh_cache_without_calling_the_api(db_session, monkeypatch):
    db_session.add(GovDataCache(
        catalog_dataset_id="fuelprice",
        raw_data=json.dumps([{"price": 2.05}]),
        row_count=1,
        fetched_at=datetime.utcnow(),
    ))
    db_session.commit()

    def _should_not_be_called(*a, **kw):
        raise AssertionError("fetch_dataset_from_api should not be called for fresh cache")

    monkeypatch.setattr(gov_data_client, "fetch_dataset_from_api", _should_not_be_called)

    df = get_dataset_dataframe(db_session, "fuelprice")

    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["price"] == 2.05


def test_fetches_live_when_no_cache_exists(db_session, monkeypatch):
    monkeypatch.setattr(gov_data_client, "fetch_dataset_from_api", lambda id, limit: [{"price": 2.10}])

    df = get_dataset_dataframe(db_session, "fuelprice")

    assert df.iloc[0]["price"] == 2.10
    cached = db_session.query(GovDataCache).filter_by(catalog_dataset_id="fuelprice").first()
    assert cached is not None
    assert cached.row_count == 1


def test_refetches_when_cache_is_stale(db_session, monkeypatch):
    db_session.add(GovDataCache(
        catalog_dataset_id="fuelprice",
        raw_data=json.dumps([{"price": 1.99}]),
        row_count=1,
        fetched_at=datetime.utcnow() - timedelta(minutes=120),  # older than default 60min
    ))
    db_session.commit()

    monkeypatch.setattr(gov_data_client, "fetch_dataset_from_api", lambda id, limit: [{"price": 2.20}])

    df = get_dataset_dataframe(db_session, "fuelprice")

    assert df.iloc[0]["price"] == 2.20  # refreshed, not the stale cached value


def test_force_refresh_ignores_fresh_cache(db_session, monkeypatch):
    db_session.add(GovDataCache(
        catalog_dataset_id="fuelprice",
        raw_data=json.dumps([{"price": 1.00}]),
        row_count=1,
        fetched_at=datetime.utcnow(),  # fresh
    ))
    db_session.commit()

    monkeypatch.setattr(gov_data_client, "fetch_dataset_from_api", lambda id, limit: [{"price": 9.99}])

    df = get_dataset_dataframe(db_session, "fuelprice", force_refresh=True)

    assert df.iloc[0]["price"] == 9.99


def test_handles_empty_result_set(db_session, monkeypatch):
    monkeypatch.setattr(gov_data_client, "fetch_dataset_from_api", lambda id, limit: [])

    df = get_dataset_dataframe(db_session, "empty_dataset")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
