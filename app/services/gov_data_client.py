"""
Fetches real data for a specific dataset id from api.data.gov.my's query
API (distinct from the static parquet catalog list catalog_sync.py pulls
from). This is the rate-limited one — 4 requests/minute, confirmed at
https://developer.data.gov.my/rate-limit — so this module exists
specifically to make sure JimmyCore never exceeds that, and to avoid
re-fetching the same dataset repeatedly via caching.

Design: fetch_dataset_from_api() is the ONLY function that touches the
network. RateLimiter is a standalone, injectable-clock class so its
timing logic is testable without real sleeps. get_dataset_dataframe() is
the main entry point everything else should call — it handles the
cache-hit/cache-miss/stale decision and always returns a DataFrame ready
for the profiler.
"""
import json
import time
from collections import deque
from datetime import datetime, timedelta

import pandas as pd
import requests
from sqlalchemy.orm import Session

from app.models.gov_data_cache import GovDataCache

BASE_URL = "https://api.data.gov.my/data-catalogue"

# Confirmed at https://developer.data.gov.my/rate-limit — exceeding this
# gets a 429. This client enforces it client-side rather than just
# reacting to 429s after the fact, since a 429 means we already made a
# request we shouldn't have.
MAX_REQUESTS_PER_MINUTE = 4

# No pagination exists on this API — `limit` is the only size control,
# and an unset limit appears to return everything. Defaulting to a cap
# rather than unbounded, so a popular high-frequency dataset (e.g. daily
# fuel prices going back years) doesn't pull an unexpectedly huge payload
# on a casual search-driven fetch. Callers can override.
DEFAULT_FETCH_LIMIT = 1000

DEFAULT_CACHE_MAX_AGE_MINUTES = 60


class RateLimiter:
    """
    Enforces at most `max_calls` calls per rolling 60-second window.
    Blocks (sleeps) rather than rejecting — this client is used from
    synchronous FastAPI route handlers, matching the rest of this
    codebase's synchronous style (no async anywhere else either).

    time_func/sleep_func are injectable so tests can verify the actual
    waiting logic without real wall-clock delays.
    """

    def __init__(self, max_calls: int = MAX_REQUESTS_PER_MINUTE, window_seconds: float = 60.0,
                 time_func=time.monotonic, sleep_func=time.sleep):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._time_func = time_func
        self._sleep_func = sleep_func
        self._call_times = deque()

    def wait_if_needed(self) -> float:
        """
        Blocks until a call is allowed under the rate limit, then records
        this call's timestamp. Returns how long it slept, in seconds (0.0
        if no wait was needed) — mainly so tests can assert on it.
        """
        now = self._time_func()

        while self._call_times and now - self._call_times[0] >= self.window_seconds:
            self._call_times.popleft()

        if len(self._call_times) < self.max_calls:
            self._call_times.append(now)
            return 0.0

        oldest_call = self._call_times[0]
        wait_time = self.window_seconds - (now - oldest_call)
        if wait_time > 0:
            self._sleep_func(wait_time)

        self._call_times.popleft()
        self._call_times.append(self._time_func())
        return max(wait_time, 0.0)


# Module-level singleton — one rate limiter shared across every call this
# process makes, since the 4/minute limit is per-client, not per-dataset.
_rate_limiter = RateLimiter()


class GovAPIError(Exception):
    """Raised for non-200 responses or unexpected response shapes."""
    pass


def fetch_dataset_from_api(dataset_id: str, limit: int = DEFAULT_FETCH_LIMIT) -> list[dict]:
    """
    The ONE network-touching function in this module. Blocks on the rate
    limiter first, then makes the actual request.

    Per https://developer.data.gov.my/response-format: a successful
    response (without ?meta=true, which we don't use) is a raw JSON list
    of row dicts. An error response is {"status": int, "errors": [...]}.
    """
    _rate_limiter.wait_if_needed()

    response = requests.get(BASE_URL, params={"id": dataset_id, "limit": limit}, timeout=30,)

    if response.status_code == 429:
        raise GovAPIError(
            f"Rate limited by api.data.gov.my fetching '{dataset_id}' despite "
            f"client-side throttling — this shouldn't normally happen; check "
            f"whether something else is sharing this rate limit budget."
        )

    if response.status_code != 200:
        raise GovAPIError(
            f"api.data.gov.my returned {response.status_code} for dataset "
            f"'{dataset_id}': {response.text[:500]}"
        )

    body = response.json()

    if isinstance(body, dict) and "errors" in body:
        raise GovAPIError(f"api.data.gov.my error for dataset '{dataset_id}': {body['errors']}")

    if not isinstance(body, list):
        raise GovAPIError(
            f"Unexpected response shape for dataset '{dataset_id}' — expected a "
            f"JSON list, got {type(body).__name__}"
        )

    return body


def _get_cache_row(db: Session, dataset_id: str) -> GovDataCache | None:
    return db.query(GovDataCache).filter_by(catalog_dataset_id=dataset_id).first()


def _is_fresh(cache_row: GovDataCache, max_age_minutes: int) -> bool:
    if cache_row.fetched_at is None:
        return False
    return datetime.utcnow() - cache_row.fetched_at < timedelta(minutes=max_age_minutes)


def get_dataset_dataframe(
    db: Session,
    dataset_id: str,
    max_age_minutes: int = DEFAULT_CACHE_MAX_AGE_MINUTES,
    limit: int = DEFAULT_FETCH_LIMIT,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Main entry point. Returns a DataFrame of the dataset's rows, using a
    cached copy if one exists and is still fresh, otherwise fetching live
    (respecting the rate limiter) and updating the cache.
    """
    cache_row = _get_cache_row(db, dataset_id)

    if cache_row is not None and not force_refresh and _is_fresh(cache_row, max_age_minutes):
        rows = json.loads(cache_row.raw_data)
        return pd.DataFrame(rows)

    rows = fetch_dataset_from_api(dataset_id, limit=limit)

    db.merge(GovDataCache(
        catalog_dataset_id=dataset_id,
        raw_data=json.dumps(rows),
        row_count=len(rows),
        fetched_at=datetime.utcnow(),
    ))
    db.commit()

    return pd.DataFrame(rows)
