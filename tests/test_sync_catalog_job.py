"""
Tests for scripts/sync_catalog_job.py's run() function. Verifies the
actual exit-code contract Railway's cron scheduler depends on (0 =
success, 1 = failure), and that the DB session is always closed via the
finally block — including on the failure path, which is easy to get
wrong if someone "simplifies" this later by removing the try/finally.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import scripts.sync_catalog_job as job


def test_run_returns_zero_on_success():
    fake_db = MagicMock()

    with patch.object(job, "SessionLocal", return_value=fake_db), \
         patch.object(job, "sync_catalog", return_value={"inserted": 2, "updated": 1, "total": 3}), \
         patch.object(job, "generate_catalog_embeddings", return_value={"embedded": 2, "skipped": 1, "total": 3}):

        exit_code = job.run()

    assert exit_code == 0
    fake_db.close.assert_called_once()


def test_run_returns_one_and_closes_db_on_sync_failure():
    fake_db = MagicMock()

    with patch.object(job, "SessionLocal", return_value=fake_db), \
         patch.object(job, "sync_catalog", side_effect=ConnectionError("could not reach data.gov.my")):

        exit_code = job.run()

    assert exit_code == 1
    # The whole point of the try/finally — DB session must close even
    # when sync_catalog blows up partway through.
    fake_db.close.assert_called_once()


def test_run_returns_one_and_closes_db_on_embedding_failure():
    fake_db = MagicMock()

    with patch.object(job, "SessionLocal", return_value=fake_db), \
         patch.object(job, "sync_catalog", return_value={"inserted": 0, "updated": 0, "total": 0}), \
         patch.object(job, "generate_catalog_embeddings", side_effect=RuntimeError("OpenRouter unreachable")):

        exit_code = job.run()

    assert exit_code == 1
    fake_db.close.assert_called_once()


def test_run_calls_sync_before_embeddings():
    """Embeddings should only be generated after sync brings in fresh rows."""
    fake_db = MagicMock()
    call_order = []

    def fake_sync(db):
        call_order.append("sync")
        return {"inserted": 0, "updated": 0, "total": 0}

    def fake_embed(db):
        call_order.append("embed")
        return {"embedded": 0, "skipped": 0, "total": 0}

    with patch.object(job, "SessionLocal", return_value=fake_db), \
         patch.object(job, "sync_catalog", side_effect=fake_sync), \
         patch.object(job, "generate_catalog_embeddings", side_effect=fake_embed):

        job.run()

    assert call_order == ["sync", "embed"]
