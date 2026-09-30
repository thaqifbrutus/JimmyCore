"""
Standalone script for keeping the catalog index fresh — syncs the
official dataset list, then embeds any new/updated rows.

Designed to run as its own short-lived process rather than as an HTTP
call to /catalog/sync:

- This script imports sync_catalog/generate_catalog_embeddings directly
  and opens its own DB session, rather than making an HTTP request to a
  running web service. One fewer network hop, and it works correctly even
  if the web service happens to be redeploying at the moment the job runs.

- The GitHub Actions workflow in .github/workflows/sync-catalog.yml
  invokes it daily at 02:00 UTC via `python scripts/sync_catalog_job.py`.

- Requires DATABASE_URL and OPENROUTER_API_KEY in the environment.
  Locally that's .env; in CI they come from repository secrets.

Note: a failed run is not retried. A missed daily sync isn't a big deal
for a search index — it just means the catalog is up to a day stale until
the next scheduled run — so this deliberately does not implement its own
retry logic. Failures are visible in the Actions tab.
"""
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `python scripts/sync_catalog_job.py`
# can import `db` and `app` — without this, only scripts/ is on the path,
# and direct invocation fails with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.database import SessionLocal  # noqa: E402
from app.services.catalog_sync import sync_catalog  # noqa: E402
from app.services.catalog_search import generate_catalog_embeddings  # noqa: E402


def run() -> int:
    """
    Returns a process exit code (0 = success, 1 = failure) rather than
    raising, so main() below has one clear place that decides how the
    process actually exits — and so this function itself stays testable
    (call it directly and assert on the return value, no need to spawn a
    subprocess or catch SystemExit).
    """
    db = SessionLocal()
    try:
        print("Starting catalog sync...")
        sync_result = sync_catalog(db)
        print(f"Sync complete: {sync_result}")

        print("Starting embedding generation for new/updated rows...")
        embed_result = generate_catalog_embeddings(db)
        print(f"Embedding generation complete: {embed_result}")

        return 0

    except Exception as e:
        print(f"ERROR: Catalog sync job failed: {e}", file=sys.stderr)
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(run())