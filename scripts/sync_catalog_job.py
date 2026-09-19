"""
Standalone script for keeping the catalog index fresh — syncs the
official dataset list, then embeds any new/updated rows. Designed to run
as its OWN Railway service (separate from the FastAPI web service),
configured with Railway's native Cron Schedule (Settings > Cron Schedule
on that service) rather than as an HTTP call to /catalog/sync.

Why a separate service instead of calling the existing endpoint:
- Railway cron services run a start command and are expected to EXIT when
  done — this script does exactly that (see the exit code at the bottom),
  whereas the FastAPI web service is meant to stay running. They're
  different service shapes, so they're different Railway services within
  the same project.
- This script imports sync_catalog/generate_catalog_embeddings directly
  and opens its own short-lived DB session, rather than making an HTTP
  request to the running web service. One fewer network hop, and it still
  works correctly even if the web service happens to be redeploying at
  the moment the cron fires.
- Both services share the same DATABASE_URL and OPENROUTER_API_KEY env
  vars (set them on this service too, same values as the web service).

Railway setup (done in the dashboard, not in code):
1. In your Railway project, create a new service from the same repo.
2. Set its start command to: python scripts/sync_catalog_job.py
3. Settings > Cron Schedule: e.g. "0 2 * * *" (02:00 UTC daily) — the
   data.gov.my catalog itself updates roughly daily, so there's no benefit
   to syncing more often than that.
4. Copy DATABASE_URL and OPENROUTER_API_KEY from the web service's
   variables (or reference them, if your Railway plan supports shared
   variables) — this script needs both.

Note per Railway's own docs: cron jobs that exit non-zero are NOT
automatically retried. A missed daily sync isn't a big deal for a
search index — it just means the catalog is up to a day stale until the
next scheduled run — so this deliberately does not implement its own
retry logic. If you want alerting on failures, that's a separate,
optional concern (Railway's dashboard shows failed runs either way).
"""
import sys

from db.database import SessionLocal
from app.services.catalog_sync import sync_catalog
from app.services.catalog_search import generate_catalog_embeddings


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
