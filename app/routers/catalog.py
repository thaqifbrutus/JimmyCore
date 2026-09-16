from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from app.services.catalog_sync import sync_catalog

router = APIRouter()


@router.post("/sync")
def trigger_catalog_sync(db: Session = Depends(get_db)):
    """
    Manually triggers a catalog sync — pulls the latest
    dataset_list.parquet from data.gov.my and upserts it into
    catalog_datasets.

    No auth on this yet since JimmyCore has no auth system at all right
    now (same as every other endpoint). Worth revisiting before this is
    reachable by the public internet — right now anyone who finds this
    endpoint could trigger a sync, which is annoying (unnecessary load)
    but not dangerous (idempotent upsert, no data loss risk).

    Manual for now by design — no scheduler wired up yet. Trigger this
    by hand, or from a GitHub Actions scheduled workflow, until a real
    scheduling decision is made.
    """
    try:
        result = sync_catalog(db)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Catalog sync failed: {str(e)}")

    return {
        "message": "Catalog sync complete",
        **result,
    }
