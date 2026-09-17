from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from db.database import get_db
from app.services.catalog_sync import sync_catalog
from app.services.catalog_search import generate_catalog_embeddings, search_catalog

router = APIRouter()


@router.post("/sync")
def trigger_catalog_sync(db: Session = Depends(get_db)):
    """
    Manually triggers a catalog sync — pulls the latest
    dataset_list.parquet from data.gov.my and upserts it into
    catalog_datasets.

    No auth on this yet since JimmyCore has no auth system at all right
    now (same as every other endpoint). Not dangerous (idempotent upsert),
    just unnecessary load if hit repeatedly by the public internet.

    Manual for now by design — no scheduler wired up yet.
    """
    try:
        result = sync_catalog(db)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Catalog sync failed: {str(e)}")

    return {
        "message": "Catalog sync complete",
        **result,
    }


@router.post("/embeddings/generate")
def trigger_embedding_generation(
    force: bool = Query(False, description="Re-embed every row, not just un-embedded ones"),
    db: Session = Depends(get_db),
):
    """
    Embeds every catalog row that doesn't have an embedding yet (or every
    row, if force=true). Run this after a sync brings in new/updated rows
    — sync and embedding are deliberately separate steps, since embedding
    costs real API calls and shouldn't fire on every sync automatically.
    """
    try:
        result = generate_catalog_embeddings(db, force=force)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {str(e)}")

    return {
        "message": "Embedding generation complete",
        **result,
    }


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Free-text search query"),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """
    Semantic search over the local catalog index. Returns the top_k
    closest-matching datasets by embedding similarity, not exact keyword
    matching — a query like "drunk driving accidents" can match a dataset
    titled "Road Traffic Accident Statistics" even with no shared words.
    """
    try:
        results = search_catalog(db, query=q, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {str(e)}")

    return {
        "query": q,
        "results": [
            {
                "id": r["dataset"].id,
                "title_en": r["dataset"].title_en,
                "category_en": r["dataset"].category_en,
                "subcategory_en": r["dataset"].subcategory_en,
                "source": r["dataset"].source,
                "frequency": r["dataset"].frequency,
                "dataset_begin": r["dataset"].dataset_begin,
                "dataset_end": r["dataset"].dataset_end,
                "score": round(r["score"], 4),
            }
            for r in results
        ],
    }
