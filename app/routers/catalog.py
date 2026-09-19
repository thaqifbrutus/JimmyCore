from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.catalog_dataset import CatalogDataset
from app.services.catalog_sync import sync_catalog
from app.services.catalog_search import generate_catalog_embeddings, search_catalog
from app.services.gov_data_client import get_dataset_dataframe, GovAPIError
from app.services.profiler import profile_dataframe
from app.services.ai_service import generate_dataset_summary
from app.services.report_builder import persist_report

router = APIRouter()


@router.post("/sync")
def trigger_catalog_sync(db: Session = Depends(get_db)):
    try:
        result = sync_catalog(db)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Catalog sync failed: {str(e)}")

    return {"message": "Catalog sync complete", **result}


@router.post("/embeddings/generate")
def trigger_embedding_generation(
    force: bool = Query(False, description="Re-embed every row, not just un-embedded ones"),
    db: Session = Depends(get_db),
):
    try:
        result = generate_catalog_embeddings(db, force=force)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {str(e)}")

    return {"message": "Embedding generation complete", **result}


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Free-text search query"),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
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


@router.post("/{catalog_dataset_id}/analyze")
def analyze_catalog_dataset(
    catalog_dataset_id: str,
    force_refresh: bool = Query(False, description="Bypass the cached government data, fetch live"),
    db: Session = Depends(get_db),
):
    """
    The endpoint that actually closes the loop: given a dataset id (found
    via /catalog/search), fetches its real data (cached or live, see
    gov_data_client.py), profiles it, generates an AI summary, and
    persists a QualityReport — the same report shape as the upload flow,
    so every existing /reports/{id}, /reports/{id}/technical-context, and
    /reports/{id}/ask endpoint works on the result with no changes.
    """
    catalog_dataset = db.query(CatalogDataset).filter(CatalogDataset.id == catalog_dataset_id).first()
    if not catalog_dataset:
        raise HTTPException(status_code=404, detail="Catalog dataset not found")

    try:
        df = get_dataset_dataframe(db, catalog_dataset_id, force_refresh=force_refresh)
    except GovAPIError as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch government data: {str(e)}")

    if len(df) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{catalog_dataset_id}' returned no rows from the government API — nothing to analyze.",
        )

    try:
        profile = profile_dataframe(df)

        ai_summary = generate_dataset_summary(
            profile_data=profile,
            original_filename=catalog_dataset.title_en,
        )

        report = persist_report(
            db, profile, ai_summary,
            catalog_dataset_id=catalog_dataset.id,
            audit_action="catalog_profile_completed",
        )

        return {
            "message": "Government dataset fetched, profiled, and AI-summarized",
            "report_id": str(report.id),
            "catalog_dataset_id": catalog_dataset.id,
            "overall_status": report.overall_status,
            "overview": profile["overview"],
            "issues": profile["issues"],
            "ai_summary": ai_summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
