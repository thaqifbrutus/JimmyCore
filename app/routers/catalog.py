from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.catalog_dataset import CatalogDataset
from app.services import data_tools
from app.services.catalog_sync import sync_catalog
from app.services.catalog_search import generate_catalog_embeddings, search_catalog
from app.services.gov_data_client import get_dataset_dataframe, GovAPIError
from app.services.profiler import profile_dataframe
from app.services.ai_service import generate_dataset_overview, build_source_context
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
    Given a dataset id (found via /catalog/search), fetches its real data
    (cached or live), profiles it, generates an AI overview plus suggested
    starter questions plus a chart, and persists a QualityReport.

    The profile's "source" field (see build_source_context in
    ai_service.py) tells the model this is an official government dataset
    rather than a user upload, so it cites the source agency and sticks to
    summarizing rather than interpreting.
    """
    catalog_dataset = (
        db.query(CatalogDataset).filter(CatalogDataset.id == catalog_dataset_id).first()
    )
    if not catalog_dataset:
        raise HTTPException(status_code=404, detail="Catalog dataset not found")

    try:
        df = get_dataset_dataframe(db, catalog_dataset_id, force_refresh=force_refresh)
    except GovAPIError as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch government data: {str(e)}")

    if len(df) == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset '{catalog_dataset_id}' returned no rows from the "
                f"government API — nothing to analyze."
            ),
        )

    try:
        profile = profile_dataframe(df)
        source = build_source_context(catalog_dataset)
        profile["source"] = source

        overview = generate_dataset_overview(
            profile_data=profile,
            original_filename=catalog_dataset.title_en,
        )

        # Attach a chart for the overview, based on the model's
        # primary_column + primary_metric hints. When both are set and
        # valid, this produces a sum-based chart ("sum of total_cases by
        # state"); otherwise it falls back to a count-based chart. The
        # chart lives on the overview dict so it survives the /reports/{id}
        # round-trip (ai_summary is where the overview is persisted).
        if overview.get("status") == "ok" and isinstance(overview.get("content"), dict):
            primary = overview["content"].get("primary_column")
            metric = overview["content"].get("primary_metric")
            overview["chart"] = data_tools.chart_data_for_overview(
                df, primary_column=primary, primary_metric=metric
            )
        else:
            overview["chart"] = None

        report = persist_report(
            db, profile, overview,
            catalog_dataset_id=catalog_dataset.id,
            audit_action="catalog_overview_generated",
        )

        return {
            "report_id": str(report.id),
            "catalog_dataset_id": catalog_dataset.id,
            "dataset_stats": profile["overview"],
            "columns": profile["columns"],
            "overview": overview,
            "source": source,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")