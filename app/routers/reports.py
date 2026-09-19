from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.dataset import Dataset
from app.models.report import QualityReport
from app.models.catalog_dataset import CatalogDataset
from app.models.audit_log import AuditLog
from app.services.profiler import profile_dataset
from app.services.report_builder import persist_report
from app.services.ai_service import (
    generate_dataset_summary,
    generate_technical_context,
    answer_dataset_question,
)
from pydantic import BaseModel
import os
import json


class QuestionRequest(BaseModel):
    question: str
    conversation_history: list = []


router = APIRouter()

UPLOAD_DIR = "file_uploads"


def _resolve_source_name(dataset: Dataset | None, catalog_dataset: CatalogDataset | None) -> str:
    """
    A report's "original filename" for AI-prompt purposes now has two
    possible sources — an uploaded file's original_name, or a government
    catalog dataset's title_en — since QualityReport is shared across both
    flows (see app/models/report.py's docstring). Pure function, no DB
    access, so it's directly unit-testable without a database at all.
    """
    if dataset is not None:
        return dataset.original_name
    if catalog_dataset is not None:
        return catalog_dataset.title_en
    raise ValueError(
        "Report has neither an uploaded dataset nor a catalog dataset — "
        "this should be impossible given the ck_report_exactly_one_source "
        "constraint, so something is badly wrong if this is ever raised."
    )


def _get_report_source(report: QualityReport, db: Session) -> tuple[Dataset | None, CatalogDataset | None]:
    """
    Looks up whichever source this report actually has, based on which of
    dataset_id / catalog_dataset_id is set. Exactly one will be, per the
    DB-level CHECK constraint.
    """
    dataset = None
    catalog_dataset = None

    if report.dataset_id is not None:
        dataset = db.query(Dataset).filter(Dataset.id == report.dataset_id).first()
    elif report.catalog_dataset_id is not None:
        catalog_dataset = db.query(CatalogDataset).filter(CatalogDataset.id == report.catalog_dataset_id).first()

    return dataset, catalog_dataset


@router.post("/datasets/{dataset_id}/profile")
def trigger_profile(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    dataset.status = "processing"
    db.commit()

    try:
        profile = profile_dataset(file_path)

        ai_summary = generate_dataset_summary(
            profile_data=profile,
            original_filename=dataset.original_name
        )

        dataset.row_count = profile["overview"]["row_count"]
        dataset.column_count = profile["overview"]["column_count"]
        dataset.status = "complete"
        # Deliberately not committed here — persist_report's first commit
        # (when it creates the report) flushes these dataset changes too,
        # matching the original code's atomicity: dataset status and its
        # report are committed together, not as two separate transactions.

        report = persist_report(
            db, profile, ai_summary,
            dataset_id=dataset.id,
            audit_action="profile_completed",
        )

        return {
            "message": "Profiling and AI analysis complete",
            "report_id": str(report.id),
            "overall_status": report.overall_status,
            "overview": profile["overview"],
            "issues": profile["issues"],
            "ai_summary": ai_summary,
        }

    except Exception as e:
        dataset.status = "failed"
        db.commit()

        log = AuditLog(
            dataset_id=dataset.id,
            action="profile_failed",
            detail=str(e)
        )
        db.add(log)
        db.commit()

        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")


@router.get("/{report_id}")
def get_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(QualityReport).filter(QualityReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    ai_summary = report.ai_summary
    if isinstance(ai_summary, str):
        try:
            ai_summary = json.loads(ai_summary)
        except (json.JSONDecodeError, TypeError):
            ai_summary = {"status": "ok", "reason": None, "content": ai_summary}

    return {
        "id": str(report.id),
        "dataset_id": str(report.dataset_id) if report.dataset_id else None,
        "catalog_dataset_id": report.catalog_dataset_id,
        "profile_data": report.profile_data,
        "ai_summary": ai_summary,
        "overall_status": report.overall_status,
        "created_at": report.created_at.isoformat()
    }


@router.post("/{report_id}/technical-context")
def get_technical_context(
    report_id: str,
    db: Session = Depends(get_db)
):
    report = db.query(QualityReport).filter(QualityReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # FIX: this used to do db.query(Dataset).filter(Dataset.id ==
    # report.dataset_id).first() unconditionally, then reach for
    # dataset.original_name — which breaks with an AttributeError on
    # None for any catalog-sourced report, since report.dataset_id is
    # None there. Now resolves whichever source the report actually has.
    dataset, catalog_dataset = _get_report_source(report, db)
    original_filename = _resolve_source_name(dataset, catalog_dataset)

    technical_brief = generate_technical_context(
        profile_data=report.profile_data,
        original_filename=original_filename
    )

    log = AuditLog(
        dataset_id=dataset.id if dataset else None,
        report_id=report.id,
        action="technical_context_generated",
        detail=(
            f"Technical brief generated. "
            f"Status: {technical_brief.get('status')}"
        )
    )
    db.add(log)
    db.commit()

    return {
        "report_id": report_id,
        "technical_brief": technical_brief,
    }


@router.post("/{report_id}/ask")
def ask_about_dataset(
    report_id: str,
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    report = db.query(QualityReport).filter(QualityReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Same fix as get_technical_context above.
    dataset, catalog_dataset = _get_report_source(report, db)
    original_filename = _resolve_source_name(dataset, catalog_dataset)

    answer = answer_dataset_question(
        profile_data=report.profile_data,
        original_filename=original_filename,
        question=request.question,
        conversation_history=request.conversation_history,
    )

    return {
        "question": request.question,
        "answer": answer,
        "report_id": report_id,
    }
