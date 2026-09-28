from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import os
import json

import pandas as pd

from db.database import get_db
from app.models.dataset import Dataset
from app.models.report import QualityReport
from app.models.catalog_dataset import CatalogDataset
from app.models.audit_log import AuditLog
from app.services import data_loader, data_tools
from app.services.profiler import profile_dataset
from app.services.report_builder import persist_report
from app.services.ai_service import (
    generate_dataset_overview,
    answer_dataset_question,
)
from pydantic import BaseModel


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
    flows. Pure function, no DB access, so it's directly unit-testable
    without a database at all.
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

        overview = generate_dataset_overview(
            profile_data=profile,
            original_filename=dataset.original_name,
        )

        # Attach a chart to the overview, matching the catalog flow's
        # response shape. Chart is presentation — if the file can't be
        # re-read for charting, the overview just ships without a chart
        # rather than failing the whole profile request.
        if overview.get("status") == "ok" and isinstance(overview.get("content"), dict):
            try:
                chart_df = pd.read_csv(file_path)
                primary = overview["content"].get("primary_column")
                metric = overview["content"].get("primary_metric")
                overview["chart"] = data_tools.chart_data_for_overview(
                    chart_df, primary_column=primary, primary_metric=metric
                )
            except Exception:
                overview["chart"] = None
        else:
            overview["chart"] = None

        dataset.row_count = profile["overview"]["row_count"]
        dataset.column_count = profile["overview"]["column_count"]
        dataset.status = "complete"
        # Deliberately not committed here — persist_report's first commit
        # (when it creates the report) flushes these dataset changes too,
        # matching the original code's atomicity: dataset status and its
        # report are committed together, not as two separate transactions.

        report = persist_report(
            db, profile, overview,
            dataset_id=dataset.id,
            audit_action="profile_completed",
        )

        return {
            "report_id": str(report.id),
            "dataset_id": str(dataset.id),
            "dataset_stats": profile["overview"],
            "columns": profile["columns"],
            "overview": overview,
            "source": None,
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
        "created_at": report.created_at.isoformat()
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

    dataset, catalog_dataset = _get_report_source(report, db)
    original_filename = _resolve_source_name(dataset, catalog_dataset)

    try:
        df = data_loader.load_dataframe_for_report(report, db)
    except data_loader.DataLoadError as e:
        raise HTTPException(status_code=502, detail=str(e))

    answer = answer_dataset_question(
        df=df,
        profile_data=report.profile_data,
        original_filename=original_filename,
        question=request.question,
        conversation_history=request.conversation_history,
    )

    return {
        "question": request.question,
        "answer": answer,
        "report_id": report_id,
        "tool_calls_log": answer.get("tool_calls_log", []),
    }