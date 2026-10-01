from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import datetime
import os
import json

import pandas as pd

from db.database import get_db, SessionLocal
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
    answer_dataset_question_streaming,
)
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    # conversation_history used to be a client-supplied field. It's now
    # server-managed (persisted on report.chat_messages), so we don't
    # accept it anymore.
    question: str


router = APIRouter()

UPLOAD_DIR = "file_uploads"


def _resolve_source_name(dataset: Dataset | None, catalog_dataset: CatalogDataset | None) -> str:
    """A report's "original filename" for AI-prompt purposes."""
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
    """Looks up whichever source this report actually has."""
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

    source_title = None
    if report.catalog_dataset_id:
        cd = (
            db.query(CatalogDataset)
            .filter(CatalogDataset.id == report.catalog_dataset_id)
            .first()
        )
        if cd:
            source_title = cd.title_en
    elif report.dataset_id:
        d = db.query(Dataset).filter(Dataset.id == report.dataset_id).first()
        if d:
            source_title = d.original_name

    return {
        "id": str(report.id),
        "dataset_id": str(report.dataset_id) if report.dataset_id else None,
        "catalog_dataset_id": report.catalog_dataset_id,
        "profile_data": report.profile_data,
        "ai_summary": ai_summary,
        "chat_messages": report.chat_messages or [],
        "source_title": source_title,
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

    prior_messages = list(report.chat_messages or [])

    answer = answer_dataset_question(
        df=df,
        profile_data=report.profile_data,
        original_filename=original_filename,
        question=request.question,
        conversation_history=prior_messages,
    )

    now = datetime.datetime.utcnow().isoformat()
    prior_messages.append({
        "role": "user",
        "content": request.question,
        "tool_calls": None,
        "error_type": None,
        "timestamp": now,
    })
    prior_messages.append({
        "role": "assistant",
        "content": answer.get("content"),
        "tool_calls": answer.get("tool_calls_log", []),
        "error_type": answer.get("error_type"),
        "timestamp": now,
    })
    report.chat_messages = prior_messages
    db.commit()

    return {
        "question": request.question,
        "answer": answer,
        "report_id": report_id,
        "tool_calls_log": answer.get("tool_calls_log", []),
    }


@router.post("/{report_id}/ask/stream")
def ask_about_dataset_streaming(
    report_id: str,
    request: QuestionRequest,
    db: Session = Depends(get_db),
):
    """
    Streaming variant of /ask. Returns text/event-stream; each SSE data
    line is a JSON-encoded event of the shape documented on
    tool_runner.run_tool_loop_streaming.

    The chat turn is persisted once the generator finishes, using a
    FRESH DB session — the request-scoped `db` is closed by the time
    StreamingResponse runs the generator, so calling db.commit() here
    would fail. See the persist block inside event_stream.
    """
    report = db.query(QualityReport).filter(QualityReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    dataset, catalog_dataset = _get_report_source(report, db)
    original_filename = _resolve_source_name(dataset, catalog_dataset)

    try:
        df = data_loader.load_dataframe_for_report(report, db)
    except data_loader.DataLoadError as e:
        raise HTTPException(status_code=502, detail=str(e))

    prior_messages = list(report.chat_messages or [])

    def event_stream():
        full_text = ""
        tool_calls_log: list = []
        error_type = None

        try:
            for event in answer_dataset_question_streaming(
                df=df,
                profile_data=report.profile_data,
                original_filename=original_filename,
                question=request.question,
                conversation_history=prior_messages,
            ):
                if event["type"] == "token":
                    full_text += event.get("content", "")
                elif event["type"] == "done":
                    tool_calls_log = event.get("tool_calls_log", []) or []
                    # The done event carries the full content — prefer it
                    # in case stream assembly dropped a final fragment.
                    done_content = event.get("content")
                    if done_content and not full_text:
                        full_text = done_content
                elif event["type"] == "error":
                    error_type = event.get("error_type")

                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e), "error_type": None}
            yield f"data: {json.dumps(error_event)}\n\n"
            error_type = None
        finally:
            # Persist using a FRESH session. The request-scoped `db` was
            # closed by the get_db dependency by the time this generator
            # runs — SQLAlchemy would raise if we tried to use it here.
            #
            # This lives in `finally` so we still write what we have if
            # the client disconnects mid-stream (GeneratorExit skips the
            # normal post-loop code path otherwise).
            persist_db = SessionLocal()
            try:
                fresh_report = (
                    persist_db.query(QualityReport)
                    .filter(QualityReport.id == report_id)
                    .first()
                )
                if fresh_report is not None:
                    now = datetime.datetime.utcnow().isoformat()
                    msgs = list(fresh_report.chat_messages or [])
                    msgs.append({
                        "role": "user",
                        "content": request.question,
                        "tool_calls": None,
                        "error_type": None,
                        "timestamp": now,
                    })
                    msgs.append({
                        "role": "assistant",
                        "content": full_text,
                        "tool_calls": tool_calls_log,
                        "error_type": error_type,
                        "timestamp": now,
                    })
                    fresh_report.chat_messages = msgs
                    persist_db.commit()
            except Exception as e:
                print(f"WARNING: failed to persist streaming chat turn: {e}")
            finally:
                persist_db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/{report_id}/reset")
def reset_conversation(report_id: str, db: Session = Depends(get_db)):
    """Clears the chat history for a report."""
    report = db.query(QualityReport).filter(QualityReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report.chat_messages = []
    db.commit()
    return {"report_id": report_id, "reset": True}