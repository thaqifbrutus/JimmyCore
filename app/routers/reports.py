from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.dataset import Dataset
from app.models.report import QualityReport
from app.models.audit_log import AuditLog
from app.services.profiler import profile_dataset, determine_overall_status
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

        # ai_summary is now a structured result dict:
        # {"status": "ok"|"failed", "reason": ..., "content": ...}
        # Store it as JSON so get_report can return it with the same shape.
        ai_summary = generate_dataset_summary(
            profile_data=profile,
            original_filename=dataset.original_name
        )

        dataset.row_count = profile["overview"]["row_count"]
        dataset.column_count = profile["overview"]["column_count"]
        dataset.status = "complete"

        overall_status = determine_overall_status(profile["issues"])
        report = QualityReport(
            dataset_id=dataset.id,
            profile_data=profile,
            # Persist the full result dict so get_report returns the same
            # shape as this endpoint — frontend always gets a result dict,
            # never a raw string or a double-encoded JSON string.
            ai_summary=json.dumps(ai_summary),
            overall_status=overall_status
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        log = AuditLog(
            dataset_id=dataset.id,
            report_id=report.id,
            action="profile_completed",
            detail=(
                f"Profile and AI summary generated. "
                f"Status: {overall_status}. "
                f"Issues found: {len(profile['issues'])}. "
                f"AI summary status: {ai_summary.get('status')}"
            )
        )
        db.add(log)
        db.commit()

        return {
            "message": "Profiling and AI analysis complete",
            "report_id": str(report.id),
            "overall_status": overall_status,
            "overview": profile["overview"],
            "issues": profile["issues"],
            # Return the result dict directly — frontend's extract_ai_content()
            # reads {"status", "reason", "content"} and handles ok/failed both.
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

    # ai_summary is stored as a JSON string — parse it back to a dict so
    # the frontend always receives the same shape regardless of whether it
    # fetched via this endpoint or the /profile endpoint above.
    ai_summary = report.ai_summary
    if isinstance(ai_summary, str):
        try:
            ai_summary = json.loads(ai_summary)
        except (json.JSONDecodeError, TypeError):
            # Defensive: if for any reason it's a raw string (e.g. a report
            # created before the hardening pass), wrap it in a result dict
            # so the frontend's extract_ai_content() can handle it cleanly.
            ai_summary = {"status": "ok", "reason": None, "content": ai_summary}

    return {
        "id": str(report.id),
        "dataset_id": str(report.dataset_id),
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

    dataset = db.query(Dataset).filter(Dataset.id == report.dataset_id).first()

    # technical_brief is a result dict:
    # {"status": "ok"|"failed", "reason": ..., "content": <TechnicalContext dict>}
    # Returned directly — frontend's extract_ai_content() handles both statuses.
    technical_brief = generate_technical_context(
        profile_data=report.profile_data,
        original_filename=dataset.original_name
    )

    log = AuditLog(
        dataset_id=dataset.id,
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

    dataset = db.query(Dataset).filter(Dataset.id == report.dataset_id).first()

    # answer is a result dict:
    # {"status": "ok"|"failed", "reason": ..., "content": <answer string>}
    # Returned under the "answer" key — matches what the frontend expects
    # (response.get("answer") -> extract_ai_content()).
    answer = answer_dataset_question(
        profile_data=report.profile_data,
        original_filename=dataset.original_name,
        question=request.question,
        conversation_history=request.conversation_history,
    )

    return {
        "question": request.question,
        "answer": answer,
        "report_id": report_id,
    }