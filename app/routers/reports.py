from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.dataset import Dataset
from app.models.report import QualityReport
from app.models.audit_log import AuditLog
from app.services.profiler import profile_dataset, determine_overall_status
from app.services.ai_service import (generate_dataset_summary, generate_technical_context, answer_dataset_question)
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
    # Fetch the dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Make sure the file actually exists on disk
    file_path = os.path.join(UPLOAD_DIR, dataset.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Update dataset status to processing
    dataset.status = "processing"
    db.commit()

    # Run the profiler
    try:
        profile = profile_dataset(file_path)

        ai_summary = generate_dataset_summary(
            profile_data=profile,
            original_filename=dataset.original_name
        )

        # Update dataset with row/column counts 
        dataset.row_count = profile["overview"]["row_count"]
        dataset.column_count = profile["overview"]["column_count"]
        dataset.status = "complete"

        # Create the quality report
        overall_status = determine_overall_status(profile["issues"])
        report = QualityReport(
            dataset_id=dataset.id,
            profile_data=profile,
            ai_summary=json.dumps(ai_summary),
            overall_status=overall_status
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        # Log it
        log = AuditLog(
            dataset_id=dataset.id,
            report_id=report.id,
            action="profile_completed",
            detail=f"Profile and AI summary generated. Status: {overall_status}. Issues found: {len(profile['issues'])}"
        )
        db.add(log)
        db.commit()

        return {
            "message": "Profiling and AI analysis complete",
            "report_id": str(report.id),
            "overall_status": overall_status,
            "overview": profile["overview"],
            "issues": profile["issues"],
            "ai_summary": ai_summary
        }

    except Exception as e:
        # If anything goes wrong, mark the dataset as failed
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
    
    return {
        "id": str(report.id),
        "dataset_id": str(report.dataset_id),
        "profile_data": report.profile_data,
        "ai_summary": report.ai_summary,
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

    technical_brief = generate_technical_context(
        profile_data=report.profile_data,
        original_filename=dataset.original_name
    )

    # Log this action
    log = AuditLog(
        dataset_id=dataset.id,
        report_id=report.id,
        action="technical_context_generated",
        detail="Technical brief generated for development team"
    )
    db.add(log)
    db.commit()

    return {
        "report_id": report_id,
        "technical_brief": technical_brief
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

    answer = answer_dataset_question(
        profile_data=report.profile_data,
        original_filename=dataset.original_name,
        question=request.question,
        conversation_history=request.conversation_history
    )

    return {
        "question": request.question,
        "answer": answer,
        "report_id": report_id
    }