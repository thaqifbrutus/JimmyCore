from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.dataset import Dataset
from app.models.report import QualityReport
from app.models.audit_log import AuditLog
from app.services.profiler import profile_dataset, determine_overall_status
import os

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

        # Update dataset with row/column counts 
        dataset.row_count = profile["overview"]["row_count"]
        dataset.column_count = profile["overview"]["column_count"]
        dataset.status = "complete"

        # Create the quality report
        overall_status = determine_overall_status(profile["issues"])
        report = QualityReport(
            dataset_id=dataset.id,
            profile_data=profile,
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
            detail=f"Profile generated. Status: {overall_status}. Issues found: {len(profile['issues'])}"
        )
        db.add(log)
        db.commit()

        return {
            "message": "Profiling complete",
            "report_id": str(report.id),
            "overall_status": overall_status,
            "overview": profile["overview"],
            "issues": profile["issues"]
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