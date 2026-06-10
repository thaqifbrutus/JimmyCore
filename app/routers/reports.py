from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.report import QualityReport


router = APIRouter()

@router.get("/reports/{report_id}")
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

#@router.get("/reports")
#def get_reports():
   # return {"reports": []}