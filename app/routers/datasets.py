from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.dataset import Dataset
from app.models.report import QualityReport
router = APIRouter()

@router.get("") ###########
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()

    return [{
        "id": str(d.id),
        "original_name": d.original_name,
        "row_count": d.row_count,
        "column_count": d.column_count,
        "status": d.status,
        "uploaded_at": d.uploaded_at.isoformat()
    }for d in datasets
]

@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Fetch associated quality report if it exists
    report = (db.query(QualityReport).filter(QualityReport.dataset_id == dataset_id).order_by(QualityReport.created_at.desc()).first())

    return {
        "id": str(dataset.id),
        "original_name": dataset.original_name,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "status": dataset.status,
        "uploaded_at": dataset.uploaded_at.isoformat(),
        "latest_report": {"id": str(report.id),
                          "overall_status": report.overall_status,
                          "ai_summary": report.ai_summary,
                          "created_at": report.created_at.isoformat()
                          } if report else None
    }

#@router.get("/datasets")
#def get_datasets():
    #return {"datasets": []}