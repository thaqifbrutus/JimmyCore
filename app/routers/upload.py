import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from app.models.dataset import Dataset
from app.models.audit_log import AuditLog

router = APIRouter()

UPLOAD_DIR = "file_uploads"
ALLOWED_TYPES = ["text/csv", "application/vnd.ms-excel"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# FastAPI endpoint for uploading datasets via UploadFile, File and Depends
@router.post("")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):

    # Validate file type via HTTPException
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Only CSV files are allowed.")
    
    #Read file into memory and check size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {size_mb:.1f}MB. Maximum allowed size is {MAX_FILE_SIZE}MB.")
    
    # Generate a unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    #Save file to disk
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(contents)

    #Create a record in the database for the uploaded dataset
    dataset = Dataset(filename=unique_filename, original_name=file.filename, status="pending")
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # Log the upload action in the audit log
    log = AuditLog(dataset_id=dataset.id, action="dataset_uploaded", detail=f"File '{file.filename}' uploaded. Size: {size_mb:.2f}MB.")
    db.add(log)
    db.commit()

    return {"message": f"File '{file.filename}' uploaded successfully.", "dataset_id": str(dataset.id),
            "original_name": dataset.original_name, "status": dataset.status}

    
# @router.post("/upload")
# def upload_file():
#     return {"message": "File uploaded"}
