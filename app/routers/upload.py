from fastapi import APIRouter

router = APIRouter()

@router.post("/upload")
def upload_file():
    return {"message": "File uploaded"}