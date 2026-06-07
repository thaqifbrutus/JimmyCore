from fastapi import APIRouter

router = APIRouter()

@router.get("/reports")
def get_reports():
    return {"reports": []}