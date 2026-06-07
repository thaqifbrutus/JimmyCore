from fastapi import FastAPI
from app.routers import upload, datasets, reports

app = FastAPI(title="AI Data Processing Platform",
              description="A platform for uploading datasets, processing them with AI, and generating reports.",
              version="0.1.0")

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Data Processing Platform!"}