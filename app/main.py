from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.routers import upload, datasets, reports, catalog
from db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="JimmyCore",
    description="A RAG + tool-calling analyst over Malaysian government open data and uploaded CSVs.",
    version="0.2.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])
app.include_router(catalog.router, prefix ="/catalog", tags=["Catalog"]) 

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Data Processing Platform!"}