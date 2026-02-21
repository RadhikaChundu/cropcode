"""Health check router"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "KrishiSahay API", "version": "1.0.0"}
