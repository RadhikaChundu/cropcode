"""Feedback router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from utils.database import save_feedback

router = APIRouter()

class FeedbackRequest(BaseModel):
    query_id: int
    rating: int  # 1 = helpful, -1 = not helpful
    comment: Optional[str] = ""

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    if request.rating not in [1, -1]:
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
    save_feedback(request.query_id, request.rating, request.comment or "")
    return {"success": True, "message": "Thank you for your feedback!"}
