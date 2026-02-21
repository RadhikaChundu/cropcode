"""Query Router — Main agricultural Q&A endpoint"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
import time

from utils.rag_engine import get_rag_engine
from utils.llm_client import generate_answer
from utils.translator import detect_language, translate_to_english, translate_from_english, SUPPORTED_LANGUAGES
from utils.database import save_query

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None  # Auto-detect if not provided


class Source(BaseModel):
    id: str
    title: str
    category: str
    subcategory: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    detected_language: str
    query_id: int
    processing_time_ms: int
    is_cached: bool = False


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()

    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    if len(request.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")

    # Detect language
    detected_lang = request.language or await detect_language(request.query)
    logger.info(f"Query language: {detected_lang}")

    # Translate to English for retrieval
    english_query = request.query
    if detected_lang != "en":
        english_query = await translate_to_english(request.query, detected_lang)
        logger.info(f"Translated query: {english_query}")

    # RAG retrieval
    rag = get_rag_engine()
    context, source_docs = rag.get_context(english_query, top_k=4)

    # Generate answer in English
    answer_en = await generate_answer(english_query, context)

    # Translate answer back if needed
    final_answer = answer_en
    if detected_lang != "en":
        final_answer = await translate_from_english(answer_en, detected_lang)

    # Save to DB
    sources_data = [
        {"id": d["id"], "title": d["title"], "category": d["category"]}
        for d in source_docs
    ]
    query_id = save_query(request.query, detected_lang, final_answer, sources_data)

    processing_time = int((time.time() - start_time) * 1000)

    return QueryResponse(
        answer=final_answer,
        sources=sources_data,
        detected_language=detected_lang,
        query_id=query_id,
        processing_time_ms=processing_time
    )


@router.get("/languages")
async def get_languages():
    return {"languages": SUPPORTED_LANGUAGES}
