from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from loguru import logger

from .ml_service import MLService

app = FastAPI(title="AI Agent ML Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleAnalysisRequest(BaseModel):
    document_id: str
    title_ru: str
    abstract_ru: str

class ArticleTopic(BaseModel):
    topic_name: str
    confidence: float
    topic_type: str

class ArticleAnalysisResponse(BaseModel):
    topics: List[ArticleTopic]
    title_embedding: str  # base64 string
    abstract_embedding: str  # base64 string

# Инициализация ML сервиса
ml_service = MLService()

@app.post("/api/analyze-article", response_model=ArticleAnalysisResponse)
async def analyze_article(request: ArticleAnalysisRequest):
    """Анализ тематик статьи"""
    try:
        result = ml_service.analyze_article_topics(
            request.document_id,
            request.title_ru,
            request.abstract_ru
        )
        return result
    except Exception as e:
        logger.error(f"Error in analyze_article: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-query")
async def analyze_query(request: dict):
    """Анализ пользовательского запроса"""
    try:
        result = ml_service.analyze_user_query(
            request["user_query"],
            request.get("context", "article_search")
        )
        return result
    except Exception as e:
        logger.error(f"Error in analyze_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/semantic-search")
async def semantic_search(request: dict):
    try:
        import base64
        import numpy as np

        # 1) decode query vector
        raw_q = request.get("query_vector")
        if raw_q is None:
            raise ValueError("missing query_vector")

        query_vec = np.frombuffer(base64.b64decode(raw_q), dtype=np.float32)

        # 2) pass article embeddings AS BASE64
        articles = request.get("articles", [])

        return ml_service.semantic_article_search(
            raw_q,      # base64 string
            articles,   # unchanged
            request.get("max_results", 10)
        )

    except Exception as e:
        logger.error(f"Error in semantic_search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-experts")
async def analyze_experts(request: dict):
    """Анализ экспертов по теме"""
    try:
        result = ml_service.analyze_experts_by_topic(
            request["topic"],
            request["authors"]
        )
        return result
    except Exception as e:
        logger.error(f"Error in analyze_experts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-departments")
async def analyze_departments(request: dict):
    """Анализ кафедр по теме"""
    try:
        result = ml_service.analyze_departments_by_topic(
            request["topic"],
            request["departments"]
        )
        return result
    except Exception as e:
        logger.error(f"Error in analyze_departments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "ai-agent-ml"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )