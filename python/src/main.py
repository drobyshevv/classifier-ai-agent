from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Инициализация ML сервиса
ml_service = MLService()

@app.post("/api/analyze-article")
async def analyze_article(request: dict):
    """Анализ тематик статьи"""
    try:
        result = ml_service.analyze_article_topics(
            request["document_id"],
            request["title_ru"],
            request["abstract_ru"]
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
    """Семантический поиск статей"""
    try:
        result = ml_service.semantic_article_search(
            request["query_vector"],
            request["articles"],
            request.get("max_results", 10)
        )
        return result
    except Exception as e:
        logger.error(f"Error in semantic_search: {e}")
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