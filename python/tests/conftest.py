# tests/conftest.py
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock

from src.main import app
from src.ml_service import MLService
from src.services.expert_analyzer import ExpertAnalyzerService
from src.services.semantic_search import SemanticSearchService
from src.services.topic_analyzer import TopicAnalyzerService
from src.models.bert_model import RuBERTModel


@pytest.fixture
def client():
    """Фикстура для тестового клиента FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_bert_model():
    """Мок RuBERT модели"""
    mock_model = Mock(spec=RuBERTModel)
    
    # Используем реалистичную размерность (например, 384 для sentence-transformers)
    embedding_dim = 384
    mock_model.encode_text.return_value = np.random.rand(embedding_dim).astype(np.float32)
    mock_model.encode_batch.return_value = np.random.rand(2, embedding_dim).astype(np.float32)
    
    # Мок анализа тем
    mock_model.analyze_topics.return_value = [
        {"topic_name": "машинное обучение", "confidence": 0.8, "topic_type": "main"},
        {"topic_name": "анализ данных", "confidence": 0.6, "topic_type": "secondary"}
    ]
    
    # Мок косинусного сходства
    mock_model._cosine_similarity.return_value = 0.75
    
    return mock_model


@pytest.fixture
def expert_analyzer_service(mock_bert_model):
    """Фикстура сервиса анализа экспертов"""
    return ExpertAnalyzerService(mock_bert_model)


@pytest.fixture
def semantic_search_service(mock_bert_model):
    """Фикстура сервиса семантического поиска"""
    return SemanticSearchService(mock_bert_model)


@pytest.fixture
def topic_analyzer_service(mock_bert_model):
    """Фикстура сервиса анализа тематик"""
    return TopicAnalyzerService(mock_bert_model)


@pytest.fixture
def ml_service(mock_bert_model):
    """Фикстура основного ML сервиса"""
    return MLService()


@pytest.fixture
def sample_article_data():
    """Пример данных статьи"""
    return {
        "document_id": "test_doc_001",
        "title_ru": "Искусственный интеллект в медицине",
        "abstract_ru": "Исследование применения ИИ для диагностики заболеваний"
    }


@pytest.fixture
def sample_query_data():
    """Пример данных запроса"""
    return {
        "user_query": "найти статьи про машинное обучение",
        "context": "article_search"
    }


@pytest.fixture
def sample_authors_data():
    """Пример данных авторов"""
    return [
        {
            "author_id": "author_001",
            "name": "Иван Петров",
            "article_ids": ["art1", "art2"],
            "article_topics": ["глубокое обучение", "компьютерное зрение"],
            "department": "Кафедра информатики"
        },
        {
            "author_id": "author_002",
            "name": "Мария Сидорова",
            "article_ids": ["art3", "art4"],
            "article_topics": ["биохимия", "молекулярная биология"],
            "department": "Кафедра биологии"
        }
    ]


@pytest.fixture
def sample_departments_data():
    """Пример данных кафедр"""
    return [
        {
            "organization_id": "dept_001",
            "name": "Кафедра информатики",
            "author_ids": ["author_001", "author_003"],
            "article_topics": ["машинное обучение", "большие данные"],
            "research_areas": ["AI", "Data Science"],
            "faculty": "Факультет информационных технологий"
        }
    ]