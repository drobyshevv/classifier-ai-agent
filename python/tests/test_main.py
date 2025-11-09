# tests/test_main.py
import pytest
from unittest.mock import patch


class TestMainEndpoints:
    """Тесты основных API endpoints"""
    
    def test_health_endpoint(self, client):
        """Тест health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai-agent-ml"
    
    def test_analyze_article_endpoint(self, client, sample_article_data):
        """Тест анализа статьи"""
        with patch('src.main.ml_service') as mock_service:
            mock_service.analyze_article_topics.return_value = {
                "topics": [{"topic_name": "AI", "confidence": 0.8, "topic_type": "main"}],
                "title_embedding": "test_embedding",
                "abstract_embedding": "test_embedding"
            }
            
            response = client.post("/api/analyze-article", json=sample_article_data)
            
            assert response.status_code == 200
            mock_service.analyze_article_topics.assert_called_once_with(
                sample_article_data["document_id"],
                sample_article_data["title_ru"],
                sample_article_data["abstract_ru"]
            )
    
    def test_analyze_query_endpoint(self, client, sample_query_data):
        """Тест анализа запроса"""
        with patch('src.main.ml_service') as mock_service:
            mock_service.analyze_user_query.return_value = {
                "interpreted_query": "машинное обучение",
                "key_concepts": ["машинное", "обучение"],
                "query_vector": "test_vector",
                "query_type": "article_search"
            }
            
            response = client.post("/api/analyze-query", json=sample_query_data)
            
            assert response.status_code == 200
            mock_service.analyze_user_query.assert_called_once_with(
                sample_query_data["user_query"],
                sample_query_data["context"]
            )
    
    def test_analyze_article_missing_fields(self, client):
        """Тест анализа статьи с отсутствующими полями"""
        response = client.post("/api/analyze-article", json={})
        
        assert response.status_code == 500
    
    def test_semantic_search_endpoint(self, client):
        """Тест семантического поиска"""
        search_data = {
            "query_vector": "test_vector",
            "articles": [{"document_id": "art1", "title_ru": "test", "abstract_ru": "test"}],
            "max_results": 5
        }
        
        with patch('src.main.ml_service') as mock_service:
            mock_service.semantic_article_search.return_value = {
                "results": [{"document_id": "art1", "relevance_score": 0.8}],
                "total_found": 1
            }
            
            response = client.post("/api/semantic-search", json=search_data)
            
            assert response.status_code == 200
            mock_service.semantic_article_search.assert_called_once_with(
                search_data["query_vector"],
                search_data["articles"],
                search_data["max_results"]
            )