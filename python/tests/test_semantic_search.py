# tests/test_semantic_search.py
import pytest
import numpy as np


class TestSemanticSearchService:
    """Тесты сервиса семантического поиска"""
    
    def test_search_articles(self, semantic_search_service):
        """Тест поиска статей"""
        # Используем реалистичную размерность
        embedding_dim = 384
        query_vector = np.random.rand(embedding_dim).astype(np.float32)
        
        articles = [
            {
                "document_id": "art1",
                "title_ru": "Нейронные сети",
                "abstract_ru": "Исследование нейронных сетей",
                "title_embedding": np.random.rand(embedding_dim).astype(np.float32).tobytes(),
                "abstract_embedding": np.random.rand(embedding_dim).astype(np.float32).tobytes()
            },
            {
                "document_id": "art2", 
                "title_ru": "Биология",
                "abstract_ru": "Исследование в биологии", 
                "title_embedding": np.random.rand(embedding_dim).astype(np.float32).tobytes(),
                "abstract_embedding": np.random.rand(embedding_dim).astype(np.float32).tobytes()
            }
        ]
        
        results = semantic_search_service.search_articles(
            query_vector, articles, max_results=2
        )
        
        assert isinstance(results, list)
        assert len(results) <= 2
        
        for result in results:
            assert "document_id" in result
            assert "relevance_score" in result
            assert "matched_concepts" in result
            
            assert isinstance(result["relevance_score"], float)
            assert isinstance(result["matched_concepts"], list)
    
    def test_search_articles_empty(self, semantic_search_service):
        """Тест поиска с пустым списком статей"""
        # Используем ту же размерность что и в моке
        embedding_dim = 384
        query_vector = np.random.rand(embedding_dim).astype(np.float32)
        
        results = semantic_search_service.search_articles(
            query_vector, [], max_results=10
        )
        
        assert results == []
    
    def test_vector_similarity(self, semantic_search_service):
        """Тест вычисления косинусного сходства"""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([1.0, 0.0])
        
        similarity = semantic_search_service._vector_similarity(vec1, vec2)
        
        assert isinstance(similarity, float)
        assert similarity == 1.0  # Идентичные векторы
    
    def test_extract_matched_concepts(self, semantic_search_service):
        """Тест извлечения совпавших концептов"""
        article = {"document_id": "test"}
        
        # Высокая релевантность
        concepts_high = semantic_search_service._extract_matched_concepts(
            article, 0.9
        )
        assert "высокая релевантность" in concepts_high
        
        # Средняя релевантность
        concepts_medium = semantic_search_service._extract_matched_concepts(
            article, 0.7
        )
        assert "средняя релевантность" in concepts_medium
        
        # Низкая релевантность
        concepts_low = semantic_search_service._extract_matched_concepts(
            article, 0.5
        )
        assert "низкая релевантность" in concepts_low