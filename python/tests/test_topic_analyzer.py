# tests/test_topic_analyzer.py
import pytest
import numpy as np


class TestTopicAnalyzerService:
    """Тесты сервиса анализа тематик"""
    
    def test_analyze_article_topics(self, topic_analyzer_service):
        """Тест анализа тематик статьи"""
        document_id = "test_doc"
        title_ru = "Искусственный интеллект в медицине"
        abstract_ru = "Применение ИИ для диагностики заболеваний"
        
        result = topic_analyzer_service.analyze_article_topics(
            document_id, title_ru, abstract_ru
        )
        
        assert isinstance(result, dict)
        assert "topics" in result
        assert "title_embedding" in result
        assert "abstract_embedding" in result
        
        assert isinstance(result["topics"], list)
        assert isinstance(result["title_embedding"], bytes)
        assert isinstance(result["abstract_embedding"], bytes)
    
    def test_analyze_user_query(self, topic_analyzer_service):
        """Тест анализа пользовательского запроса"""
        user_query = "найти статьи про машинное обучение"
        context = "article_search"
        
        result = topic_analyzer_service.analyze_user_query(user_query, context)
        
        assert isinstance(result, dict)
        assert "interpreted_query" in result
        assert "key_concepts" in result
        assert "query_vector" in result
        assert "query_type" in result
        
        assert isinstance(result["interpreted_query"], str)
        assert isinstance(result["key_concepts"], list)
        assert isinstance(result["query_vector"], bytes)
        assert result["query_type"] == context
    
    def test_combine_topics(self, topic_analyzer_service):
        """Тест объединения тем"""
        title_topics = [
            {"topic_name": "AI", "confidence": 0.8, "topic_type": "main"},
            {"topic_name": "ML", "confidence": 0.6, "topic_type": "secondary"}
        ]
        abstract_topics = [
            {"topic_name": "AI", "confidence": 0.9, "topic_type": "main"},
            {"topic_name": "Data", "confidence": 0.7, "topic_type": "secondary"}
        ]
        
        combined = topic_analyzer_service._combine_topics(title_topics, abstract_topics)
        
        assert isinstance(combined, list)
        assert len(combined) <= 3  # Топ-3 темы
        
        for topic in combined:
            assert "topic_name" in topic
            assert "confidence" in topic
            assert "topic_type" in topic
    
    def test_interpret_query(self, topic_analyzer_service):
        """Тест интерпретации запроса"""
        # Тест для article_search
        query1 = "найти статьи про машинное обучение"
        interpreted1 = topic_analyzer_service._interpret_query(query1, "article_search")
        assert "статьи" not in interpreted1.lower()
        
        # Тест для expert_search
        query2 = "найди экспертов по биологии"
        interpreted2 = topic_analyzer_service._interpret_query(query2, "expert_search")
        assert "экспертов" not in interpreted2.lower()
    
    def test_extract_key_concepts(self, topic_analyzer_service):
        """Тест извлечения ключевых концептов"""
        query = "научные статьи про машинное обучение и анализ данных"
        
        concepts = topic_analyzer_service._extract_key_concepts(query)
        
        assert isinstance(concepts, list)
        assert len(concepts) <= 5  # Не более 5 концептов
        
        for concept in concepts:
            assert isinstance(concept, str)
            assert len(concept) > 3  # Слова длиннее 3 символов