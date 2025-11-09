# tests/test_expert_analyzer.py
import pytest
import numpy as np


class TestExpertAnalyzerService:
    """Тесты сервиса анализа экспертов"""
    
    def test_analyze_experts_by_topic(self, expert_analyzer_service, sample_authors_data):
        """Тест анализа экспертов по теме"""
        topic = "машинное обучение"
        
        results = expert_analyzer_service.analyze_experts_by_topic(
            topic, sample_authors_data
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for expert in results:
            assert "author_id" in expert
            assert "expertise_score" in expert
            assert "topic_article_count" in expert
            assert "total_citations" in expert
            assert "last_activity_year" in expert
            assert "related_topics" in expert
            
            # Проверяем типы данных
            assert isinstance(expert["expertise_score"], float)
            assert isinstance(expert["topic_article_count"], int)
            assert isinstance(expert["total_citations"], int)
            assert isinstance(expert["last_activity_year"], int)
            assert isinstance(expert["related_topics"], list)
    
    def test_analyze_experts_empty_authors(self, expert_analyzer_service):
        """Тест анализа экспертов с пустым списком авторов"""
        results = expert_analyzer_service.analyze_experts_by_topic(
            "тема", []
        )
        
        assert results == []
    
    def test_analyze_departments_by_topic(self, expert_analyzer_service, sample_departments_data):
        """Тест анализа кафедр по теме"""
        topic = "искусственный интеллект"
        
        results = expert_analyzer_service.analyze_departments_by_topic(
            topic, sample_departments_data
        )
        
        assert isinstance(results, list)
        
        if results:  # Если есть результаты
            for dept in results:
                assert "organization_id" in dept
                assert "strength_score" in dept
                assert "expert_count" in dept
                assert "total_articles" in dept
                assert "key_author_ids" in dept
    
    def test_calculate_expertise_score(self, expert_analyzer_service):
        """Тест вычисления оценки экспертизы"""
        author = {
            "article_topics": ["машинное обучение", "глубокое обучение"]
        }
        topic = "искусственный интеллект"
        topic_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        score = expert_analyzer_service._calculate_expertise_score(
            author, topic, topic_vector
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_count_topic_articles(self, expert_analyzer_service):
        """Тест подсчета статей по теме"""
        author = {
            "article_topics": ["машинное обучение", "биология", "компьютерное зрение"]
        }
        topic = "машинное обучение"
        
        count = expert_analyzer_service._count_topic_articles(author, topic)
        
        assert isinstance(count, int)
        assert count >= 0