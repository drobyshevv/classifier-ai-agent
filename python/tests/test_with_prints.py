# tests/test_with_prints.py
import pytest
from unittest.mock import patch
import json
import base64
import numpy as np

class TestWithPrints:
    """Тесты с выводом реальных response"""
    
    def test_real_health_check(self, client):
        """Реальный тест health check"""
        print("\n" + "="*60)
        print("🩺 REAL HEALTH CHECK TEST")
        print("="*60)
        
        response = client.get("/health")
        
        print(f"📤 Request: GET /health")
        print(f"📥 Response:")
        print(f"   Status: {response.status_code}")
        print(f"   Body: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✅ Health check passed!")
    
    def test_real_article_analysis(self, client):
        """Реальный тест анализа статьи"""
        print("\n" + "="*60)
        print("📝 REAL ARTICLE ANALYSIS TEST")
        print("="*60)
        
        test_data = {
            "document_id": "real_test_001",
            "title_ru": "Реальное тестирование анализа статей",
            "abstract_ru": "Это реальный тест для проверки работы анализа тематик и создания эмбеддингов с использованием искусственного интеллекта и машинного обучения"
        }
        
        print(f"📤 Request: POST /api/analyze-article")
        print(f"   Data: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
        
        response = client.post("/api/analyze-article", json=test_data)
        
        print(f"📥 Response:")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Topics found: {len(data.get('topics', []))}")
            for i, topic in enumerate(data.get('topics', []), 1):
                print(f"     {i}. {topic['topic_name']}: {topic['confidence']:.3f} ({topic['topic_type']})")
            print(f"   Title embedding size: {len(data.get('title_embedding', ''))} bytes")
            print(f"   Abstract embedding size: {len(data.get('abstract_embedding', ''))} bytes")
            
            # Декодируем и покажем размерность эмбеддингов
            try:
                title_embedding = np.frombuffer(base64.b64decode(data['title_embedding']), dtype=np.float32)
                abstract_embedding = np.frombuffer(base64.b64decode(data['abstract_embedding']), dtype=np.float32)
                print(f"   Title embedding shape: {title_embedding.shape}")
                print(f"   Abstract embedding shape: {abstract_embedding.shape}")
            except:
                print(f"   Could not decode embeddings")
        else:
            print(f"   Error: {response.text}")
        
        assert response.status_code == 200
        print("✅ Article analysis passed!")
    
    def test_real_query_analysis(self, client):
        """Реальный тест анализа запроса"""
        print("\n" + "="*60)
        print("🔍 REAL QUERY ANALYSIS TEST")
        print("="*60)
        
        test_data = {
            "user_query": "найти научные статьи про искусственный интеллект в медицине",
            "context": "article_search"
        }
        
        print(f"📤 Request: POST /api/analyze-query")
        print(f"   Data: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
        
        response = client.post("/api/analyze-query", json=test_data)
        
        print(f"📥 Response:")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Interpreted query: '{data.get('interpreted_query')}'")
            print(f"   Key concepts: {data.get('key_concepts', [])}")
            print(f"   Query type: {data.get('query_type')}")
            print(f"   Query vector size: {len(data.get('query_vector', ''))} bytes")
            
            # Декодируем и покажем размерность вектора запроса
            try:
                query_vector = np.frombuffer(base64.b64decode(data['query_vector']), dtype=np.float32)
                print(f"   Query vector shape: {query_vector.shape}")
            except:
                print(f"   Could not decode query vector")
        else:
            print(f"   Error: {response.text}")
        
        assert response.status_code == 200
        print("✅ Query analysis passed!")
    
    def test_real_experts_analysis(self, client):
        """Реальный тест анализа экспертов"""
        print("\n" + "="*60)
        print("👨‍🔬 REAL EXPERTS ANALYSIS TEST")
        print("="*60)
        
        test_data = {
            "topic": "машинное обучение в медицине",
            "authors": [
                {
                    "author_id": "real_author_001",
                    "name": "Реальный Исследователь ИИ",
                    "article_ids": ["real_art1", "real_art2", "real_art3"],
                    "article_topics": ["глубокое обучение", "компьютерное зрение", "медицинская диагностика", "нейронные сети"],
                    "department": "Кафедра информатики"
                },
                {
                    "author_id": "real_author_002", 
                    "name": "Биолог Исследователь",
                    "article_ids": ["bio_art1", "bio_art2"],
                    "article_topics": ["биохимия", "молекулярная биология", "генетика"],
                    "department": "Кафедра биологии"
                }
            ]
        }
        
        print(f"📤 Request: POST /api/analyze-experts")
        print(f"   Topic: {test_data['topic']}")
        print(f"   Authors: {len(test_data['authors'])}")
        
        response = client.post("/api/analyze-experts", json=test_data)
        
        print(f"📥 Response:")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            experts = data.get('experts', [])
            print(f"   Experts found: {len(experts)}")
            for i, expert in enumerate(experts, 1):
                print(f"     {i}. {expert['author_id']}:")
                print(f"        - Expertise score: {expert['expertise_score']:.3f}")
                print(f"        - Topic articles: {expert['topic_article_count']}")
                print(f"        - Total citations: {expert['total_citations']}")
                print(f"        - Related topics: {expert['related_topics']}")
        else:
            print(f"   Error: {response.text}")
        
        assert response.status_code == 200
        print("✅ Experts analysis passed!")
    
    def test_real_departments_analysis(self, client):
        """Реальный тест анализа кафедр"""
        print("\n" + "="*60)
        print("🏛️ REAL DEPARTMENTS ANALYSIS TEST") 
        print("="*60)
        
        test_data = {
            "topic": "искусственный интеллект и большие данные",
            "departments": [
                {
                    "organization_id": "real_dept_001",
                    "name": "Кафедра искусственного интеллекта",
                    "author_ids": ["author_001", "author_002", "author_003"],
                    "article_topics": ["машинное обучение", "большие данные", "нейронные сети", "компьютерное зрение"],
                    "research_areas": ["AI", "Machine Learning", "Data Science"],
                    "faculty": "Факультет информационных технологий"
                },
                {
                    "organization_id": "real_dept_002",
                    "name": "Кафедра биоинформатики",
                    "author_ids": ["author_004", "author_005"],
                    "article_topics": ["биоинформатика", "геномика", "анализ данных"],
                    "research_areas": ["Bioinformatics", "Genomics"],
                    "faculty": "Биологический факультет"
                }
            ]
        }
        
        print(f"📤 Request: POST /api/analyze-departments")
        print(f"   Topic: {test_data['topic']}")
        print(f"   Departments: {len(test_data['departments'])}")
        
        response = client.post("/api/analyze-departments", json=test_data)
        
        print(f"📥 Response:")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            departments = data.get('departments', [])
            print(f"   Departments found: {len(departments)}")
            for i, dept in enumerate(departments, 1):
                print(f"     {i}. {dept['organization_id']}:")
                print(f"        - Strength score: {dept['strength_score']:.3f}")
                print(f"        - Expert count: {dept['expert_count']}")
                print(f"        - Total articles: {dept['total_articles']}")
                print(f"        - Key authors: {dept['key_author_ids']}")
        else:
            print(f"   Error: {response.text}")
        
        assert response.status_code == 200
        print("✅ Departments analysis passed!")