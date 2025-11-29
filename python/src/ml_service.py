import numpy as np
from loguru import logger
from typing import List, Dict, Any
import base64

from .models.bert_model import RuBERTModel
from .services.topic_analyzer import TopicAnalyzerService
from .services.semantic_search import SemanticSearchService
from .services.expert_analyzer import ExpertAnalyzerService
from .topic.intelligent_topics import extended_topics


class MLService:
    """Основной ML сервис"""
    
    def __init__(self):
        logger.info("Инициализация ML сервиса...")

        self.config = {
            'models': {
                'bert_model': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                'topic_model': "cointegrated/rubert-tiny2", 
                'device': "cpu"
            },
            'embeddings': {
                'dimension': 384,
                'normalize': True
            },
            'topics': {
                'predefined_topics': extended_topics
            }
        }
        
        self.bert_model = RuBERTModel(self.config)
        self.topic_analyzer = TopicAnalyzerService(self.bert_model)
        self.semantic_search = SemanticSearchService(self.bert_model)
        self.expert_analyzer = ExpertAnalyzerService(self.bert_model)
        
        logger.info("ML сервис инициализирован")
    
    def analyze_article_topics(self, document_id: str, title_ru: str, abstract_ru: str) -> Dict[str, Any]:
        """Анализ тематик статьи"""
        logger.info(f"Анализ статьи {document_id}")
        
        result = self.topic_analyzer.analyze_article_topics(document_id, title_ru, abstract_ru)
        
        logger.info(f"Title embedding type from topic_analyzer: {type(result['title_embedding'])}")
        logger.info(f"Abstract embedding type from topic_analyzer: {type(result['abstract_embedding'])}")
        
        return {
            "topics": result["topics"],
            "title_embedding": result["title_embedding"],  
            "abstract_embedding": result["abstract_embedding"]  
        }
    
    def analyze_user_query(self, user_query: str, context: str) -> Dict[str, Any]:
        """Анализ пользовательского запроса"""
        logger.info(f"Анализ запроса: {user_query}")
        
        result = self.topic_analyzer.analyze_user_query(user_query, context)
        
        return {
            "interpreted_query": result["interpreted_query"],
            "key_concepts": result["key_concepts"],
            "query_vector": base64.b64encode(result["query_vector"]).decode('utf-8'),
            "query_type": result["query_type"]
        }
    
    def semantic_article_search(self, query_vector: str, articles: List[Dict], max_results: int):
        logger.info(f"Семантический поиск по {len(articles)} статьям")

        # 1) decode query vector
        query_vec = np.frombuffer(base64.b64decode(query_vector), dtype=np.float32)

        processed_articles = []
        for art in articles:
            processed_articles.append({
                "document_id": art["document_id"],
                "title_ru": art["title_ru"],
                "abstract_ru": art["abstract_ru"],
                # 2) RAW base64, decode later in SemanticSearchService
                "title_embedding": art["title_embedding"],
                "abstract_embedding": art["abstract_embedding"]
            })

        results = self.semantic_search.search_articles(query_vec, processed_articles, max_results)

        return {
            "results": results,
            "total_found": len(results)
        }

    
    def analyze_experts_by_topic(self, topic: str, authors: List[Dict]) -> Dict[str, Any]:
        """Анализ экспертов по теме"""
        logger.info(f"Анализ экспертов по теме: {topic}")
        
        experts = self.expert_analyzer.analyze_experts_by_topic(topic, authors)
        
        return {
            "experts": experts
        }
    
    def analyze_departments_by_topic(self, topic: str, departments: List[Dict]) -> Dict[str, Any]:
        """Анализ кафедр по теме"""
        logger.info(f"Анализ кафедр по теме: {topic}")
        
        dept_analysis = self.expert_analyzer.analyze_departments_by_topic(topic, departments)
        
        return {
            "departments": dept_analysis
        }