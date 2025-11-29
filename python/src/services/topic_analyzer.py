from typing import List, Dict
import numpy as np
from loguru import logger

from src.utils.vector_utils import vector_to_base64


class TopicAnalyzerService:
    """Сервис анализа тематик"""
    
    def __init__(self, bert_model):
        self.bert_model = bert_model
    
    def analyze_article_topics(self, document_id: str, title_ru: str, abstract_ru: str) -> Dict:
        """Анализ тематик статьи"""
        logger.info(f"Анализ тематик для статьи {document_id}")
        
        # Анализ тем по заголовку и аннотации
        title_topics = self.bert_model.analyze_topics(title_ru)
        abstract_topics = self.bert_model.analyze_topics(abstract_ru)
        
        # Объединяем и усредняем уверенность
        combined_topics = self._combine_topics(title_topics, abstract_topics)
        
        # Создаем эмбеддинги
        title_embedding = self.bert_model.encode_text(title_ru)
        abstract_embedding = self.bert_model.encode_text(abstract_ru)
        
        # ЛОГИ ДЛЯ ДЕБАГА
        logger.info(f"Title: '{title_ru}'")
        logger.info(f"Abstract: '{abstract_ru}'")
        logger.info(f"Title embedding shape: {title_embedding.shape}, dtype: {title_embedding.dtype}")
        logger.info(f"Abstract embedding shape: {abstract_embedding.shape}, dtype: {abstract_embedding.dtype}")
        logger.info(f"Title embedding sample: {title_embedding[:5]}")  # первые 5 значений
        logger.info(f"Abstract embedding sample: {abstract_embedding[:5]}")
        
        
        title_b64 = vector_to_base64(title_embedding)
        abstract_b64 = vector_to_base64(abstract_embedding)
        
        logger.info(f"Title base64 length: {len(title_b64)}")
        logger.info(f"Abstract base64 length: {len(abstract_b64)}")
        logger.info(f"Title base64 preview: {title_b64[:50]}...")
        logger.info(f"Abstract base64 preview: {abstract_b64[:50]}...")
        
        return {
            "topics": combined_topics,
            "title_embedding": title_b64, 
            "abstract_embedding": abstract_b64 
        }
    
    def analyze_user_query(self, user_query: str, context: str = "article_search") -> Dict:
        """Анализ пользовательского запроса"""
        logger.info(f"Анализ запроса: '{user_query}' в контексте: {context}")
        
        # Очистка и интерпретация запроса
        interpreted_query = self._interpret_query(user_query, context)
        
        # Извлечение ключевых концептов
        key_concepts = self._extract_key_concepts(interpreted_query)
        
        # Векторизация запроса
        query_vector = self.bert_model.encode_text(interpreted_query)
        
        return {
            "interpreted_query": interpreted_query,
            "key_concepts": key_concepts,
            "query_vector": query_vector.tobytes(),
            "query_type": context
        }
    
    def _combine_topics(self, title_topics: List[Dict], abstract_topics: List[Dict]) -> List[Dict]:
        """Объединение тем из заголовка и аннотации"""
        topic_dict = {}
        
        # Обрабатываем темы заголовка
        for topic in title_topics:
            name = topic["topic_name"]
            if name in topic_dict:
                topic_dict[name]["confidence"] = max(topic_dict[name]["confidence"], topic["confidence"])
            else:
                topic_dict[name] = topic.copy()
        
        # Обрабатываем темы аннотации
        for topic in abstract_topics:
            name = topic["topic_name"]
            if name in topic_dict:
                topic_dict[name]["confidence"] = (topic_dict[name]["confidence"] + topic["confidence"]) / 2
            else:
                topic_dict[name] = topic.copy()
        
        # Сортируем по уверенности
        combined = list(topic_dict.values())
        combined.sort(key=lambda x: x["confidence"], reverse=True)
        return combined[:3]  # Возвращаем топ-3 темы
    
    def _interpret_query(self, query: str, context: str) -> str:
        """Интерпретация и очистка запроса"""
        # Удаляем стоп-слова в зависимости от контекста
        stop_words = {
            "expert_search": ["найди", "покажи", "ищи", "экспертов", "авторов"],
            "department_search": ["кафедр", "подразделений", "факультетов", "где"],
            "article_search": ["статьи", "публикации", "работы", "про"]
        }
        
        words = query.lower().split()
        context_stop_words = stop_words.get(context, [])
        filtered_words = [word for word in words if word not in context_stop_words]
        
        return " ".join(filtered_words).strip()
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Извлечение ключевых концептов из запроса"""
        # Простая реализация - разбиваем на слова и фильтруем
        words = query.split()
        # Оставляем только существительные и прилагательные (упрощенно)
        concepts = [word for word in words if len(word) > 3]
        return concepts[:5]  # Не более 5 концептов