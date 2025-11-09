from typing import List, Dict
import numpy as np
from loguru import logger
from collections import Counter


class ExpertAnalyzerService:
    """Сервис анализа экспертов и кафедр"""
    
    def __init__(self, bert_model):
        self.bert_model = bert_model
    
    def analyze_experts_by_topic(self, topic: str, authors: List[Dict]) -> List[Dict]:
        """Анализ экспертов по теме"""
        logger.info(f"Анализ экспертов по теме: {topic}")
        
        topic_vector = self.bert_model.encode_text(topic)
        experts = []
        
        for author in authors:
            try:
                expertise_score = self._calculate_expertise_score(author, topic, topic_vector)
                
                if expertise_score > 0.3:  # Порог экспертизы
                    experts.append({
                        "author_id": author["author_id"],
                        "expertise_score": float(expertise_score),
                        "topic_article_count": self._count_topic_articles(author, topic),
                        "total_citations": self._estimate_citations(author),
                        "last_activity_year": self._get_last_activity(author),
                        "related_topics": self._get_related_topics(author, topic)
                    })
                    
            except Exception as e:
                logger.error(f"Ошибка анализа автора {author.get('author_id')}: {e}")
                continue
        
        # Сортируем по оценке экспертизы
        experts.sort(key=lambda x: x["expertise_score"], reverse=True)
        return experts
    
    def analyze_departments_by_topic(self, topic: str, departments: List[Dict]) -> List[Dict]:
        """Анализ кафедр по теме"""
        logger.info(f"Анализ кафедр по теме: {topic}")
        
        departments_analysis = []
        
        for dept in departments:
            try:
                strength_score = self._calculate_department_strength(dept, topic)
                expert_count = self._count_experts_in_department(dept, topic)
                total_articles = self._count_department_articles(dept, topic)
                
                if strength_score > 0.2:  # Порог значимости
                    departments_analysis.append({
                        "organization_id": dept["organization_id"],
                        "strength_score": strength_score,
                        "expert_count": expert_count,
                        "total_articles": total_articles,
                        "key_author_ids": self._get_key_authors(dept, topic)
                    })
                    
            except Exception as e:
                logger.error(f"Ошибка анализа кафедры {dept.get('organization_id')}: {e}")
                continue
        
        # Сортируем по силе тематики
        departments_analysis.sort(key=lambda x: x["strength_score"], reverse=True)
        return departments_analysis
    
    def _calculate_expertise_score(self, author: Dict, topic: str, topic_vector: np.ndarray) -> float:
        """Вычисление оценки экспертизы автора"""
        # Анализируем темы статей автора
        article_topics = author.get("article_topics", [])
        
        if not article_topics:
            return 0.0
        
        # Вычисляем сходство тем автора с целевой темой
        topic_similarities = []
        for author_topic in article_topics:
            author_topic_vector = self.bert_model.encode_text(author_topic)
            similarity = self.bert_model._cosine_similarity(topic_vector, author_topic_vector)
            topic_similarities.append(similarity)
        
        # Усредненное сходство + бонус за количество статей
        avg_similarity = np.mean(topic_similarities) if topic_similarities else 0.0
        article_count_bonus = min(len(article_topics) * 0.1, 0.3)  # Максимум +0.3 за много статей
        
        return min(avg_similarity + article_count_bonus, 1.0)
    
    def _count_topic_articles(self, author: Dict, topic: str) -> int:
        """Подсчет статей автора по теме"""
        article_topics = author.get("article_topics", [])
        topic_vector = self.bert_model.encode_text(topic)
        
        count = 0
        for author_topic in article_topics:
            author_topic_vector = self.bert_model.encode_text(author_topic)
            similarity = self.bert_model._cosine_similarity(topic_vector, author_topic_vector)
            if similarity > 0.5:
                count += 1
        
        return count
    
    def _estimate_citations(self, author: Dict) -> int:
        """Оценка цитирований (упрощенно)"""
        article_count = len(author.get("article_ids", []))
        # Упрощенная формула: в среднем 20 цитирований на статью
        return article_count * 20
    
    def _get_last_activity(self, author: Dict) -> int:
        """Получение года последней активности"""
        return 2023  # Заглушка
    
    def _get_related_topics(self, author: Dict, main_topic: str) -> List[str]:
        """Получение смежных тем автора"""
        article_topics = author.get("article_topics", [])
        # Исключаем основную тему и берем самые частые
        other_topics = [topic for topic in article_topics if topic != main_topic]
        topic_counts = Counter(other_topics)
        return [topic for topic, _ in topic_counts.most_common(3)]
    
    def _calculate_department_strength(self, department: Dict, topic: str) -> float:
        """Вычисление силы кафедры в теме"""
        all_topics = department.get("article_topics", [])
        topic_count = sum(1 for dept_topic in all_topics 
                         if self.bert_model._cosine_similarity(
                             self.bert_model.encode_text(dept_topic),
                             self.bert_model.encode_text(topic)
                         ) > 0.5)
        
        total_topics = len(all_topics)
        if total_topics == 0:
            return 0.0
        
        # Доля статей по теме + бонус за абсолютное количество
        topic_ratio = topic_count / total_topics
        count_bonus = min(topic_count * 0.05, 0.3)  # Максимум +0.3
        
        return min(topic_ratio + count_bonus, 1.0)
    
    def _count_experts_in_department(self, department: Dict, topic: str) -> int:
        """Подсчет экспертов в кафедре"""
        author_ids = department.get("author_ids", [])
        return min(len(author_ids) // 2, 10)  # Заглушка
    
    def _count_department_articles(self, department: Dict, topic: str) -> int:
        """Подсчет статей кафедры по теме"""
        all_topics = department.get("article_topics", [])
        return sum(1 for dept_topic in all_topics 
                  if self.bert_model._cosine_similarity(
                      self.bert_model.encode_text(dept_topic),
                      self.bert_model.encode_text(topic)
                  ) > 0.5)
    
    def _get_key_authors(self, department: Dict, topic: str) -> List[str]:
        """Получение ключевых авторов кафедры"""
        author_ids = department.get("author_ids", [])
        return author_ids[:3]  # Берем первых трех авторов