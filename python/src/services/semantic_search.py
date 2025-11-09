import numpy as np
from typing import List, Dict
from loguru import logger


class SemanticSearchService:
    """Сервис семантического поиска"""
    
    def __init__(self, bert_model):
        self.bert_model = bert_model
    
    def search_articles(self, query_vector: np.ndarray, articles: List[Dict], max_results: int = 10) -> List[Dict]:
        """Семантический поиск статей по вектору запроса"""
        if not articles:
            return []
        
        logger.info(f"Поиск по {len(articles)} статьям")
        
        # Вычисляем сходство для каждой статьи
        results = []
        for article in articles:
            try:
                # Среднее сходство по заголовку и аннотации
                title_similarity = self._vector_similarity(
                    query_vector, 
                    np.frombuffer(article['title_embedding'], dtype=np.float32)
                )
                abstract_similarity = self._vector_similarity(
                    query_vector,
                    np.frombuffer(article['abstract_embedding'], dtype=np.float32)
                )
                
                # Общая релевантность (взвешенная сумма)
                relevance_score = 0.6 * title_similarity + 0.4 * abstract_similarity
                
                results.append({
                    "document_id": article["document_id"],
                    "relevance_score": float(relevance_score),
                    "matched_concepts": self._extract_matched_concepts(article, relevance_score)
                })
                
            except Exception as e:
                logger.error(f"Ошибка обработки статьи {article.get('document_id')}: {e}")
                continue
        
        # Сортируем и ограничиваем результаты
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]
    
    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисление косинусного сходства между векторами"""
        dot_product = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm if norm > 0 else 0.0
    
    def _extract_matched_concepts(self, article: Dict, relevance_score: float) -> List[str]:
        """Извлечение совпавших концептов на основе релевантности"""
        concepts = []
        
        if relevance_score > 0.8:
            concepts.append("высокая релевантность")
        elif relevance_score > 0.6:
            concepts.append("средняя релевантность")
        else:
            concepts.append("низкая релевантность")
            
        return concepts