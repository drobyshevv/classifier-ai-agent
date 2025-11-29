import base64
import numpy as np
from typing import List, Dict
from loguru import logger


class SemanticSearchService:
    """Сервис семантического поиска"""
    
    def __init__(self, bert_model):
        self.bert_model = bert_model
    
    def search_articles(self, query_vector: np.ndarray, articles: List[Dict], max_results: int = 10):
        results = []

        for article in articles:
            try:
                title_vec = np.frombuffer(base64.b64decode(article["title_embedding"]), dtype=np.float32)
                abstract_vec = np.frombuffer(base64.b64decode(article["abstract_embedding"]), dtype=np.float32)

                title_sim = self._vector_similarity(query_vector, title_vec)
                abstract_sim = self._vector_similarity(query_vector, abstract_vec)

                relevance = 0.6 * title_sim + 0.4 * abstract_sim

                results.append({
                    "document_id": article["document_id"],
                    "relevance_score": float(relevance)
                })

            except Exception as e:
                logger.error(f"Error processing {article.get('document_id')}: {e}")

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