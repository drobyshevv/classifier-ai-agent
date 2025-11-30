import base64
import numpy as np
from typing import List, Dict
from loguru import logger


class SemanticSearchService:
    """Сервис семантического поиска"""
    
    def __init__(self, bert_model):
        self.bert_model = bert_model
    
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def search_articles(self, query_vector, articles, max_results=10):
        # normalize query
        query_vector = self._normalize(query_vector.astype(np.float32))

        results = []

        for article in articles:
            try:
                title_vec = np.frombuffer(
                    base64.b64decode(article["title_embedding"]),
                    dtype=np.float32
                )
                abstract_vec = np.frombuffer(
                    base64.b64decode(article["abstract_embedding"]),
                    dtype=np.float32
                )

                # CRITICAL FIX
                title_vec = self._normalize(title_vec)
                abstract_vec = self._normalize(abstract_vec)

                title_sim = float(np.dot(query_vector, title_vec))
                abstract_sim = float(np.dot(query_vector, abstract_vec))

                relevance = 0.6 * title_sim + 0.4 * abstract_sim

                results.append({
                    "document_id": article["document_id"],
                    "relevance_score": relevance
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