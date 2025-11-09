import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from loguru import logger
import typing as tp


class RuBERTModel:
    """Класс для работы с ruBERT моделью для эмбеддингов и анализа текстов"""
    
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = None
        self.topic_model = None
        self.tokenizer = None
        self.device = config['models']['device']
        self._load_models()
    
    def _load_models(self):
        """Загрузка моделей"""
        try:
            logger.info("Загрузка embedding модели...")
            self.embedding_model = SentenceTransformer(
                self.config['models']['bert_model'],
                device=self.device
            )
            
            logger.info("Загрузка topic модели...")
            self.topic_model = AutoModel.from_pretrained(
                self.config['models']['topic_model']
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['models']['topic_model']
            )
            
            logger.info("Модели успешно загружены")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Создание эмбеддинга для текста"""
        if not text or not text.strip():
            return np.zeros(self.config['embeddings']['dimension'])
        
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=self.config['embeddings']['normalize']
        )
        return embedding
    
    def encode_batch(self, texts: tp.List[str]) -> np.ndarray:
        """Создание эмбеддингов для батча текстов"""
        if not texts:
            return np.array([])
        
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=self.config['embeddings']['normalize'],
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings
    
    def analyze_topics(self, text: str, predefined_topics: tp.List[str] = None) -> tp.List[tp.Dict]:
        """Анализ тематик текста"""
        if predefined_topics is None:
            predefined_topics = self.config['topics']['predefined_topics']
        
        # Сравниваем текст с каждой предопределенной темой
        text_embedding = self.encode_text(text)
        topic_embeddings = self.encode_batch(predefined_topics)
        
        # Вычисляем косинусное сходство
        similarities = self._cosine_similarity(text_embedding, topic_embeddings)
        
        # Формируем результаты
        topics = []
        for topic, similarity in zip(predefined_topics, similarities):
            if similarity > 0.3:  # Порог релевантности
                topic_type = "main" if similarity > 0.7 else "secondary"
                topics.append({
                    "topic_name": topic,
                    "confidence": float(similarity),
                    "topic_type": topic_type
                })
        
        # Сортируем по уверенности
        topics.sort(key=lambda x: x["confidence"], reverse=True)
        return topics[:5]  # Возвращаем топ-5 тем
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Вычисление косинусного сходства"""
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)
        
        dot_product = np.dot(vec2, vec1)
        norms = np.linalg.norm(vec2, axis=1) * np.linalg.norm(vec1)
        return dot_product / norms