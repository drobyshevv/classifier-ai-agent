import numpy as np
import base64


def vector_to_base64(vector: np.ndarray) -> str:
    """Конвертирует вектор numpy в base64 строку"""
    return base64.b64encode(vector.tobytes()).decode('utf-8')


def base64_to_vector(base64_str: str, dtype=np.float32) -> np.ndarray:
    """Конвертирует base64 строку обратно в numpy вектор"""
    bytes_data = base64.b64decode(base64_str)
    return np.frombuffer(bytes_data, dtype=dtype)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Нормализует вектор к единичной длине"""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector