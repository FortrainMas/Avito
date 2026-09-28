"""Кодирование текстов в нормированные векторы (fp16, чтобы корпус занимал меньше места)."""

import numpy as np
from sentence_transformers import SentenceTransformer


def encode(model: SentenceTransformer, texts: list[str], prefix: str, batch_size: int = 128) -> np.ndarray:
    embeddings = model.encode(
        [prefix + text for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float16)
