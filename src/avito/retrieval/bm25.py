"""BM25 на разреженных матрицах: скоры всего корпуса для батча запросов одним умножением."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    """Классическая формула с множителем (k1 + 1) и idf = log(1 + (N - df + 0.5) / (df + 0.5)).
    Документы и запросы — строки уже простемленных токенов через пробел."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, documents: list[str]) -> "BM25":
        self.vectorizer = CountVectorizer(analyzer=str.split)
        counts = self.vectorizer.fit_transform(documents).tocsr().astype(np.float32)

        document_count = counts.shape[0]
        document_lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = document_lengths.mean()
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))

        term_frequency = counts.data
        lengths = np.repeat(document_lengths, np.diff(counts.indptr))
        normalizer = term_frequency + self.k1 * (1 - self.b + self.b * lengths / average_length)
        counts.data = (term_frequency * (self.k1 + 1) / normalizer * idf[counts.indices]).astype(np.float32)

        self.weights = counts.T.tocsr()
        self.presence = self.weights.copy()
        self.presence.data[:] = 1.0
        return self

    def query_terms(self, queries: list[str]):
        terms = self.vectorizer.transform(queries).astype(np.float32)
        terms.data[:] = 1.0
        return terms

    def score(self, queries: list[str]) -> np.ndarray:
        return (self.query_terms(queries) @ self.weights).toarray()

    def coverage(self, queries: list[str]) -> np.ndarray:
        """Доля уникальных слов запроса, которые есть в документе."""
        matched = (self.query_terms(queries) @ self.presence).toarray()
        lengths = np.array([len(set(query.split())) for query in queries], dtype=np.float32)
        return matched / np.maximum(lengths, 1)[:, None]
