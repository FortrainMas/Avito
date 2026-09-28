"""Пул кандидатов: объединение top-500 BM25 и top-200 эмбеддингов со скорами обоих каналов."""

import numpy as np
import polars as pl

from avito.retrieval.bm25 import BM25
from avito.retrieval.location import LocationPrior
from avito.retrieval.search import build_documents, log_prior_matrix, score_batch, top_k


def build_indexes(corpus: pl.DataFrame) -> dict[str, BM25]:
    """Индекс по всему документу (заголовок ×3 + параметры + описание) и по каждому полю отдельно."""
    return {
        "full": BM25().fit(build_documents(corpus)),
        "title": BM25().fit(build_documents(corpus, title_repeats=1, params=False, description=False)),
        "params": BM25().fit(build_documents(corpus, title_repeats=0, params=True, description=False)),
        "description": BM25().fit(build_documents(corpus, title_repeats=0, params=False, description=True)),
    }


def rank_matrix(top: np.ndarray, document_count: int, depth: int) -> np.ndarray:
    """Место каждого документа в top-k запроса; документы вне top-k получают depth."""
    ranks = np.full((top.shape[0], document_count), depth, dtype=np.int32)
    rows = np.repeat(np.arange(top.shape[0]), top.shape[1])
    ranks[rows, top.ravel()] = np.tile(np.arange(top.shape[1]), top.shape[0])
    return ranks


def build_candidates(
    queries: pl.DataFrame,
    indexes: dict[str, BM25],
    item_location_ids: pl.Series,
    location_prior: LocationPrior,
    alpha: float = 0.1,
    filter_weight: float = 0.3,
    pool_size: int = 500,
    query_vectors: np.ndarray | None = None,
    document_vectors: np.ndarray | None = None,
    dense_alpha: float = 0.05,
    dense_pool_size: int = 200,
    batch_size: int = 128,
) -> pl.DataFrame:
    """Для батча запросов считает скоры каналов по всему корпусу, отбирает объединение их top-k
    и возвращает по строке на пару «запрос — кандидат» со скорами и местами в каналах."""
    document_count = len(item_location_ids)
    if document_vectors is not None:
        document_matrix = document_vectors.astype(np.float32).T

    parts = []
    for start in range(0, queries.height, batch_size):
        batch = queries.slice(start, batch_size)
        query_stems = batch["query_stems"].to_list()
        scores = score_batch(batch, indexes["full"], item_location_ids, location_prior, alpha=alpha, filter_weight=filter_weight)
        ranks = rank_matrix(top_k(scores, pool_size), document_count, pool_size)
        selected = ranks < pool_size

        dense_matrices = {}
        if query_vectors is not None:
            dense = query_vectors[start:start + batch.height].astype(np.float32) @ document_matrix
            dense_relative = dense / np.maximum(dense.max(axis=1, keepdims=True), 1e-9)
            dense_combined = dense_relative + dense_alpha * log_prior_matrix(batch, item_location_ids, location_prior)
            dense_ranks = rank_matrix(top_k(dense_combined, dense_pool_size), document_count, dense_pool_size)
            selected |= dense_ranks < dense_pool_size
            dense_matrices = {
                "dense_score": dense,
                "dense_relative": dense_relative,
                "dense_combined": dense_combined,
                "dense_rank": dense_ranks,
            }

        rows, columns = np.nonzero(selected)
        features = {
            "query_id": batch["query_id"].to_numpy()[rows],
            "item_index": columns,
            "retrieval_rank": ranks[rows, columns],
            "retrieval_score": scores[rows, columns],
            "bm25_filters": indexes["full"].score(batch["filter_stems"].to_list())[rows, columns],
        }
        for name, index in indexes.items():
            features[f"bm25_{name}"] = index.score(query_stems)[rows, columns]
            features[f"coverage_{name}"] = index.coverage(query_stems)[rows, columns]
        for name, matrix in dense_matrices.items():
            features[name] = matrix[rows, columns]
        parts.append(pl.DataFrame(features))
    return pl.concat(parts)
