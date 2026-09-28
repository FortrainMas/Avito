"""Recall поиска по эмбеддингам на валидации: косинус / максимум по запросу + alpha * log-доля выборов по локациям."""

import numpy as np
import polars as pl

from avito.metrics.recall import recalls_at_k
from avito.retrieval.location import LocationPrior
from avito.retrieval.search import log_prior_matrix, top_k

ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.1]
DEPTH = 1000


def evaluate_dense(
    queries: pl.DataFrame,
    query_embeddings: np.ndarray,
    documents: pl.DataFrame,
    document_embeddings: np.ndarray,
    location_prior: LocationPrior,
    alphas: list[float] = ALPHAS,
    batch_size: int = 256,
) -> list[dict]:
    item_ids = documents["item_id"].to_numpy()
    matrix = document_embeddings.astype(np.float32).T
    depth = min(DEPTH, len(item_ids))

    tops = {alpha: [] for alpha in alphas}
    for start in range(0, queries.height, batch_size):
        batch = queries.slice(start, batch_size)
        scores = query_embeddings[start:start + batch.height].astype(np.float32) @ matrix
        scores /= np.maximum(scores.max(axis=1, keepdims=True), 1e-9)
        log_prior = log_prior_matrix(batch, documents["item_location_id"], location_prior)
        for alpha in alphas:
            tops[alpha].append(top_k(scores + alpha * log_prior, depth))

    relevant = queries["relevant_item_ids"].to_list()
    results = []
    for alpha in alphas:
        predicted = item_ids[np.vstack(tops[alpha])].tolist()
        results.append({
            "alpha": alpha,
            "recall@50": float(recalls_at_k(predicted, relevant, 50).mean()),
            "recall@200": float(recalls_at_k(predicted, relevant, 200).mean()),
            "recall@1000": float(recalls_at_k(predicted, relevant, 1000).mean()),
        })
    return results
