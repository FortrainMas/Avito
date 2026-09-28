"""Пул кандидатов и признаки к нему для набора запросов."""

import numpy as np
import polars as pl

from avito.ranking.candidates import build_candidates
from avito.ranking.features import build_features
from avito.retrieval.bm25 import BM25
from avito.retrieval.location import LocationPrior


def build_dataset(
    queries: pl.DataFrame,
    items: pl.DataFrame,
    indexes: dict[str, BM25],
    location_prior: LocationPrior,
    centroids: pl.DataFrame,
    pool_size: int = 500,
    query_vectors: np.ndarray | None = None,
    document_vectors: np.ndarray | None = None,
) -> pl.DataFrame:
    candidates = build_candidates(
        queries,
        indexes,
        items["item_location_id"],
        location_prior,
        pool_size=pool_size,
        query_vectors=query_vectors,
        document_vectors=document_vectors,
    )
    return build_features(candidates, queries, items, location_prior.table, centroids)
