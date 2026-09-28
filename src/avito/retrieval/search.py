"""BM25-поиск с учётом долей выборов по локациям — бейзлайн (notebooks/retrieval/baseline.ipynb)."""

import numpy as np
import polars as pl

from avito.retrieval.bm25 import BM25
from avito.retrieval.location import LocationPrior

# сглаживание под логарифмом: локации, где ни разу не выбирали, получают большой штраф, а не -inf
EPSILON = 3e-4


def build_documents(items: pl.DataFrame, title_repeats: int = 3, params: bool = True, description: bool = True) -> list[str]:
    """Документ для BM25 — стемы заголовка (повторённого title_repeats раз, чтобы весил больше), параметров и описания."""
    documents = []
    for row in items.iter_rows(named=True):
        parts = [row["title_stems"]] * title_repeats
        if params:
            parts.append(row["params_stems"])
        if description:
            parts.append(row["description_stems"])
        documents.append(" ".join(parts))
    return documents


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы k лучших документов в каждой строке, по убыванию скора."""
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(-top_scores, axis=1)
    return np.take_along_axis(top, order, axis=1)


def log_prior_matrix(batch: pl.DataFrame, item_location_ids: pl.Series, location_prior: LocationPrior) -> np.ndarray:
    """log(доля выборов локации объявления при локации поиска + EPSILON) для каждой пары «запрос — объявление»."""
    matrix = np.zeros((batch.height, len(item_location_ids)), dtype=np.float32)
    for row, search_location_id in enumerate(batch["search_location_id"].to_list()):
        matrix[row] = np.log(location_prior.probabilities(search_location_id, item_location_ids) + EPSILON)
    return matrix


def score_batch(
    batch: pl.DataFrame,
    bm25: BM25,
    item_location_ids: pl.Series,
    location_prior: LocationPrior | None = None,
    alpha: float = 0.1,
    same_location_bonus: float = 0.0,
    filter_weight: float = 0.0,
) -> np.ndarray:
    """Скор по всему корпусу: (BM25 запроса + filter_weight * BM25 фильтров) / максимум по запросу
    + alpha * log-доля выборов по локациям."""
    scores = bm25.score(batch["query_stems"].to_list())
    if filter_weight > 0:
        scores += filter_weight * bm25.score(batch["filter_stems"].to_list())
    scores /= np.maximum(scores.max(axis=1, keepdims=True), 1e-9)

    if location_prior is not None:
        scores += alpha * log_prior_matrix(batch, item_location_ids, location_prior)
    if same_location_bonus > 0:
        item_locations = item_location_ids.to_numpy()
        for row, search_location_id in enumerate(batch["search_location_id"].to_list()):
            scores[row] += same_location_bonus * (item_locations == search_location_id)
    return scores


def retrieve(
    queries: pl.DataFrame,
    bm25: BM25,
    item_location_ids: pl.Series,
    location_prior: LocationPrior | None = None,
    alpha: float = 0.1,
    same_location_bonus: float = 0.0,
    filter_weight: float = 0.0,
    k: int = 50,
    batch_size: int = 256,
) -> np.ndarray:
    results = []
    for batch in queries.iter_slices(batch_size):
        scores = score_batch(batch, bm25, item_location_ids, location_prior, alpha, same_location_bonus, filter_weight)
        results.append(top_k(scores, k))
    return np.vstack(results)
