"""Recall@k по запросам и бутстреп-интервалы."""

import numpy as np
import polars as pl


def recall_at_k(predicted: list[str], relevant: list[str], k: int = 50) -> float:
    hits = set(predicted[:k]) & set(relevant)
    return len(hits) / len(relevant)


def recalls_at_k(predicted: list[list[str]], relevant: list[list[str]], k: int = 50) -> np.ndarray:
    return np.array([recall_at_k(p, r, k) for p, r in zip(predicted, relevant, strict=True)])


def mean_recall_at_k(predicted: list[list[str]], relevant: list[list[str]], k: int = 50) -> float:
    return float(recalls_at_k(predicted, relevant, k).mean())


def query_recalls(predictions: pl.DataFrame, queries: pl.DataFrame, k: int = 50) -> np.ndarray:
    joined = queries.select("query_id", "relevant_item_ids").join(predictions, on="query_id", how="left", maintain_order="left")
    predicted = [items if items is not None else [] for items in joined["item_id"].to_list()]
    return recalls_at_k(predicted, joined["relevant_item_ids"].to_list(), k)


def bootstrap_interval(values: np.ndarray, samples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% интервал среднего. Для сравнения двух методов передаём разницу их recall по запросам."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values))).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
