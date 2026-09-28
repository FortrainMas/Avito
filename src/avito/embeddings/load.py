"""Загрузка сохранённых векторов в порядке нужных объявлений или запросов (без torch)."""

import numpy as np
import polars as pl

from avito import paths

# эмбеддер итогового решения
DENSE_MODEL = "e5-large-instruct-finetuned"


def aligned_vectors(model: str, kind: str, id_column: str, ids: pl.Series) -> np.ndarray:
    """Строки сохранённой матрицы векторов в порядке ids; падает, если какого-то id нет."""
    directory = paths.EMBEDDINGS_OUTPUT_DIR / model
    stored = pl.read_parquet(directory / f"{kind}.parquet", columns=[id_column]).with_row_index("position")
    order = pl.DataFrame({id_column: ids}).join(stored, on=id_column, how="left", maintain_order="left")["position"]
    assert order.null_count() == 0
    return np.load(directory / f"{kind}.npy")[order.to_numpy()]


def load_document_vectors(model: str, item_ids: pl.Series) -> np.ndarray:
    return aligned_vectors(model, "documents", "item_id", item_ids)


def load_query_vectors(model: str, query_ids: pl.Series) -> np.ndarray:
    return aligned_vectors(model, "queries", "query_id", query_ids)
