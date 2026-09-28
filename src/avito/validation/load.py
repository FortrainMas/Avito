"""Загрузка валидации, запросов реранкера и частей train."""

import polars as pl

from avito import paths


def load_val_queries() -> pl.DataFrame:
    return pl.read_parquet(paths.VAL_QUERIES)


def load_train_fold(columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(paths.TRAIN_FOLD, columns=columns)


def load_ranker_queries() -> pl.DataFrame:
    return pl.read_parquet(paths.RANKER_QUERIES)


def load_retriever_fold(columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(paths.RETRIEVER_FOLD, columns=columns)
