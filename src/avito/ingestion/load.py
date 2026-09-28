"""Загрузка таблиц этапа ingestion."""

import polars as pl

from avito import paths


def load_items(columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(paths.INGESTED_ITEMS, columns=columns)


def load_corpus(columns: list[str] | None = None) -> pl.DataFrame:
    df = pl.scan_parquet(paths.INGESTED_ITEMS).filter(pl.col("in_corpus"))
    if columns is not None:
        df = df.select(columns)
    return df.collect()


def load_train(columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(paths.INGESTED_TRAIN, columns=columns)


def load_bench_queries() -> pl.DataFrame:
    return pl.read_parquet(paths.INGESTED_BENCH_QUERIES)
