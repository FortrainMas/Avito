"""Сырые parquet → таблица объявлений (корпус и объявления из train), пары train и тестовые запросы."""

import polars as pl

from avito.columns import ITEM_COLUMNS, QUERY_COLUMNS


def clean_items(df: pl.DataFrame) -> pl.DataFrame:
    """Цена -1, рейтинг 0 при нуле отзывов и координаты (0, 0) означают «нет значения» — заменяем на null."""
    df = df.select(ITEM_COLUMNS)
    df = df.with_columns(
        pl.col("item_title_raw").fill_null(""),
        pl.col("item_description_raw").fill_null(""),
        pl.col("item_infm_params_text").fill_null(""),
        pl.col("item_price").cast(pl.Float64),
        pl.col("item_latitude").cast(pl.Float64),
        pl.col("item_longitude").cast(pl.Float64),
    )
    df = df.with_columns(
        pl.when(pl.col("item_price") >= 0).then(pl.col("item_price")).otherwise(None).alias("item_price"),
        pl.when(pl.col("item_rating") > 0).then(pl.col("item_rating")).otherwise(None).alias("item_rating"),
        pl.when(pl.col("item_latitude") != 0).then(pl.col("item_latitude")).otherwise(None).alias("item_latitude"),
        pl.when(pl.col("item_longitude") != 0).then(pl.col("item_longitude")).otherwise(None).alias("item_longitude"),
    )
    return df


def build_items(train_raw: pl.DataFrame, bench_items_raw: pl.DataFrame) -> pl.DataFrame:
    corpus = clean_items(bench_items_raw)
    corpus = corpus.with_columns(pl.lit(True).alias("in_corpus"))

    train_items = clean_items(train_raw)
    train_items = train_items.unique(subset="item_id", keep="first", maintain_order=True)
    train_only = train_items.join(corpus.select("item_id"), on="item_id", how="anti")
    train_only = train_only.with_columns(pl.lit(False).alias("in_corpus"))

    return pl.concat([corpus, train_only])


def build_train(train_raw: pl.DataFrame) -> pl.DataFrame:
    df = train_raw.select(QUERY_COLUMNS + ["item_id"])
    df = df.with_row_index("row_id")
    return df


def build_bench_queries(bench_queries_raw: pl.DataFrame) -> pl.DataFrame:
    return bench_queries_raw.select(["query_id"] + QUERY_COLUMNS)


def check(items: pl.DataFrame, train: pl.DataFrame, bench_queries: pl.DataFrame) -> None:
    assert items["item_id"].is_unique().all()
    assert items["item_id"].str.contains(r"^[0-9a-f]{16}$").all()
    assert train.join(items, on="item_id", how="anti").height == 0

    assert bench_queries["query_id"].is_unique().all()
    assert (bench_queries["query_id"].str.len_chars() == 16).all()

    for column in QUERY_COLUMNS:
        assert train[column].null_count() == 0
        assert bench_queries[column].null_count() == 0
