"""Разбиение train на валидацию, запросы реранкера и retriever_fold (notebooks/validation/split.ipynb)."""

import polars as pl

from avito.columns import QUERY_COLUMNS

# фильтра «Автосервис, аренда» нет среди тестовых запросов, поэтому такие запросы не берём
AUTO_FILTER = "Автосервис, аренда"
# корзины популярности текста: сколько запросов с тем же текстом есть в train
POPULARITY_BREAKS = [1, 3, 10, 50]


def popularity_bin(text_tuples: pl.Expr) -> pl.Expr:
    return text_tuples.cut(POPULARITY_BREAKS).cast(pl.String).alias("popularity_bin")


def build_tuples(train: pl.DataFrame, corpus_item_ids: pl.DataFrame) -> pl.DataFrame:
    """Запрос — набор всех полей поиска; ответы — выбранные по нему объявления, которые есть в корпусе.

    seen_text: тот же текст встречается в train и в других запросах (с другой локацией или фильтрами).
    """
    corpus = corpus_item_ids.select("item_id").with_columns(pl.lit(True).alias("in_corpus"))
    rows = train.join(corpus, on="item_id", how="left")
    rows = rows.with_columns(pl.col("in_corpus").fill_null(False))

    tuples = rows.group_by(QUERY_COLUMNS + ["query_stems", "filter_stems"]).agg(
        pl.col("row_id"),
        pl.col("item_id").filter(pl.col("in_corpus")).unique().sort().alias("relevant_item_ids"),
        pl.col("in_corpus").all().alias("all_in_corpus"),
        pl.col("in_corpus").any().alias("any_in_corpus"),
    )
    tuples = tuples.sort(QUERY_COLUMNS)

    text_tuples = tuples.group_by("search_query").len("text_tuples")
    tuples = tuples.join(text_tuples, on="search_query", maintain_order="left")
    tuples = tuples.with_columns((pl.col("text_tuples") > 1).alias("seen_text"))
    return tuples


def bench_seen_profile(bench_queries: pl.DataFrame, tuples: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    """Доля тестовых текстов, встречающихся в train, и их распределение по корзинам популярности. Ответы теста не нужны."""
    text_tuples = tuples.select("search_query", "text_tuples").unique()
    seen = bench_queries.join(text_tuples, on="search_query")
    seen = seen.with_columns(popularity_bin(pl.col("text_tuples")))

    bins = seen.group_by("popularity_bin").len("bench_texts").sort("popularity_bin")
    bins = bins.with_columns((pl.col("bench_texts") / seen.height).alias("share"))
    return seen.height / bench_queries.height, bins


def select_val_tuples(tuples: pl.DataFrame, bench_queries: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Валидация как тест: один запрос на текст, все ответы в корпусе.

    Берём все новые тексты и добираем знакомые так, чтобы их доля и популярность по корзинам были как в тесте.
    """
    candidates = tuples.filter(pl.col("all_in_corpus"))
    candidates = candidates.filter(~pl.col("search_infm_params_text").str.contains(AUTO_FILTER, literal=True))
    candidates = candidates.sample(fraction=1.0, shuffle=True, seed=seed)
    candidates = candidates.unique(subset="search_query", keep="first", maintain_order=True)
    candidates = candidates.with_columns((pl.col("text_tuples") - 1).alias("text_tuples_in_fold"))
    candidates = candidates.with_columns(popularity_bin(pl.col("text_tuples_in_fold")))

    unseen = candidates.filter(~pl.col("seen_text"))
    seen = candidates.filter(pl.col("seen_text"))

    seen_share, bins = bench_seen_profile(bench_queries, tuples)
    seen_total = unseen.height * seen_share / (1 - seen_share)

    parts = [unseen]
    for popularity, share in bins.select("popularity_bin", "share").iter_rows():
        pool = seen.filter(pl.col("popularity_bin") == popularity)
        count = min(round(seen_total * share), pool.height)
        parts.append(pool.sample(count, seed=seed))

    val = pl.concat(parts)
    val = val.with_row_index("query_id")
    val = val.with_columns(pl.col("query_id").cast(pl.String))
    return val


def split(train: pl.DataFrame, corpus_item_ids: pl.DataFrame, bench_queries: pl.DataFrame, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Валидационные запросы и train без их строк."""
    tuples = build_tuples(train, corpus_item_ids)
    val = select_val_tuples(tuples, bench_queries, seed)

    val_rows = val.select(pl.col("row_id").explode())
    train_fold = train.join(val_rows, on="row_id", how="anti", maintain_order="left")

    columns = ["query_id"] + QUERY_COLUMNS + ["query_stems", "filter_stems", "relevant_item_ids", "seen_text", "text_tuples_in_fold"]
    return val.select(columns), train_fold


def select_ranker_tuples(tuples: pl.DataFrame, val: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Запросы для обучения реранкера: один на текст, текст не из валидации, хотя бы один ответ в корпусе."""
    candidates = tuples.filter(pl.col("any_in_corpus"))
    candidates = candidates.filter(~pl.col("search_infm_params_text").str.contains(AUTO_FILTER, literal=True))
    candidates = candidates.join(val.select("search_query"), on="search_query", how="anti", maintain_order="left")
    candidates = candidates.sample(fraction=1.0, shuffle=True, seed=seed)
    candidates = candidates.unique(subset="search_query", keep="first", maintain_order=True)
    candidates = candidates.with_columns((pl.col("text_tuples") - 1).alias("text_tuples_in_fold"))

    ranker = candidates.with_row_index("query_id")
    ranker = ranker.with_columns(pl.format("r{}", pl.col("query_id")).alias("query_id"))
    return ranker


def split_ranker(train_fold: pl.DataFrame, corpus_item_ids: pl.DataFrame, val: pl.DataFrame, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Запросы реранкера и retriever_fold — train без строк валидации и реранкера.

    По retriever_fold считаются доли выборов по локациям и дообучается эмбеддер, поэтому они не видели ответов реранкера.
    """
    tuples = build_tuples(train_fold, corpus_item_ids)
    ranker = select_ranker_tuples(tuples, val, seed)

    ranker_rows = ranker.select(pl.col("row_id").explode())
    retriever_fold = train_fold.join(ranker_rows, on="row_id", how="anti", maintain_order="left")

    columns = ["query_id"] + QUERY_COLUMNS + ["query_stems", "filter_stems", "relevant_item_ids", "seen_text", "text_tuples_in_fold"]
    return ranker.select(columns), retriever_fold


def check_ranker(ranker: pl.DataFrame, retriever_fold: pl.DataFrame, val: pl.DataFrame, corpus_item_ids: pl.DataFrame) -> None:
    """Тексты реранкера не пересекаются с валидацией, ни один запрос реранкера и валидации не остался в retriever_fold."""
    assert ranker["search_query"].is_unique().all()
    assert ranker.join(val, on="search_query", how="semi").height == 0
    assert (ranker["relevant_item_ids"].list.len() > 0).all()

    relevant = ranker.select(pl.col("relevant_item_ids").explode().alias("item_id"))
    assert relevant.join(corpus_item_ids, on="item_id", how="anti").height == 0

    fold_tuples = retriever_fold.select(QUERY_COLUMNS).unique()
    assert ranker.join(fold_tuples, on=QUERY_COLUMNS, how="semi").height == 0
    assert val.join(fold_tuples, on=QUERY_COLUMNS, how="semi").height == 0


def check(val: pl.DataFrame, train_fold: pl.DataFrame, corpus_item_ids: pl.DataFrame) -> None:
    """Тексты валидации уникальны, ответы в корпусе, seen_text совпадает с train fold, запросы валидации удалены из train."""
    assert val["search_query"].is_unique().all()

    relevant = val.select(pl.col("relevant_item_ids").explode().alias("item_id"))
    assert relevant.join(corpus_item_ids, on="item_id", how="anti").height == 0

    fold_texts = train_fold.select("search_query").unique().with_columns(pl.lit(True).alias("text_in_fold"))
    val_texts = val.join(fold_texts, on="search_query", how="left")
    val_texts = val_texts.with_columns(pl.col("text_in_fold").fill_null(False))
    assert (val_texts["text_in_fold"] == val_texts["seen_text"]).all()

    fold_tuples = train_fold.select(QUERY_COLUMNS).unique()
    assert val.join(fold_tuples, on=QUERY_COLUMNS, how="semi").height == 0
