"""Признаки пары «запрос — кандидат» для реранкера (список и абляции — notebooks/ranking/ranker.ipynb)."""

import polars as pl

from avito.retrieval.search import EPSILON

# по top-30 кандидатов канала оцениваем, объявления каких подкатегорий нужны запросу
INTENT_DEPTH = 30
# item_category_id категории «Услуги»
SERVICES_CATEGORY = 114

FEATURES = [
    "retrieval_rank",
    "retrieval_score",
    "bm25_full",
    "bm25_title",
    "bm25_params",
    "bm25_description",
    "bm25_filters",
    "coverage_full",
    "coverage_title",
    "coverage_params",
    "coverage_description",
    "bm25_full_relative",
    "query_max_bm25",
    "query_terms",
    "has_filters",
    "search_category",
    "log_prior",
    "same_location",
    "distance_km",
    "same_location_share",
    "microcat_share",
    "is_service",
    "log_price",
    "item_rating",
    "log_reviews",
    "title_length",
    "description_length",
    "phone_hidden",
    "message_forbidden",
]

# свойства объявления: на валидации помогают, но в итог не входят — ответы валидации смещены к популярным объявлениям
ITEM_FEATURES = [
    "is_service",
    "log_price",
    "item_rating",
    "log_reviews",
    "title_length",
    "description_length",
    "phone_hidden",
    "message_forbidden",
]

RANKING_FEATURES = [feature for feature in FEATURES if feature not in ITEM_FEATURES]

DENSE_FEATURES = [
    "dense_score",
    "dense_relative",
    "dense_combined",
    "dense_rank",
    "dense_microcat_share",
]


def item_table(corpus: pl.DataFrame) -> pl.DataFrame:
    items = corpus.with_row_index("item_index")
    return items.select(
        "item_index",
        "item_id",
        "item_location_id",
        "item_microcat_id",
        "item_latitude",
        "item_longitude",
        (pl.col("item_category_id") == SERVICES_CATEGORY).alias("is_service"),
        pl.col("item_price").log1p().alias("log_price"),
        "item_rating",
        pl.col("item_rating_reviews_count").log1p().alias("log_reviews"),
        pl.col("item_title_raw").str.len_chars().alias("title_length"),
        pl.col("item_description_raw").str.len_chars().alias("description_length"),
        pl.col("item_is_phone_hidden").alias("phone_hidden"),
        pl.col("item_is_message_forbidden").alias("message_forbidden"),
    )


def haversine_km(latitude_a: pl.Expr, longitude_a: pl.Expr, latitude_b: pl.Expr, longitude_b: pl.Expr) -> pl.Expr:
    latitude_a, longitude_a = latitude_a.radians(), longitude_a.radians()
    latitude_b, longitude_b = latitude_b.radians(), longitude_b.radians()
    a = ((latitude_b - latitude_a) / 2).sin() ** 2 + latitude_a.cos() * latitude_b.cos() * ((longitude_b - longitude_a) / 2).sin() ** 2
    return 2 * 6371 * a.sqrt().arcsin()


def add_query_features(candidates: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    queries = queries.select(
        "query_id",
        "search_location_id",
        "search_category",
        pl.col("query_stems").str.split(" ").list.unique().list.len().alias("query_terms"),
        (pl.col("filter_stems") != "").alias("has_filters"),
    )
    return candidates.join(queries, on="query_id", how="left", maintain_order="left")


def add_location_features(candidates: pl.DataFrame, prior_table: pl.DataFrame, centroids: pl.DataFrame) -> pl.DataFrame:
    """log-доля выборов локации объявления при этой локации поиска, та же ли локация, расстояние до центра локации поиска."""
    candidates = candidates.join(prior_table, on=["search_location_id", "item_location_id"], how="left", maintain_order="left")
    candidates = candidates.join(centroids, on="search_location_id", how="left", maintain_order="left")
    return candidates.with_columns(
        (pl.col("probability").fill_null(0.0) + EPSILON).log().alias("log_prior"),
        (pl.col("item_location_id") == pl.col("search_location_id")).alias("same_location"),
        haversine_km(
            pl.col("search_latitude"),
            pl.col("search_longitude"),
            pl.col("item_latitude"),
            pl.col("item_longitude"),
        ).alias("distance_km"),
    )


def add_pool_features(candidates: pl.DataFrame) -> pl.DataFrame:
    """Признаки по всему пулу запроса: максимум BM25, доля кандидатов в локации поиска и доля подкатегории
    кандидата среди top-30 BM25 — грубая оценка того, какую услугу ищут."""
    candidates = candidates.with_columns(
        pl.col("bm25_full").max().over("query_id").alias("query_max_bm25"),
        pl.col("same_location").mean().over("query_id").alias("same_location_share"),
    )
    candidates = candidates.with_columns(
        (pl.col("bm25_full") / pl.col("query_max_bm25").clip(lower_bound=1e-9)).alias("bm25_full_relative"),
    )

    intent = candidates.filter(pl.col("retrieval_rank") < INTENT_DEPTH)
    intent = intent.group_by("query_id", "item_microcat_id").len("microcat_count")
    candidates = candidates.join(intent, on=["query_id", "item_microcat_id"], how="left", maintain_order="left")
    return candidates.with_columns((pl.col("microcat_count").fill_null(0) / INTENT_DEPTH).alias("microcat_share"))


def add_dense_intent(candidates: pl.DataFrame) -> pl.DataFrame:
    """Доля подкатегории кандидата среди top-30 канала эмбеддингов."""
    intent = candidates.filter(pl.col("dense_rank") < INTENT_DEPTH)
    intent = intent.group_by("query_id", "item_microcat_id").len("dense_microcat_count")
    candidates = candidates.join(intent, on=["query_id", "item_microcat_id"], how="left", maintain_order="left")
    return candidates.with_columns((pl.col("dense_microcat_count").fill_null(0) / INTENT_DEPTH).alias("dense_microcat_share"))


def add_labels(candidates: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    relevant = queries.select("query_id", pl.col("relevant_item_ids").alias("item_id")).explode("item_id")
    relevant = relevant.with_columns(pl.lit(1).alias("label"))
    candidates = candidates.join(relevant, on=["query_id", "item_id"], how="left", maintain_order="left")
    return candidates.with_columns(pl.col("label").fill_null(0))


def build_features(
    candidates: pl.DataFrame,
    queries: pl.DataFrame,
    items: pl.DataFrame,
    prior_table: pl.DataFrame,
    centroids: pl.DataFrame,
) -> pl.DataFrame:
    candidates = candidates.join(items, on="item_index", how="left", maintain_order="left")
    candidates = add_query_features(candidates, queries)
    candidates = add_location_features(candidates, prior_table, centroids)
    candidates = add_pool_features(candidates)
    if "dense_rank" in candidates.columns:
        candidates = add_dense_intent(candidates)
    if "relevant_item_ids" in queries.columns:
        candidates = add_labels(candidates, queries)
    return candidates
