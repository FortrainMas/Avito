"""Тексты объявлений и запросов для эмбеддера и обучающие пары (notebooks/embeddings/models.ipynb)."""

import re

import polars as pl

from avito.processing.params import clean_params

# начало описания: вместе с заголовком и параметрами укладывается в 256 токенов модели
DESCRIPTION_CHARS = 700
# не больше 20 пар на текст запроса, чтобы частые запросы вроде «маникюр» не забили обучение
MAX_PAIRS_PER_QUERY = 20
SPACES = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    return SPACES.sub(" ", text).strip()


def document_text(title: str, params: str, description: str) -> str:
    parts = [normalize_spaces(title), clean_params(params), normalize_spaces(description)[:DESCRIPTION_CHARS]]
    return ". ".join(part for part in parts if part)


def query_text(query: str, filters: str) -> str:
    cleaned = clean_params(filters)
    if cleaned:
        return f"{normalize_spaces(query)}. {cleaned}"
    return normalize_spaces(query)


def build_documents(items: pl.DataFrame) -> pl.DataFrame:
    titles = items["item_title_raw"].to_list()
    params = items["item_infm_params_text"].to_list()
    descriptions = items["item_description_raw"].to_list()
    texts = [document_text(title, param, description) for title, param, description in zip(titles, params, descriptions)]
    return items.select("item_id", "in_corpus", "item_location_id").with_columns(pl.Series("text", texts))


def build_queries(queries: pl.DataFrame, name: str) -> pl.DataFrame:
    texts = [query_text(query, filters) for query, filters in zip(queries["search_query"].to_list(), queries["search_infm_params_text"].to_list())]
    result = queries.select("query_id", "search_location_id").with_columns(
        pl.lit(name).alias("set"),
        pl.Series("text", texts),
    )
    if "relevant_item_ids" in queries.columns:
        return result.with_columns(queries["relevant_item_ids"])
    return result.with_columns(pl.lit(None, dtype=pl.List(pl.String)).alias("relevant_item_ids"))


def build_pairs(train: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Уникальные пары «текст запроса → выбранное объявление» для дообучения эмбеддера."""
    queries = train.select("search_query", "search_infm_params_text").unique()
    texts = [query_text(query, filters) for query, filters in zip(queries["search_query"].to_list(), queries["search_infm_params_text"].to_list())]
    queries = queries.with_columns(pl.Series("query_text", texts))

    pairs = train.select("search_query", "search_infm_params_text", "item_id").join(queries, on=["search_query", "search_infm_params_text"])
    pairs = pairs.select("query_text", "item_id").unique().sort("query_text", "item_id")
    pairs = pairs.sample(fraction=1.0, shuffle=True, seed=seed)
    return pairs.group_by("query_text", maintain_order=True).head(MAX_PAIRS_PER_QUERY)
