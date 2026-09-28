"""Стемы заголовка, параметров и описания объявлений, текста запросов и их фильтров."""

import polars as pl

from avito.processing.text import stem

# стеммим кусками, чтобы описания всех 515 тыс. объявлений не держать в памяти питоновскими строками
CHUNK_SIZE = 50_000


def stem_column(df: pl.DataFrame, column: str) -> list[str]:
    return [stem(text) for text in df[column].to_list()]


def process_items(items: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for chunk in items.iter_slices(CHUNK_SIZE):
        part = pl.DataFrame({
            "item_id": chunk["item_id"],
            "in_corpus": chunk["in_corpus"],
            "title_stems": stem_column(chunk, "item_title_raw"),
            "params_stems": stem_column(chunk, "item_infm_params_text"),
            "description_stems": stem_column(chunk, "item_description_raw"),
        })
        parts.append(part)
    return pl.concat(parts)


def process_queries(queries: pl.DataFrame) -> pl.DataFrame:
    return queries.with_columns(
        pl.Series("query_stems", stem_column(queries, "search_query")),
        pl.Series("filter_stems", stem_column(queries, "search_infm_params_text")),
    )
