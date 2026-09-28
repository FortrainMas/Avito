"""Шаг 4а: тексты объявлений и запросов и обучающие пары для эмбеддера → data/embeddings/input/."""

import polars as pl

from avito import paths
from avito.embeddings.texts import build_documents, build_pairs, build_queries
from avito.ingestion import load as ingested
from avito.validation.load import load_ranker_queries, load_retriever_fold, load_val_queries


def main() -> None:
    fold = load_retriever_fold()
    pairs = build_pairs(fold)

    items = ingested.load_items(["item_id", "in_corpus", "item_location_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    pair_items = pairs.select("item_id").unique().with_columns(pl.lit(True).alias("in_pairs"))
    items = items.join(pair_items, on="item_id", how="left", maintain_order="left")
    items = items.filter(pl.col("in_corpus") | pl.col("in_pairs").fill_null(False))
    documents = build_documents(items)

    queries = pl.concat([
        build_queries(load_val_queries(), "val"),
        build_queries(load_ranker_queries(), "ranker"),
        build_queries(ingested.load_bench_queries(), "bench"),
    ])

    item_locations = ingested.load_items(["item_id", "item_location_id"])
    location_pairs = fold.select("search_location_id", "item_id").join(item_locations, on="item_id")
    location_pairs = location_pairs.select("search_location_id", "item_location_id")

    paths.EMBEDDINGS_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents.write_parquet(paths.EMBEDDING_DOCUMENTS)
    queries.write_parquet(paths.EMBEDDING_QUERIES)
    pairs.write_parquet(paths.EMBEDDING_PAIRS)
    location_pairs.write_parquet(paths.EMBEDDING_LOCATION_PAIRS)

    print(f"documents: {documents.height}, in corpus: {documents['in_corpus'].sum()}")
    print(f"queries: {queries.group_by('set').len().sort('set').rows()}")
    print(f"pairs: {pairs.height}, query texts: {pairs['query_text'].n_unique()}")
    print(f"location pairs: {location_pairs.height}")


if __name__ == "__main__":
    main()
