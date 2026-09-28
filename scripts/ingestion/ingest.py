"""Шаг 1: сырые parquet из data/ → data/ingestion/."""

import polars as pl

from avito import paths
from avito.ingestion.ingest import build_bench_queries, build_items, build_train, check


def main() -> None:
    train_raw = pl.read_parquet(paths.TRAIN_RAW)
    bench_items_raw = pl.read_parquet(paths.BENCH_ITEMS_RAW)
    bench_queries_raw = pl.read_parquet(paths.BENCH_QUERIES_RAW)

    items = build_items(train_raw, bench_items_raw)
    train = build_train(train_raw)
    bench_queries = build_bench_queries(bench_queries_raw)
    check(items, train, bench_queries)

    paths.INGESTION_DIR.mkdir(parents=True, exist_ok=True)
    items.write_parquet(paths.INGESTED_ITEMS)
    train.write_parquet(paths.INGESTED_TRAIN)
    bench_queries.write_parquet(paths.INGESTED_BENCH_QUERIES)

    print(f"items: {items.height}, in corpus: {items['in_corpus'].sum()}")
    print(f"train: {train.height}")
    print(f"bench queries: {bench_queries.height}")


if __name__ == "__main__":
    main()
