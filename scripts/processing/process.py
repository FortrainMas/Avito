"""Шаг 2: стемы текстов объявлений и запросов → data/processing/."""

from avito import paths
from avito.ingestion import load
from avito.processing.process import process_items, process_queries


def main() -> None:
    items = process_items(load.load_items(["item_id", "in_corpus", "item_title_raw", "item_infm_params_text", "item_description_raw"]))
    train = process_queries(load.load_train())
    bench_queries = process_queries(load.load_bench_queries())

    paths.PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    items.write_parquet(paths.PROCESSED_ITEMS)
    train.write_parquet(paths.PROCESSED_TRAIN)
    bench_queries.write_parquet(paths.PROCESSED_BENCH_QUERIES)

    print(f"items: {items.height}, in corpus: {items['in_corpus'].sum()}")
    print(f"train: {train.height}")
    print(f"bench queries: {bench_queries.height}")


if __name__ == "__main__":
    main()
