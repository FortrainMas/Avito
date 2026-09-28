"""Шаг 5: пулы кандидатов с признаками для запросов реранкера, валидации и теста → data/ranking/.

Для реранкера и валидации доли выборов по локациям считаются по retriever_fold, для теста — по всему train.
"""

import argparse
import time

from avito import paths
from avito.embeddings.load import DENSE_MODEL, load_document_vectors, load_query_vectors
from avito.ingestion import load as ingested
from avito.processing import load as processed
from avito.ranking.candidates import build_indexes
from avito.ranking.dataset import build_dataset
from avito.ranking.features import item_table
from avito.retrieval.location import fit_location_stats
from avito.validation.load import load_ranker_queries, load_retriever_fold, load_val_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()
    dense_model = None if args.bm25_only else DENSE_MODEL

    start = time.time()
    corpus = processed.load_corpus()
    items = item_table(ingested.load_corpus())
    assert corpus["item_id"].equals(items["item_id"])
    all_items = ingested.load_items()

    fold_stats = fit_location_stats(load_retriever_fold(), all_items)
    full_stats = fit_location_stats(processed.load_train(), all_items)
    indexes = build_indexes(corpus)
    document_vectors = None
    if dense_model is not None:
        document_vectors = load_document_vectors(dense_model, items["item_id"])
    print(f"indexes ready: {time.time() - start:.0f}s")

    paths.RANKING_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("ranker", load_ranker_queries(), fold_stats),
        ("val", load_val_queries(), fold_stats),
        ("bench", processed.load_bench_queries(), full_stats),
    ]
    for name, queries, (location_prior, centroids) in datasets:
        query_vectors = None
        if dense_model is not None:
            query_vectors = load_query_vectors(dense_model, queries["query_id"])
        dataset = build_dataset(queries, items, indexes, location_prior, centroids, query_vectors=query_vectors, document_vectors=document_vectors)
        path = paths.candidates_path(name, dense_model)
        dataset.write_parquet(path)
        print(f"{path.name}: {dataset.height} rows, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
