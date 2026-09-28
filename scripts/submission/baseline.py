"""Ответ бейзлайна: BM25 + доли выборов по локациям (+ фильтры) → data/submissions/bm25_prior*.csv."""

from avito import paths
from avito.ingestion import load as ingested
from avito.processing import load as processed
from avito.retrieval.bm25 import BM25
from avito.retrieval.location import LocationPrior
from avito.retrieval.search import build_documents, retrieve
from avito.submission.answer import build_answer, check_answer

CONFIGS = {
    "bm25_prior": {"alpha": 0.1, "filter_weight": 0.0},
    "bm25_prior_filters": {"alpha": 0.1, "filter_weight": 0.3},
}


def main() -> None:
    corpus = processed.load_corpus()
    corpus = corpus.join(ingested.load_corpus(["item_id", "item_location_id"]), on="item_id", maintain_order="left")
    item_ids = corpus["item_id"].to_numpy()
    bench_queries = processed.load_bench_queries()

    item_locations = ingested.load_items(["item_id", "item_location_id"])
    pairs = processed.load_train(["search_location_id", "item_id"]).join(item_locations, on="item_id")
    location_prior = LocationPrior().fit(pairs)

    bm25 = BM25().fit(build_documents(corpus))

    paths.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for name, config in CONFIGS.items():
        top = retrieve(bench_queries, bm25, corpus["item_location_id"], location_prior=location_prior, k=50, **config)
        answer = build_answer(bench_queries["query_id"].to_list(), item_ids[top].tolist())
        check_answer(answer, bench_queries, corpus.select("item_id"))
        answer.write_csv(paths.SUBMISSIONS_DIR / f"{name}.csv")
        print(f"{name}: {answer.height} queries -> {paths.SUBMISSIONS_DIR / f'{name}.csv'}")


if __name__ == "__main__":
    main()
