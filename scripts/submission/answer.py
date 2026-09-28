"""Шаг 7: реранкер выбирает top-50 из пула каждого тестового запроса → data/submissions/answer.csv."""

import time

import lightgbm as lgb
import polars as pl

from avito import paths
from avito.embeddings.load import DENSE_MODEL
from avito.processing import load as processed
from avito.ranking.features import DENSE_FEATURES, RANKING_FEATURES
from avito.ranking.model import rerank
from avito.submission.answer import build_answer, check_answer


def main() -> None:
    start = time.time()
    model = lgb.Booster(model_file=str(paths.RANKER_MODEL))
    bench_queries = processed.load_bench_queries()
    bench_candidates = pl.read_parquet(paths.candidates_path("bench", DENSE_MODEL))

    predictions = rerank(bench_candidates, model, features=RANKING_FEATURES + DENSE_FEATURES)
    predictions = bench_queries.select("query_id").join(predictions, on="query_id", how="left", maintain_order="left")

    answer = build_answer(predictions["query_id"].to_list(), predictions["item_id"].to_list())
    check_answer(answer, bench_queries, processed.load_corpus(["item_id"]))
    paths.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    answer.write_csv(paths.ANSWER)
    print(f"{answer.height} queries, {time.time() - start:.0f}s -> {paths.ANSWER}")


if __name__ == "__main__":
    main()
