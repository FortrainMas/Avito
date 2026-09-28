"""Шаг 6: обучить итоговый реранкер на запросах реранкера и всей валидации → data/models/ranker_lightgbm.txt."""

import time

import polars as pl

from avito import paths
from avito.embeddings.load import DENSE_MODEL
from avito.ranking.features import DENSE_FEATURES, RANKING_FEATURES
from avito.ranking.model import fit_ranker


def main() -> None:
    start = time.time()
    ranker_candidates = pl.read_parquet(paths.candidates_path("ranker", DENSE_MODEL))
    val_candidates = pl.read_parquet(paths.candidates_path("val", DENSE_MODEL))

    # каждый десятый запрос реранкера (id вида r0, r10, ...) — для ранней остановки
    early_stopping_mask = pl.col("query_id").str.slice(1).cast(pl.Int64) % 10 == 0
    train = pl.concat([ranker_candidates.filter(~early_stopping_mask), val_candidates])
    early_stopping = ranker_candidates.filter(early_stopping_mask)
    model = fit_ranker(train, early_stopping, features=RANKING_FEATURES + DENSE_FEATURES)

    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths.RANKER_MODEL))
    print(f"best iteration {model.best_iteration_}, {time.time() - start:.0f}s -> {paths.RANKER_MODEL}")


if __name__ == "__main__":
    main()
