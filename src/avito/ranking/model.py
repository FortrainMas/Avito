"""Реранкер LightGBM LambdaRank: обучение на пулах кандидатов и выбор top-50."""

import lightgbm as lgb
import numpy as np
import polars as pl

from avito.ranking.features import FEATURES

# выставлены один раз заранее и на валидации не перебирались; число деревьев выбирает ранняя остановка,
# градиенты считаются по парам в top-100, потому что метрика — top-50 из ~600 кандидатов
PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "lambdarank_truncation_level": 100,
    "verbose": -1,
}


def feature_matrix(candidates: pl.DataFrame, features: list[str]) -> np.ndarray:
    return np.ascontiguousarray(candidates.select(pl.col(features).cast(pl.Float32)).to_numpy())


def labels(candidates: pl.DataFrame) -> np.ndarray:
    return np.array(candidates["label"].to_list(), dtype=np.float32)


def group_sizes(candidates: pl.DataFrame) -> np.ndarray:
    return np.array(candidates.group_by("query_id", maintain_order=True).len()["len"].to_list(), dtype=np.int32)


def training_rows(candidates: pl.DataFrame) -> pl.DataFrame:
    """Запросы без правильного объявления в пуле ничему не учат LambdaRank — убираем их."""
    candidates = candidates.filter(pl.col("label").max().over("query_id") == 1)
    return candidates.sort("query_id", "retrieval_rank")


def fit_ranker(
    train: pl.DataFrame,
    valid: pl.DataFrame | None = None,
    features: list[str] = FEATURES,
    params: dict = PARAMS,
) -> lgb.LGBMRanker:
    train = training_rows(train)
    model = lgb.LGBMRanker(**params)
    if valid is None:
        model.fit(feature_matrix(train, features), labels(train), group=group_sizes(train))
        return model

    valid = training_rows(valid)
    model.fit(
        feature_matrix(train, features),
        labels(train),
        group=group_sizes(train),
        eval_set=[(feature_matrix(valid, features), labels(valid))],
        eval_group=[group_sizes(valid)],
        eval_at=[50],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def rerank(candidates: pl.DataFrame, model: lgb.Booster, features: list[str] = FEATURES, k: int = 50) -> pl.DataFrame:
    scores = model.predict(feature_matrix(candidates, features))
    candidates = candidates.with_columns(pl.Series("ranker_score", scores))
    candidates = candidates.sort(["query_id", "ranker_score"], descending=[False, True])
    top = candidates.group_by("query_id", maintain_order=True).head(k)
    return top.group_by("query_id", maintain_order=True).agg(pl.col("item_id"))
