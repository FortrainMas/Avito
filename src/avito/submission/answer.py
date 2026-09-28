"""Сборка answer.csv и проверка формата из условия задачи."""

import polars as pl


def build_answer(query_ids: list[str], predictions: list[list[str]]) -> pl.DataFrame:
    return pl.DataFrame({
        "query_id": query_ids,
        "answer": [" ".join(items) for items in predictions],
    })


def check_answer(answer: pl.DataFrame, bench_queries: pl.DataFrame, corpus_item_ids: pl.DataFrame, k: int = 50) -> None:
    assert answer.columns == ["query_id", "answer"]
    assert answer.height == bench_queries.height
    assert answer["query_id"].is_unique().all()
    assert answer.join(bench_queries, on="query_id", how="anti").height == 0

    items = answer.select("query_id", pl.col("answer").str.split(" ").alias("item_id"))
    assert (items["item_id"].list.len() == k).all()
    assert (items["item_id"].list.n_unique() == k).all()

    items = items.explode("item_id")
    assert items["item_id"].str.contains(r"^[0-9a-f]{16}$").all()
    assert items.join(corpus_item_ids, on="item_id", how="anti").height == 0
