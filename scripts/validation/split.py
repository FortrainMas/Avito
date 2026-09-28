"""Шаг 3: валидация, запросы реранкера и retriever_fold с проверками на утечки → data/validation/."""

from avito import paths
from avito.processing import load
from avito.validation.split import check, check_ranker, split, split_ranker


def main() -> None:
    train = load.load_train()
    corpus_item_ids = load.load_corpus(["item_id"])
    bench_queries = load.load_bench_queries()

    val, train_fold = split(train, corpus_item_ids, bench_queries)
    check(val, train_fold, corpus_item_ids)

    ranker, retriever_fold = split_ranker(train_fold, corpus_item_ids, val)
    check_ranker(ranker, retriever_fold, val, corpus_item_ids)

    paths.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    val.write_parquet(paths.VAL_QUERIES)
    train_fold.write_parquet(paths.TRAIN_FOLD)
    ranker.write_parquet(paths.RANKER_QUERIES)
    retriever_fold.write_parquet(paths.RETRIEVER_FOLD)

    print(f"val queries: {val.height}, seen text share: {val['seen_text'].mean():.3f}")
    print(f"train fold: {train_fold.height} of {train.height}")
    print(f"ranker queries: {ranker.height}, retriever fold: {retriever_fold.height}")


if __name__ == "__main__":
    main()
