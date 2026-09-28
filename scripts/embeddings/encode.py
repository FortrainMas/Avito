"""Шаг 4б (GPU): закодировать корпус и запросы моделями и сравнить их на валидации → data/embeddings/output/<модель>/."""

import argparse

import polars as pl

from avito.embeddings.pipeline import encode_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    summary = []
    for name in args.models:
        for result in encode_and_evaluate(name, args.limit, args.batch_size):
            summary.append({"model": name} | result)
    print(pl.DataFrame(summary).sort("recall@50", descending=True))


if __name__ == "__main__":
    main()
