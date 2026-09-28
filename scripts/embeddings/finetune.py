"""Шаг 4в (GPU): дообучить эмбеддер на парах из retriever_fold, затем закодировать и оценить."""

import argparse
import json
import time

import polars as pl

from avito import paths
from avito.embeddings.models import FINETUNED_SUFFIX, load_model, prefixes
from avito.embeddings.pipeline import encode_and_evaluate
from avito.embeddings.train import finetune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    start = time.time()
    pairs = pl.read_parquet(paths.EMBEDDING_PAIRS)
    documents = pl.read_parquet(paths.EMBEDDING_DOCUMENTS, columns=["item_id", "text"])
    pairs = pairs.join(documents, on="item_id", maintain_order="left")
    pairs = pairs.sample(fraction=1.0, shuffle=True, seed=0)
    if args.limit is not None:
        pairs = pairs.head(args.limit)
    print(f"pairs: {pairs.height}", flush=True)

    name = args.model + FINETUNED_SUFFIX
    output_dir = paths.EMBEDDINGS_MODELS_DIR / name
    model = load_model(args.model)
    query_prefix, document_prefix = prefixes(args.model)
    history = finetune(
        model,
        pairs,
        query_prefix,
        document_prefix,
        str(paths.EMBEDDINGS_DIR / "checkpoints" / name),
        epochs=args.epochs,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
    )
    model.half()
    model.save(str(output_dir))
    paths.EMBEDDINGS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.EMBEDDINGS_LOGS_DIR / f"{name}.json").write_text(json.dumps(history, indent=2))
    print(f"{name}: trained in {time.time() - start:.0f}s", flush=True)

    encode_and_evaluate(name, args.limit)


if __name__ == "__main__":
    main()
