"""Закодировать корпус и все запросы моделью, сохранить векторы и метрики на валидации."""

import json
import time

import numpy as np
import polars as pl
import torch

from avito import paths
from avito.embeddings.encode import encode
from avito.embeddings.evaluate import evaluate_dense
from avito.embeddings.models import load_model, prefixes
from avito.retrieval.location import LocationPrior


def encode_and_evaluate(name: str, limit: int | None = None, batch_size: int = 128) -> list[dict]:
    start = time.time()
    documents = pl.read_parquet(paths.EMBEDDING_DOCUMENTS).filter(pl.col("in_corpus"))
    queries = pl.read_parquet(paths.EMBEDDING_QUERIES)
    if limit is not None:
        documents = documents.head(limit)
        queries = queries.head(limit)

    model = load_model(name)
    if torch.cuda.is_available():
        model.half()
    query_prefix, document_prefix = prefixes(name)
    document_embeddings = encode(model, documents["text"].to_list(), document_prefix, batch_size)
    query_embeddings = encode(model, queries["text"].to_list(), query_prefix, batch_size)
    print(f"{name}: encoded in {time.time() - start:.0f}s", flush=True)

    output_dir = paths.EMBEDDINGS_OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "documents.npy", document_embeddings)
    np.save(output_dir / "queries.npy", query_embeddings)
    documents.select("item_id").write_parquet(output_dir / "documents.parquet")
    queries.select("query_id", "set").write_parquet(output_dir / "queries.parquet")

    location_prior = LocationPrior().fit(pl.read_parquet(paths.EMBEDDING_LOCATION_PAIRS))
    is_val = (queries["set"] == "val").to_numpy()
    results = evaluate_dense(queries.filter(pl.col("set") == "val"), query_embeddings[is_val], documents, document_embeddings, location_prior)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"{name}: evaluated in {time.time() - start:.0f}s", flush=True)
    print(pl.DataFrame(results), flush=True)
    return results
