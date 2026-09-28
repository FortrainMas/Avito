"""Пути ко всем файлам данных и моделей, по этапам пайплайна."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

TRAIN_RAW = DATA_DIR / "train.parquet"
BENCH_QUERIES_RAW = DATA_DIR / "benchmark_queries.parquet"
BENCH_ITEMS_RAW = DATA_DIR / "benchmark_items.parquet"

INGESTION_DIR = DATA_DIR / "ingestion"
INGESTED_ITEMS = INGESTION_DIR / "items.parquet"
INGESTED_TRAIN = INGESTION_DIR / "train.parquet"
INGESTED_BENCH_QUERIES = INGESTION_DIR / "bench_queries.parquet"

PROCESSING_DIR = DATA_DIR / "processing"
PROCESSED_ITEMS = PROCESSING_DIR / "items.parquet"
PROCESSED_TRAIN = PROCESSING_DIR / "train.parquet"
PROCESSED_BENCH_QUERIES = PROCESSING_DIR / "bench_queries.parquet"

VALIDATION_DIR = DATA_DIR / "validation"
VAL_QUERIES = VALIDATION_DIR / "val_queries.parquet"
TRAIN_FOLD = VALIDATION_DIR / "train_fold.parquet"
RANKER_QUERIES = VALIDATION_DIR / "ranker_queries.parquet"
RETRIEVER_FOLD = VALIDATION_DIR / "retriever_fold.parquet"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_INPUT_DIR = EMBEDDINGS_DIR / "input"
EMBEDDINGS_OUTPUT_DIR = EMBEDDINGS_DIR / "output"
EMBEDDINGS_MODELS_DIR = EMBEDDINGS_DIR / "models"
EMBEDDINGS_LOGS_DIR = EMBEDDINGS_DIR / "logs"
EMBEDDING_DOCUMENTS = EMBEDDINGS_INPUT_DIR / "documents.parquet"
EMBEDDING_QUERIES = EMBEDDINGS_INPUT_DIR / "queries.parquet"
EMBEDDING_PAIRS = EMBEDDINGS_INPUT_DIR / "pairs.parquet"
EMBEDDING_LOCATION_PAIRS = EMBEDDINGS_INPUT_DIR / "location_pairs.parquet"

RANKING_DIR = DATA_DIR / "ranking"

MODELS_DIR = DATA_DIR / "models"
RANKER_MODEL = MODELS_DIR / "ranker_lightgbm.txt"

SUBMISSIONS_DIR = DATA_DIR / "submissions"
ANSWER = SUBMISSIONS_DIR / "answer.csv"


def candidates_path(name: str, dense_model: str | None) -> Path:
    if dense_model is None:
        return RANKING_DIR / f"{name}_candidates.parquet"
    return RANKING_DIR / f"{name}_candidates_{dense_model}.parquet"
