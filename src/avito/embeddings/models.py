"""Эмбеддеры, которые сравнивали, их префиксы для запросов и документов и загрузка моделей."""

from sentence_transformers import SentenceTransformer

from avito import paths

MAX_LENGTH = 256
FINETUNED_SUFFIX = "-finetuned"

INSTRUCTION = "Instruct: Given a search query for a local service, retrieve service listings that match it\nQuery: "

# префиксы — те, что рекомендуют авторы моделей
MODELS = {
    "rubert-mini-frida": {"hf_id": "sergeyzh/rubert-mini-frida", "query_prefix": "search_query: ", "document_prefix": "search_document: "},
    "frida": {"hf_id": "ai-forever/FRIDA", "query_prefix": "search_query: ", "document_prefix": "search_document: "},
    "e5-small": {"hf_id": "intfloat/multilingual-e5-small", "query_prefix": "query: ", "document_prefix": "passage: "},
    "e5-base": {"hf_id": "intfloat/multilingual-e5-base", "query_prefix": "query: ", "document_prefix": "passage: "},
    "e5-large": {"hf_id": "intfloat/multilingual-e5-large", "query_prefix": "query: ", "document_prefix": "passage: "},
    "e5-large-instruct": {"hf_id": "intfloat/multilingual-e5-large-instruct", "query_prefix": INSTRUCTION, "document_prefix": ""},
    "bge-m3": {"hf_id": "BAAI/bge-m3", "query_prefix": "", "document_prefix": ""},
    "user-bge-m3": {"hf_id": "deepvk/USER-bge-m3", "query_prefix": "", "document_prefix": ""},
}


def base_name(name: str) -> str:
    return name.removesuffix(FINETUNED_SUFFIX)


def prefixes(name: str) -> tuple[str, str]:
    config = MODELS[base_name(name)]
    return config["query_prefix"], config["document_prefix"]


def load_model(name: str, device: str | None = None) -> SentenceTransformer:
    if name.endswith(FINETUNED_SUFFIX):
        source = str(paths.EMBEDDINGS_MODELS_DIR / name)
    else:
        source = MODELS[name]["hf_id"]
    model = SentenceTransformer(source, device=device)
    model.max_seq_length = MAX_LENGTH
    # у части моделей в конфиге есть промпт по умолчанию, он добавился бы к нашему префиксу
    model.default_prompt_name = None
    return model
