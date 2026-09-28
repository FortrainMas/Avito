"""Нормализация и стемминг текстов: нижний регистр, ё → е, токены [a-z0-9а-я]+, стоп-слова, Snowball."""

import re

import Stemmer

STOPWORDS = {
    "а", "без", "бы", "в", "во", "для", "до", "же", "за", "и", "из", "или", "к", "как", "ко", "ли",
    "на", "над", "не", "но", "о", "об", "от", "по", "под", "при", "про", "с", "со", "то", "у", "что", "это",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9а-я]+")

stemmer = Stemmer.Stemmer("russian")
stemmer.maxCacheSize = 1_000_000


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    tokens = TOKEN_PATTERN.findall(text)
    return [token for token in tokens if token not in STOPWORDS]


def stem(text: str) -> str:
    return " ".join(stemmer.stemWords(tokenize(text)))
