"""Дообучение эмбеддера на парах «запрос → выбранное объявление» (запускается на GPU)."""

import polars as pl
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers


def finetune(
    model: SentenceTransformer,
    pairs: pl.DataFrame,
    query_prefix: str,
    document_prefix: str,
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 256,
    mini_batch_size: int = 64,
    learning_rate: float = 2e-5,
) -> list[dict]:
    """Одна эпоха по парам с отрицательными примерами из той же пачки, без повторов текстов в пачке.

    На HuggingFace e5-large-instruct лежит в fp16, а обучение в чистом fp16 разваливается, поэтому веса
    переводим в fp32 (bf16 остаётся только в autocast). Возвращает лог обучения.
    """
    model.float()
    dataset = Dataset.from_dict({
        "anchor": [query_prefix + text for text in pairs["query_text"].to_list()],
        "positive": [document_prefix + text for text in pairs["text"].to_list()],
    })
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=0.05,
        bf16=bf16,
        fp16=cuda and not bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()
    return trainer.state.log_history
