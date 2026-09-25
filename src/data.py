"""Deterministic GLUE SST-2 validation inputs for the benchmark."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DATASET_ID = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
DATASET_CONFIG = "sst2"
DATASET_SPLIT = "validation"
PRIMARY_SEQUENCE_LENGTH = 64
BATCH_SIZES_PROTOTYPE = (1, 8)
BATCH_SIZES_PRIMARY = (1, 2, 4, 8, 16, 32, 64)


def load_pool():
    """Use published validation order; every condition draws a prefix of it."""
    pool = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
    if len(pool) < max(BATCH_SIZES_PRIMARY):
        raise ValueError(f"SST-2 validation pool is too small: {len(pool)}")
    if not all(isinstance(row["sentence"], str) for row in pool):
        raise ValueError("SST-2 pool contains a non-string sentence")
    return pool


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)


def token_lengths(pool, tokenizer, output_path: Path):
    """Lengths include special tokens, with neither truncation nor padding."""
    sentences = list(pool["sentence"])
    encoded = tokenizer(sentences, add_special_tokens=True, truncation=False,
                        padding=False, return_attention_mask=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    if len(lengths) != len(pool) or any(length <= 0 for length in lengths):
        raise ValueError("Invalid token-length results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_index", "token_length"))
        writer.writerows(enumerate(lengths))
    ordered = sorted(lengths)

    def nearest_rank(percentile: float) -> int:
        return ordered[math.ceil(percentile * len(ordered)) - 1]

    stats = {
        "count": len(lengths),
        "median": statistics.median(lengths),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "max": max(lengths),
        "percentile_method": "nearest rank (median is conventional midpoint)",
        "fits_without_truncation": {
            str(limit): sum(length <= limit for length in lengths)
            for limit in (32, 64, 128)
        },
        "candidate_lengths": [32, 64, 128],
    }
    stats["recommended_length"] = next(
        (limit for limit in (32, 64, 128) if stats["p95"] <= limit), 128
    )
    stats["recommendation_rule"] = "smallest candidate covering at least p95"
    stats["primary_length_selected"] = False
    return stats


def make_batch(pool, tokenizer, batch_size: int, sequence_length: int):
    if batch_size < 1 or batch_size > len(pool):
        raise ValueError("Invalid batch size")
    if sequence_length != PRIMARY_SEQUENCE_LENGTH:
        raise ValueError(f"Primary sequence length must be {PRIMARY_SEQUENCE_LENGTH}")
    sentences = [pool[index]["sentence"] for index in range(batch_size)]
    encoded = tokenizer(sentences, padding="max_length", truncation=True,
                        max_length=sequence_length, return_tensors="pt")
    return {key: encoded[key] for key in ("input_ids", "attention_mask")}
