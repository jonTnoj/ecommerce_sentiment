"""
Ingestion: gzipped JSONL  ->  stratified sample  ->  document store.

The raw Appliances dataset has ~2.13M reviews and is heavily skewed toward
5-star ratings (~70%). Working on the full file is slow and produces
imbalanced metrics, so we draw a stratified sample by star rating using
reservoir sampling -- this gives an unbiased per-class sample in a single
pass without having to load the whole file into memory.
"""
from __future__ import annotations

import gzip
import json
import logging
import random
from pathlib import Path
from typing import Iterator

from . import config
from .db import ReviewStore

log = logging.getLogger(__name__)


def _stream_jsonl(path: Path) -> Iterator[dict]:
    """Yield one parsed record per line from a (possibly gzipped) JSONL file.

    Skips malformed lines rather than crashing -- public datasets occasionally
    have a stray bad row, and we don't want one bad line to abort an ingest.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed line %d", i)
                continue


def _is_usable(rec: dict) -> bool:
    """Filter out records we can't analyze: missing rating or empty text."""
    rating = rec.get("rating")
    text = rec.get("text") or ""
    return rating in (1.0, 2.0, 3.0, 4.0, 5.0) and len(text.strip()) >= 3


def stratified_sample(
    path: Path,
    sample_size: int,
    seed: int = 42,
) -> list[dict]:
    """Reservoir-sample ``sample_size // 5`` records from each star rating.

    Reservoir sampling lets us draw a uniform-random sample without ever
    holding the full 2M-row file in memory, which keeps the ingest step
    runnable on a laptop.
    """
    rng = random.Random(seed)
    per_class = sample_size // 5
    reservoirs: dict[float, list[dict]] = {r: [] for r in (1.0, 2.0, 3.0, 4.0, 5.0)}
    seen: dict[float, int] = {r: 0 for r in reservoirs}

    for rec in _stream_jsonl(path):
        if not _is_usable(rec):
            continue
        r = rec["rating"]
        bucket = reservoirs[r]
        seen[r] += 1
        # Standard Algorithm R: fill, then probabilistically replace.
        if len(bucket) < per_class:
            bucket.append(rec)
        else:
            j = rng.randint(0, seen[r] - 1)
            if j < per_class:
                bucket[j] = rec

        if sum(seen.values()) % 250_000 == 0:
            log.info("Streamed %s records...", f"{sum(seen.values()):,}")

    log.info(
        "Stream complete. Per-rating counts seen: %s",
        {k: f"{v:,}" for k, v in seen.items()},
    )
    sampled: list[dict] = []
    for bucket in reservoirs.values():
        sampled.extend(bucket)
    rng.shuffle(sampled)
    return sampled


def _normalize(rec: dict) -> dict:
    """Light normalization: keep the fields the pipeline uses, coerce types."""
    return {
        "rating": float(rec["rating"]),
        "title": (rec.get("title") or "").strip(),
        "text": (rec.get("text") or "").strip(),
        "asin": rec.get("asin", ""),
        "parent_asin": rec.get("parent_asin", ""),
        "user_id": rec.get("user_id", ""),
        "timestamp": int(rec.get("timestamp", 0)),
        "helpful_vote": int(rec.get("helpful_vote", 0)),
        "verified_purchase": bool(rec.get("verified_purchase", False)),
        "has_image": bool(rec.get("images")),
    }


def ingest(
    path: Path | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    fresh: bool = True,
) -> int:
    """Top-level ingest entry point. Returns count of inserted documents."""
    path = path or config.RAW_DATA_FILE
    sample_size = sample_size or config.SAMPLE_SIZE
    seed = seed if seed is not None else config.RANDOM_SEED

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. "
            "Drop the Appliances_jsonl.gz file into ./data/ "
            "(see README) and re-run."
        )

    log.info("Sampling up to %d reviews from %s (seed=%d)", sample_size, path, seed)
    sampled = stratified_sample(path, sample_size, seed=seed)
    log.info("Sampled %d reviews", len(sampled))

    docs = [_normalize(r) for r in sampled]

    store = ReviewStore()
    if fresh:
        store.drop()
    inserted = store.insert_many(docs)
    store.create_indexes()
    log.info("Inserted %d documents into store (%s)", inserted, store.backend)
    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest()
