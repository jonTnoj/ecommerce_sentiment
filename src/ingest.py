"""Ingest gzipped JSONL reviews into the document store via stratified sampling."""
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

STAR_RATINGS = (1.0, 2.0, 3.0, 4.0, 5.0)


def _stream_jsonl(path: Path) -> Iterator[dict]:
    """Yield one parsed record per line; skip malformed lines rather than crashing."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed line %d", line_number)


def _is_usable(record: dict) -> bool:
    rating = record.get("rating")
    text = record.get("text") or ""
    return rating in STAR_RATINGS and len(text.strip()) >= 3


def stratified_sample(
    path: Path,
    sample_size: int,
    seed: int = 42,
) -> list[dict]:
    """Draw ``sample_size // 5`` records from each star bucket using reservoir sampling.

    Reservoir sampling lets us pull a balanced sample in a single pass
    without loading the full ~2M-row file into memory.
    """
    rng = random.Random(seed)
    per_star = sample_size // 5

    reservoir: dict[float, list[dict]] = {star: [] for star in STAR_RATINGS}
    seen: dict[float, int] = {star: 0 for star in STAR_RATINGS}

    for record in _stream_jsonl(path):
        if not _is_usable(record):
            continue
        star = record["rating"]
        bucket = reservoir[star]
        seen[star] += 1
        total = sum(seen.values())

        # Algorithm R: fill the reservoir first, then replace slots at random.
        if len(bucket) < per_star:
            bucket.append(record)
        else:
            slot = rng.randint(0, seen[star] - 1)
            if slot < per_star:
                bucket[slot] = record

        if total % 250_000 == 0:
            log.info("Streamed %s records...", f"{total:,}")

    log.info("Stream complete. Per-rating counts: %s", {k: f"{v:,}" for k, v in seen.items()})

    samples: list[dict] = []
    for bucket in reservoir.values():
        samples.extend(bucket)
    rng.shuffle(samples)
    return samples


def _normalize(record: dict) -> dict:
    """Keep only the fields the pipeline uses and coerce types."""
    return {
        "rating": float(record["rating"]),
        "title": (record.get("title") or "").strip(),
        "text": (record.get("text") or "").strip(),
        "asin": record.get("asin", ""),
        "parent_asin": record.get("parent_asin", ""),
        "user_id": record.get("user_id", ""),
        "timestamp": int(record.get("timestamp", 0)),
        "helpful_vote": int(record.get("helpful_vote", 0)),
        "verified_purchase": bool(record.get("verified_purchase", False)),
        "has_image": bool(record.get("images")),
    }


def ingest(
    path: Path | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    fresh: bool = True,
) -> int:
    """Run the full ingest pipeline and return the number of documents inserted."""
    path = path or config.RAW_DATA_FILE
    sample_size = sample_size or config.SAMPLE_SIZE
    seed = seed if seed is not None else config.RANDOM_SEED

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. "
            "Drop the Appliances_jsonl.gz file into ./data/ and re-run."
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
    log.info("Inserted %d documents into %s store", inserted, store.backend)
    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest()
