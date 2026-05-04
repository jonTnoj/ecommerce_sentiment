"""MongoDB document store for review data."""
from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

from pymongo import MongoClient

from . import config

log = logging.getLogger(__name__)


class ReviewStore:
    """Thin wrapper around a pymongo collection for the operations the pipeline needs."""

    def __init__(self) -> None:
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")  # fail fast if MongoDB isn't running
        self._collection = client[config.MONGO_DB_NAME][config.MONGO_COLLECTION]
        log.info("Connected to MongoDB at %s", config.MONGO_URI)

    def insert_many(self, docs: Iterable[dict[str, Any]]) -> int:
        docs = list(docs)
        if not docs:
            return 0
        self._collection.insert_many(docs, ordered=False)
        return len(docs)

    def drop(self) -> None:
        self._collection.drop()

    def create_indexes(self) -> None:
        # Compound index for product-level time-series queries on the dashboard.
        self._collection.create_index([("parent_asin", 1), ("timestamp", 1)])
        self._collection.create_index("rating")
        self._collection.create_index("predicted_polarity")
        log.info("Created indexes on parent_asin+timestamp, rating, predicted_polarity.")

    def count(self) -> int:
        return self._collection.count_documents({})

    def iter_all(self, projection: list[str] | None = None) -> Iterator[dict[str, Any]]:
        """Stream every document, optionally restricted to specific fields."""
        proj = {f: 1 for f in projection} if projection else None
        if proj:
            proj["_id"] = 0
        yield from self._collection.find({}, proj)
