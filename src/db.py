"""Document store with MongoDB backend and TinyDB fallback.

Set DB_BACKEND = "tinydb" in config to skip MongoDB entirely (useful for
grading without a local server). The default "mongo" tries MongoDB first
and falls back to TinyDB automatically if the server is unreachable.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

from . import config

log = logging.getLogger(__name__)


class ReviewStore:
    """Thin wrapper exposing insert / find / count / drop for either backend."""

    def __init__(self) -> None:
        self.backend, self._impl = self._connect(config.DB_BACKEND)
        log.info("ReviewStore using backend: %s", self.backend)

    @staticmethod
    def _connect(backend: str):
        if backend == "mongo":
            try:
                from pymongo import MongoClient

                client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=2000)
                client.admin.command("ping")  # confirm server is actually reachable
                collection = client[config.MONGO_DB_NAME][config.MONGO_COLLECTION]
                return "mongo", collection
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "MongoDB unreachable (%s) — falling back to TinyDB.",
                    exc.__class__.__name__,
                )

        # Pure-Python, file-based fallback — no server needed.
        from tinydb import TinyDB

        config.TINYDB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = TinyDB(config.TINYDB_PATH)
        return "tinydb", db.table("reviews")

    def insert_many(self, docs: Iterable[dict[str, Any]]) -> int:
        docs = list(docs)
        if not docs:
            return 0
        if self.backend == "mongo":
            self._impl.insert_many(docs, ordered=False)
        else:
            self._impl.insert_multiple(docs)
        return len(docs)

    def drop(self) -> None:
        if self.backend == "mongo":
            self._impl.drop()
        else:
            self._impl.truncate()

    def create_indexes(self) -> None:
        if self.backend != "mongo":
            return  # TinyDB has no index support
        # Compound index for product-level time-series queries on the dashboard.
        self._impl.create_index([("parent_asin", 1), ("timestamp", 1)])
        self._impl.create_index("rating")
        self._impl.create_index("predicted_polarity")
        log.info("Created indexes on parent_asin+timestamp, rating, predicted_polarity.")

    def count(self) -> int:
        if self.backend == "mongo":
            return self._impl.count_documents({})
        return len(self._impl)

    def iter_all(self, projection: list[str] | None = None) -> Iterator[dict[str, Any]]:
        """Stream every document, optionally restricted to specific fields."""
        if self.backend == "mongo":
            proj = {f: 1 for f in projection} if projection else None
            if proj:
                proj["_id"] = 0
            yield from self._impl.find({}, proj)
        else:
            for doc in self._impl.all():
                if projection:
                    yield {k: doc.get(k) for k in projection}
                else:
                    yield doc
