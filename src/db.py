"""
Storage layer.

Exposes a single :class:`ReviewStore` class with the same interface
regardless of whether the underlying backend is MongoDB or TinyDB.
This is what lets the rest of the pipeline stay backend-agnostic.

Backend selection rules (see config.DB_BACKEND):
  * "mongo"        -- try MongoDB; silently fall back to TinyDB if unreachable
  * "mongo_strict" -- try MongoDB; raise if unreachable
  * "tinydb"       -- always use TinyDB
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

from . import config

log = logging.getLogger(__name__)


class ReviewStore:
    """Document-store wrapper exposing only the operations the pipeline needs.

    The intent is to give the rest of the code a small, clean surface --
    insert_many / find / count / create_index / drop -- so we can swap
    backends without touching the pipeline modules.
    """

    def __init__(self, backend: str | None = None) -> None:
        backend = backend or config.DB_BACKEND
        self.backend, self._impl = self._connect(backend)
        log.info("ReviewStore using backend: %s", self.backend)

    # ------------------------------------------------------------------ init

    @staticmethod
    def _connect(backend: str):
        if backend in ("mongo", "mongo_strict"):
            try:
                from pymongo import MongoClient
                from pymongo.errors import ServerSelectionTimeoutError

                client = MongoClient(
                    config.MONGO_URI, serverSelectionTimeoutMS=2000
                )
                # Force a round-trip to confirm the server is actually up.
                client.admin.command("ping")
                coll = client[config.MONGO_DB_NAME][config.MONGO_COLLECTION]
                return "mongo", coll
            except Exception as exc:  # noqa: BLE001
                if backend == "mongo_strict":
                    raise RuntimeError(
                        f"MongoDB required but unreachable at {config.MONGO_URI}"
                    ) from exc
                log.warning(
                    "MongoDB unreachable (%s) -- falling back to TinyDB. "
                    "Install MongoDB or run 'docker run -d -p 27017:27017 mongo' "
                    "for the production-style backend.",
                    exc.__class__.__name__,
                )

        # TinyDB fallback. Pure-Python, file-based document store.
        from tinydb import TinyDB

        config.TINYDB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = TinyDB(config.TINYDB_PATH)
        return "tinydb", db.table("reviews")

    # ------------------------------------------------------------------ writes

    def insert_many(self, docs: Iterable[dict[str, Any]]) -> int:
        """Bulk insert. Returns count of inserted documents."""
        docs = list(docs)
        if not docs:
            return 0
        if self.backend == "mongo":
            self._impl.insert_many(docs, ordered=False)
        else:
            self._impl.insert_multiple(docs)
        return len(docs)

    def drop(self) -> None:
        """Wipe the collection. Used at the start of a fresh ingest."""
        if self.backend == "mongo":
            self._impl.drop()
        else:
            self._impl.truncate()

    def create_indexes(self) -> None:
        """Build the indexes the dashboard needs.

        MongoDB only -- TinyDB doesn't support indexes, so this is a no-op
        there. The README discusses the indexing strategy in prose.
        """
        if self.backend != "mongo":
            log.info("Index creation skipped (TinyDB has no index support).")
            return

        # Compound: product-level time-series queries on the dashboard.
        self._impl.create_index([("parent_asin", 1), ("timestamp", 1)])
        # Single-field: rating filter, polarity filter.
        self._impl.create_index("rating")
        self._impl.create_index("predicted_polarity")
        log.info("Created MongoDB indexes on parent_asin+timestamp, rating, polarity.")

    # ------------------------------------------------------------------ reads

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
