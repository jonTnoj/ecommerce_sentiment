"""Stage 1: stream the gzipped JSONL into the document store."""
import logging
import sys
from pathlib import Path

# Make `src` importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import ingest  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inserted = ingest()
    print(f"\nInserted {inserted:,} reviews.")
