"""Stage 2: score sentiment, tag aspects, and emit metrics + scored CSV."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyze import run  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run()
    print(f"\nScored {len(df):,} reviews. Outputs in ./output/.")
