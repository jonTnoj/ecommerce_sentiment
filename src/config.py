"""
Project configuration.

Single source of truth for paths, sampling parameters, DB settings,
and the aspect-keyword lexicon. Edit values here rather than scattering
constants through the pipeline.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

# The raw input file. Either drop the gzipped JSONL here, or change this path.
RAW_DATA_FILE = DATA_DIR / "Appliances_jsonl.gz"

# Where intermediate cleaned/scored CSVs land. Useful for the Streamlit app
# so it doesn't have to hit the database for every page render.
SCORED_CSV = OUTPUT_DIR / "scored_reviews.csv"
METRICS_JSON = OUTPUT_DIR / "metrics.json"

# --- Sampling ------------------------------------------------------------

# Stratified sample size: total reviews to keep after ingestion.
# Per-star quota is SAMPLE_SIZE / 5. With 50_000 we get 10k per star bucket,
# which is plenty for VADER comparison and quick to score (~30s on a laptop).
SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

# --- Database ------------------------------------------------------------

# "mongo" tries pymongo against MONGO_URI; if it can't connect, falls back
# to TinyDB automatically. "tinydb" forces TinyDB. "mongo_strict" raises if
# MongoDB isn't reachable (use this when grading the DB-implementation rubric
# line so you know which backend you actually demoed).
DB_BACKEND = "mongo"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "cs210_reviews"
MONGO_COLLECTION = "appliances"

TINYDB_PATH = DATA_DIR / "tinydb_store.json"

# --- Aspect / complaint categories --------------------------------------
#
# Keyword-based aspect tagging. Each review is checked against every
# category; matches are non-exclusive (a review can fit several aspects).
# Phrases are matched as whole-word substrings, case-insensitive.
#
# These lexicons were hand-curated for the Appliances category. Adding
# domain-specific phrases here is the easiest way to improve aspect recall
# without changing model code.

ASPECT_KEYWORDS = {
    "Product Quality": [
        "quality", "build", "material", "cheaply made", "well made", "sturdy",
        "flimsy", "defective", "broken", "broke", "doesn't work", "stopped working",
        "not working", "malfunction", "faulty", "poor quality", "great quality",
    ],
    "Durability": [
        "lasted", "lasts", "last long", "fell apart", "wore out", "years", "months",
        "weeks", "durable", "durability", "lifespan", "longevity", "still works",
        "broke after", "stopped after",
    ],
    "Price / Value": [
        "price", "expensive", "cheap", "overpriced", "worth", "value", "money",
        "pricey", "affordable", "deal", "bargain", "waste of money", "good buy",
    ],
    "Shipping & Packaging": [
        "shipping", "shipped", "package", "packaging", "delivery", "delivered",
        "arrived", "box was", "damaged in", "fast shipping", "slow shipping",
        "packed", "in transit",
    ],
    "Installation / Ease of Use": [
        "install", "installation", "setup", "set up", "instructions", "manual",
        "easy to use", "hard to use", "complicated", "simple to", "directions",
        "assembly", "assemble",
    ],
    "Customer Service": [
        "customer service", "return", "returned", "refund", "warranty",
        "support", "contacted", "replacement", "exchange", "seller", "rep ",
    ],
}
