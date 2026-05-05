from pathlib import Path

# paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

RAW_DATA_FILE = DATA_DIR / "Appliances.jsonl.gz"
SCORED_CSV = OUTPUT_DIR / "scored_reviews.csv"
METRICS_JSON = OUTPUT_DIR / "metrics.json"

# sampling
SAMPLE_SIZE = 100_000   # 10k reviews per star bucket
RANDOM_SEED = 42

# database
# "mongo" tries MongoDB first, falls back to TinyDB if unreachable.
# "tinydb" skips MongoDB entirely — useful for grading without a local server.
DB_BACKEND = "mongo"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "cs210_reviews"
MONGO_COLLECTION = "appliances"

TINYDB_PATH = DATA_DIR / "tinydb_store.json"

# aspect keyword lexicons
# Matched against review text (whole-word, case-insensitive).
# A review can match multiple aspects. Extend these lists to improve recall.

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
