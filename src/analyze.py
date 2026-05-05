"""Pipeline stage 2: score sentiment, tag aspects, compute metrics, write outputs."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .categorize import categorize
from .db import ReviewStore
from .preprocess import join_title_text
from .sentiment import rating_to_polarity, score_text

log = logging.getLogger(__name__)


def load_reviews_to_df() -> pd.DataFrame:
    """Pull every review from the store into a pandas DataFrame."""
    store = ReviewStore()
    rows = list(store.iter_all())
    if not rows:
        raise RuntimeError("Review store is empty. Run scripts/01_ingest.py first.")
    df = pd.DataFrame(rows)
    log.info("Loaded %d reviews from %s store", len(df), store.backend)
    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Score each review with VADER and attach the result columns."""
    log.info("Scoring sentiment with VADER (this is the slow step)...")
    results = [score_text(title, body) for title, body in zip(df["title"], df["text"])]
    df["compound"] = [r.compound for r in results]
    df["pos_score"] = [r.pos for r in results]
    df["neu_score"] = [r.neu for r in results]
    df["neg_score"] = [r.neg for r in results]
    df["predicted_polarity"] = [r.polarity for r in results]
    df["predicted_stars"] = [r.predicted_stars for r in results]
    df["actual_polarity"] = df["rating"].apply(rating_to_polarity)
    log.info("Sentiment scoring complete.")
    return df


def add_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each review with matching aspect categories."""
    log.info("Tagging aspects...")
    full_texts = [join_title_text(t, b) for t, b in zip(df["title"], df["text"])]
    df["aspects"] = [categorize(t) for t in full_texts]
    df["aspect_count"] = df["aspects"].apply(len)
    log.info("Aspect tagging complete.")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetime, year_month, and review_length columns used by the viz layer."""
    # Timestamps are stored in milliseconds since epoch.
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Strip timezone before period conversion — pandas warns otherwise.
    df["year_month"] = df["datetime"].dt.tz_localize(None).dt.to_period("M").astype(str)
    df["review_length"] = df["text"].str.len()
    return df


def _aspect_count(df: pd.DataFrame, aspect: str) -> int:
    return int(df["aspects"].apply(lambda al: aspect in al).sum())


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute regression (MAE, RMSE) and classification (accuracy, F1) metrics."""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        mean_absolute_error,
    )

    mae = mean_absolute_error(df["rating"], df["predicted_stars"])
    rmse = float(np.sqrt(((df["rating"] - df["predicted_stars"]) ** 2).mean()))

    polarity_labels = ["negative", "neutral", "positive"]
    cls_report = classification_report(
        df["actual_polarity"],
        df["predicted_polarity"],
        labels=polarity_labels,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        df["actual_polarity"], df["predicted_polarity"], labels=polarity_labels
    ).tolist()
    accuracy = float((df["actual_polarity"] == df["predicted_polarity"]).mean())

    return {
        "n_reviews": int(len(df)),
        "regression": {"mae": float(mae), "rmse": rmse},
        "classification": {
            "accuracy": accuracy,
            "per_class": {label: cls_report[label] for label in polarity_labels},
            "macro_avg": cls_report["macro avg"],
            "weighted_avg": cls_report["weighted avg"],
            "labels": polarity_labels,
            "confusion_matrix": confusion,
        },
        "rating_distribution": (
            df["rating"].value_counts().sort_index().astype(int).to_dict()
        ),
        "polarity_distribution": df["predicted_polarity"].value_counts().to_dict(),
        "aspect_distribution": {
            aspect: _aspect_count(df, aspect)
            for aspect in config.ASPECT_KEYWORDS
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(write_csv: bool = True, write_metrics: bool = True) -> pd.DataFrame:
    """Full analysis pass: load → score → aspect-tag → derive → persist."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_reviews_to_df()
    df = add_sentiment(df)
    df = add_aspects(df)
    df = add_derived_columns(df)

    if write_csv:
        out = df.copy()
        out["aspects"] = out["aspects"].apply(json.dumps)
        out.to_csv(config.SCORED_CSV, index=False)
        log.info("Wrote scored reviews to %s", config.SCORED_CSV)

    metrics = compute_metrics(df)
    if write_metrics:
        with open(config.METRICS_JSON, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Wrote metrics to %s", config.METRICS_JSON)

    log.info("=== Headline metrics ===")
    log.info("MAE (predicted stars vs actual): %.3f", metrics["regression"]["mae"])
    log.info("RMSE: %.3f", metrics["regression"]["rmse"])
    log.info("3-class polarity accuracy: %.3f", metrics["classification"]["accuracy"])
    log.info("Macro F1: %.3f", metrics["classification"]["macro_avg"]["f1-score"])

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
