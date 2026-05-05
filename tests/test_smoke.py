"""Smoke tests: catch broken imports, sentiment regressions, and silent data-loss bugs.

Runnable with `pytest tests/` or directly: `python tests/test_smoke.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import categorize, preprocess, sentiment


def test_clean_text_strips_html_and_urls():
    raw = "Great <b>product</b>! See https://example.com  for  more"
    cleaned = preprocess.clean_text(raw)
    assert "<b>" not in cleaned
    assert "https://" not in cleaned
    assert "  " not in cleaned  # whitespace collapsed


def test_clean_text_preserves_intensity_signals():
    # VADER reads ALL-CAPS and "!!!" as intensity boosters — must not strip them.
    raw = "AMAZING product!!!"
    cleaned = preprocess.clean_text(raw)
    assert "AMAZING" in cleaned
    assert "!!!" in cleaned


def test_sentiment_polarity_thresholds():
    pos = sentiment.score_text("Great", "I love this product, it works perfectly!")
    neg = sentiment.score_text("Terrible", "Awful. Broke immediately. Total waste.")
    assert pos.polarity == "positive"
    assert neg.polarity == "negative"
    assert pos.compound > 0
    assert neg.compound < 0


def test_predicted_stars_in_range():
    s = sentiment.score_text("ok", "It's fine I guess")
    assert 1.0 <= s.predicted_stars <= 5.0


def test_rating_to_polarity_buckets():
    assert sentiment.rating_to_polarity(1.0) == "negative"
    assert sentiment.rating_to_polarity(2.0) == "negative"
    assert sentiment.rating_to_polarity(3.0) == "neutral"
    assert sentiment.rating_to_polarity(4.0) == "positive"
    assert sentiment.rating_to_polarity(5.0) == "positive"


def test_aspect_categorization():
    text = "The shipping was fast but the build quality is awful, broke after a week"
    aspects = categorize.categorize(text)
    assert "Shipping & Packaging" in aspects
    assert "Product Quality" in aspects
    assert len(aspects) >= 2  # "broke after a week" also hits Durability


def test_aspect_no_false_positive():
    assert categorize.categorize("I really enjoy this!") == []


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    sys.exit(1 if failed else 0)
