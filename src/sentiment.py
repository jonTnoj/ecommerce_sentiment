"""
Sentiment scoring with VADER.

Why VADER:
  * Lexicon-based, works without training data, runs in milliseconds per review.
  * Tuned for social/short-form text but performs well on product reviews.
  * Returns a normalized compound score in [-1, 1] which we map two ways:
      - to a 1-5 scale  (regression-style evaluation: MAE / RMSE vs star rating)
      - to {neg, neutral, pos} (classification-style: precision / recall / F1)

Reference: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious
Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM-14.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocess import join_title_text

# Module-level singleton -- the analyzer loads its lexicon on construction,
# which is non-trivial. One instance is plenty.
_ANALYZER: SentimentIntensityAnalyzer | None = None


def _analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


Polarity = Literal["negative", "neutral", "positive"]


@dataclass
class SentimentResult:
    compound: float          # raw VADER compound, in [-1, 1]
    pos: float               # share of text in positive lexicon
    neu: float
    neg: float
    polarity: Polarity       # 3-class label derived from compound
    predicted_stars: float   # compound rescaled to [1.0, 5.0]


def _polarity_from_compound(compound: float) -> Polarity:
    """Standard VADER thresholds: |0.05| separates pos / neutral / neg."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _stars_from_compound(compound: float) -> float:
    """Linearly rescale [-1, 1] -> [1, 5] for MAE/RMSE comparison with ratings."""
    return 1.0 + (compound + 1.0) * 2.0  # -1 -> 1.0, +1 -> 5.0


def score_text(title: str, text: str) -> SentimentResult:
    """Score one review. Combines title + body before analysis."""
    full = join_title_text(title, text)
    if not full:
        # Treat empty input as neutral; VADER would also return all-zero.
        return SentimentResult(0.0, 0.0, 1.0, 0.0, "neutral", 3.0)
    s = _analyzer().polarity_scores(full)
    compound = s["compound"]
    return SentimentResult(
        compound=compound,
        pos=s["pos"],
        neu=s["neu"],
        neg=s["neg"],
        polarity=_polarity_from_compound(compound),
        predicted_stars=_stars_from_compound(compound),
    )


def rating_to_polarity(rating: float) -> Polarity:
    """Bucket a 1-5 star rating into the same 3 classes VADER outputs.

    1 or 2 stars  -> negative
    3 stars       -> neutral
    4 or 5 stars  -> positive

    The 3-star bucket is genuinely "neutral" in product-review settings
    -- discussion of why this isn't trivially obvious belongs in the report.
    """
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"
