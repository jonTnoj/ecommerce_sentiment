"""VADER-based sentiment scoring for product reviews."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocess import join_title_text

# Loaded once at first use — the lexicon is non-trivial to construct.
_ANALYZER: SentimentIntensityAnalyzer | None = None


def _analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


Polarity = Literal["negative", "neutral", "positive"]


@dataclass
class SentimentResult:
    compound: float        # raw VADER score in [-1, 1]
    pos: float             # share of text matching positive lexicon
    neu: float
    neg: float
    polarity: Polarity     # 3-class label derived from compound
    predicted_stars: float # compound rescaled to [1.0, 5.0]


def _polarity_from_compound(compound: float) -> Polarity:
    """Standard VADER thresholds: ±0.05 separates the three classes."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _stars_from_compound(compound: float) -> float:
    """Rescale [-1, 1] → [1, 5] for MAE/RMSE comparison against star ratings."""
    return 1.0 + (compound + 1.0) * 2.0  # -1 → 1.0, +1 → 5.0


def score_text(title: str, text: str) -> SentimentResult:
    """Score one review; joins title and body before analysis."""
    full_text = join_title_text(title, text)
    if not full_text:
        # Empty input — VADER would also return all-zero, so we short-circuit.
        return SentimentResult(0.0, 0.0, 1.0, 0.0, "neutral", 3.0)
    scores = _analyzer().polarity_scores(full_text)
    compound = scores["compound"]
    return SentimentResult(
        compound=compound,
        pos=scores["pos"],
        neu=scores["neu"],
        neg=scores["neg"],
        polarity=_polarity_from_compound(compound),
        predicted_stars=_stars_from_compound(compound),
    )


def rating_to_polarity(rating: float) -> Polarity:
    """Map a 1–5 star rating to the same three-class polarity VADER uses.

    1–2 stars → negative, 3 stars → neutral, 4–5 stars → positive.
    """
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"
