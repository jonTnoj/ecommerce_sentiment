"""Keyword-based aspect tagging for review text."""
from __future__ import annotations

import re
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _aspect_patterns() -> dict[str, re.Pattern]:
    """Compile each aspect's keyword list into a single alternation regex.

    Longer phrases are tried first so "customer service" beats "service" alone.
    """
    patterns: dict[str, re.Pattern] = {}
    for aspect, keywords in config.ASPECT_KEYWORDS.items():
        keywords_longest_first = sorted(keywords, key=len, reverse=True)
        escaped_keywords = [re.escape(kw) for kw in keywords_longest_first]
        alternation = "|".join(escaped_keywords)
        patterns[aspect] = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
    return patterns


def categorize(text: str) -> list[str]:
    """Return which aspect categories appear in ``text`` (zero or more)."""
    if not text:
        return []
    matched_aspects = [
        aspect
        for aspect, pattern in _aspect_patterns().items()
        if pattern.search(text)
    ]
    return matched_aspects


def categorize_batch(texts: list[str]) -> list[list[str]]:
    """Vectorized convenience wrapper for pandas .apply use."""
    return [categorize(t) for t in texts]
