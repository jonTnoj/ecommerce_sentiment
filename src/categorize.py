"""
Aspect-based categorization (the "feedback sorter").

We tag each review with zero or more *aspect categories* (Product Quality,
Shipping, Price, etc.) using keyword-based matching against the lexicons
in :mod:`config`. This is intentionally simple -- the proposal does not
promise a learned classifier, and there is no labeled training data for
aspects in the public Amazon Reviews 2023 release. A keyword approach is
honest, transparent, and easy for the report to discuss.

Each matched aspect inherits the review's overall VADER polarity, which
gives us aspect-level sentiment aggregates (e.g. "shipping is mostly
discussed negatively for this product").
"""
from __future__ import annotations

import re
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _aspect_patterns() -> dict[str, re.Pattern]:
    """Compile each aspect's keyword list into one alternation regex.

    Phrase boundaries: we use \b on either side so "lasted" doesn't match
    inside "everlasting." Multi-word phrases ("set up", "customer service")
    are kept intact and just embedded in the alternation.
    """
    compiled = {}
    for aspect, kws in config.ASPECT_KEYWORDS.items():
        # Sort by length descending so multi-word phrases are tried first.
        kws_sorted = sorted(kws, key=len, reverse=True)
        escaped = [re.escape(k) for k in kws_sorted]
        # Word-boundary on outside; \W* in middle for the regex engine.
        pattern = r"\b(?:" + "|".join(escaped) + r")\b"
        compiled[aspect] = re.compile(pattern, re.IGNORECASE)
    return compiled


def categorize(text: str) -> list[str]:
    """Return the list of aspect categories matched in ``text``.

    A review can match zero, one, or many. Order in the returned list is
    deterministic (same as :data:`config.ASPECT_KEYWORDS` insertion order).
    """
    if not text:
        return []
    matched = []
    for aspect, pattern in _aspect_patterns().items():
        if pattern.search(text):
            matched.append(aspect)
    return matched


def categorize_batch(texts: list[str]) -> list[list[str]]:
    """Vectorized convenience wrapper for pandas .apply use."""
    return [categorize(t) for t in texts]
