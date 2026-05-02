"""
Text preprocessing utilities.

Light cleaning only -- VADER works on raw-ish text and over-cleaning
(e.g. lowercasing, removing punctuation) actually *hurts* it because it
relies on capitalization and exclamation marks as intensity signals.
So we only strip HTML, collapse whitespace, and remove obvious noise.
"""
from __future__ import annotations

import re

# Pre-compiled patterns, applied in order. Each tuple: (regex, replacement).
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTI_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return a cleaned copy of ``text`` suitable for VADER scoring.

    Notes
    -----
    We deliberately preserve case and punctuation. VADER treats
    ALL-CAPS and "!!!" as intensity boosters, so stripping them would
    silently degrade sentiment quality.
    """
    if not text:
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


def join_title_text(title: str, text: str) -> str:
    """Combine review title + body for scoring.

    Titles often carry a lot of polarity ("Terrible!", "Love it!") but are
    very short, so we join them onto the body so VADER sees both. The
    duplicate punctuation is intentional -- VADER reads it as emphasis.
    """
    title = clean_text(title or "")
    text = clean_text(text or "")
    if title and text:
        return f"{title}. {text}"
    return title or text
