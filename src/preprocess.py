"""Light text cleaning for VADER scoring."""
from __future__ import annotations

import re

# Compiled once at import time for performance.
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTI_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML, URLs, and extra whitespace — but leave case and punctuation intact.

    VADER treats ALL-CAPS and '!!!' as intensity signals, so stripping them
    would silently degrade sentiment quality.
    """
    if not text:
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


def join_title_text(title: str, text: str) -> str:
    """Combine review title and body so VADER sees both polarity signals."""
    title = clean_text(title or "")
    text = clean_text(text or "")
    if title and text:
        return f"{title}. {text}"
    return title or text
