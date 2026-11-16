"""Article text extraction (trafilatura, with a plain-HTML fallback) and quote verification."""

from __future__ import annotations

import logging
import re
from html import unescape

log = logging.getLogger(__name__)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def extract_text(html: str) -> str:
    try:
        import trafilatura

        text = trafilatura.extract(
            html, include_comments=False, include_tables=False, favor_precision=True
        )
        if text:
            return _WS.sub(" ", text).strip()
    except Exception as e:  # noqa: BLE001 - fall back to crude stripping
        log.debug("trafilatura failed: %s", e)
    stripped = _TAG.sub(" ", re.sub(r"(?is)<(script|style).*?</\1>", " ", html))
    return _WS.sub(" ", unescape(stripped)).strip()


def normalise(s: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s%₹$.,-]", " ", s.lower())).strip()


def quote_present(quote: str, text: str, min_overlap: float = 0.85) -> bool:
    """True if `quote` is in `text` (normalised) or >= `min_overlap` of its 6-grams are."""
    q, t = normalise(quote), normalise(text)
    if not q or not t:
        return False
    if q in t:
        return True
    words = q.split()
    if len(words) < 6:
        return False
    grams = [" ".join(words[i : i + 6]) for i in range(len(words) - 5)]
    hits = sum(1 for g in grams if g in t)
    return hits / len(grams) >= min_overlap


def lead_sentences(text: str, n: int = 2, max_chars: int = 320) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n])[:max_chars].strip()
