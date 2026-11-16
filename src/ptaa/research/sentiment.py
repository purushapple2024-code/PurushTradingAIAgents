"""Financial sentiment: FinBERT (ProsusAI/finbert via transformers) when installed, else a
transparent Loughran-McDonald-style lexicon. Both return (p_neg, p_neu, p_pos) and a label.

FinBERT needs the `nlp` extra (transformers, torch). It is loaded lazily and once; on an 8 GB laptop
the int8/CPU path scores a headline in ~40 ms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

POSITIVE = {
    "beat",
    "beats",
    "surge",
    "surges",
    "surged",
    "jump",
    "jumps",
    "jumped",
    "rally",
    "rallies",
    "record",
    "upgrade",
    "upgraded",
    "outperform",
    "profit",
    "profits",
    "growth",
    "grew",
    "strong",
    "robust",
    "gain",
    "gains",
    "gained",
    "higher",
    "rise",
    "rises",
    "rose",
    "buyback",
    "dividend",
    "expansion",
    "approval",
    "approves",
    "approved",
    "wins",
    "won",
    "order",
    "orders",
    "bullish",
    "positive",
    "exceeds",
    "exceeded",
    "improves",
    "improved",
    "upbeat",
    "boost",
}
NEGATIVE = {
    "miss",
    "misses",
    "missed",
    "fall",
    "falls",
    "fell",
    "drop",
    "drops",
    "dropped",
    "plunge",
    "plunges",
    "plunged",
    "slump",
    "slumps",
    "downgrade",
    "downgraded",
    "underperform",
    "loss",
    "losses",
    "weak",
    "weaker",
    "decline",
    "declines",
    "declined",
    "lower",
    "cut",
    "cuts",
    "probe",
    "penalty",
    "fine",
    "fined",
    "fraud",
    "default",
    "lawsuit",
    "bearish",
    "negative",
    "warns",
    "warning",
    "concern",
    "concerns",
    "pressure",
    "slowdown",
    "halt",
    "halted",
    "recall",
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed"}


@dataclass
class Sentiment:
    p_neg: float
    p_neu: float
    p_pos: float
    label: str  # negative | neutral | positive
    engine: str  # finbert | lexicon

    @property
    def score(self) -> float:
        """-1..+1"""
        return self.p_pos - self.p_neg


def lexicon_sentiment(text: str) -> Sentiment:
    words = re.findall(r"[a-z']+", text.lower())
    pos = neg = 0
    for i, w in enumerate(words):
        negated = i > 0 and words[i - 1] in NEGATORS
        if w in POSITIVE:
            neg, pos = (neg + 1, pos) if negated else (neg, pos + 1)
        elif w in NEGATIVE:
            neg, pos = (neg, pos + 1) if negated else (neg + 1, pos)
    total = pos + neg
    if total == 0:
        return Sentiment(0.15, 0.70, 0.15, "neutral", "lexicon")
    p_pos = 0.15 + 0.7 * pos / total
    p_neg = 0.15 + 0.7 * neg / total
    p_neu = max(0.0, 1 - p_pos - p_neg)
    label = "positive" if p_pos > p_neg + 0.1 else "negative" if p_neg > p_pos + 0.1 else "neutral"
    return Sentiment(round(p_neg, 3), round(p_neu, 3), round(p_pos, 3), label, "lexicon")


@lru_cache(maxsize=1)
def _finbert() -> Any | None:
    try:
        from transformers import pipeline  # type: ignore[import-not-found]

        return pipeline(
            "text-classification", model="ProsusAI/finbert", top_k=None, truncation=True
        )
    except Exception as e:  # noqa: BLE001 - transformers/torch missing or model not downloadable
        log.info("FinBERT unavailable (%s); using lexicon sentiment", type(e).__name__)
        return None


def finbert_sentiment(text: str) -> Sentiment | None:
    clf = _finbert()
    if clf is None:
        return None
    out = clf(text[:512])[0]
    probs = {d["label"].lower(): float(d["score"]) for d in out}
    p_neg, p_neu, p_pos = (
        probs.get("negative", 0),
        probs.get("neutral", 0),
        probs.get("positive", 0),
    )
    label = max(probs, key=probs.get)  # type: ignore[arg-type]
    return Sentiment(round(p_neg, 3), round(p_neu, 3), round(p_pos, 3), label, "finbert")


def sentiment(text: str, prefer_finbert: bool = True) -> Sentiment:
    if prefer_finbert:
        s = finbert_sentiment(text)
        if s is not None:
            return s
    return lexicon_sentiment(text)
