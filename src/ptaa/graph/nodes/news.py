"""News analyst: discover -> fetch (allow-listed) -> extract -> sentiment -> Signal with quoted evidence.

Every Evidence carries the URL and a verbatim lead quote from the fetched article. The Verifier later
re-reads the cached page and drops any evidence whose quote is not in the text.
Fetching is injectable (`ctx.fetcher`) so tests and the Colab demo run offline on recorded pages.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from ptaa.graph.context import GraphContext, audited
from ptaa.graph.state import Evidence, GraphState, Signal
from ptaa.research.discover import NewsItem, discover
from ptaa.research.extract import extract_text, lead_sentences
from ptaa.research.sentiment import sentiment

log = logging.getLogger(__name__)


def _recency_weight(published: datetime | None, half_life_days: float = 3.0) -> float:
    if published is None:
        return 0.5
    age = max(0.0, (datetime.now(tz=timezone.utc) - published).total_seconds() / 86400)
    return float(0.5 ** (age / half_life_days))


@audited("news_analyst")
def news_analyst(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    allowed = ctx.cfg.research.allowed_domains
    items: list[NewsItem] = (
        ctx.news_items if ctx.news_items is not None else discover(state["symbol"], allowed)
    )
    if not items:
        return {}
    fetch = ctx.fetcher
    evidence: list[Evidence] = []
    scores: list[tuple[float, float]] = []  # (sentiment score, weight)
    for it in items[:15]:
        try:
            page = fetch(it.url)
        except Exception as e:  # noqa: BLE001 - one bad page must not sink the signal
            log.debug("fetch skipped %s: %s", it.url, e)
            continue
        text = extract_text(page.text) if page.text else ""
        if len(text) < 200:
            continue
        quote = lead_sentences(text)
        s = sentiment(f"{it.title}. {quote}")
        w = _recency_weight(it.published)
        scores.append((s.score, w))
        evidence.append(
            Evidence(
                claim=f"{it.title} [{s.label} {s.score:+.2f}, {s.engine}]",
                source_kind="web",
                source_url=it.url,
                fetched_at=datetime.fromtimestamp(page.fetched_at, tz=timezone.utc),
                quote=quote,
                attribution={"p_pos": s.p_pos, "p_neg": s.p_neg, "recency_w": round(w, 3)},
            )
        )
    if not evidence:
        return {}
    w = np.array([x[1] for x in scores])
    agg = float(np.average([x[0] for x in scores], weights=w)) if w.sum() > 0 else 0.0
    # map -1..1 sentiment to probabilities with a damped slope: news alone is a weak predictor
    p_up = float(np.clip(0.5 + 0.35 * agg, 0.05, 0.95))
    sig = Signal(
        signal_id="S3",
        analyst="news",
        horizon="5d",
        p_up=p_up,
        p_down=1 - p_up,
        evidence=evidence,
        summary=f"{len(evidence)} articles, weighted tone {agg:+.2f} ({evidence[0].claim.split('[')[-1].split(',')[-1].strip(' ]')})",
    )
    return {
        "signals": [sig],
        "news_cutoff": (datetime.now(tz=timezone.utc) - timedelta(days=10)).isoformat(),
    }
