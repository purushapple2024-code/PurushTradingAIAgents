from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ptaa.graph.graph import build_graph, run_once
from ptaa.research.discover import NewsItem
from ptaa.research.extract import extract_text, lead_sentences, quote_present
from ptaa.research.http import DomainNotAllowedError, Fetched, GuardedHTTP, host_allowed
from ptaa.research.sentiment import lexicon_sentiment
from ptaa.settings import AppConfig

from .test_graph import _ctx, _snapshot

ARTICLE_POS = """<html><head><title>x</title><script>var a=1;</script></head><body><nav>menu</nav>
<article><p>Reliance Industries reported a strong quarter as retail profit surged 18% and the board approved
a record dividend, beating analyst estimates comfortably. The company said growth in new energy remains robust.</p>
<p>Shares rose 2% in early trade. Analysts upgraded the stock after the results, citing improved margins and
higher cash flows across segments over the coming year.</p></article></body></html>"""

ARTICLE_NEG = """<html><body><article><p>Reliance Industries fell sharply after a brokerage downgraded the stock and cut
its refining margin estimates, warning of weak demand and pressure on profits in the coming quarters. Concerns
about a slowdown weighed on sentiment and the shares declined 3% in heavy volume during the session.</p></article></body></html>"""


def test_host_allowed_and_guarded_http_refuses(tmp_path: Path) -> None:
    allowed = ["moneycontrol.com", "nseindia.com"]
    assert host_allowed("https://www.moneycontrol.com/news/x", allowed)
    assert host_allowed("https://nseindia.com/api", allowed)
    assert not host_allowed("https://evil.com/moneycontrol.com", allowed)
    assert not host_allowed("https://moneycontrol.com.evil.com/x", allowed)
    http = GuardedHTTP(allowed, tmp_path)
    with pytest.raises(DomainNotAllowedError):
        http.get("https://example.com/anything")


def test_extract_and_quote_present() -> None:
    text = extract_text(ARTICLE_POS)
    assert (
        "menu" not in text.lower() or "surged" in text
    )  # nav stripped by trafilatura or fallback keeps body
    assert "var a=1" not in text
    quote = lead_sentences(text)
    assert quote_present(quote, text)
    assert not quote_present(
        "The company announced a merger with a rival worth ten billion dollars today.", text
    )
    assert quote_present(quote.upper(), text)  # normalisation


def test_lexicon_sentiment() -> None:
    pos = lexicon_sentiment(
        "Profit surged and the board approved a record dividend; analysts upgraded the stock"
    )
    neg = lexicon_sentiment(
        "Brokerage downgraded the stock on weak demand and cut estimates amid losses"
    )
    neu = lexicon_sentiment("The company will hold its annual general meeting on Tuesday")
    assert pos.label == "positive" and pos.score > 0.3
    assert neg.label == "negative" and neg.score < -0.3
    assert neu.label == "neutral" and abs(neu.score) < 0.1
    flipped = lexicon_sentiment("Profit did not rise and growth failed to improve")
    assert flipped.score < pos.score


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls = 0

    def __call__(self, url: str) -> Fetched:
        self.calls += 1
        if url not in self.pages:
            raise RuntimeError("404")
        return Fetched(url=url, status=200, text=self.pages[url], fetched_at=time.time())


def test_news_signal_and_verifier_content_check(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.55)
    now = datetime.now(tz=timezone.utc)
    ctx.news_items = [
        NewsItem(
            "Reliance retail profit surges, record dividend",
            "https://www.moneycontrol.com/a1",
            "moneycontrol",
            now,
        ),
        NewsItem(
            "Brokerage downgrades Reliance",
            "https://economictimes.indiatimes.com/a2",
            "economictimes",
            now,
        ),
        NewsItem("Missing page", "https://www.livemint.com/a3", "livemint", now),
    ]
    fetcher = FakeFetcher(
        {
            "https://www.moneycontrol.com/a1": ARTICLE_POS,
            "https://economictimes.indiatimes.com/a2": ARTICLE_NEG,
        }
    )
    ctx.set_fetcher(fetcher)
    out = run_once(build_graph(ctx), ctx, "RELIANCE", "swing", _snapshot())
    assert out["errors"] == [], out["errors"]
    news = [s for s in out["verified_signals"] if s.analyst == "news"]
    assert len(news) == 1 and len(news[0].evidence) == 2  # the 404 page produced no evidence
    assert all(e.source_url and e.quote for e in news[0].evidence)
    assert 0.3 < news[0].p_up < 0.7  # one positive, one negative article -> near neutral
    assert out["ensemble"].weights.get("news", 0) > 0

    # Fabricated quote: inject a signal whose evidence quote is not in the page -> verifier drops it
    from ptaa.graph.nodes.gates import verifier
    from ptaa.graph.state import Evidence, Signal

    fake = Signal(
        signal_id="S9",
        analyst="news",
        horizon="5d",
        p_up=0.9,
        p_down=0.1,
        evidence=[
            Evidence(
                claim="made up",
                source_kind="web",
                source_url="https://www.moneycontrol.com/a1",
                quote="Reliance will acquire a rival for ten billion dollars tomorrow morning",
                fetched_at=now,
            )
        ],
    )
    res = (
        verifier.__wrapped__(dict(out, signals=[fake]), ctx)
        if hasattr(verifier, "__wrapped__")
        else verifier(dict(out, signals=[fake]), ctx)
    )
    assert res["verified_signals"] == [] and len(res["rejected"]) == 1
