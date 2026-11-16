"""News discovery: DuckDuckGo news search (no key) + RSS feeds from allow-listed Indian outlets.

Returns candidate items (title, url, published, source). Only allow-listed hosts survive.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ptaa.research.http import host_allowed

log = logging.getLogger(__name__)

RSS_FEEDS: dict[str, str] = {
    "moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
    "economictimes": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint": "https://www.livemint.com/rss/markets",
    "business-standard": "https://www.business-standard.com/rss/markets-106.rss",
}

COMPANY_NAMES: dict[str, list[str]] = {
    "RELIANCE": ["Reliance Industries", "RIL"],
    "HDFCBANK": ["HDFC Bank"],
    "ICICIBANK": ["ICICI Bank"],
    "INFY": ["Infosys"],
    "TCS": ["Tata Consultancy", "TCS"],
    "SBIN": ["State Bank of India", "SBI"],
    "AXISBANK": ["Axis Bank"],
    "KOTAKBANK": ["Kotak Mahindra Bank", "Kotak Bank"],
    "BHARTIARTL": ["Bharti Airtel", "Airtel"],
    "LT": ["Larsen & Toubro", "L&T"],
    "TATAMOTORS": ["Tata Motors"],
    "BAJFINANCE": ["Bajaj Finance"],
    "ITC": ["ITC Ltd", "ITC"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL"],
    "TATASTEEL": ["Tata Steel"],
    "MARUTI": ["Maruti Suzuki"],
    "SUNPHARMA": ["Sun Pharma"],
    "TITAN": ["Titan Company"],
    "WIPRO": ["Wipro"],
    "HCLTECH": ["HCL Technologies", "HCLTech"],
}


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published: datetime | None
    snippet: str = ""


def names_for(symbol: str) -> list[str]:
    return COMPANY_NAMES.get(symbol.upper(), [symbol.upper()])


def ddg_news(
    symbol: str, allowed: list[str], max_results: int = 15, days: int = 10
) -> list[NewsItem]:
    try:
        from ddgs import DDGS
    except ImportError:
        log.warning("ddgs not installed; skipping web discovery")
        return []
    query = " OR ".join(f'"{n}"' for n in names_for(symbol)) + " stock"
    out: list[NewsItem] = []
    try:
        with DDGS() as d:
            for r in d.news(
                query, region="in-en", safesearch="off", timelimit="m", max_results=max_results * 2
            ):
                url = r.get("url") or ""
                if not host_allowed(url, allowed):
                    continue
                pub = None
                if r.get("date"):
                    try:
                        pub = datetime.fromisoformat(str(r["date"]).replace("Z", "+00:00"))
                    except ValueError:
                        pub = None
                if pub and pub < datetime.now(tz=timezone.utc) - timedelta(days=days):
                    continue
                out.append(
                    NewsItem(
                        title=r.get("title", ""),
                        url=url,
                        source=r.get("source", ""),
                        published=pub,
                        snippet=r.get("body", ""),
                    )
                )
                if len(out) >= max_results:
                    break
    except Exception as e:  # noqa: BLE001 - discovery is best effort
        log.warning("ddg news failed: %s", e)
    return out


def rss_news(symbol: str, allowed: list[str], max_results: int = 15) -> list[NewsItem]:
    try:
        import feedparser
    except ImportError:
        return []
    names = [n.lower() for n in names_for(symbol)]
    out: list[NewsItem] = []
    for src, feed_url in RSS_FEEDS.items():
        try:
            feed: Any = feedparser.parse(feed_url)
        except Exception as e:  # noqa: BLE001
            log.debug("rss %s failed: %s", src, e)
            continue
        for e in feed.get("entries", []):
            text = f"{e.get('title', '')} {e.get('summary', '')}".lower()
            if not any(n in text for n in names):
                continue
            url = e.get("link", "")
            if not host_allowed(url, allowed):
                continue
            pub = None
            if e.get("published_parsed"):
                pub = datetime.fromtimestamp(time.mktime(e["published_parsed"]), tz=timezone.utc)
            out.append(
                NewsItem(
                    title=e.get("title", ""),
                    url=url,
                    source=src,
                    published=pub,
                    snippet=e.get("summary", ""),
                )
            )
    out.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return out[:max_results]


def discover(symbol: str, allowed: list[str], max_results: int = 15) -> list[NewsItem]:
    seen: set[str] = set()
    items: list[NewsItem] = []
    for it in rss_news(symbol, allowed, max_results) + ddg_news(symbol, allowed, max_results):
        if it.url in seen:
            continue
        seen.add(it.url)
        items.append(it)
    return items[:max_results]
