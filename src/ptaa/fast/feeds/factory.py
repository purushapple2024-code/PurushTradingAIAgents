"""Build the configured feed chain: primary -> secondary -> fallback.

Feeds without credentials are skipped; yfinance polling is always the last resort.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ptaa.data.adapter import MarketFeed
from ptaa.data.yfinance_adapter import YFinancePollingFeed
from ptaa.fast.feeds.failover import FailoverFeed
from ptaa.settings import AppConfig, Settings

log = logging.getLogger(__name__)


def build_feed(settings: Settings, cfg: AppConfig) -> FailoverFeed:
    cache_dir = Path(settings.data_dir) / "ref"
    chain: list[MarketFeed] = []
    for name in dict.fromkeys([cfg.feeds.primary, cfg.feeds.secondary, cfg.feeds.fallback]):
        if name == "angelone":
            if settings.has_angel:
                from ptaa.fast.feeds.angelone_ws import AngelOneFeed

                chain.append(AngelOneFeed(settings, cache_dir))
            else:
                log.info("feed angelone skipped: no credentials")
        elif name == "upstox":
            if settings.upstox_access_token:
                from ptaa.fast.feeds.upstox_ws import UpstoxFeed

                chain.append(UpstoxFeed(settings, cache_dir))
            else:
                log.info("feed upstox skipped: no access token")
        elif name == "yfinance":
            chain.append(YFinancePollingFeed())
    if not chain:
        chain.append(YFinancePollingFeed())
    log.info("feed chain: %s", " -> ".join(f.name for f in chain))
    return FailoverFeed(chain, failover_after_ms=cfg.feeds.failover_after_ms)
