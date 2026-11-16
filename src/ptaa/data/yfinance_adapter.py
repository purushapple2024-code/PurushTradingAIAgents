"""yfinance adapter: free, 15-minute delayed, long daily history. Default when no broker keys exist.

Also provides a polling MarketFeed so the fast path runs end to end without a broker account;
it is honest about being delayed (`Tick.source == "yfinance-delayed"`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ptaa.data.adapter import (
    BrokerAdapter,
    CorporateAction,
    FeedMode,
    Fundamentals,
    validate_candles,
)
from ptaa.data.cache import DataCache
from ptaa.data.universe import sector_of, to_yf
from ptaa.models import Interval, Tick

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_YF_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}


def _yf() -> Any:
    import yfinance as yf  # lazy: tests without network still import this module

    return yf


class YFinanceAdapter(BrokerAdapter):
    name = "yfinance"

    def __init__(self, cache: DataCache | None = None):
        self.cache = cache

    def candles(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> pd.DataFrame:
        params = {"interval": interval, "start": start.date(), "end": end.date()}
        if self.cache:
            hit = self.cache.get_frame(self.name, symbol, params)
            if hit is not None:
                return hit
        df = _yf().download(
            to_yf(symbol),
            start=start,
            end=end,
            interval=_YF_INTERVAL[interval],
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        if df is None or df.empty:
            return validate_candles(pd.DataFrame())
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        out = validate_candles(df)
        if self.cache:
            self.cache.put_frame(self.name, symbol, params, out)
        return out

    def fundamentals(self, symbol: str) -> Fundamentals:
        params = {"kind": "fundamentals"}
        if self.cache:
            hit = self.cache.get_json(self.name, symbol, params)
            if hit:
                return Fundamentals.model_validate(hit)
        info: dict[str, Any] = _yf().Ticker(to_yf(symbol)).info or {}
        mcap = info.get("marketCap")
        f = Fundamentals(
            symbol=symbol,
            market_cap_cr=(mcap / 1e7) if mcap else None,
            pe_ttm=info.get("trailingPE"),
            pb=info.get("priceToBook"),
            roe_pct=(info["returnOnEquity"] * 100)
            if info.get("returnOnEquity") is not None
            else None,
            debt_to_equity=info.get("debtToEquity"),
            sector=info.get("sector") or sector_of(symbol),
            as_of=datetime.now(tz=IST),
        )
        if self.cache:
            self.cache.put_json(self.name, symbol, params, f.model_dump())
        return f

    def corporate_actions(self, symbol: str) -> list[CorporateAction]:
        t = _yf().Ticker(to_yf(symbol))
        out: list[CorporateAction] = []
        try:
            for ts, amt in t.dividends.items():
                out.append(
                    CorporateAction(
                        symbol=symbol, kind="dividend", ex_date=ts.to_pydatetime(), detail=f"{amt}"
                    )
                )
            for ts, ratio in t.splits.items():
                out.append(
                    CorporateAction(
                        symbol=symbol, kind="split", ex_date=ts.to_pydatetime(), detail=f"{ratio}"
                    )
                )
        except Exception as e:  # noqa: BLE001 - best effort
            log.debug("ignored: %s", e)
        return out


class YFinancePollingFeed:
    """Polls last price every `poll_s` seconds. Delayed data; for development and fallback only."""

    name = "yfinance-delayed"

    def __init__(self, poll_s: float = 5.0):
        self.poll_s = poll_s
        self._closed = False

    async def subscribe(self, symbols: list[str], mode: FeedMode = "ltp") -> AsyncIterator[Tick]:
        yf = _yf()
        tickers = {s: yf.Ticker(to_yf(s)) for s in symbols}
        while not self._closed:
            now = datetime.now(tz=timezone.utc).astimezone(IST)
            for s, t in tickers.items():
                try:
                    fi = t.fast_info
                    ltp = float(fi["last_price"])
                    vol = int(fi.get("last_volume") or 0)
                except Exception as e:  # noqa: BLE001 - skip this symbol this round
                    log.debug("poll skipped: %s", e)
                    continue
                yield Tick(symbol=s, ts=now, ltp=ltp, volume=vol, source=self.name)
            await asyncio.sleep(self.poll_s)

    async def close(self) -> None:
        self._closed = True
