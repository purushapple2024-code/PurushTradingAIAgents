"""Protocols every data source implements. Downstream code depends only on these."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel

from ptaa.models import Interval, Tick

FeedMode = Literal["ltp", "quote", "depth"]


class Fundamentals(BaseModel):
    symbol: str
    market_cap_cr: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None
    roe_pct: float | None = None
    debt_to_equity: float | None = None
    promoter_holding_pct: float | None = None
    fii_holding_pct: float | None = None
    next_results_date: datetime | None = None
    sector: str | None = None
    as_of: datetime


class CorporateAction(BaseModel):
    symbol: str
    kind: Literal["dividend", "split", "bonus", "rights", "buyback", "other"]
    ex_date: datetime
    detail: str = ""


@runtime_checkable
class MarketFeed(Protocol):
    """Streaming source of ticks. Implementations must be cancellation-safe."""

    name: str

    async def subscribe(
        self, symbols: list[str], mode: FeedMode = "ltp"
    ) -> AsyncIterator[Tick]: ...

    async def close(self) -> None: ...


@runtime_checkable
class BrokerAdapter(Protocol):
    """Historical and reference data."""

    name: str

    def candles(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return a frame indexed by IST timestamps with open, high, low, close, volume."""
        ...

    def fundamentals(self, symbol: str) -> Fundamentals: ...

    def corporate_actions(self, symbol: str) -> list[CorporateAction]: ...


CANDLE_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a candle frame: tz-aware IST index, sorted, float OHLC, int volume, no NaN rows."""
    if df.empty:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    missing = [c for c in CANDLE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"candles missing columns: {missing}")
    out = out[CANDLE_COLUMNS]
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    out.index = idx
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].astype(float)
    out["volume"] = out["volume"].fillna(0).astype("int64")
    return out
