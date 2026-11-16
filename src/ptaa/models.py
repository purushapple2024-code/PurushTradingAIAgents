"""Core data models shared by the fast path, the research graph and the ledger."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Interval = Literal["1m", "5m", "15m", "1h", "1d"]
Side = Literal["BUY", "SELL"]
Module = Literal["swing", "intraday"]

INTERVAL_SECONDS: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}


class Tick(BaseModel):
    """One market data update. `ts` is exchange time in IST, timezone-aware."""

    symbol: str
    ts: datetime
    ltp: float = Field(gt=0)
    volume: int = Field(default=0, ge=0)  # cumulative day volume when the feed provides it
    bid: float | None = None
    ask: float | None = None
    source: str = "unknown"


class Bar(BaseModel):
    symbol: str
    interval: Interval
    start: datetime  # bar open time, IST
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    ticks: int = 0
    complete: bool = False

    @property
    def end(self) -> datetime:
        from datetime import timedelta

        return self.start + timedelta(seconds=INTERVAL_SECONDS[self.interval])


class Position(BaseModel):
    symbol: str
    module: Module
    side: Side
    quantity: int = Field(gt=0)
    avg_price: float = Field(gt=0)
    stop_loss: float
    target: float
    trailing_after_r: float | None = None
    trailed: bool = False  # set once the stop has been ratcheted by the trailing rule
    initial_risk: float | None = None  # |entry - initial stop|, fixed at entry for trailing math
    highest_since_entry: float | None = None
    lowest_since_entry: float | None = None
    opened_at: datetime
    strategy: str = "manual"


class ExitSignal(BaseModel):
    symbol: str
    module: Module
    reason: Literal["stop", "target", "trailing_stop", "circuit", "square_off", "manual"]
    price: float
    ts: datetime
    quantity: int
