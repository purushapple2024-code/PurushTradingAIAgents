"""Bar builder: aggregates ticks into exchange-clock-aligned bars for several intervals at once.

Bars close on the exchange clock (09:15, 09:16, ... for 1m; 09:15, 09:20 for 5m; the daily bar
spans the session). A tick that arrives after a bar's end closes that bar and starts the next.
Cumulative day volume from the feed is converted to per-bar volume by differencing.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from ptaa.models import INTERVAL_SECONDS, Bar, Interval, Tick

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def bar_start(ts: datetime, interval: Interval) -> datetime:
    """Align a timestamp to the start of its bar on the exchange clock."""
    ts = ts.astimezone(IST)
    if interval == "1d":
        return ts.replace(
            hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute, second=0, microsecond=0
        )
    secs = INTERVAL_SECONDS[interval]
    open_dt = ts.replace(
        hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute, second=0, microsecond=0
    )
    elapsed = int((ts - open_dt).total_seconds())
    if elapsed < 0:
        return open_dt
    return open_dt + timedelta(seconds=(elapsed // secs) * secs)


class BarBuilder:
    """Feed ticks in; receive completed bars through `on_bar` callbacks."""

    def __init__(self, intervals: list[Interval]):
        self.intervals = intervals
        self._open: dict[tuple[str, str], Bar] = {}
        self._last_cum_vol: dict[str, int] = {}
        self._callbacks: list[Callable[[Bar], None]] = []

    def on_bar(self, cb: Callable[[Bar], None]) -> None:
        self._callbacks.append(cb)

    def current(self, symbol: str, interval: Interval) -> Bar | None:
        return self._open.get((symbol, interval))

    def push(self, tick: Tick) -> list[Bar]:
        """Process one tick; returns any bars completed by it."""
        completed: list[Bar] = []
        prev_cum = self._last_cum_vol.get(tick.symbol)
        delta_vol = 0
        if tick.volume:
            delta_vol = max(0, tick.volume - prev_cum) if prev_cum is not None else 0
            self._last_cum_vol[tick.symbol] = tick.volume

        for iv in self.intervals:
            key = (tick.symbol, iv)
            start = bar_start(tick.ts, iv)
            bar = self._open.get(key)
            if bar is not None and start > bar.start:
                bar.complete = True
                completed.append(bar)
                bar = None
            if bar is None:
                bar = Bar(
                    symbol=tick.symbol,
                    interval=iv,
                    start=start,
                    open=tick.ltp,
                    high=tick.ltp,
                    low=tick.ltp,
                    close=tick.ltp,
                    volume=delta_vol,
                    ticks=1,
                )
                self._open[key] = bar
            else:
                bar.high = max(bar.high, tick.ltp)
                bar.low = min(bar.low, tick.ltp)
                bar.close = tick.ltp
                bar.volume += delta_vol
                bar.ticks += 1
        for b in completed:
            for cb in self._callbacks:
                cb(b)
        return completed

    def flush(self, symbol: str | None = None) -> list[Bar]:
        """Close all open bars (end of session)."""
        out: list[Bar] = []
        for key in list(self._open):
            if symbol is None or key[0] == symbol:
                b = self._open.pop(key)
                b.complete = True
                out.append(b)
                for cb in self._callbacks:
                    cb(b)
        return out
