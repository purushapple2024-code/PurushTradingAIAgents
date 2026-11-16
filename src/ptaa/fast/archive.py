"""Tick archive: buffers ticks in memory and flushes them to daily Parquet files.

Grows the intraday training dataset every session; ~50 MB/day for 50 symbols.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ptaa.models import Tick


class TickArchive:
    def __init__(self, root: Path, flush_every: int = 5000):
        self.root = Path(root) / "ticks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buf: list[dict[str, object]] = []

    def append(self, tick: Tick) -> None:
        self._buf.append(
            {
                "symbol": tick.symbol,
                "ts": tick.ts,
                "ltp": tick.ltp,
                "volume": tick.volume,
                "bid": tick.bid,
                "ask": tick.ask,
                "source": tick.source,
            }
        )
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buf:
            return None
        df = pd.DataFrame(self._buf)
        self._buf = []
        day: date = pd.Timestamp(df["ts"].iloc[-1]).date()
        path = self.root / f"{day.isoformat()}.parquet"
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path, index=False)
        return path

    def load_day(self, day: date) -> pd.DataFrame:
        path = self.root / f"{day.isoformat()}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
