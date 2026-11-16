"""Simulated tick feed: seeded random walk per symbol at a configurable rate.

For offline demos, latency benchmarks and tests. Clearly labelled `source="simulated"`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from ptaa.data.adapter import FeedMode
from ptaa.models import Tick

IST = ZoneInfo("Asia/Kolkata")


class SimulatedFeed:
    name = "simulated"

    def __init__(
        self,
        ticks_per_second: float = 200.0,
        seed: int = 7,
        start: datetime | None = None,
        realtime: bool = True,
        max_ticks: int | None = None,
        sim_step_s: float | None = None,
    ):
        self.tps = ticks_per_second
        self.rng = np.random.default_rng(seed)
        self.start = start or datetime.now(tz=IST).replace(
            hour=9, minute=15, second=0, microsecond=0
        )
        self.realtime = realtime
        self.max_ticks = max_ticks
        self.sim_step_s = sim_step_s  # simulated seconds per tick; None = 1/tps (wall-clock pace)
        self._closed = False

    async def subscribe(self, symbols: list[str], mode: FeedMode = "ltp") -> AsyncIterator[Tick]:
        px = {s: float(self.rng.uniform(200, 4000)) for s in symbols}
        vol = dict.fromkeys(symbols, 0)
        n = 0
        t = self.start
        step = timedelta(seconds=self.sim_step_s if self.sim_step_s else 1.0 / self.tps)
        while not self._closed and (self.max_ticks is None or n < self.max_ticks):
            s = symbols[n % len(symbols)]
            px[s] *= float(np.exp(self.rng.normal(0, 0.0004)))
            vol[s] += int(self.rng.integers(1, 500))
            t += step
            yield Tick(symbol=s, ts=t, ltp=round(px[s], 2), volume=vol[s], source=self.name)
            n += 1
            if self.realtime:
                await asyncio.sleep(1.0 / self.tps)
            elif n % 1000 == 0:
                await asyncio.sleep(0)  # let other tasks run

    async def close(self) -> None:
        self._closed = True
