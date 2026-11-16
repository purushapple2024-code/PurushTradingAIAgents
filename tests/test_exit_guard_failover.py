from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from datetime import time as dtime

import pytest

from ptaa.data.adapter import FeedMode
from ptaa.fast.exit_guard import ExitGuard
from ptaa.fast.feeds.failover import FailoverFeed
from ptaa.models import Position, Tick

from .conftest import IST


def _pos(**kw: object) -> Position:
    base = dict(
        symbol="LT",
        module="swing",
        side="BUY",
        quantity=14,
        avg_price=3588.2,
        stop_loss=3512.0,
        target=3700.0,
        opened_at=datetime(2026, 11, 16, 10, 0, tzinfo=IST),
    )
    base.update(kw)
    return Position(**base)  # type: ignore[arg-type]


def _tick(px: float, t: datetime | None = None, sym: str = "LT") -> Tick:
    return Tick(symbol=sym, ts=t or datetime(2026, 11, 16, 11, 0, tzinfo=IST), ltp=px, source="t")


def test_stop_and_target() -> None:
    g = ExitGuard()
    g.add(_pos())
    assert g.evaluate(_tick(3600.0)) is None
    sig = g.evaluate(_tick(3511.0))
    assert sig is not None and sig.reason == "stop" and sig.quantity == 14
    assert g.positions() == []
    g.add(_pos())
    assert g.evaluate(_tick(3701.0)).reason == "target"  # type: ignore[union-attr]


def test_trailing_stop_ratchets() -> None:
    g = ExitGuard()
    g.add(_pos(trailing_after_r=1.0))  # R = 76.2
    assert g.evaluate(_tick(3660.0)) is None  # below +1R (3664.4)
    assert g.evaluate(_tick(3700.0 - 1)) is None  # above +1R; stop moves to 3699 - 76.2 = 3622.8
    pos = g.positions()[0]
    assert pos.stop_loss == pytest.approx(3622.8)
    sig = g.evaluate(_tick(3620.0))
    assert sig is not None and sig.reason == "trailing_stop"


def test_short_position() -> None:
    g = ExitGuard()
    g.add(_pos(side="SELL", avg_price=1000.0, stop_loss=1020.0, target=960.0))
    assert g.evaluate(_tick(1010.0)) is None
    assert g.evaluate(_tick(959.0)).reason == "target"  # type: ignore[union-attr]
    g.add(_pos(side="SELL", avg_price=1000.0, stop_loss=1020.0, target=960.0))
    assert g.evaluate(_tick(1021.0)).reason == "stop"  # type: ignore[union-attr]


def test_circuit_and_square_off() -> None:
    g = ExitGuard(square_off_ist=dtime(15, 10))
    g.add(_pos(module="intraday"))
    g.set_circuit_hits({"LT"})
    assert g.evaluate(_tick(3600.0)).reason == "circuit"  # type: ignore[union-attr]
    g.set_circuit_hits(set())
    g.add(_pos(module="intraday"))
    assert g.evaluate(_tick(3600.0, datetime(2026, 11, 16, 15, 9, tzinfo=IST))) is None
    assert (
        g.evaluate(_tick(3600.0, datetime(2026, 11, 16, 15, 10, tzinfo=IST))).reason == "square_off"
    )  # type: ignore[union-attr]
    g.add(_pos(module="swing"))
    assert (
        g.evaluate(_tick(3600.0, datetime(2026, 11, 16, 15, 20, tzinfo=IST))) is None
    )  # swing never squares off


def test_exit_guard_latency_budget() -> None:
    """p99 under 50 µs per tick for 50 positions on this machine; the live budget is 5 ms end to end."""
    g = ExitGuard()
    syms = [f"S{i}" for i in range(50)]
    for s in syms:
        g.add(_pos(symbol=s, stop_loss=1.0, target=1e9))
    t0 = datetime(2026, 11, 16, 11, 0, tzinfo=IST)
    ticks = [
        _tick(3600.0 + (i % 7), t0 + timedelta(seconds=i), syms[i % 50]) for i in range(20_000)
    ]
    lat = []
    for tk in ticks:
        a = time.perf_counter_ns()
        g.evaluate(tk)
        lat.append(time.perf_counter_ns() - a)
    lat.sort()
    assert lat[int(len(lat) * 0.99)] < 50_000, f"p99 {lat[int(len(lat) * 0.99)] / 1e3:.1f} µs"


class _FakeFeed:
    def __init__(
        self,
        name: str,
        ticks: int,
        fail_after: int | None = None,
        delay: float = 0.0,
        silent_after: int | None = None,
    ):
        self.name, self.ticks, self.fail_after, self.delay, self.silent_after = (
            name,
            ticks,
            fail_after,
            delay,
            silent_after,
        )
        self.closed = False

    async def subscribe(self, symbols: list[str], mode: FeedMode = "ltp") -> AsyncIterator[Tick]:
        for i in range(self.ticks):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("boom")
            if self.silent_after is not None and i >= self.silent_after:
                await asyncio.sleep(10)
            if self.delay:
                await asyncio.sleep(self.delay)
            yield Tick(symbol=symbols[0], ts=datetime.now(tz=IST), ltp=100 + i, source=self.name)

    async def close(self) -> None:
        self.closed = True


async def _collect(feed: FailoverFeed, n: int) -> list[Tick]:
    out: list[Tick] = []
    async for t in feed.subscribe(["X"]):
        out.append(t)
        if len(out) >= n:
            break
    await feed.close()
    return out


def test_failover_on_error() -> None:
    primary = _FakeFeed("p", ticks=10, fail_after=3)
    secondary = _FakeFeed("s", ticks=100)
    ff = FailoverFeed([primary, secondary], failover_after_ms=500)  # type: ignore[list-item]
    out = asyncio.run(_collect(ff, 8))
    assert [t.source for t in out[:3]] == ["p", "p", "p"]
    assert all(t.source == "s" for t in out[3:])
    assert ff.switches >= 1 and ff.active.name == "s"


def test_failover_on_silence() -> None:
    primary = _FakeFeed("p", ticks=10, silent_after=2)
    secondary = _FakeFeed("s", ticks=100)
    ff = FailoverFeed([primary, secondary], failover_after_ms=100)  # type: ignore[list-item]
    out = asyncio.run(_collect(ff, 6))
    assert [t.source for t in out[:2]] == ["p", "p"] and all(t.source == "s" for t in out[2:])
