from __future__ import annotations

from datetime import datetime, timedelta

from ptaa.fast.bars import BarBuilder, bar_start
from ptaa.models import Tick

from .conftest import IST, ts_at


def test_bar_start_alignment(session_start: datetime) -> None:
    t = session_start + timedelta(minutes=7, seconds=31)
    assert bar_start(t, "1m") == session_start + timedelta(minutes=7)
    assert bar_start(t, "5m") == session_start + timedelta(minutes=5)
    assert bar_start(t, "1h") == session_start
    assert bar_start(t, "1d") == session_start
    # 1h bars start at 09:15, 10:15, ...
    assert bar_start(
        session_start + timedelta(hours=1, minutes=20), "1h"
    ) == session_start + timedelta(hours=1)


def test_bar_start_before_open_clamps(session_start: datetime) -> None:
    assert bar_start(session_start - timedelta(minutes=10), "5m") == session_start


def test_builder_closes_bars_and_differences_volume(session_start: datetime) -> None:
    b = BarBuilder(["1m", "5m"])
    done = []
    b.on_bar(done.append)
    prices = [100.0, 101.0, 99.5, 100.5]
    cum_vol = [1000, 1500, 1800, 2600]
    for i, (p, v) in enumerate(zip(prices, cum_vol)):
        b.push(Tick(symbol="X", ts=ts_at(session_start, i * 15), ltp=p, volume=v, source="t"))
    assert done == []
    cur = b.current("X", "1m")
    assert cur is not None and (cur.open, cur.high, cur.low, cur.close) == (
        100.0,
        101.0,
        99.5,
        100.5,
    )
    assert (
        cur.volume == 1600
    )  # first tick's cumulative volume is a baseline, then deltas 500+300+800
    assert cur.ticks == 4

    # next minute closes the 1m bar, 5m stays open
    completed = b.push(
        Tick(symbol="X", ts=ts_at(session_start, 61), ltp=102.0, volume=3000, source="t")
    )
    assert [c.interval for c in completed] == ["1m"]
    assert completed[0].complete and completed[0].close == 100.5
    assert len(done) == 1
    assert b.current("X", "5m").high == 102.0  # type: ignore[union-attr]

    # jump past 5m boundary closes both
    completed = b.push(
        Tick(symbol="X", ts=ts_at(session_start, 305), ltp=103.0, volume=3100, source="t")
    )
    assert sorted(c.interval for c in completed) == ["1m", "5m"]


def test_builder_flush(session_start: datetime) -> None:
    b = BarBuilder(["1d"])
    b.push(Tick(symbol="A", ts=session_start, ltp=10.0, source="t"))
    b.push(Tick(symbol="B", ts=session_start, ltp=20.0, source="t"))
    out = b.flush()
    assert {o.symbol for o in out} == {"A", "B"} and all(o.complete for o in out)
    assert b.current("A", "1d") is None


def test_timezone_conversion(session_start: datetime) -> None:
    from datetime import timezone

    utc = session_start.astimezone(timezone.utc)
    assert bar_start(utc, "1m").tzinfo is not None
    assert bar_start(utc, "1m") == session_start
    assert (
        bar_start(utc, "1m").tzinfo == IST
        or bar_start(utc, "1m").utcoffset() == session_start.utcoffset()
    )
