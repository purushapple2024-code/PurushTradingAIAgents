from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ptaa.control.service import ControlService
from ptaa.engine import LiveEngine
from ptaa.fast.feeds.simulated import SimulatedFeed
from ptaa.graph.nodes.data import daily_turnover_cr
from ptaa.settings import AppConfig
from ptaa.training.pipeline import synthetic_candles

from .test_graph import _ctx

IST = ZoneInfo("Asia/Kolkata")


def test_daily_turnover_handles_intraday_bars() -> None:
    daily = synthetic_candles("AAA", n=100)
    d = daily_turnover_cr(daily)
    # split each daily bar's volume across 75 five-minute bars: daily turnover must be ~unchanged
    import pandas as pd

    rows = []
    for ts, r in daily.tail(20).iterrows():
        for i in range(75):
            rows.append(
                {
                    "ts": ts + pd.Timedelta(minutes=5 * i),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume / 75,
                }
            )
    intra = pd.DataFrame(rows).set_index("ts")
    assert abs(daily_turnover_cr(intra) - daily_turnover_cr(daily.tail(20))) / d < 0.02


def test_engine_enters_and_exits_paper_positions(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.85)
    ctx.cfg.modules.intraday.llm_top_n_per_bar = 2
    control = ControlService(tmp_path / "control.db", cfg)
    control.set_engine(actor="t", intraday_running=True)
    syms = ["RELIANCE", "TCS", "INFY"]
    # seed with daily history so features exist from the first 5m bar; intraday bars append to it
    seed = {s: synthetic_candles(s, n=400) for s in syms}
    start = datetime(2026, 11, 17, 9, 15, tzinfo=IST)
    feed = SimulatedFeed(
        ticks_per_second=5000, realtime=False, start=start, sim_step_s=2.0, max_ticks=12_000, seed=3
    )
    eng = LiveEngine(ctx, control, feed, "intraday", syms, seed_candles=seed, top_n=2)
    asyncio.run(eng.run())
    st = eng.stats
    assert st["ticks"] == 12_000 and st["bars"] > 0 and st["decisions"] > 0
    runs = ctx.ledger.runs(500)
    assert any(r.decision == "BUY" for r in runs), "expected at least one paper BUY"
    orders_buy = sum(1 for r in runs if r.decision == "BUY")
    # the guard resolved entries by stop/target/square-off during the session or they remain open
    open_pos = ctx.ledger.positions()
    assert st["exits"] + len(open_pos) >= 1
    assert len(open_pos) <= cfg.risk["intraday"].max_open_positions
    # after 15:10 IST nothing stays open in intraday (square-off)
    last_ts = start.timestamp() + 12_000 * 2.0
    if datetime.fromtimestamp(last_ts, tz=IST).time() >= datetime(2026, 1, 1, 15, 10).time():
        assert open_pos == [], open_pos
    assert ctx.ledger.fees_paid() > 0 and orders_buy >= 1


def test_engine_respects_kill_switch(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.85)
    control = ControlService(tmp_path / "control.db", cfg)
    control.kill()
    seed = {"RELIANCE": synthetic_candles("RELIANCE", n=400)}
    feed = SimulatedFeed(
        ticks_per_second=5000,
        realtime=False,
        start=datetime(2026, 11, 17, 9, 15, tzinfo=IST),
        sim_step_s=5.0,
        max_ticks=1500,
        seed=1,
    )
    eng = LiveEngine(ctx, control, feed, "intraday", ["RELIANCE"], seed_candles=seed)
    asyncio.run(eng.run())
    assert eng.stats["decisions"] == 0 and eng.stats["skipped_paused"] > 0
    assert ctx.ledger.positions() == [] and ctx.ledger.runs() == []
