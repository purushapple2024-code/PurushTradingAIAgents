"""Live engine: fast path + research graph + exits + alerts, governed by the control plane.

Loop per tick (sub-millisecond, no models):
  feed -> BarBuilder -> ExitGuard (stop/target/trailing/circuit/square-off -> paper exit) -> archive
On each closed decision bar (5m intraday, 1d swing):
  build candles from accumulated bars -> market_snapshot -> rank symbols by a cheap pre-score ->
  run the research graph for the top-N -> paper fills through the Executor node.
Engine state (running/paused, kill switch, risk limits) is re-read from the control plane every bar.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ptaa.control.service import ControlService
from ptaa.data.adapter import MarketFeed
from ptaa.fast.archive import TickArchive
from ptaa.fast.bars import BarBuilder
from ptaa.fast.exit_guard import ExitGuard
from ptaa.graph.context import GraphContext
from ptaa.graph.graph import build_graph, run_once
from ptaa.graph.nodes.data import market_snapshot
from ptaa.models import Bar, ExitSignal, Interval, Module, Position, Tick

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
DECISION_BAR: dict[str, Interval] = {"intraday": "5m", "swing": "1d"}
MIN_BARS = {"intraday": 80, "swing": 260}  # enough history for the feature engine


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


class LiveEngine:
    def __init__(
        self,
        ctx: GraphContext,
        control: ControlService,
        feed: MarketFeed,
        module: Module,
        symbols: list[str],
        seed_candles: dict[str, pd.DataFrame] | None = None,
        archive: TickArchive | None = None,
        top_n: int | None = None,
    ):
        self.ctx, self.control, self.feed, self.module, self.symbols = (
            ctx,
            control,
            feed,
            module,
            symbols,
        )
        intervals: list[Interval] = ["1m", "5m", "1h", "1d"]
        self.builder = BarBuilder(intervals)
        sched = ctx.cfg.modules.intraday
        self.guard = ExitGuard(
            square_off_ist=_parse_hhmm(sched.square_off_ist) if module == "intraday" else None,
            on_exit=self._on_exit,
        )
        self.archive = archive
        self.graph = build_graph(ctx)
        self.top_n = (
            top_n
            if top_n is not None
            else (sched.llm_top_n_per_bar if module == "intraday" else len(symbols))
        )
        self.bars: dict[str, list[Bar]] = defaultdict(list)
        self.seed = seed_candles or {}
        self.decision_bar = DECISION_BAR[module]
        self.entry_window = tuple(_parse_hhmm(x) for x in sched.entries_window_ist)
        self.stats: dict[str, Any] = {
            "ticks": 0,
            "bars": 0,
            "decisions": 0,
            "exits": 0,
            "skipped_paused": 0,
            "p99_us": 0.0,
        }
        self._lat: list[int] = []
        self._pending: set[str] = set()
        self.builder.on_bar(self._on_bar)
        for p in ctx.ledger.positions():  # resume open paper positions into the guard
            self.guard.add(
                Position(
                    symbol=p.symbol,
                    module=p.module,
                    side=p.side,
                    quantity=p.quantity,
                    avg_price=p.avg_price,
                    stop_loss=p.stop_loss,
                    target=p.target,
                    trailing_after_r=1.0,
                    opened_at=p.opened_at,
                    strategy=p.strategy,
                )
            )  # type: ignore[arg-type]

    # -- fast path callbacks --------------------------------------------------
    def _on_exit(self, sig: ExitSignal) -> None:
        fill = self.ctx.broker.close_position(sig.symbol, sig.price, sig.ts, sig.reason)
        self.stats["exits"] += 1
        if fill and self.ctx.alerts is not None:
            self.ctx.alerts.send(
                "fill", f"EXIT {sig.reason} {sig.symbol} {sig.quantity} @ ₹{sig.price:,.2f} · PAPER"
            )
        log.info("exit %s %s %d @ %.2f", sig.reason, sig.symbol, sig.quantity, sig.price)

    def _on_bar(self, bar: Bar) -> None:
        self.stats["bars"] += 1
        if bar.interval == self.decision_bar:
            self.bars[bar.symbol].append(bar)
            self._pending.add(bar.symbol)

    # -- candles for the graph -----------------------------------------------
    def candles(self, symbol: str) -> pd.DataFrame:
        live = self.bars.get(symbol, [])
        df_live = pd.DataFrame(
            [
                {"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
                for b in live
            ],
            index=pd.DatetimeIndex([b.start for b in live]),
        )
        seed = self.seed.get(symbol)
        if seed is not None and not seed.empty:
            df = pd.concat([seed[~seed.index.isin(df_live.index)], df_live]).sort_index()
        else:
            df = df_live
        return df

    def _pre_score(self, symbol: str) -> float:
        """Cheap ranking before spending model/LLM time: recent 5-bar momentum scaled by range."""
        df = self.candles(symbol).tail(6)
        if len(df) < 6:
            return -1.0
        mom = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        rng = (df["high"] - df["low"]).mean() / df["close"].iloc[-1]
        return float(abs(mom) / (rng + 1e-9))

    def _can_enter(self, ts: datetime) -> bool:
        st = self.control.engine_state()
        running = st.intraday_running if self.module == "intraday" else st.swing_running
        if st.kill_switch or not running:
            self.stats["skipped_paused"] += 1
            return False
        if self.module == "intraday":
            t = ts.astimezone(IST).time()
            return self.entry_window[0] <= t <= self.entry_window[1]
        return True

    # -- decision cycle -------------------------------------------------------
    def decide(self, ts: datetime) -> list[str]:
        """Run the research graph for the top-N pending symbols. Returns run ids."""
        pending, self._pending = list(self._pending), set()
        if not pending or not self._can_enter(ts):
            return []
        self.ctx.cfg.risk[self.module] = self.control.risk_limits(self.module)  # live limits
        held = {p.symbol for p in self.ctx.ledger.positions()}
        ranked = sorted((s for s in pending if s not in held), key=self._pre_score, reverse=True)[
            : self.top_n
        ]
        runs = []
        for sym in ranked:
            df = self.candles(sym)
            if len(df) < MIN_BARS[self.module]:
                continue
            try:
                snap = market_snapshot(df)
                snap["as_of"] = ts
                out = run_once(self.graph, self.ctx, sym, self.module, snap)
                runs.append(out["run_id"])
                self.stats["decisions"] += 1
                d = out.get("decision")
                if d is not None and d.action in ("BUY", "SELL") and out.get("proposal"):
                    p = out["proposal"]
                    self.guard.add(
                        Position(
                            symbol=sym,
                            module=self.module,
                            side=p.side,
                            quantity=d.quantity,
                            avg_price=p.entry,
                            stop_loss=p.stop_loss,
                            target=p.target,
                            trailing_after_r=1.0,
                            opened_at=ts,
                            strategy=p.strategy,
                        )
                    )
            except Exception as e:  # noqa: BLE001 - one symbol must not stop the session
                log.exception("decision failed for %s: %s", sym, e)
        return runs

    def on_tick(self, tick: Tick) -> None:
        a = time.perf_counter_ns()
        self.builder.push(tick)
        self.guard.evaluate(tick)
        if self.archive is not None:
            self.archive.append(tick)
        self._lat.append(time.perf_counter_ns() - a)
        self.stats["ticks"] += 1
        if len(self._lat) >= 2000:
            self._lat.sort()
            self.stats["p99_us"] = self._lat[int(len(self._lat) * 0.99)] / 1e3
            self._lat = []
        if self._pending:
            self.decide(tick.ts)

    async def run(self, seconds: int = 0, on_status: Any | None = None) -> None:
        t0 = time.monotonic()
        try:
            async for tick in self.feed.subscribe(self.symbols, "ltp"):
                self.on_tick(tick)
                if on_status and self.stats["ticks"] % 1000 == 0:
                    on_status(self.stats)
                if seconds and time.monotonic() - t0 > seconds:
                    break
        finally:
            await self.feed.close()
            self.builder.flush()
            if self.archive is not None:
                self.archive.flush()
            prices = {s: b[-1].close for s, b in self.bars.items() if b}
            if prices:
                self.ctx.ledger.mark(prices)
                self.ctx.ledger.snapshot(prices)


def default_data_dir(ctx: GraphContext) -> Path:
    return Path(ctx.settings.data_dir)
