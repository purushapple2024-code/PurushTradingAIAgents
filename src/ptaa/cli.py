"""ptaa command line: data, features, universe, live, fees, settings, train, models."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ptaa import __version__
from ptaa.settings import get_settings, load_config

if TYPE_CHECKING:
    from ptaa.graph.context import GraphContext
    from ptaa.mlmodels.registry import ModelRegistry

UNIVERSE_OPT = typer.Option(None, help="NIFTY50 | NIFTY100 | ... (default: config swing universe)")

app = typer.Typer(
    help="PurushTradingAIAgents: zero-cost, ML-first paper trading for NSE. Not investment advice."
)
console = Console()
IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=False, show_path=False)],
)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
) -> None:
    if version:
        console.print(f"ptaa {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def settings() -> None:
    """Show effective settings with secrets masked."""
    s = get_settings()
    t = Table(title="Settings (secrets masked)")
    t.add_column("key")
    t.add_column("value")
    for k, v in s.safe_dict().items():
        t.add_row(k, str(v))
    console.print(t)
    console.print(
        f"feeds available: angelone={s.has_angel} upstox={bool(s.upstox_access_token)} yfinance=True"
    )


@app.command()
def universe(name: str = typer.Argument("NIFTY50"), config: Path = Path("config.yaml")) -> None:
    """List a universe (NIFTY50, NIFTYNEXT50, NIFTY100, NIFTY50_LIQUID15, custom)."""
    from ptaa.data.universe import resolve_universe, sector_of

    cfg = load_config(config)
    syms = resolve_universe(name, cfg.universe.custom)
    t = Table(title=f"{name} · {len(syms)} symbols")
    t.add_column("symbol")
    t.add_column("sector")
    for s in syms:
        t.add_row(s, sector_of(s))
    console.print(t)


@app.command()
def data(
    symbol: str,
    interval: str = typer.Option("1d", help="1m 5m 15m 1h 1d"),
    days: int = typer.Option(400, help="lookback in calendar days"),
    tail: int = typer.Option(10, help="rows to show"),
) -> None:
    """Fetch candles via the yfinance adapter (cached) and show the tail."""
    from ptaa.data.cache import DataCache
    from ptaa.data.yfinance_adapter import YFinanceAdapter

    s = get_settings()
    adapter = YFinanceAdapter(DataCache(Path(s.data_dir)))
    end = datetime.now(tz=IST)
    df = adapter.candles(symbol.upper(), interval, end - timedelta(days=days), end)  # type: ignore[arg-type]
    if df.empty:
        console.print(f"[red]no data for {symbol} ({interval})[/red]")
        raise typer.Exit(1)
    console.print(
        f"{symbol.upper()} {interval}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}"
    )
    console.print(df.tail(tail).to_string())


@app.command()
def features(symbol: str, interval: str = "1d", days: int = 600, top: int = 20) -> None:
    """Compute the deterministic feature row for the latest bar."""
    from ptaa.data.cache import DataCache
    from ptaa.data.yfinance_adapter import YFinanceAdapter
    from ptaa.fast.features import (
        FEATURE_NAMES,
        compute_features,
        feature_row_hash,
        feature_schema_hash,
    )

    s = get_settings()
    adapter = YFinanceAdapter(DataCache(Path(s.data_dir)))
    end = datetime.now(tz=IST)
    df = adapter.candles(symbol.upper(), interval, end - timedelta(days=days), end)  # type: ignore[arg-type]
    # NIFTY index series for relative strength is wired in milestone 3 (research graph).
    f = compute_features(df, None).dropna(how="all")
    row = f.iloc[-1]
    t = Table(
        title=f"{symbol.upper()} features @ {f.index[-1]} · schema {feature_schema_hash()} · row {feature_row_hash(row)}"
    )
    t.add_column("feature")
    t.add_column("value", justify="right")
    for name in FEATURE_NAMES[:top]:
        t.add_row(name, f"{row[name]:.5f}" if row[name] == row[name] else "nan")
    console.print(t)
    console.print(f"{len(FEATURE_NAMES)} features total")


@app.command()
def fees(price: float, quantity: int, module: str = "swing") -> None:
    """Show the fee breakdown for a round trip at one price."""
    from ptaa.risk.fees import compute_fees, round_trip_cost

    cfg = load_config()
    for side in ("BUY", "SELL"):
        fb = compute_fees(cfg.fees, side, module, price, quantity)  # type: ignore[arg-type]
        console.print(f"{side}: {fb.model_dump()} total ₹{fb.total}")
    rt = round_trip_cost(cfg.fees, module, price, quantity)  # type: ignore[arg-type]
    console.print(
        f"round trip ₹{rt:.2f} = {rt / (price * quantity) * 100:.4f}% of one side's turnover"
    )


@app.command()
def live(
    module: str = typer.Option("intraday", help="swing | intraday (sets universe and square-off)"),
    symbols: str | None = typer.Option(None, help="comma-separated override"),  # noqa: UP007
    seconds: int = typer.Option(0, help="stop after N seconds (0 = run until Ctrl+C)"),
    simulate: bool = typer.Option(
        False, help="use a seeded simulated feed (offline demo / benchmark)"
    ),
    tps: float = typer.Option(200.0, help="simulated ticks per second"),
    sim_step: float = typer.Option(
        0.0, help="simulated seconds per tick (0 = wall clock); e.g. 2 replays a day fast"
    ),
    trade: bool = typer.Option(
        False, help="also run the research graph on closed decision bars and paper-trade"
    ),
    config: Path = Path("config.yaml"),
) -> None:
    """Fast path: feed → bars → features → exit guard → archive. With --trade: decide + paper-fill."""
    from ptaa.data.universe import resolve_universe
    from ptaa.fast.archive import TickArchive
    from ptaa.fast.bars import BarBuilder
    from ptaa.fast.exit_guard import ExitGuard
    from ptaa.fast.feeds.factory import build_feed
    from ptaa.fast.feeds.failover import FailoverFeed
    from ptaa.models import Bar

    s = get_settings()
    cfg = load_config(config)
    uni = cfg.universe.intraday if module == "intraday" else cfg.universe.swing
    syms = (
        [x.strip().upper() for x in symbols.split(",")]
        if symbols
        else resolve_universe(uni, cfg.universe.custom)
    )
    if simulate:
        from ptaa.fast.feeds.simulated import SimulatedFeed

        feed = FailoverFeed(
            [SimulatedFeed(ticks_per_second=tps, sim_step_s=sim_step or None)],
            failover_after_ms=cfg.feeds.failover_after_ms,
        )
    else:
        feed = build_feed(s, cfg)
    archive = TickArchive(Path(s.data_dir)) if cfg.feeds.archive_ticks else None
    if trade:
        from ptaa.control.service import ControlService
        from ptaa.engine import LiveEngine
        from ptaa.training.pipeline import synthetic_candles, yfinance_candle_source

        ctx = _context(config)
        if simulate:
            ctx.news_items = []
        control = ControlService(Path(s.data_dir) / "control.db", cfg)
        control.set_engine(actor="cli", **{f"{module}_running": True, "kill_switch": False})
        source = synthetic_candles if simulate else yfinance_candle_source(Path(s.data_dir))
        seed = {sym: source(sym) for sym in syms} if module == "swing" or simulate else {}
        engine = LiveEngine(ctx, control, feed, module, syms, seed_candles=seed, archive=archive)  # type: ignore[arg-type]
        console.print(
            f"live+trade · {module} · {len(syms)} symbols · top-{engine.top_n} per bar · feed {' → '.join(f.name for f in feed.feeds)} · PAPER"
        )

        def status(st: dict) -> None:
            console.print(
                f"[dim]ticks {st['ticks']} bars {st['bars']} decisions {st['decisions']} exits {st['exits']} paused-skips {st['skipped_paused']} p99 {st['p99_us']:.0f} µs[/dim]"
            )

        try:
            asyncio.run(engine.run(seconds=seconds, on_status=status))
        except KeyboardInterrupt:
            pass
        console.print(f"done · {engine.stats}")
        return
    builder = BarBuilder(cfg.bars["intervals"])  # type: ignore[arg-type]
    guard = ExitGuard()

    def on_bar(b: Bar) -> None:
        console.print(
            f"[dim]{b.start:%H:%M}[/dim] {b.symbol:<12} {b.interval:<3} o={b.open:.2f} h={b.high:.2f} l={b.low:.2f} c={b.close:.2f} v={b.volume} n={b.ticks}"
        )

    builder.on_bar(on_bar)
    console.print(
        f"live · {module} · {len(syms)} symbols · feed chain {' → '.join(f.name for f in feed.feeds)} · PAPER"
    )

    async def run() -> None:
        n, t0, lat = 0, time.monotonic(), []
        try:
            async for tick in feed.subscribe(syms, "ltp"):
                a = time.perf_counter_ns()
                builder.push(tick)
                guard.evaluate(tick)
                if archive:
                    archive.append(tick)
                lat.append(time.perf_counter_ns() - a)
                n += 1
                if n % 500 == 0:
                    lat.sort()
                    console.print(
                        f"[dim]{n} ticks · p50 {lat[len(lat) // 2] / 1e3:.0f} µs · p99 {lat[int(len(lat) * 0.99)] / 1e3:.0f} µs · feed {feed.active.name}[/dim]"
                    )
                    lat = []
                if seconds and time.monotonic() - t0 > seconds:
                    break
        finally:
            await feed.close()
            builder.flush()
            if archive:
                archive.flush()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("stopped")


# ----------------------------------------------------------------------------- training

train_app = typer.Typer(
    help="Train and register models (walk-forward, purged, out-of-fold metrics)."
)
models_app = typer.Typer(help="Model registry.")
app.add_typer(train_app, name="train")
app.add_typer(models_app, name="models")


def _registry() -> ModelRegistry:
    from ptaa.mlmodels.registry import ModelRegistry

    return ModelRegistry(Path(get_settings().data_dir) / "models")


@train_app.command("gbm")
def train_gbm_cmd(
    horizon: str = typer.Option("5d", help="5d (swing default) | 1d"),
    universe: str | None = UNIVERSE_OPT,
    synthetic: bool = typer.Option(False, help="use seeded synthetic candles (offline demo / CI)"),
    folds: int = 5,
    min_train: int = typer.Option(2000, help="minimum panel rows before the first test fold"),
    years: int = typer.Option(12, help="history to download with yfinance"),
    config: Path = Path("config.yaml"),
) -> None:
    """Train the technical LightGBM analyst and print out-of-fold metrics plus a fee-aware backtest."""
    from ptaa.training.pipeline import run_training, synthetic_candles, yfinance_candle_source

    cfg = load_config(config)
    s = get_settings()
    source = (
        synthetic_candles if synthetic else yfinance_candle_source(Path(s.data_dir), years=years)
    )
    index_close = None
    if not synthetic:
        try:
            from ptaa.data.yfinance_adapter import _yf

            end = datetime.now(tz=IST)
            raw = _yf().download(
                "^NSEI",
                start=end - timedelta(days=365 * years),
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=False,
                multi_level_index=False,
            )
            if raw is not None and not raw.empty:
                index_close = raw["Close"].rename("close")
                index_close.index = pd.to_datetime(index_close.index).tz_localize("Asia/Kolkata")
        except Exception as e:  # noqa: BLE001
            console.print(
                f"[yellow]NIFTY index unavailable ({e}); baseline will be the panel average[/yellow]"
            )
    rep = run_training(
        cfg,
        _registry(),
        source,
        horizon=horizon,
        universe=universe,
        index_close=index_close,
        n_folds=folds,
        min_train=min_train,
    )
    t = Table(title=f"{rep.model_id} · out-of-fold metrics")
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k, v in rep.metrics.items():
        t.add_row(k, f"{v:.4f}")
    console.print(t)
    t2 = Table(title="Backtest (fees + slippage) vs baseline")
    t2.add_column("metric")
    t2.add_column("strategy", justify="right")
    t2.add_column("baseline", justify="right")
    for k in ("sharpe", "total_return", "max_drawdown", "win_rate", "n"):
        t2.add_row(
            k,
            f"{rep.strategy.get(k, float('nan')):.4f}",
            f"{rep.baseline.get(k, float('nan')):.4f}",
        )
    console.print(t2)
    console.print(rep.sweep.to_string(index=False))
    console.print(f"report: {rep.report_path}")
    console.print("[dim]Out-of-fold, fee-aware. Not investment advice.[/dim]")


@train_app.command("seq")
def train_seq_cmd(
    horizon: int = typer.Option(5, help="bars ahead"),
    universe: str | None = UNIVERSE_OPT,
    synthetic: bool = False,
    epochs: int = 30,
    config: Path = Path("config.yaml"),
) -> None:
    """Train the LSTM sequence analyst (needs PyTorch; use Colab/Kaggle) and export to ONNX."""
    from ptaa.data.universe import resolve_universe
    from ptaa.training.pipeline import synthetic_candles, yfinance_candle_source
    from ptaa.training.train_seq import train_seq

    cfg = load_config(config)
    s = get_settings()
    uni = universe or cfg.universe.swing
    source = synthetic_candles if synthetic else yfinance_candle_source(Path(s.data_dir))
    try:
        res = train_seq(
            resolve_universe(uni, cfg.universe.custom),
            source,
            _registry(),
            horizon=horizon,
            universe=uni,
            epochs=epochs,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    console.print(res)


@train_app.command("ppo")
def train_ppo_cmd(timesteps: int = 200_000) -> None:
    """Train the PPO position sizer (stable-baselines3 + torch). Use the Colab notebook; see docs."""
    console.print(
        "PPO training needs stable-baselines3 and torch: open training/notebooks/train_ppo_colab.ipynb. "
        f"(timesteps={timesteps}). The exported ONNX policy is picked up automatically once registered "
        "and only used when its out-of-sample run beat fixed-fraction sizing."
    )


@models_app.command("list")
def models_list() -> None:
    reg = _registry()
    t = Table(title="Model registry")
    for c in (
        "model_id",
        "kind",
        "horizon",
        "train_end",
        "rows",
        "bal.acc",
        "top-decile",
        "age d",
        "ok",
    ):
        t.add_column(c)
    from ptaa.mlmodels.registry import ModelRegistry as Registry

    for r in reg.list():
        ok, _ = reg.verify(r.model_id)
        t.add_row(
            r.model_id,
            r.kind,
            r.horizon,
            str(r.train_end),
            str(r.n_samples),
            f"{r.metrics.get('balanced_accuracy', float('nan')):.3f}",
            f"{r.metrics.get('top_decile_precision_up', float('nan')):.3f}",
            str(Registry.freshness_days(r)),
            "yes" if ok else "NO",
        )
    console.print(t)


@models_app.command("verify")
def models_verify(model_id: str) -> None:
    ok, msg = _registry().verify(model_id)
    console.print(("[green]" if ok else "[red]") + f"{model_id}: {msg}")
    if not ok:
        raise typer.Exit(1)


@models_app.command("explain")
def models_explain(model_id: str, symbol: str, synthetic: bool = False) -> None:
    """Score the latest bar of a symbol with a registered GBM and show SHAP top features."""
    from ptaa.fast.features import FEATURE_NAMES, compute_features, feature_row_hash
    from ptaa.training.pipeline import synthetic_candles, yfinance_candle_source
    from ptaa.training.train_gbm import GBMModel

    s = get_settings()
    source = synthetic_candles if synthetic else yfinance_candle_source(Path(s.data_dir))
    df = source(symbol.upper())
    row = (
        compute_features(df)
        .dropna(subset=[c for c in FEATURE_NAMES if not c.startswith(("rel_", "beta"))])
        .iloc[-1]
    )
    m = GBMModel(_registry(), model_id)
    proba = m.predict_proba(row)[0]
    console.print(
        f"{symbol.upper()} @ {row.name} · P(down/flat/up) = {proba[0]:.3f} / {proba[1]:.3f} / {proba[2]:.3f} · feature row {feature_row_hash(row)}"
    )
    for k, v in m.explain(row).items():
        console.print(f"  {k:<22} {v:+.4f}")


# ----------------------------------------------------------------------------- research graph


def _context(config: Path = Path("config.yaml")) -> GraphContext:
    from ptaa.graph.context import GraphContext
    from ptaa.ledger.db import Ledger, PaperBroker

    s = get_settings()
    cfg = load_config(config)
    ledger = Ledger(Path(s.data_dir) / "ledger.db", cfg.fees, cfg.paper_book_inr)
    return GraphContext(
        cfg=cfg, settings=s, registry=_registry(), ledger=ledger, broker=PaperBroker(ledger)
    )


@app.command()
def run(
    symbol: str,
    module: str = typer.Option("swing", help="swing | intraday"),
    synthetic: bool = typer.Option(False, help="use seeded synthetic candles (offline demo)"),
    news: bool = typer.Option(True, help="--no-news skips web discovery (offline)"),
    llm: bool = typer.Option(True, help="--no-llm uses deterministic templates (same decisions)"),
    config: Path = Path("config.yaml"),
) -> None:
    """Run the research graph for one symbol and record the decision in the paper ledger."""
    from ptaa.graph.graph import build_graph, run_once
    from ptaa.graph.nodes.data import market_snapshot
    from ptaa.training.pipeline import synthetic_candles, yfinance_candle_source

    ctx = _context(config)
    if not news or synthetic:
        ctx.news_items = []  # no web discovery
    if not llm and ctx.llm is not None:
        ctx.llm.backend = "none"
    source = synthetic_candles if synthetic else yfinance_candle_source(Path(ctx.settings.data_dir))
    candles = source(symbol.upper())
    index_close = synthetic_candles("NIFTY")["close"] if synthetic else None
    snap = market_snapshot(candles, index_close)
    graph = build_graph(ctx, checkpoint_path=Path(ctx.settings.data_dir) / "checkpoints.db")
    out = run_once(graph, ctx, symbol, module, snap)  # type: ignore[arg-type]

    console.print(
        f"[bold]run {out['run_id']} · {symbol.upper()} · {module} · as of {out['as_of']}[/bold]"
    )
    for sig in out.get("verified_signals", []):
        console.print(
            f"  [{sig.signal_id}] {sig.analyst:<12} P(up) {sig.p_up:.2f}  P(down) {sig.p_down:.2f}  {sig.summary}"
        )
    if out.get("rejected"):
        console.print(f"  verifier rejected {len(out['rejected'])} evidence item(s)")
    ens = out.get("ensemble")
    if ens:
        console.print(
            f"  ensemble {ens.score:.2f} · {ens.direction} · regime {ens.regime} · weights {ens.weights}"
        )
    for t in out.get("debate", []):
        console.print(f"  [dim]{t.side.upper()} r{t.round}:[/dim] {t.text}")
    if out.get("thesis"):
        console.print(f"  thesis: {out['thesis'].narrative} · risks {out['thesis'].key_risks}")
    if out.get("proposal"):
        p = out["proposal"]
        console.print(
            f"  proposal: {p.side} {p.quantity} @ {p.entry} stop {p.stop_loss} target {p.target}"
        )
    if out.get("verdict"):
        v = out["verdict"]
        for pv in v.persona_views:
            console.print(f"  [dim]{pv}[/dim]")
        failed = [r["rule"] + ": " + r["message"] for r in v.rules if r["outcome"] != "approve"]
        console.print(
            f"  risk: {v.outcome} qty {v.quantity}"
            + (f" · {failed}" if failed else " · all rules pass")
        )
    d = out.get("decision")
    if d:
        console.print(f"[bold]{d.action}[/bold] {d.quantity} · {d.report}")
    for e in out.get("errors", []):
        console.print(f"[red]{e}[/red]")
    console.print(
        f"[dim]{len(out.get('audit', []))} audit events · paper only · not investment advice[/dim]"
    )


@app.command()
def portfolio(config: Path = Path("config.yaml")) -> None:
    """Positions, cash, equity, fees and recent runs from the paper ledger."""
    ctx = _context(config)
    cash, value, eq = ctx.ledger.equity()
    console.print(
        f"cash ₹{cash:,.0f} · positions ₹{value:,.0f} · equity ₹{eq:,.0f} · fees paid ₹{ctx.ledger.fees_paid():,.2f}"
    )
    t = Table(title="Open positions")
    for c in ("symbol", "module", "side", "qty", "avg", "last", "stop", "target", "strategy"):
        t.add_column(c)
    for p in ctx.ledger.positions():
        t.add_row(
            p.symbol,
            p.module,
            p.side,
            str(p.quantity),
            f"{p.avg_price:.2f}",
            f"{p.last_price or 0:.2f}",
            f"{p.stop_loss:.2f}",
            f"{p.target:.2f}",
            p.strategy,
        )
    console.print(t)
    t2 = Table(title="Recent runs")
    for c in ("run_id", "symbol", "module", "decision", "qty", "ensemble", "regime", "verdict"):
        t2.add_column(c)
    for r in ctx.ledger.runs(15):
        t2.add_row(
            r.run_id,
            r.symbol,
            r.module,
            r.decision or "",
            str(r.quantity or 0),
            f"{r.ensemble_score or 0:.2f}",
            r.regime or "",
            r.verdict or "",
        )
    console.print(t2)


@app.command()
def replay(run_id: str, config: Path = Path("config.yaml")) -> None:
    """Show the audit trail of a run: every node, its latency and hashes."""
    ctx = _context(config)
    rows = ctx.ledger.audit_for(run_id)
    if not rows:
        console.print(f"[red]no audit for run {run_id}[/red]")
        raise typer.Exit(1)
    t = Table(title=f"audit · run {run_id}")
    for c in ("node", "at", "ms", "inputs", "outputs", "model", "note"):
        t.add_column(c)
    for r in rows:
        t.add_row(
            r.node,
            r.at.strftime("%H:%M:%S.%f")[:-3],
            f"{r.duration_ms:.2f}",
            r.inputs_hash,
            r.outputs_hash,
            r.model_id or "",
            r.note,
        )
    console.print(t)


@app.command()
def eval(  # noqa: A001 - CLI verb
    record: int = typer.Option(0, help="record N fresh snapshots first (synthetic unless --real)"),
    real: bool = typer.Option(
        False, help="record from cached yfinance candles instead of synthetic"
    ),
    snapshots: Path = Path("evals/snapshots"),
    config: Path = Path("config.yaml"),
) -> None:
    """Run the frozen-snapshot eval suite: provenance, determinism, number guard, rules, latency."""
    from ptaa.data.universe import NIFTY50
    from ptaa.evals import LATENCY_BUDGET_US, record_snapshots, run_all
    from ptaa.training.pipeline import synthetic_candles, yfinance_candle_source

    ctx = _context(config)
    ctx.news_items = []  # evals are offline
    if record:
        src = yfinance_candle_source(Path(ctx.settings.data_dir)) if real else synthetic_candles
        paths = record_snapshots(snapshots, NIFTY50, src, n=record)
        console.print(f"recorded {len(paths)} snapshots in {snapshots}")
    results, p99 = run_all(ctx, snapshots)
    if not results:
        console.print("[yellow]no snapshots; run with --record 30[/yellow]")
        raise typer.Exit(1)
    t = Table(title="evals")
    checks = list(results[0].checks)
    t.add_column("snapshot")
    for c in checks:
        t.add_column(c)
    for r in results:
        t.add_row(
            r.snapshot,
            *[
                (
                    "[green]ok"
                    if r.checks[c]
                    else "[red]FAIL "
                    + r.notes.get("uncited" if c == "no_uncited_numbers" else "errors", "")
                )
                for c in checks
            ],
        )
    console.print(t)
    failed = [r for r in results if not r.ok]
    lat_ok = p99 < LATENCY_BUDGET_US
    console.print(
        f"fast path p99 {p99:.0f} µs (budget {LATENCY_BUDGET_US:.0f}) {'ok' if lat_ok else 'FAIL'} · {len(results) - len(failed)}/{len(results)} snapshots pass · llm {ctx.llm.backend if ctx.llm else 'none'}"
    )
    if failed or not lat_ok:
        raise typer.Exit(1)


@app.command()
def serve(
    api_port: int = typer.Option(8765, help="FastAPI control plane port (127.0.0.1)"),
    ui_port: int = typer.Option(8501, help="Streamlit terminal port (127.0.0.1)"),
    api_only: bool = typer.Option(False, help="start only the control-plane API"),
) -> None:
    """Start the control-plane API and the Streamlit terminal (local only)."""
    import subprocess
    import sys
    import threading

    import uvicorn

    from ptaa.control.api import create_app

    api = create_app()
    t = threading.Thread(
        target=lambda: uvicorn.run(api, host="127.0.0.1", port=api_port, log_level="warning"),
        daemon=True,
    )
    t.start()
    console.print(f"control plane: http://127.0.0.1:{api_port}/docs")
    if api_only:
        t.join()
        return
    app_path = Path(__file__).resolve().parents[2] / "terminal" / "app.py"
    console.print(f"terminal: http://127.0.0.1:{ui_port}  ({app_path})")
    subprocess.run(  # noqa: S603 - fixed argv
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(ui_port),
            "--server.address",
            "127.0.0.1",
        ],
        check=False,
    )


if __name__ == "__main__":
    app()
