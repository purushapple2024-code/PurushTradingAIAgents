from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ptaa.execution.broker import OrderRequest
from ptaa.graph.context import GraphContext
from ptaa.graph.graph import build_graph, run_once
from ptaa.graph.nodes.data import market_snapshot
from ptaa.graph.nodes.gates import classify_regime
from ptaa.ledger.db import Ledger, PaperBroker
from ptaa.mlmodels.registry import ModelRegistry
from ptaa.settings import AppConfig, get_settings
from ptaa.training.pipeline import synthetic_candles


class FakeGBM:
    """Stands in for GBMModel: fixed probabilities, deterministic attribution."""

    def __init__(self, record, p_up: float):
        self.record = record
        self.p = p_up

    def predict_proba(self, row):
        return np.array([[1 - self.p - 0.05, 0.05, self.p]])

    def explain(self, row, top_k=8):
        return {"ema20_slope": 0.2, "rsi14": 0.1}


def _ctx(tmp_path: Path, cfg: AppConfig, p_up: float) -> GraphContext:
    reg = ModelRegistry(tmp_path / "models")
    f = tmp_path / "models" / "gbm" / "m.txt"
    f.parent.mkdir(parents=True)
    f.write_text("x")
    rec = reg.register(
        "gbm",
        "5d",
        f,
        ["a"],
        pd.Timestamp("2020-01-01").date(),
        datetime.now(tz=timezone.utc).date(),
        "NIFTY50",
        100,
        {"balanced_accuracy": 0.45},
    )
    ledger = Ledger(tmp_path / "ledger.db", cfg.fees, cfg.paper_book_inr)
    ctx = GraphContext(
        cfg=cfg, settings=get_settings(), registry=reg, ledger=ledger, broker=PaperBroker(ledger)
    )
    ctx.models[rec.model_id] = FakeGBM(rec, p_up)
    return ctx


def _snapshot(symbol: str = "RELIANCE") -> dict:
    snap = market_snapshot(
        synthetic_candles(symbol, n=600), synthetic_candles("NIFTY", n=600)["close"]
    )
    snap["index_returns"] = [0.003] * 60  # force trend_up regime
    return snap


def test_regime_classifier() -> None:
    assert classify_regime([0.003] * 60) == "trend_up"
    assert classify_regime([-0.003] * 60) == "trend_down"
    assert classify_regime([0.0001, -0.0001] * 30) == "range"
    rng = np.random.default_rng(0)
    assert (
        classify_regime(list(rng.normal(0, 0.001, 40)) + list(rng.normal(0, 0.04, 20)))
        == "high_vol"
    )


def test_graph_buy_path_fills_paper_ledger(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.80)
    graph = build_graph(ctx)
    out = run_once(graph, ctx, "RELIANCE", "swing", _snapshot())
    assert out["errors"] == [], out["errors"]
    assert [s.signal_id for s in out["verified_signals"]] == ["S1", "S4"] or len(
        out["verified_signals"]
    ) == 2
    assert out["ensemble"].direction == "long" and out["ensemble"].regime == "trend_up"
    assert (
        out["thesis"].conviction == out["ensemble"].score
    )  # conviction is the ensemble score, never LLM
    prop = out["proposal"]
    assert prop.side == "BUY" and prop.quantity > 0 and prop.stop_loss < prop.entry < prop.target
    assert out["verdict"].outcome in ("approve", "resize")
    assert out["decision"].action == "BUY" and out["decision"].quantity == out["verdict"].quantity
    # ledger: one position, cash reduced by notional + fees, run and audit recorded
    pos = ctx.ledger.positions()
    assert (
        len(pos) == 1
        and pos[0].symbol == "RELIANCE"
        and pos[0].quantity == out["decision"].quantity
    )
    cash = ctx.ledger.cash()
    assert cash < cfg.paper_book_inr - prop.entry * pos[0].quantity
    runs = ctx.ledger.runs()
    assert runs[0].decision == "BUY" and runs[0].run_id == out["run_id"]
    nodes = [a.node for a in ctx.ledger.audit_for(out["run_id"])]
    for n in (
        "data_collector",
        "technical_analyst",
        "fundamental_analyst",
        "verifier",
        "regime",
        "ensemble",
        "research_manager",
        "trader",
        "risk_manager",
        "portfolio_manager",
        "executor",
    ):
        assert n in nodes
    # a second run on the same symbol is rejected by the open_positions rule
    out2 = run_once(graph, ctx, "RELIANCE", "swing", _snapshot())
    assert out2["decision"].action == "HOLD" and "open_positions" in out2["verdict"].rules_triggered


def test_graph_hold_when_model_bearish(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.20)
    out = run_once(build_graph(ctx), ctx, "TCS", "swing", _snapshot("TCS"))
    assert out["decision"].action == "HOLD" and out["proposal"] is None
    assert ctx.ledger.positions() == []


def test_verifier_rejects_unregistered_model(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.80)
    rec = ctx.registry.latest("gbm", "5d")
    ctx.models[rec.model_id].record = SimpleNamespace(
        model_id="gbm-5d-v99", horizon="5d"
    )  # unregistered id
    out = run_once(build_graph(ctx), ctx, "INFY", "swing", _snapshot("INFY"))
    assert all(s.analyst != "technical" for s in out["verified_signals"])
    assert any(e.model_id == "gbm-5d-v99" for e in out["rejected"])


def test_ledger_round_trip_and_exit(tmp_path: Path, cfg: AppConfig) -> None:
    ledger = Ledger(tmp_path / "l.db", cfg.fees, 1_000_000)
    broker = PaperBroker(ledger)
    at = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)
    fill = broker.place(
        OrderRequest(
            symbol="LT",
            module="swing",
            side="BUY",
            quantity=10,
            stop_loss=3500,
            target=3700,
            strategy="t",
        ),
        3600.0,
        at,
    )
    assert fill.fees > 0 and ledger.positions()[0].quantity == 10
    assert ledger.cash() == pytest.approx(1_000_000 - 36_000 - fill.fees)
    ledger.mark({"LT": 3650.0})
    _, value, eq = ledger.equity()
    assert value == pytest.approx(36_500) and eq == pytest.approx(ledger.cash() + 36_500)
    out = broker.close_position("LT", 3700.0, at, "target")
    assert out is not None and out.side == "SELL" and ledger.positions() == []
    assert ledger.cash() == pytest.approx(1_000_000 + 1_000 - fill.fees - out.fees)
    assert broker.close_position("LT", 1.0, at, "manual") is None
    ps = ledger.portfolio_state()
    assert ps.open_positions == [] and ps.equity == pytest.approx(ledger.cash())
