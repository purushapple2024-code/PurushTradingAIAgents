from __future__ import annotations

import pytest

from ptaa.risk.fees import compute_fees, round_trip_cost
from ptaa.risk.rules import OpenPosition, PortfolioState, Proposal, evaluate
from ptaa.risk.sizing import size_by_risk
from ptaa.settings import AppConfig


def test_delivery_fees_match_schedule(cfg: AppConfig) -> None:
    # Buy 100 @ 1000 = 1,00,000 turnover: STT 0.1% = 100; stamp 0.015% = 15; exch 0.00297% = 2.97; SEBI 0.1; GST 18% of (0+2.97+0.1)
    fb = compute_fees(cfg.fees, "BUY", "swing", 1000.0, 100)
    assert fb.stt == 100.0 and fb.stamp == 15.0 and fb.exchange == 2.97 and fb.sebi == 0.1
    assert fb.gst == pytest.approx(0.55, abs=0.01)
    sell = compute_fees(cfg.fees, "SELL", "swing", 1000.0, 100)
    assert sell.stamp == 0.0 and sell.stt == 100.0
    rt = round_trip_cost(cfg.fees, "swing", 1000.0, 100)
    assert rt == pytest.approx(
        222.24, abs=0.02
    )  # ~0.22% round trip on delivery with zero brokerage


def test_intraday_fees(cfg: AppConfig) -> None:
    buy = compute_fees(cfg.fees, "BUY", "intraday", 1000.0, 100)
    sell = compute_fees(cfg.fees, "SELL", "intraday", 1000.0, 100)
    assert buy.stt == 0.0 and sell.stt == 25.0  # 0.025% sell side
    assert buy.stamp == 3.0 and sell.stamp == 0.0  # 0.003% buy side
    assert buy.brokerage == 20.0  # 0.03% of 1,00,000 = 30, capped at 20


def test_sizer_caps(cfg: AppConfig) -> None:
    lim = cfg.risk["swing"]
    # 1% of 10L = 10,000 risk; stop 20 away -> 500 shares; position cap 5% = 50,000 / 1000 = 50 shares
    assert size_by_risk(1_000_000, 1000, 980, lim) == 50
    # stop 100 away -> 100 by risk, capped to 50 by position size
    assert size_by_risk(1_000_000, 1000, 900, lim) == 50
    # stop 500 away -> 20 by risk, under the position cap
    assert size_by_risk(1_000_000, 1000, 500, lim) == 20
    assert size_by_risk(1_000_000, 1000, 1000, lim) == 0
    assert size_by_risk(1_000_000, 1000, 980, lim, risk_fraction=0.5) == 50
    assert (
        size_by_risk(1_000_000, 2000, 1900, lim, risk_fraction=0.2) == 20
    )  # 2,000 budget / 100 = 20


def _proposal(**kw: object) -> Proposal:
    base = dict(
        symbol="RELIANCE",
        sector="Oil Gas & Energy",
        module="swing",
        side="BUY",
        entry=2966.0,
        stop=2924.0,
        target=3040.0,
        quantity=15,
        atr=21.4,
        avg_turnover_cr=2140.0,
        ensemble_score=0.74,
        model_age_days=3,
    )
    base.update(kw)
    return Proposal(**base)  # type: ignore[arg-type]


def _state(**kw: object) -> PortfolioState:
    base = dict(equity=1_042_318.0, peak_equity=1_064_000.0, cash=560_000.0, open_positions=[])
    base.update(kw)
    return PortfolioState(**base)  # type: ignore[arg-type]


def test_rules_approve(cfg: AppConfig) -> None:
    v = evaluate(_proposal(), _state(), cfg.risk["swing"])
    assert v.outcome == "approve" and v.quantity == 15 and v.triggered == []


def test_rules_sector_resize(cfg: AppConfig) -> None:
    held = [OpenPosition(symbol="ONGC", sector="Oil Gas & Energy", value=170_000.0)]
    v = evaluate(_proposal(), _state(open_positions=held), cfg.risk["swing"])
    # cap 20% of 1,042,318 = 208,463; room = 38,463 -> 12 shares @ 2966
    assert v.outcome == "resize" and v.quantity == 12 and v.triggered == ["sector_exposure"]


def test_rules_reject_paths(cfg: AppConfig) -> None:
    lim = cfg.risk["swing"]
    assert evaluate(_proposal(ensemble_score=0.33), _state(), lim).outcome == "reject"
    assert evaluate(_proposal(circuit_today=True), _state(), lim).outcome == "reject"
    assert evaluate(_proposal(days_to_results=1), _state(), lim).outcome == "reject"
    assert (
        evaluate(_proposal(days_to_results=1, is_event_strategy=True), _state(), lim).outcome
        == "approve"
    )
    assert evaluate(_proposal(stop=2960.0), _state(), lim).outcome == "reject"  # 0.28x ATR
    assert evaluate(_proposal(target=2990.0), _state(), lim).outcome == "reject"  # R/R 0.57
    assert evaluate(_proposal(model_age_days=30), _state(), lim).outcome == "reject"
    assert (
        evaluate(_proposal(), _state(equity=950_000.0), lim).outcome == "reject"
    )  # 10.7% drawdown
    held = [OpenPosition(symbol=f"S{i}", sector="X", value=10_000.0) for i in range(8)]
    assert evaluate(_proposal(), _state(open_positions=held), lim).outcome == "reject"


def test_rules_intraday_daily_loss(cfg: AppConfig) -> None:
    lim = cfg.risk["intraday"]
    p = _proposal(module="intraday", quantity=10, avg_turnover_cr=2140.0, ensemble_score=0.7)
    assert evaluate(p, _state(day_pnl=-25_000.0), lim).outcome == "reject"
    assert evaluate(p, _state(day_pnl=-5_000.0), lim).outcome == "approve"


def test_resize_picks_smallest(cfg: AppConfig) -> None:
    v = evaluate(_proposal(quantity=500), _state(cash=100_000.0), cfg.risk["swing"])
    # risk rule: 10,423 / 42 = 248; position: 52,116/2966 = 17; cash: 33 -> min = 17
    assert v.outcome == "resize" and v.quantity == 17
