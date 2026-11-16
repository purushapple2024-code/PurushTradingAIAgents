"""Rule engine: hard risk rules that decide approve / resize / reject. LLM personas only advise.

Each rule is a pure function `(ctx) -> RuleResult`. The engine runs all of them, so the report
shows every pass/fail, then combines: any `reject` wins; otherwise the smallest `resize`
quantity wins; otherwise approve.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from ptaa.models import Module, Side
from ptaa.settings import RiskLimits

Outcome = Literal["approve", "resize", "reject"]


class OpenPosition(BaseModel):
    symbol: str
    sector: str
    value: float  # current market value, INR


class PortfolioState(BaseModel):
    equity: float
    peak_equity: float
    cash: float
    open_positions: list[OpenPosition] = []
    day_pnl: float = 0.0
    drawdown_halted: bool = False


class Proposal(BaseModel):
    symbol: str
    sector: str
    module: Module
    side: Side
    entry: float = Field(gt=0)
    stop: float = Field(gt=0)
    target: float = Field(gt=0)
    quantity: int = Field(ge=0)
    atr: float = Field(gt=0)
    avg_turnover_cr: float = Field(ge=0)
    circuit_today: bool = False
    days_to_results: int | None = None
    ensemble_score: float = Field(ge=0, le=1)
    model_age_days: int = 0
    is_event_strategy: bool = False
    as_of: date | None = None


class RuleResult(BaseModel):
    rule: str
    outcome: Outcome
    message: str
    adjusted_quantity: int | None = None


class Verdict(BaseModel):
    outcome: Outcome
    quantity: int
    rules: list[RuleResult]

    @property
    def triggered(self) -> list[str]:
        return [r.rule for r in self.rules if r.outcome != "approve"]


Rule = Callable[[Proposal, PortfolioState, RiskLimits], RuleResult]


def _ok(name: str, msg: str) -> RuleResult:
    return RuleResult(rule=name, outcome="approve", message=msg)


def _reject(name: str, msg: str) -> RuleResult:
    return RuleResult(rule=name, outcome="reject", message=msg)


def _resize(name: str, msg: str, qty: int) -> RuleResult:
    if qty <= 0:
        return _reject(name, msg + " (no size fits)")
    return RuleResult(rule=name, outcome="resize", message=msg, adjusted_quantity=qty)


def rule_risk_per_trade(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    budget = s.equity * lim.max_risk_per_trade_pct / 100
    risk = abs(p.entry - p.stop) * p.quantity
    if risk <= budget:
        return _ok("risk_per_trade", f"risk ₹{risk:,.0f} ≤ budget ₹{budget:,.0f}")
    qty = math.floor(budget / abs(p.entry - p.stop))
    return _resize("risk_per_trade", f"risk ₹{risk:,.0f} > budget ₹{budget:,.0f}", qty)


def rule_position_size(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    cap = s.equity * lim.max_position_pct / 100
    value = p.entry * p.quantity
    if value <= cap:
        return _ok("position_size", f"₹{value:,.0f} ≤ cap ₹{cap:,.0f}")
    return _resize("position_size", f"₹{value:,.0f} > cap ₹{cap:,.0f}", math.floor(cap / p.entry))


def rule_sector_exposure(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    cap = s.equity * lim.max_sector_pct / 100
    current = sum(o.value for o in s.open_positions if o.sector == p.sector)
    room = cap - current
    value = p.entry * p.quantity
    if value <= room:
        return _ok(
            "sector_exposure",
            f"{p.sector} {((current + value) / s.equity) * 100:.1f}% ≤ {lim.max_sector_pct}%",
        )
    return _resize(
        "sector_exposure",
        f"{p.sector} would be {((current + value) / s.equity) * 100:.1f}% > {lim.max_sector_pct}%",
        math.floor(max(room, 0) / p.entry),
    )


def rule_open_positions(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    if any(o.symbol == p.symbol for o in s.open_positions):
        return _reject("open_positions", f"already holding {p.symbol}")
    n = len(s.open_positions)
    if n < lim.max_open_positions:
        return _ok("open_positions", f"{n} open < {lim.max_open_positions}")
    return _reject("open_positions", f"{n} open ≥ {lim.max_open_positions}")


def rule_gross_exposure(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    cap = s.equity * lim.max_gross_exposure_pct / 100
    current = sum(o.value for o in s.open_positions)
    room = cap - current
    value = p.entry * p.quantity
    if value <= room:
        return _ok(
            "gross_exposure",
            f"{((current + value) / s.equity) * 100:.0f}% ≤ {lim.max_gross_exposure_pct}%",
        )
    return _resize(
        "gross_exposure",
        f"gross would exceed {lim.max_gross_exposure_pct}%",
        math.floor(max(room, 0) / p.entry),
    )


def rule_daily_loss(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    if lim.daily_loss_limit_pct is None:
        return _ok("daily_loss", "n/a")
    limit = -s.equity * lim.daily_loss_limit_pct / 100
    if s.day_pnl > limit:
        return _ok("daily_loss", f"day P&L ₹{s.day_pnl:,.0f} above limit ₹{limit:,.0f}")
    return _reject("daily_loss", f"daily loss limit hit (₹{s.day_pnl:,.0f})")


def rule_drawdown_halt(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    dd = (s.peak_equity - s.equity) / s.peak_equity * 100 if s.peak_equity > 0 else 0.0
    if s.drawdown_halted or dd >= lim.drawdown_halt_pct:
        return _reject(
            "drawdown_halt", f"drawdown {dd:.1f}% ≥ {lim.drawdown_halt_pct}%; manual reset required"
        )
    return _ok("drawdown_halt", f"drawdown {dd:.1f}%")


def rule_stop_sanity(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    mult = abs(p.entry - p.stop) / p.atr
    if lim.stop_atr_min <= mult <= lim.stop_atr_max:
        return _ok("stop_sanity", f"stop {mult:.2f}× ATR")
    return _reject(
        "stop_sanity", f"stop {mult:.2f}× ATR outside [{lim.stop_atr_min}, {lim.stop_atr_max}]"
    )


def rule_reward_risk(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    risk = abs(p.entry - p.stop)
    reward = (p.target - p.entry) if p.side == "BUY" else (p.entry - p.target)
    if risk <= 0 or reward <= 0:
        return _reject("reward_risk", "target on the wrong side of entry")
    rr = reward / risk
    if rr >= lim.min_reward_risk:
        return _ok("reward_risk", f"R/R {rr:.2f}")
    return _reject("reward_risk", f"R/R {rr:.2f} < {lim.min_reward_risk}")


def rule_liquidity(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    if p.avg_turnover_cr >= lim.min_avg_turnover_cr:
        return _ok("liquidity", f"₹{p.avg_turnover_cr:,.0f} cr/day")
    return _reject("liquidity", f"₹{p.avg_turnover_cr:,.0f} cr/day < ₹{lim.min_avg_turnover_cr} cr")


def rule_circuit(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    return (
        _reject("circuit", "hit a circuit today")
        if p.circuit_today
        else _ok("circuit", "no circuit")
    )


def rule_event_blackout(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    if (
        p.days_to_results is not None
        and 0 <= p.days_to_results <= lim.results_blackout_days
        and not p.is_event_strategy
    ):
        return _reject("event_blackout", f"results in {p.days_to_results} trading day(s)")
    return _ok("event_blackout", "clear")


def rule_ensemble_floor(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    if p.ensemble_score >= lim.ensemble_floor:
        return _ok("ensemble_floor", f"{p.ensemble_score:.2f} ≥ {lim.ensemble_floor}")
    return _reject("ensemble_floor", f"{p.ensemble_score:.2f} < {lim.ensemble_floor}")


def rule_model_freshness(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    if p.model_age_days <= lim.model_max_age_days:
        return _ok("model_freshness", f"{p.model_age_days}d ≤ {lim.model_max_age_days}d")
    return _reject("model_freshness", f"model {p.model_age_days}d old > {lim.model_max_age_days}d")


def rule_cash(p: Proposal, s: PortfolioState, lim: RiskLimits) -> RuleResult:
    value = p.entry * p.quantity
    if value <= s.cash:
        return _ok("cash", f"₹{value:,.0f} ≤ cash ₹{s.cash:,.0f}")
    return _resize("cash", f"₹{value:,.0f} > cash ₹{s.cash:,.0f}", math.floor(s.cash / p.entry))


ALL_RULES: list[Rule] = [
    rule_drawdown_halt,
    rule_daily_loss,
    rule_open_positions,
    rule_circuit,
    rule_event_blackout,
    rule_liquidity,
    rule_stop_sanity,
    rule_reward_risk,
    rule_ensemble_floor,
    rule_model_freshness,
    rule_risk_per_trade,
    rule_position_size,
    rule_sector_exposure,
    rule_gross_exposure,
    rule_cash,
]


def evaluate(
    p: Proposal, s: PortfolioState, lim: RiskLimits, rules: list[Rule] | None = None
) -> Verdict:
    results = [r(p, s, lim) for r in (rules or ALL_RULES)]
    if any(r.outcome == "reject" for r in results):
        return Verdict(outcome="reject", quantity=0, rules=results)
    resizes = [
        r.adjusted_quantity
        for r in results
        if r.outcome == "resize" and r.adjusted_quantity is not None
    ]
    if resizes:
        qty = min(resizes)
        return Verdict(outcome="resize" if qty > 0 else "reject", quantity=qty, rules=results)
    return Verdict(outcome="approve", quantity=p.quantity, rules=results)
