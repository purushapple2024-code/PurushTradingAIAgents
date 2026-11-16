"""Typed hand-offs between graph nodes. Every field that reaches the ledger is produced by code."""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from ptaa.models import Module, Side

Direction = Literal["long", "short", "no_trade"]
Regime = Literal["trend_up", "trend_down", "range", "high_vol"]
Analyst = Literal["technical", "sequence", "news", "fundamental"]


class Evidence(BaseModel):
    claim: str
    source_kind: Literal["web", "filing", "model", "feature", "flows"]
    source_url: str | None = None
    model_id: str | None = None
    feature_hash: str | None = None
    attribution: dict[str, float] | None = None
    fetched_at: datetime
    quote: str | None = None


class Signal(BaseModel):
    signal_id: str  # S1, S2, ... referenced by the debate
    analyst: Analyst
    horizon: str
    p_up: float = Field(ge=0, le=1)
    p_down: float = Field(ge=0, le=1)
    expected_move_pct: float | None = None
    uncertainty: float | None = None
    evidence: list[Evidence] = Field(min_length=1)
    summary: str = ""


class EnsembleScore(BaseModel):
    score: float = Field(ge=0, le=1)  # P(profitable long after fees)
    direction: Direction
    regime: Regime
    weights: dict[str, float]


class DebateTurn(BaseModel):
    side: Literal["bull", "bear", "manager"]
    round: int
    text: str
    cited: list[str] = []


class Thesis(BaseModel):
    direction: Direction
    conviction: float = Field(ge=0, le=1)  # copied from EnsembleScore.score, never from an LLM
    key_risks: list[str] = []
    invalidation_price: float | None = None
    narrative: str = ""


class TradeProposal(BaseModel):
    symbol: str
    module: Module
    side: Side
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    target: float = Field(gt=0)
    quantity: int = Field(ge=0)
    horizon_minutes: int = Field(ge=1)
    strategy: str = "trend_pullback"
    rationale: str = ""


class RiskVerdict(BaseModel):
    outcome: Literal["approve", "resize", "reject"]
    quantity: int
    rules_triggered: list[str] = []
    rules: list[dict[str, Any]] = []
    persona_views: list[str] = []


class Decision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    quantity: int
    report: str = ""
    decided_at: datetime


class AuditEvent(BaseModel):
    node: str
    at: datetime
    duration_ms: float
    inputs_hash: str
    outputs_hash: str
    model_id: str | None = None
    prompt_version: str | None = None
    note: str = ""


class GraphState(TypedDict, total=False):
    run_id: str
    symbol: str
    module: Module
    as_of: datetime
    features: dict[str, float]
    feature_hash: str
    price: float
    atr: float
    avg_turnover_cr: float
    circuit_today: bool
    days_to_results: int | None
    index_returns: list[float]
    news_cutoff: str
    seq_window: list[list[float]] | None
    signals: Annotated[list[Signal], operator.add]
    verified_signals: list[Signal]
    rejected: list[Evidence]
    ensemble: EnsembleScore | None
    debate: list[DebateTurn]
    thesis: Thesis | None
    proposal: TradeProposal | None
    verdict: RiskVerdict | None
    decision: Decision | None
    risk_loops: int
    audit: Annotated[list[AuditEvent], operator.add]
    errors: Annotated[list[str], operator.add]
