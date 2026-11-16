"""SQLModel tables: runs, audit events, orders (fills), positions, equity snapshots."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class RunRow(SQLModel, table=True):
    __tablename__ = "runs"
    run_id: str = Field(primary_key=True)
    symbol: str
    module: str
    as_of: datetime
    decision: str | None = None
    quantity: int | None = None
    ensemble_score: float | None = None
    regime: str | None = None
    verdict: str | None = None
    report: str | None = None
    created_at: datetime


class AuditRow(SQLModel, table=True):
    __tablename__ = "audit"
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    node: str
    at: datetime
    duration_ms: float
    inputs_hash: str
    outputs_hash: str
    model_id: str | None = None
    prompt_version: str | None = None
    note: str = ""


class OrderRow(SQLModel, table=True):
    __tablename__ = "orders"
    id: int | None = Field(default=None, primary_key=True)
    run_id: str | None = Field(default=None, index=True)
    symbol: str = Field(index=True)
    module: str
    side: str
    quantity: int
    price: float
    fees: float
    reason: str  # entry | stop | target | trailing_stop | circuit | square_off | manual
    strategy: str = ""
    filled_at: datetime


class PositionRow(SQLModel, table=True):
    __tablename__ = "positions"
    symbol: str = Field(primary_key=True)
    module: str
    side: str
    quantity: int
    avg_price: float
    stop_loss: float
    target: float
    strategy: str = ""
    opened_at: datetime
    last_price: float | None = None


class EquityRow(SQLModel, table=True):
    __tablename__ = "equity"
    id: int | None = Field(default=None, primary_key=True)
    at: datetime
    cash: float
    positions_value: float
    equity: float
    realised_pnl: float
    fees_paid: float
