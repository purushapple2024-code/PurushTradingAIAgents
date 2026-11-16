"""ExecutionBroker protocol.

Live adapters (Angel One, Upstox, Flattrade) are milestone 9 and sit behind LIVE_TRADING_ENABLED
plus per-order human approval; in v1 only PaperBroker implements this.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ptaa.models import Module, Side


class OrderRequest(BaseModel):
    symbol: str
    module: Module
    side: Side
    quantity: int
    limit_price: float | None = None  # None = market / next tick
    stop_loss: float
    target: float
    strategy: str = ""
    run_id: str | None = None
    reason: str = "entry"


class Fill(BaseModel):
    symbol: str
    side: Side
    quantity: int
    price: float
    fees: float
    filled_at: datetime
    order_id: str


@runtime_checkable
class ExecutionBroker(Protocol):
    name: str
    is_live: bool

    def place(self, req: OrderRequest, price: float, at: datetime) -> Fill: ...

    def close_position(
        self, symbol: str, price: float, at: datetime, reason: str
    ) -> Fill | None: ...
