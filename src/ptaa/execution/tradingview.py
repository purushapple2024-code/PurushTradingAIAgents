"""TradingView webhook INPUT adapter (TradingView has no order API; it sends alerts).

An alert JSON like {"symbol":"RELIANCE","action":"buy","price":2966,"strategy":"p","secret":"..."}
becomes an external Signal-like hint recorded in the control plane and an alert; it never places
an order. The research graph can consume recent hints as a low-weight "external" evidence source.
The shared secret must match PTAA_TRADINGVIEW_SECRET (compared in constant time).
"""

from __future__ import annotations

import hmac
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from ptaa.control.service import ControlService


class TVAlert(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    action: Literal["buy", "sell", "exit", "note"]
    price: float | None = None
    strategy: str = "tradingview"
    message: str = ""
    secret: str = ""


def verify_secret(provided: str, expected: str | None) -> bool:
    if not expected:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


def record_alert(control: ControlService, alert: TVAlert) -> dict[str, Any]:
    doc = control.get("settings", "tradingview_alerts") or {"alerts": []}
    item = {
        "symbol": alert.symbol.upper(),
        "action": alert.action,
        "price": alert.price,
        "strategy": alert.strategy,
        "message": alert.message[:200],
        "at": datetime.now(tz=timezone.utc).isoformat(),
    }
    doc["alerts"] = (doc.get("alerts", []) + [item])[-200:]
    control.put(
        "settings",
        "tradingview_alerts",
        doc,
        actor="tradingview",
        note=f"{alert.action} {alert.symbol}",
    )
    return item


def recent_alerts(control: ControlService, symbol: str, hours: int = 24) -> list[dict[str, Any]]:
    doc = control.get("settings", "tradingview_alerts") or {"alerts": []}
    cutoff = datetime.now(tz=timezone.utc).timestamp() - hours * 3600
    return [
        a
        for a in doc.get("alerts", [])
        if a["symbol"] == symbol.upper() and datetime.fromisoformat(a["at"]).timestamp() >= cutoff
    ]
