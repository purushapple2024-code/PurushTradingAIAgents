"""Live broker adapters: Angel One SmartAPI, Upstox v2, Flattrade (NorenAPI).

All three implement ExecutionBroker with `is_live = True` and are constructed ONLY by
`build_live_broker`, which refuses unless the LiveGate's environment checks pass. Every `place`
call additionally requires a GateDecision with allowed=True for that exact order.

Payloads follow each broker's documented order endpoints; they have not been exercised against
live accounts in this repository. Treat the first live order as a test with quantity 1.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any

import requests

from ptaa.execution.broker import Fill, OrderRequest
from ptaa.execution.gate import GateDecision, LiveGate
from ptaa.settings import Settings

log = logging.getLogger(__name__)


class _LiveBase:
    name = "live"
    is_live = True

    def __init__(self, settings: Settings, gate: LiveGate, session: requests.Session | None = None):
        self.s, self.gate = settings, gate
        self.http = session or requests.Session()
        self.last_decision: GateDecision | None = None

    def _guard(self, req: OrderRequest, price: float, approval_id: str | None) -> GateDecision:
        d = self.gate.check(req, price, approval_id)
        self.last_decision = d
        if not d.allowed:
            log.warning("live order blocked: %s", d.reason)
        return d

    def _send(self, req: OrderRequest, price: float) -> str:  # pragma: no cover - broker specific
        raise NotImplementedError

    def place(
        self, req: OrderRequest, price: float, at: datetime, approval_id: str | None = None
    ) -> Fill:
        d = self._guard(req, price, approval_id)
        if not d.allowed:
            raise PermissionError(d.reason)
        order_id = self._send(req, price)
        self.gate.mark_executed(d.approval_id or "")
        # live fees come from the contract note; the ledger estimates them with the paper model
        return Fill(
            symbol=req.symbol,
            side=req.side,
            quantity=req.quantity,
            price=price,
            fees=0.0,
            filled_at=at,
            order_id=order_id,
        )

    def close_position(self, symbol: str, price: float, at: datetime, reason: str) -> Fill | None:
        raise PermissionError(
            "live close_position requires a fresh approval; use place() with reason='exit'"
        )


class AngelOneBroker(_LiveBase):
    name = "angelone"
    BASE = "https://apiconnect.angelone.in"

    def __init__(
        self,
        settings: Settings,
        gate: LiveGate,
        token_map: dict[str, str],
        jwt: str,
        session: requests.Session | None = None,
    ):
        super().__init__(settings, gate, session)
        self.tokens, self.jwt = token_map, jwt

    def _send(self, req: OrderRequest, price: float) -> str:
        payload = {
            "variety": "NORMAL",
            "tradingsymbol": f"{req.symbol}-EQ",
            "symboltoken": self.tokens[req.symbol],
            "transactiontype": req.side,
            "exchange": "NSE",
            "ordertype": "LIMIT" if req.limit_price else "MARKET",
            "producttype": "DELIVERY" if req.module == "swing" else "INTRADAY",
            "duration": "DAY",
            "price": str(req.limit_price or 0),
            "quantity": str(req.quantity),
        }
        r = self.http.post(
            f"{self.BASE}/rest/secure/angelbroking/order/v1/placeOrder",
            json=payload,
            timeout=10,
            headers={
                "Authorization": f"Bearer {self.jwt}",
                "X-PrivateKey": self.s.angel_api_key.get_secret_value(),  # type: ignore[union-attr]
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "Content-Type": "application/json",
            },
        )
        r.raise_for_status()
        body = r.json()
        if not body.get("status"):
            raise RuntimeError(f"angelone order rejected: {body.get('message', 'unknown')}")
        return str(body["data"]["orderid"])


class UpstoxBroker(_LiveBase):
    name = "upstox"
    BASE = "https://api.upstox.com/v2"

    def __init__(
        self,
        settings: Settings,
        gate: LiveGate,
        instrument_keys: dict[str, str],
        session: requests.Session | None = None,
    ):
        super().__init__(settings, gate, session)
        self.keys = instrument_keys

    def _send(self, req: OrderRequest, price: float) -> str:
        payload = {
            "quantity": req.quantity,
            "product": "D" if req.module == "swing" else "I",
            "validity": "DAY",
            "price": req.limit_price or 0,
            "tag": "ptaa",
            "instrument_token": self.keys[req.symbol],
            "order_type": "LIMIT" if req.limit_price else "MARKET",
            "transaction_type": req.side,
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
        }
        r = self.http.post(
            f"{self.BASE}/order/place",
            json=payload,
            timeout=10,
            headers={
                "Authorization": f"Bearer {self.s.upstox_access_token.get_secret_value()}",
                "Accept": "application/json",
            },
        )  # type: ignore[union-attr]
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "success":
            raise RuntimeError(f"upstox order rejected: {body.get('errors', body)}")
        return str(body["data"]["order_id"])


class FlattradeBroker(_LiveBase):
    name = "flattrade"
    BASE = "https://piconnect.flattrade.in/PiConnectTP"

    def __init__(
        self,
        settings: Settings,
        gate: LiveGate,
        session_token: str,
        session: requests.Session | None = None,
    ):
        super().__init__(settings, gate, session)
        self.token = session_token

    def _send(self, req: OrderRequest, price: float) -> str:
        import json

        uid = self.s.flattrade_user_id or ""
        jdata = {
            "uid": uid,
            "actid": uid,
            "exch": "NSE",
            "tsym": f"{req.symbol}-EQ",
            "qty": str(req.quantity),
            "prc": str(req.limit_price or 0),
            "prd": "C" if req.module == "swing" else "I",
            "trantype": "B" if req.side == "BUY" else "S",
            "prctyp": "LMT" if req.limit_price else "MKT",
            "ret": "DAY",
            "ordersource": "API",
        }
        r = self.http.post(
            f"{self.BASE}/PlaceOrder",
            data=f"jData={json.dumps(jdata)}&jKey={self.token}",
            timeout=10,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("stat") != "Ok":
            raise RuntimeError(f"flattrade order rejected: {body.get('emsg', body)}")
        return str(body["norenordno"])


def build_live_broker(settings: Settings, gate: LiveGate, **kw: Any) -> _LiveBase:
    """Construct the configured live adapter, or raise; only when the gate's environment passes."""
    ok, why = gate.environment_ok()
    if not ok:
        raise PermissionError(f"live broker refused: {why}")
    adapter = gate.control.engine_state().execution_adapter
    if adapter == "angelone":
        return AngelOneBroker(settings, gate, kw["token_map"], kw["jwt"], kw.get("session"))
    if adapter == "upstox":
        return UpstoxBroker(settings, gate, kw["instrument_keys"], kw.get("session"))
    if adapter == "flattrade":
        return FlattradeBroker(settings, gate, kw["session_token"], kw.get("session"))
    raise PermissionError(f"unknown live adapter {adapter}")


def fingerprint(req: OrderRequest) -> str:
    """Stable id for an order request, used to de-duplicate approvals."""
    return hashlib.sha256(
        f"{req.symbol}|{req.side}|{req.quantity}|{req.module}|{req.run_id}".encode()
    ).hexdigest()[:10]
