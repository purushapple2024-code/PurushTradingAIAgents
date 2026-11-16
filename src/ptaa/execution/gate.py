"""LiveGate: every live order must pass ALL of these, in order, or it becomes a paper order.

1. PTAA_LIVE_TRADING_ENABLED=true in the environment (never settable from the UI).
2. Control-plane engine state: execution_mode == "live" and kill_switch == False.
3. A broker adapter other than paper is configured with credentials.
4. Daily live notional cap not exceeded (control-plane setting, default INR 50,000).
5. A human approved THIS order (approval queue in the control plane; terminal or Telegram).

Approvals expire after `approval_ttl_s`. Nothing here can be bypassed by prompts, models or settings
edits: step 1 is an environment variable and step 5 is a human click per order.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel

from ptaa.control.service import ControlService
from ptaa.execution.broker import OrderRequest
from ptaa.settings import Settings

ApprovalStatus = Literal["pending", "approved", "rejected", "expired", "executed"]


class Approval(BaseModel):
    approval_id: str
    order: dict
    notional: float
    created_at: datetime
    status: ApprovalStatus = "pending"
    decided_by: str | None = None
    decided_at: datetime | None = None
    reason: str = ""


class GateDecision(BaseModel):
    allowed: bool
    reason: str
    approval_id: str | None = None


class LiveGate:
    def __init__(self, settings: Settings, control: ControlService, approval_ttl_s: int = 600):
        self.settings, self.control, self.ttl = settings, control, approval_ttl_s

    # -- static checks --------------------------------------------------------
    def environment_ok(self) -> tuple[bool, str]:
        if not self.settings.live_trading_enabled:
            return False, "PTAA_LIVE_TRADING_ENABLED is not true"
        st = self.control.engine_state()
        if st.kill_switch:
            return False, "kill switch is on"
        if st.execution_mode != "live":
            return False, f"execution_mode is {st.execution_mode!r}, not 'live'"
        if st.execution_adapter == "paper":
            return False, "execution_adapter is paper"
        has = {
            "angelone": self.settings.has_angel,
            "upstox": bool(self.settings.upstox_access_token),
            "flattrade": self.settings.has_flattrade,
        }
        if not has.get(st.execution_adapter, False):
            return False, f"no credentials for {st.execution_adapter}"
        return True, "ok"

    def daily_cap_inr(self) -> float:
        doc = self.control.get("settings", "live") or {}
        return float(doc.get("daily_notional_cap_inr", 50_000.0))

    def notional_today(self) -> float:
        today = date.today().isoformat()
        return sum(
            a["notional"]
            for a in (self.control.get("settings", "live_ledger") or {}).get(today, [])
        )

    def _record_executed(self, approval: Approval) -> None:
        led = self.control.get("settings", "live_ledger") or {}
        today = date.today().isoformat()
        led.setdefault(today, []).append(
            {
                "approval_id": approval.approval_id,
                "notional": approval.notional,
                "at": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        self.control.put("settings", "live_ledger", led, actor="gate", note="live fill recorded")

    # -- approval queue (stored in the control plane, versioned and audited) ---------
    def _queue(self) -> dict[str, dict]:
        return self.control.get("settings", "approvals") or {}

    def _save_queue(self, q: dict[str, dict], actor: str, note: str) -> None:
        self.control.put("settings", "approvals", q, actor=actor, note=note)

    def request_approval(self, req: OrderRequest, price: float) -> Approval:
        ap = Approval(
            approval_id=uuid.uuid4().hex[:8],
            order=req.model_dump(),
            notional=req.quantity * price,
            created_at=datetime.now(tz=timezone.utc),
        )
        q = self._queue()
        q[ap.approval_id] = ap.model_dump(mode="json")
        self._save_queue(q, actor="gate", note=f"approval requested {ap.approval_id}")
        return ap

    def pending(self) -> list[Approval]:
        out = []
        for a in self._queue().values():
            ap = Approval.model_validate(a)
            if (
                ap.status == "pending"
                and (datetime.now(tz=timezone.utc) - ap.created_at).total_seconds() > self.ttl
            ):
                ap.status = "expired"
            out.append(ap)
        return [a for a in out if a.status == "pending"]

    def decide(self, approval_id: str, approve: bool, by: str, reason: str = "") -> Approval:
        q = self._queue()
        if approval_id not in q:
            raise KeyError(approval_id)
        ap = Approval.model_validate(q[approval_id])
        if ap.status != "pending":
            raise ValueError(f"approval {approval_id} is {ap.status}")
        if (datetime.now(tz=timezone.utc) - ap.created_at).total_seconds() > self.ttl:
            ap.status = "expired"
        else:
            ap.status = "approved" if approve else "rejected"
        ap.decided_by, ap.decided_at, ap.reason = by, datetime.now(tz=timezone.utc), reason
        q[approval_id] = ap.model_dump(mode="json")
        self._save_queue(q, actor=by, note=f"approval {ap.status} {approval_id}")
        return ap

    # -- the check the executor calls ---------------------------------------
    def check(self, req: OrderRequest, price: float, approval_id: str | None) -> GateDecision:
        ok, why = self.environment_ok()
        if not ok:
            return GateDecision(allowed=False, reason=why)
        notional = req.quantity * price
        if self.notional_today() + notional > self.daily_cap_inr():
            return GateDecision(
                allowed=False,
                reason=f"daily live notional cap ₹{self.daily_cap_inr():,.0f} would be exceeded",
            )
        if approval_id is None:
            ap = self.request_approval(req, price)
            return GateDecision(
                allowed=False, reason="awaiting human approval", approval_id=ap.approval_id
            )
        q = self._queue()
        if approval_id not in q:
            return GateDecision(allowed=False, reason="unknown approval id")
        ap = Approval.model_validate(q[approval_id])
        if ap.status != "approved":
            return GateDecision(
                allowed=False, reason=f"approval is {ap.status}", approval_id=approval_id
            )
        if (
            ap.order.get("symbol") != req.symbol
            or ap.order.get("side") != req.side
            or int(ap.order.get("quantity", -1)) != req.quantity
        ):
            return GateDecision(
                allowed=False, reason="approval does not match this order", approval_id=approval_id
            )
        return GateDecision(allowed=True, reason="approved", approval_id=approval_id)

    def mark_executed(self, approval_id: str) -> None:
        q = self._queue()
        ap = Approval.model_validate(q[approval_id])
        ap.status = "executed"
        q[approval_id] = ap.model_dump(mode="json")
        self._save_queue(q, actor="gate", note=f"executed {approval_id}")
        self._record_executed(ap)
