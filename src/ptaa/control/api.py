"""FastAPI control plane. Local-only by default (bind 127.0.0.1). The Streamlit terminal, the CLI,
the Telegram bot and a future React UI all talk to this."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ptaa.control.service import Collection, ControlService
from ptaa.execution.gate import LiveGate
from ptaa.execution.tradingview import TVAlert, record_alert, verify_secret
from ptaa.ledger.db import Ledger
from ptaa.settings import AppConfig, Settings, get_settings, load_config


class PutBody(BaseModel):
    body: dict[str, Any]
    note: str = ""
    actor: str = "api"


def create_app(
    settings: Settings | None = None, cfg: AppConfig | None = None, data_dir: Path | None = None
) -> FastAPI:
    settings = settings or get_settings()
    cfg = cfg or load_config()
    data_dir = data_dir or Path(settings.data_dir)
    svc = ControlService(data_dir / "control.db", cfg)
    ledger = Ledger(data_dir / "ledger.db", cfg.fees, cfg.paper_book_inr)

    app = FastAPI(title="PurushTradingAIAgents control plane", version="0.1.0")
    app.state.svc, app.state.ledger = svc, ledger

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "engine": svc.engine_state().model_dump(), "paper_only": True}

    @app.get("/engine")
    def engine() -> dict[str, Any]:
        return svc.engine_state().model_dump()

    @app.post("/engine")
    def engine_set(changes: dict[str, Any]) -> dict[str, Any]:
        try:
            return svc.set_engine(actor="api", **changes).model_dump()
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/engine/kill")
    def engine_kill() -> dict[str, Any]:
        return svc.kill(actor="api").model_dump()

    @app.get("/cfg/{collection}")
    def list_docs(collection: Collection) -> dict[str, Any]:
        return svc.list(collection)

    @app.get("/cfg/{collection}/{key}")
    def get_doc(collection: Collection, key: str) -> dict[str, Any]:
        doc = svc.get(collection, key)
        if doc is None:
            raise HTTPException(404, f"{collection}/{key} not found")
        return {"key": key, "version": svc.version_of(collection, key), "body": doc}

    @app.put("/cfg/{collection}/{key}")
    def put_doc(collection: Collection, key: str, payload: PutBody) -> dict[str, Any]:
        try:
            v = svc.put(collection, key, payload.body, actor=payload.actor, note=payload.note)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"key": key, "version": v}

    @app.delete("/cfg/{collection}/{key}")
    def delete_doc(collection: Collection, key: str) -> dict[str, Any]:
        if not svc.delete(collection, key, actor="api"):
            raise HTTPException(404, f"{collection}/{key} not found")
        return {"deleted": key}

    @app.get("/cfg/{collection}/{key}/history")
    def history(collection: Collection, key: str) -> list[dict[str, Any]]:
        return [
            {
                "version": h.version,
                "actor": h.actor,
                "note": h.note,
                "deleted": h.deleted,
                "at": h.at.isoformat(),
            }
            for h in svc.history(collection, key)
        ]

    @app.post("/cfg/{collection}/{key}/revert/{version}")
    def revert(collection: Collection, key: str, version: int) -> dict[str, Any]:
        try:
            return {"key": key, "version": svc.revert(collection, key, version, actor="api")}
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    gate = LiveGate(settings, svc)

    @app.get("/approvals")
    def approvals() -> list[dict[str, Any]]:
        return [a.model_dump(mode="json") for a in gate.pending()]

    @app.post("/approvals/{approval_id}/{verdict}")
    def decide(
        approval_id: str, verdict: Literal["approve", "reject"], by: str = "api"
    ) -> dict[str, Any]:
        try:
            return gate.decide(approval_id, verdict == "approve", by=by).model_dump(mode="json")
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/live/status")
    def live_status() -> dict[str, Any]:
        ok, why = gate.environment_ok()
        return {
            "live_possible": ok,
            "reason": why,
            "daily_cap_inr": gate.daily_cap_inr(),
            "notional_today": gate.notional_today(),
            "pending_approvals": len(gate.pending()),
        }

    @app.post("/webhooks/tradingview")
    def tradingview(alert: TVAlert) -> dict[str, Any]:
        if not verify_secret(
            alert.secret,
            settings.tradingview_secret.get_secret_value() if settings.tradingview_secret else None,
        ):
            raise HTTPException(401, "bad secret")
        return record_alert(svc, alert)

    @app.get("/ledger/portfolio")
    def portfolio() -> dict[str, Any]:
        cash, value, eq = ledger.equity()
        return {
            "cash": cash,
            "positions_value": value,
            "equity": eq,
            "fees_paid": ledger.fees_paid(),
            "positions": [p.model_dump() for p in ledger.positions()],
            "runs": [r.model_dump() for r in ledger.runs(20)],
        }

    @app.get("/ledger/runs/{run_id}/audit")
    def run_audit(run_id: str) -> list[dict[str, Any]]:
        return [a.model_dump() for a in ledger.audit_for(run_id)]

    return app
