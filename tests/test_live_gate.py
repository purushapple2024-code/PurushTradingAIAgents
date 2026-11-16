from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ptaa.control.api import create_app
from ptaa.control.service import ControlService
from ptaa.execution.broker import OrderRequest
from ptaa.execution.gate import LiveGate
from ptaa.execution.live_brokers import AngelOneBroker, build_live_broker
from ptaa.execution.tradingview import TVAlert, recent_alerts, record_alert, verify_secret
from ptaa.settings import AppConfig, Settings

REQ = OrderRequest(
    symbol="RELIANCE",
    module="swing",
    side="BUY",
    quantity=10,
    stop_loss=2924,
    target=3040,
    strategy="t",
    run_id="r1",
)


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(data_dir=tmp_path, _env_file=None, **kw)  # type: ignore[call-arg]


def test_live_is_locked_without_env_flag(tmp_path: Path, cfg: AppConfig) -> None:
    svc = ControlService(tmp_path / "c.db", cfg)
    gate = LiveGate(_settings(tmp_path), svc)
    ok, why = gate.environment_ok()
    assert not ok and "PTAA_LIVE_TRADING_ENABLED" in why
    d = gate.check(REQ, 2966.0, None)
    assert not d.allowed and d.approval_id is None  # not even an approval is created
    with pytest.raises(PermissionError):
        build_live_broker(_settings(tmp_path), gate)


def test_gate_requires_mode_adapter_creds_cap_and_approval(
    tmp_path: Path, cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PTAA_LIVE_TRADING_ENABLED", "true")
    from ptaa.settings import get_settings

    get_settings.cache_clear()
    svc = ControlService(tmp_path / "c.db", cfg)
    s = _settings(
        tmp_path,
        live_trading_enabled=True,
        angel_api_key="k",
        angel_client_code="c",
        angel_pin="p",
        angel_totp_secret="t",
    )
    gate = LiveGate(s, svc)
    assert gate.environment_ok()[1].startswith("execution_mode")
    svc.set_engine(actor="t", execution_mode="live")  # unlocked because the env flag is set
    assert "adapter is paper" in gate.environment_ok()[1]
    svc.set_engine(actor="t", execution_adapter="upstox")
    assert "no credentials for upstox" in gate.environment_ok()[1]
    svc.set_engine(actor="t", execution_adapter="angelone")
    assert gate.environment_ok() == (True, "ok")

    # daily cap
    svc.put("settings", "live", {"daily_notional_cap_inr": 20_000.0}, actor="t")
    d = gate.check(REQ, 2966.0, None)  # 29,660 notional > 20,000
    assert not d.allowed and "cap" in d.reason
    svc.put("settings", "live", {"daily_notional_cap_inr": 100_000.0}, actor="t")

    # approval flow
    d = gate.check(REQ, 2966.0, None)
    assert not d.allowed and d.reason == "awaiting human approval" and d.approval_id
    assert [a.approval_id for a in gate.pending()] == [d.approval_id]
    assert not gate.check(REQ, 2966.0, d.approval_id).allowed  # still pending
    gate.decide(d.approval_id, approve=True, by="purush")
    assert gate.check(REQ, 2966.0, d.approval_id).allowed
    # approval is bound to the exact order
    other = REQ.model_copy(update={"quantity": 11})
    assert "does not match" in gate.check(other, 2966.0, d.approval_id).reason
    with pytest.raises(ValueError):
        gate.decide(d.approval_id, approve=False, by="x")  # already decided

    # kill switch overrides everything
    svc.kill()
    assert "kill switch" in gate.check(REQ, 2966.0, d.approval_id).reason

    # broker: refuses without approval, sends with it, records live notional
    svc.set_engine(
        actor="t", kill_switch=False, execution_mode="live", execution_adapter="angelone"
    )
    sent = []

    class FakeSession:
        def post(self, url, json, timeout, headers):
            sent.append((url, json, headers))
            return SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {"status": True, "data": {"orderid": "A1"}},
            )

    broker = build_live_broker(
        s, gate, token_map={"RELIANCE": "2885"}, jwt="jwt", session=FakeSession()
    )
    assert isinstance(broker, AngelOneBroker) and broker.is_live
    with pytest.raises(PermissionError):
        broker.place(REQ, 2966.0, datetime.now(tz=timezone.utc))  # no approval id
    fill = broker.place(REQ, 2966.0, datetime.now(tz=timezone.utc), approval_id=d.approval_id)
    assert (
        fill.order_id == "A1"
        and sent[0][1]["tradingsymbol"] == "RELIANCE-EQ"
        and sent[0][1]["quantity"] == "10"
    )
    assert "jwt" not in str(sent[0][1])  # token in header only
    assert gate.notional_today() == pytest.approx(29_660.0)
    with pytest.raises(PermissionError):
        broker.place(
            REQ, 2966.0, datetime.now(tz=timezone.utc), approval_id=d.approval_id
        )  # executed, not reusable
    get_settings.cache_clear()


def test_tradingview_webhook(tmp_path: Path, cfg: AppConfig) -> None:
    svc = ControlService(tmp_path / "c.db", cfg)
    assert (
        not verify_secret("x", None)
        and verify_secret("abc", "abc")
        and not verify_secret("abd", "abc")
    )
    item = record_alert(svc, TVAlert(symbol="infy", action="buy", price=1866.9, strategy="pine-1"))
    assert item["symbol"] == "INFY" and recent_alerts(svc, "INFY")[0]["action"] == "buy"
    app = create_app(
        settings=_settings(tmp_path, tradingview_secret="s3cret"), cfg=cfg, data_dir=tmp_path
    )
    c = TestClient(app)
    assert (
        c.post(
            "/webhooks/tradingview", json={"symbol": "TCS", "action": "sell", "secret": "wrong"}
        ).status_code
        == 401
    )
    assert (
        c.post(
            "/webhooks/tradingview", json={"symbol": "TCS", "action": "sell", "secret": "s3cret"}
        ).json()["symbol"]
        == "TCS"
    )
    st = c.get("/live/status").json()
    assert st["live_possible"] is False and st["pending_approvals"] == 0
    assert c.get("/approvals").json() == []
