from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ptaa.alerts.telegram import TelegramAlerts
from ptaa.control.api import create_app
from ptaa.control.service import ControlService
from ptaa.settings import AppConfig, Settings


def test_service_seed_versioning_revert_delete(tmp_path: Path, cfg: AppConfig) -> None:
    svc = ControlService(tmp_path / "c.db", cfg)
    assert svc.count("agents") == 18 and svc.count("strategies") == 3
    assert svc.risk_limits("swing").max_sector_pct == 20.0
    # idempotent seed
    ControlService(tmp_path / "c.db", cfg)
    assert svc.version_of("settings", "risk_swing") == 1
    # validated write -> v2; invalid write refused
    lim = svc.get("settings", "risk_swing")
    lim["max_sector_pct"] = 25.0
    assert svc.put("settings", "risk_swing", lim, actor="t", note="loosen") == 2
    with pytest.raises(ValueError):
        svc.put("settings", "risk_swing", {**lim, "max_sector_pct": 500})
    with pytest.raises(ValueError):
        svc.put("settings", "nonsense", {"a": 1})
    assert svc.risk_limits("swing").max_sector_pct == 25.0
    # revert creates v3 with v1 content; history has 3 rows; audit recorded the diff
    assert svc.revert("settings", "risk_swing", 1) == 3
    assert svc.risk_limits("swing").max_sector_pct == 20.0
    assert [h.version for h in svc.history("settings", "risk_swing")] == [1, 2, 3]
    assert any("max_sector_pct" in a.diff for a in svc.audit())
    # strategies CRUD
    assert svc.delete("strategies", "Mean Reversion Bands")
    assert "Mean Reversion Bands" not in svc.list("strategies")
    assert not svc.delete("strategies", "Mean Reversion Bands")
    with pytest.raises(ValueError):
        svc.put("strategies", "X", {"name": "X", "module": "options"})
    # engine: live is locked
    st = svc.set_engine(actor="t", swing_running=True)
    assert st.swing_running and not st.kill_switch
    with pytest.raises(ValueError):
        svc.set_engine(execution_mode="live")
    st = svc.kill()
    assert st.kill_switch and not st.swing_running and not st.intraday_running


def test_api_roundtrip(tmp_path: Path, cfg: AppConfig) -> None:
    app = create_app(settings=Settings(data_dir=tmp_path), cfg=cfg, data_dir=tmp_path)
    c = TestClient(app)
    assert c.get("/health").json()["paper_only"] is True
    strategies = c.get("/cfg/strategies").json()
    assert "Trend Pullback" in strategies
    r = c.put(
        "/cfg/strategies/Trend Pullback",
        json={"body": {**strategies["Trend Pullback"], "ensemble_floor": 0.7}, "note": "tighter"},
    )
    assert r.status_code == 200 and r.json()["version"] == 2
    assert c.get("/cfg/strategies/Trend Pullback").json()["body"]["ensemble_floor"] == 0.7
    assert (
        c.put(
            "/cfg/strategies/Trend Pullback",
            json={"body": {**strategies["Trend Pullback"], "ensemble_floor": 7}},
        ).status_code
        == 422
    )
    assert len(c.get("/cfg/strategies/Trend Pullback/history").json()) == 2
    assert c.post("/cfg/strategies/Trend Pullback/revert/1").json()["version"] == 3
    assert c.get("/cfg/strategies/nope").status_code == 404
    assert c.post("/engine", json={"execution_mode": "live"}).status_code == 400
    assert c.post("/engine/kill").json()["kill_switch"] is True
    port = c.get("/ledger/portfolio").json()
    assert port["equity"] == cfg.paper_book_inr and port["positions"] == []
    assert c.get("/cfg/bogus").status_code == 422


def test_telegram_noop_without_token_and_sends_with(tmp_path: Path) -> None:
    off = TelegramAlerts(Settings(data_dir=tmp_path))
    assert off.enabled is False and off.fill("BUY", 10, "LT", 3600.0, 62.4, "swing") is False
    assert off.sent and "PAPER" in off.sent[-1]

    calls: list[dict] = []

    class FakeSession:
        def post(self, url, json, timeout):
            calls.append({"url": url, "json": json})
            return SimpleNamespace(status_code=200)

    on = TelegramAlerts(
        Settings(data_dir=tmp_path, telegram_bot_token="123:abc", telegram_chat_id="42"),
        min_gap_s=0,
        session=FakeSession(),
    )
    assert on.enabled and on.halt("drawdown 10.2%") is True
    assert calls[0]["json"]["chat_id"] == "42" and "TRADING HALTED" in calls[0]["json"]["text"]
    assert "123:abc" in calls[0]["url"] and "123:abc" not in calls[0]["json"]["text"]


def test_terminal_app_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    from ptaa.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PTAA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PTAA_CONFIG", str(Path(__file__).resolve().parents[1] / "config.yaml"))
    at = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "terminal" / "app.py"), default_timeout=60
    )
    at.run()
    assert not at.exception, at.exception
    assert any("KILL SWITCH" in b.label for b in at.button)
    assert len(at.tabs) == 5
