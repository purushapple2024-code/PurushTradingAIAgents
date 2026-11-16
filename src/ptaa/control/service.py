"""ControlService: SQLite-backed documents with full version history and an audit log.

Collections: settings (one doc per group: risk, feeds, llm, universe, schedule, alerts, execution),
strategies, agents. Every write creates a new version row; `revert` restores an older version as a
new version. `engine` state (running/paused per module, kill switch) is a special settings group.
Values are validated against the config models before they are stored, so a bad limit never reaches
the engine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func
from sqlmodel import Field as SField
from sqlmodel import Session, SQLModel, create_engine, select

from ptaa.settings import AppConfig, LLMConfig, RiskLimits

Collection = Literal["settings", "strategies", "agents"]


class DocVersion(SQLModel, table=True):
    __tablename__ = "control_docs"
    id: int | None = SField(default=None, primary_key=True)
    collection: str = SField(index=True)
    key: str = SField(index=True)
    version: int
    body: str  # JSON
    actor: str
    note: str = ""
    deleted: bool = False
    at: datetime


class ControlAudit(SQLModel, table=True):
    __tablename__ = "control_audit"
    id: int | None = SField(default=None, primary_key=True)
    at: datetime
    actor: str
    action: str
    collection: str
    key: str
    version: int | None = None
    diff: str = ""


# ----------------------------------------------------------------------------- document schemas


class StrategyDoc(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    module: Literal["swing", "intraday"]
    universe: str = "NIFTY50"
    enabled: bool = True
    status: Literal["draft", "backtest", "shadow", "live_paper"] = "draft"
    decision_bar: str = "1d"
    signals: list[str] = ["technical", "news", "fundamental"]
    signal_floors: dict[str, float] = {}
    ensemble_floor: float = Field(default=0.60, ge=0, le=1)
    stop_atr: float = Field(default=2.0, gt=0)
    target_atr: float = Field(default=3.6, gt=0)
    max_hold_days: int = Field(default=12, ge=1)
    trail_after_r: float | None = 1.0
    sizer: Literal["fixed_risk", "ppo"] = "fixed_risk"
    regime_gate: list[str] = ["trend_up", "range"]
    yaml_extra: str = ""
    metrics: dict[str, float] = {}


class AgentDoc(BaseModel):
    name: str
    kind: Literal["ml", "llm", "code"]
    layer: Literal["fast", "analysts", "gates", "reasoning", "decision", "execution"]
    enabled: bool = True
    runs_on: Literal["every_closed_bar", "daily_after_close", "on_demand"] = "every_closed_bar"
    scope_modules: list[str] = ["swing", "intraday"]
    scope_universe: list[str] = ["NIFTY50"]
    scope_horizons: list[str] = ["1h", "1d", "5d"]
    model_id: str | None = None
    max_model_age_days: int | None = None
    min_confidence: float | None = None
    permissions: dict[str, bool] = {
        "read_features": True,
        "emit_signals": True,
        "network": False,
        "ledger_write": False,
    }


class EngineState(BaseModel):
    swing_running: bool = False
    intraday_running: bool = False
    kill_switch: bool = False
    execution_mode: Literal["paper", "shadow", "live"] = "paper"
    execution_adapter: Literal["paper", "angelone", "upstox", "flattrade"] = "paper"
    updated_at: datetime | None = None


SETTINGS_SCHEMAS: dict[str, type[BaseModel] | None] = {
    "risk_swing": RiskLimits,
    "risk_intraday": RiskLimits,
    "llm": LLMConfig,
    "engine": EngineState,
    # free-form groups validated loosely:
    "feeds": None,
    "universe": None,
    "schedule": None,
    "alerts": None,
    "live": None,  # daily_notional_cap_inr etc.
    "live_ledger": None,
    "approvals": None,
    "tradingview_alerts": None,
}

DEFAULT_AGENTS: list[AgentDoc] = [
    AgentDoc(name="Tick ingester", kind="code", layer="fast"),
    AgentDoc(name="Bar + feature engine", kind="code", layer="fast"),
    AgentDoc(name="Exit guard", kind="code", layer="fast"),
    AgentDoc(
        name="Technical analyst",
        kind="ml",
        layer="analysts",
        max_model_age_days=14,
        min_confidence=0.55,
    ),
    AgentDoc(
        name="Sequence analyst", kind="ml", layer="analysts", enabled=False, max_model_age_days=30
    ),
    AgentDoc(
        name="News + sentiment analyst",
        kind="ml",
        layer="analysts",
        permissions={
            "read_features": True,
            "emit_signals": True,
            "network": True,
            "ledger_write": False,
        },
    ),
    AgentDoc(
        name="Fundamental analyst", kind="code", layer="analysts", runs_on="daily_after_close"
    ),
    AgentDoc(name="Verifier", kind="code", layer="gates"),
    AgentDoc(name="Market regime", kind="code", layer="gates"),
    AgentDoc(name="Ensemble + strategy selector", kind="code", layer="gates"),
    AgentDoc(name="Bull researcher", kind="llm", layer="reasoning"),
    AgentDoc(name="Bear researcher", kind="llm", layer="reasoning"),
    AgentDoc(name="Research manager", kind="llm", layer="reasoning"),
    AgentDoc(name="Risk personas", kind="llm", layer="reasoning"),
    AgentDoc(name="Trader + sizer", kind="code", layer="decision"),
    AgentDoc(name="Risk manager", kind="code", layer="decision"),
    AgentDoc(name="Portfolio manager", kind="llm", layer="decision"),
    AgentDoc(
        name="Executor",
        kind="code",
        layer="execution",
        permissions={
            "read_features": False,
            "emit_signals": False,
            "network": False,
            "ledger_write": True,
        },
    ),
]

DEFAULT_STRATEGIES: list[StrategyDoc] = [
    StrategyDoc(
        name="Trend Pullback",
        module="swing",
        status="live_paper",
        signals=["technical", "news", "fundamental"],
    ),
    StrategyDoc(
        name="Opening Range Breakout",
        module="intraday",
        universe="NIFTY50_LIQUID15",
        decision_bar="5m",
        status="live_paper",
        stop_atr=1.5,
        target_atr=2.0,
        max_hold_days=1,
        signals=["technical"],
        regime_gate=["trend_up", "trend_down", "range"],
    ),
    StrategyDoc(
        name="Mean Reversion Bands",
        module="swing",
        status="shadow",
        signals=["technical", "fundamental"],
        regime_gate=["range"],
    ),
]


class ControlService:
    def __init__(self, path: Path, cfg: AppConfig | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(
            self.engine, tables=[DocVersion.__table__, ControlAudit.__table__]
        )  # type: ignore[attr-defined]
        if cfg is not None:
            self.seed(cfg)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=timezone.utc)

    def _latest(self, s: Session, collection: str, key: str) -> DocVersion | None:
        return s.exec(
            select(DocVersion)
            .where(DocVersion.collection == collection, DocVersion.key == key)
            .order_by(DocVersion.version.desc())
        ).first()  # type: ignore[attr-defined]

    def _validate(self, collection: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
        schema: type[BaseModel] | None
        if collection == "strategies":
            schema = StrategyDoc
        elif collection == "agents":
            schema = AgentDoc
        else:
            if key not in SETTINGS_SCHEMAS:
                raise ValueError(
                    f"unknown settings group {key!r}; allowed: {sorted(SETTINGS_SCHEMAS)}"
                )
            schema = SETTINGS_SCHEMAS[key]
        if schema is None:
            return body
        try:
            return json.loads(schema.model_validate(body).model_dump_json())
        except ValidationError as e:
            raise ValueError(
                f"invalid {collection}/{key}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}"
            ) from e

    def _audit(
        self,
        s: Session,
        actor: str,
        action: str,
        collection: str,
        key: str,
        version: int | None,
        diff: str = "",
    ) -> None:
        s.add(
            ControlAudit(
                at=self._now(),
                actor=actor,
                action=action,
                collection=collection,
                key=key,
                version=version,
                diff=diff[:2000],
            )
        )

    # -- seed ----------------------------------------------------------------
    def seed(self, cfg: AppConfig) -> None:
        """Insert defaults only where no document exists yet (idempotent)."""
        groups = {
            "risk_swing": cfg.risk["swing"].model_dump(),
            "risk_intraday": cfg.risk["intraday"].model_dump(),
            "llm": cfg.llm.model_dump(),
            "feeds": cfg.feeds.model_dump(),
            "universe": cfg.universe.model_dump(),
            "schedule": {
                "swing": cfg.modules.swing.model_dump(),
                "intraday": cfg.modules.intraday.model_dump(),
            },
            "alerts": cfg.alerts.model_dump(),
            "engine": EngineState().model_dump(),
        }
        for k, v in groups.items():
            if self.get("settings", k) is None:
                self.put("settings", k, v, actor="seed", note="seeded from config.yaml")
        for st in DEFAULT_STRATEGIES:
            if self.get("strategies", st.name) is None:
                self.put("strategies", st.name, st.model_dump(), actor="seed")
        for ag in DEFAULT_AGENTS:
            if self.get("agents", ag.name) is None:
                self.put("agents", ag.name, ag.model_dump(), actor="seed")

    # -- CRUD ----------------------------------------------------------------
    def get(self, collection: Collection, key: str) -> dict[str, Any] | None:
        with Session(self.engine) as s:
            row = self._latest(s, collection, key)
            return None if row is None or row.deleted else json.loads(row.body)

    def version_of(self, collection: Collection, key: str) -> int:
        with Session(self.engine) as s:
            row = self._latest(s, collection, key)
            return row.version if row else 0

    def list(self, collection: Collection) -> dict[str, dict[str, Any]]:
        with Session(self.engine) as s:
            keys = s.exec(
                select(DocVersion.key).where(DocVersion.collection == collection).distinct()
            ).all()
            out: dict[str, dict[str, Any]] = {}
            for k in keys:
                row = self._latest(s, collection, k)
                if row and not row.deleted:
                    out[k] = json.loads(row.body)
            return out

    def put(
        self,
        collection: Collection,
        key: str,
        body: dict[str, Any],
        actor: str = "user",
        note: str = "",
    ) -> int:
        body = self._validate(collection, key, body)
        if collection == "settings" and key == "engine":
            body["updated_at"] = self._now().isoformat()
        with Session(self.engine) as s:
            prev = self._latest(s, collection, key)
            version = (prev.version + 1) if prev else 1
            diff = ""
            if prev and not prev.deleted:
                old = json.loads(prev.body)
                changed = {
                    k: (old.get(k), body.get(k))
                    for k in set(old) | set(body)
                    if old.get(k) != body.get(k)
                }
                diff = json.dumps(changed, default=str)
            s.add(
                DocVersion(
                    collection=collection,
                    key=key,
                    version=version,
                    body=json.dumps(body, default=str),
                    actor=actor,
                    note=note,
                    at=self._now(),
                )
            )
            self._audit(s, actor, "put", collection, key, version, diff)
            s.commit()
        return version

    def delete(self, collection: Collection, key: str, actor: str = "user") -> bool:
        with Session(self.engine) as s:
            prev = self._latest(s, collection, key)
            if prev is None or prev.deleted:
                return False
            s.add(
                DocVersion(
                    collection=collection,
                    key=key,
                    version=prev.version + 1,
                    body=prev.body,
                    actor=actor,
                    deleted=True,
                    at=self._now(),
                )
            )
            self._audit(s, actor, "delete", collection, key, prev.version + 1)
            s.commit()
        return True

    def history(self, collection: Collection, key: str) -> list[DocVersion]:
        with Session(self.engine) as s:
            return list(
                s.exec(
                    select(DocVersion)
                    .where(DocVersion.collection == collection, DocVersion.key == key)
                    .order_by(DocVersion.version)
                )
            )

    def revert(self, collection: Collection, key: str, to_version: int, actor: str = "user") -> int:
        with Session(self.engine) as s:
            row = s.exec(
                select(DocVersion).where(
                    DocVersion.collection == collection,
                    DocVersion.key == key,
                    DocVersion.version == to_version,
                )
            ).first()
        if row is None:
            raise KeyError(f"{collection}/{key} v{to_version} not found")
        return self.put(
            collection, key, json.loads(row.body), actor=actor, note=f"revert to v{to_version}"
        )

    def audit(self, limit: int = 100) -> list[ControlAudit]:
        with Session(self.engine) as s:
            return list(s.exec(select(ControlAudit).order_by(ControlAudit.at.desc()).limit(limit)))  # type: ignore[attr-defined]

    # -- typed helpers -------------------------------------------------------
    def engine_state(self) -> EngineState:
        return EngineState.model_validate(self.get("settings", "engine") or {})

    def set_engine(self, actor: str = "user", **changes: Any) -> EngineState:
        st = self.engine_state().model_dump()
        st.update(changes)
        if st.get("execution_mode") == "live":
            from ptaa.settings import get_settings

            if not get_settings().live_trading_enabled:
                raise ValueError(
                    "live execution is locked: set PTAA_LIVE_TRADING_ENABLED=true in the environment "
                    "(not in the UI), configure a broker adapter, and approve each order"
                )
        self.put("settings", "engine", st, actor=actor, note="engine state")
        return self.engine_state()

    def kill(self, actor: str = "user") -> EngineState:
        return self.set_engine(
            actor=actor, kill_switch=True, swing_running=False, intraday_running=False
        )

    def risk_limits(self, module: str) -> RiskLimits:
        return RiskLimits.model_validate(self.get("settings", f"risk_{module}") or {})

    def count(self, collection: Collection) -> int:
        with Session(self.engine) as s:
            return int(
                s.exec(
                    select(func.count(func.distinct(DocVersion.key))).where(
                        DocVersion.collection == collection
                    )
                ).one()
            )
