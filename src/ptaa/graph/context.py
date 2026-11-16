"""Runtime context shared by graph nodes (models, registry, ledger, config) + audit wrapper."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ptaa.graph.state import AuditEvent, GraphState
from ptaa.ledger.db import Ledger, PaperBroker
from ptaa.mlmodels.registry import ModelRegistry
from ptaa.settings import AppConfig, Settings


@dataclass
class GraphContext:
    cfg: AppConfig
    settings: Settings
    registry: ModelRegistry
    ledger: Ledger
    broker: PaperBroker
    models: dict[str, Any] = field(default_factory=dict)  # loaded model objects by model_id
    llm: Any | None = None  # milestone 5
    news_items: list[Any] | None = None  # injected discovery results (tests / offline); None = live
    alerts: Any | None = None  # TelegramAlerts or None
    _fetcher: Callable[[str], Any] | None = None

    @property
    def fetcher(self) -> Callable[[str], Any]:
        """Allow-listed, cached page fetcher. Injectable for offline runs."""
        if self._fetcher is None:
            from pathlib import Path

            from ptaa.research.http import GuardedHTTP

            http = GuardedHTTP(
                self.cfg.research.allowed_domains,
                Path(self.settings.data_dir) / "research",
                timeout_s=self.cfg.research.fetch_timeout_s,
                max_bytes=self.cfg.research.fetch_max_bytes,
            )
            self._fetcher = http.get
        return self._fetcher

    def set_fetcher(self, fn: Callable[[str], Any]) -> None:
        self._fetcher = fn

    now: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc)


def _hash(obj: Any) -> str:
    try:
        blob = json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        blob = repr(obj)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def audited(
    name: str, model_id_key: str | None = None
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Wrap a node so it always appends an AuditEvent and converts exceptions to `errors`."""

    def deco(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        def wrapper(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
            t0 = time.perf_counter()
            inputs_hash = _hash({k: v for k, v in state.items() if k not in ("audit", "errors")})
            try:
                out = fn(state, ctx)
                note = ""
            except Exception as e:  # noqa: BLE001 - a failing node must not kill the run
                out = {"errors": [f"{name}: {type(e).__name__}: {e}"]}
                note = f"error: {type(e).__name__}"
            dur = (time.perf_counter() - t0) * 1000
            ev = AuditEvent(
                node=name,
                at=ctx.now(),
                duration_ms=round(dur, 3),
                inputs_hash=inputs_hash,
                outputs_hash=_hash(out),
                model_id=out.get(model_id_key) if model_id_key else None,
                note=note,
            )
            out.setdefault("audit", [])
            out["audit"] = list(out["audit"]) + [ev]
            return out

        wrapper.__name__ = name
        return wrapper

    return deco
