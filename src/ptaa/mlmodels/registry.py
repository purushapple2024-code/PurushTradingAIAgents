"""Model registry: `models/registry.json` plus one artifact file per model version.

Every registered model records its kind, horizon, feature schema hash, training data range,
metrics, file SHA-256 and git commit. `load()` refuses a model whose file hash or feature schema
no longer matches, and `freshness_days()` feeds the model_freshness risk rule.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ptaa.fast.features import feature_schema_hash

ModelKind = Literal[
    "gbm", "seq_lstm", "seq_transformer", "finbert", "quality_gbm", "ensemble", "ppo"
]


class ModelRecord(BaseModel):
    model_id: str  # e.g. gbm-1d-v3
    kind: ModelKind
    horizon: str  # 1h | 1d | 5d | n/a
    version: int
    file: str  # relative to registry dir
    sha256: str
    feature_schema: str
    feature_names: list[str]
    train_start: date
    train_end: date
    trained_at: datetime
    universe: str
    n_samples: int
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    notes: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no user input
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self._records: dict[str, ModelRecord] = {}
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._records = {k: ModelRecord.model_validate(v) for k, v in raw.items()}

    # -- write ---------------------------------------------------------------
    def _save(self) -> None:
        payload = {k: json.loads(v.model_dump_json()) for k, v in self._records.items()}
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def next_version(self, kind: str, horizon: str) -> int:
        prefix = f"{kind}-{horizon}-v"
        versions = [r.version for r in self._records.values() if r.model_id.startswith(prefix)]
        return (max(versions) + 1) if versions else 1

    def register(
        self,
        kind: ModelKind,
        horizon: str,
        file: Path,
        feature_names: list[str],
        train_start: date,
        train_end: date,
        universe: str,
        n_samples: int,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        notes: str = "",
    ) -> ModelRecord:
        version = self.next_version(kind, horizon)
        model_id = f"{kind}-{horizon}-v{version}"
        rec = ModelRecord(
            model_id=model_id,
            kind=kind,
            horizon=horizon,
            version=version,
            file=str(Path(file).relative_to(self.root)) if Path(file).is_absolute() else str(file),
            sha256=sha256_file(self.root / file if not Path(file).is_absolute() else Path(file)),
            feature_schema=feature_schema_hash()
            if kind in ("gbm", "quality_gbm", "ensemble")
            else "n/a",
            feature_names=list(feature_names),
            train_start=train_start,
            train_end=train_end,
            trained_at=datetime.now(tz=timezone.utc),
            universe=universe,
            n_samples=n_samples,
            metrics={k: float(v) for k, v in metrics.items()},
            params=params or {},
            git_commit=git_commit(),
            notes=notes,
        )
        self._records[model_id] = rec
        self._save()
        return rec

    # -- read ----------------------------------------------------------------
    def get(self, model_id: str) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError as e:
            raise KeyError(f"model {model_id!r} is not registered") from e

    def latest(self, kind: str, horizon: str) -> ModelRecord | None:
        cands = [r for r in self._records.values() if r.kind == kind and r.horizon == horizon]
        return max(cands, key=lambda r: r.version) if cands else None

    def list(self) -> list[ModelRecord]:
        return sorted(self._records.values(), key=lambda r: (r.kind, r.horizon, r.version))

    def verify(self, model_id: str) -> tuple[bool, str]:
        rec = self.get(model_id)
        path = self.root / rec.file
        if not path.exists():
            return False, f"file missing: {path}"
        if sha256_file(path) != rec.sha256:
            return False, "sha256 mismatch: file changed since registration"
        if rec.feature_schema != "n/a" and rec.feature_schema != feature_schema_hash():
            return False, "feature schema changed since training; retrain required"
        return True, "ok"

    def path(self, model_id: str) -> Path:
        ok, msg = self.verify(model_id)
        if not ok:
            raise RuntimeError(f"refusing to load {model_id}: {msg}")
        return self.root / self.get(model_id).file

    @staticmethod
    def freshness_days(rec: ModelRecord, today: date | None = None) -> int:
        today = today or datetime.now(tz=timezone.utc).date()
        return (today - rec.train_end).days
