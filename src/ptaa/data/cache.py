"""Parquet/JSON cache keyed by (source, symbol, params, date) with TTLs.

A re-run on the same inputs makes zero network calls, which keeps tests deterministic and
makes the Colab demo reproducible.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TTL = {"bars": 24 * 3600, "news": 3600, "ref": 7 * 24 * 3600}


class DataCache:
    def __init__(self, root: Path):
        self.root = Path(root) / "cache"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(source: str, symbol: str, params: dict[str, Any]) -> str:
        blob = json.dumps({"s": source, "y": symbol, "p": params}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _path(self, kind: str, key: str, ext: str) -> Path:
        d = self.root / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.{ext}"

    def _fresh(self, path: Path, ttl_s: int) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) < ttl_s

    def get_frame(
        self, source: str, symbol: str, params: dict[str, Any], ttl_s: int = DEFAULT_TTL["bars"]
    ) -> pd.DataFrame | None:
        p = self._path("frames", self._key(source, symbol, params), "parquet")
        if self._fresh(p, ttl_s):
            return pd.read_parquet(p)
        return None

    def put_frame(self, source: str, symbol: str, params: dict[str, Any], df: pd.DataFrame) -> None:
        p = self._path("frames", self._key(source, symbol, params), "parquet")
        df.to_parquet(p)

    def get_json(
        self, source: str, symbol: str, params: dict[str, Any], ttl_s: int = DEFAULT_TTL["ref"]
    ) -> Any | None:
        p = self._path("json", self._key(source, symbol, params), "json")
        if self._fresh(p, ttl_s):
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def put_json(self, source: str, symbol: str, params: dict[str, Any], value: Any) -> None:
        p = self._path("json", self._key(source, symbol, params), "json")
        p.write_text(json.dumps(value, default=str), encoding="utf-8")
