"""NSE India public endpoints: index constituents, FII/DII flows, circuit list, announcements.

Polite client: cookie bootstrap from the home page, browser-like headers, a minimum gap between
requests, and every response cached. These endpoints are unofficial and change without notice,
so every call degrades to `None`/empty rather than raising into the engine.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from ptaa.data.cache import DataCache

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE,
}


class NSEPublicClient:
    def __init__(self, cache: DataCache | None = None, min_gap_s: float = 1.5, timeout_s: int = 10):
        self.cache = cache
        self.min_gap_s = min_gap_s
        self.timeout_s = timeout_s
        self._s = requests.Session()
        self._s.headers.update(HEADERS)
        self._last = 0.0
        self._booted = False

    def _boot(self) -> None:
        if not self._booted:
            self._s.get(BASE, timeout=self.timeout_s)
            self._booted = True

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any | None:
        key_params = {"path": path, **(params or {})}
        if self.cache:
            hit = self.cache.get_json("nse", "_", key_params, ttl_s=900)
            if hit is not None:
                return hit
        try:
            self._boot()
            gap = self.min_gap_s - (time.time() - self._last)
            if gap > 0:
                time.sleep(gap)
            r = self._s.get(f"{BASE}{path}", params=params, timeout=self.timeout_s)
            self._last = time.time()
            if r.status_code != 200:
                return None
            data = r.json()
        except (requests.RequestException, ValueError):
            return None
        if self.cache:
            self.cache.put_json("nse", "_", key_params, data)
        return data

    def index_constituents(self, index: str = "NIFTY 50") -> list[str]:
        data = self._get("/api/equity-stockIndices", {"index": index})
        if not data:
            return []
        return [row["symbol"] for row in data.get("data", []) if row.get("symbol") != index]

    def fii_dii(self) -> list[dict[str, Any]]:
        data = self._get("/api/fiidiiTradeReact")
        return list(data) if isinstance(data, list) else []

    def circuit_hits(self) -> dict[str, list[str]]:
        """Symbols at upper/lower circuit today, from the price-band hitters endpoint."""
        data = self._get("/api/live-analysis-price-band-hitter")
        if not data:
            return {"upper": [], "lower": []}
        upper = [r.get("symbol") for r in data.get("upper", {}).get("data", []) if r.get("symbol")]
        lower = [r.get("symbol") for r in data.get("lower", {}).get("data", []) if r.get("symbol")]
        return {"upper": upper, "lower": lower}

    def announcements(self, symbol: str) -> list[dict[str, Any]]:
        data = self._get("/api/corporate-announcements", {"index": "equities", "symbol": symbol})
        return list(data) if isinstance(data, list) else []
