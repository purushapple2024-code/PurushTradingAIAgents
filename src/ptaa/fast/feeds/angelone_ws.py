"""Angel One SmartAPI WebSocket 2.0 feed (free with an Angel One account).

Login: SmartConnect.generateSession(client_code, pin, TOTP). Streaming: SmartWebSocketV2 with
LTP (mode 1) or Quote (mode 2). The SDK is callback/thread based; ticks are bridged into the
asyncio loop with `call_soon_threadsafe`. Prices arrive in paise and are converted to rupees.

Requires the `brokers` extra: smartapi-python, pyotp. Credentials come from Settings only.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from ptaa.data.adapter import FeedMode
from ptaa.models import Tick
from ptaa.settings import Settings

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
NSE_CM_EXCHANGE_TYPE = 1
MODE = {"ltp": 1, "quote": 2, "depth": 3}


def load_token_map(cache_dir: Path, force: bool = False) -> dict[str, str]:
    """NSE equity symbol -> Angel One token, from the public scrip master (cached daily)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "angel_scrip_master.json.gz"
    if force or not p.exists() or (datetime.now().timestamp() - p.stat().st_mtime) > 86400:
        r = requests.get(SCRIP_MASTER_URL, timeout=30)
        r.raise_for_status()
        p.write_bytes(gzip.compress(r.content))
    rows = json.loads(gzip.decompress(p.read_bytes()))
    out: dict[str, str] = {}
    for row in rows:
        if row.get("exch_seg") == "NSE" and str(row.get("symbol", "")).endswith("-EQ"):
            out[row["name"].upper()] = str(row["token"])
    return out


class AngelOneFeed:
    name = "angelone"

    def __init__(self, settings: Settings, cache_dir: Path):
        if not settings.has_angel:
            raise RuntimeError("Angel One credentials missing (PTAA_ANGEL_*)")
        self.s = settings
        self.cache_dir = cache_dir
        self._ws: Any = None
        self._closed = False

    def _login(self) -> tuple[str, str]:
        import pyotp
        from SmartApi import SmartConnect  # type: ignore[import-not-found]

        api = SmartConnect(api_key=self.s.angel_api_key.get_secret_value())  # type: ignore[union-attr]
        totp = pyotp.TOTP(self.s.angel_totp_secret.get_secret_value()).now()  # type: ignore[union-attr]
        data = api.generateSession(
            self.s.angel_client_code, self.s.angel_pin.get_secret_value(), totp
        )  # type: ignore[union-attr]
        if not data or not data.get("status"):
            raise RuntimeError(
                "Angel One login failed (never logging the response: may contain tokens)"
            )
        auth_token = data["data"]["jwtToken"]
        feed_token = api.getfeedToken()
        return auth_token, feed_token

    async def subscribe(self, symbols: list[str], mode: FeedMode = "ltp") -> AsyncIterator[Tick]:
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore[import-not-found]

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Tick | Exception] = asyncio.Queue()
        token_map = await asyncio.to_thread(load_token_map, self.cache_dir)
        tokens = {token_map[s.upper()]: s.upper() for s in symbols if s.upper() in token_map}
        missing = [s for s in symbols if s.upper() not in token_map]
        if missing:
            log.warning("angelone: no token for %s", missing)
        auth_token, feed_token = await asyncio.to_thread(self._login)

        ws = SmartWebSocketV2(
            auth_token,
            self.s.angel_api_key.get_secret_value(),
            self.s.angel_client_code,
            feed_token,
        )  # type: ignore[union-attr]
        self._ws = ws

        def on_data(_wsapp: Any, msg: dict[str, Any]) -> None:
            try:
                sym = tokens.get(str(msg.get("token")))
                if sym is None:
                    return
                ts_ms = msg.get("exchange_timestamp") or 0
                ts = (
                    datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(IST)
                    if ts_ms
                    else datetime.now(tz=IST)
                )
                tick = Tick(
                    symbol=sym,
                    ts=ts,
                    ltp=float(msg["last_traded_price"]) / 100.0,
                    volume=int(msg.get("volume_trade_for_the_day") or 0),
                    bid=(float(msg["best_5_buy_data"][0]["price"]) / 100.0)
                    if msg.get("best_5_buy_data")
                    else None,
                    ask=(float(msg["best_5_sell_data"][0]["price"]) / 100.0)
                    if msg.get("best_5_sell_data")
                    else None,
                    source=self.name,
                )
                loop.call_soon_threadsafe(q.put_nowait, tick)
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, e)

        def on_open(_wsapp: Any) -> None:
            ws.subscribe(
                "ptaa", MODE[mode], [{"exchangeType": NSE_CM_EXCHANGE_TYPE, "tokens": list(tokens)}]
            )

        def on_error(_wsapp: Any, err: Any) -> None:
            loop.call_soon_threadsafe(q.put_nowait, RuntimeError(f"angelone ws error: {err}"))

        ws.on_open, ws.on_data, ws.on_error = on_open, on_data, on_error
        ws.on_close = lambda *_: loop.call_soon_threadsafe(
            q.put_nowait, RuntimeError("angelone ws closed")
        )
        threading.Thread(target=ws.connect, daemon=True, name="angelone-ws").start()

        while not self._closed:
            item = await q.get()
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                self._ws.close_connection()
            except Exception as e:  # noqa: BLE001 - best effort
                log.debug("ignored: %s", e)
