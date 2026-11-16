"""Upstox Market Data Feed v3 (WebSocket, protobuf), free with an Upstox account.

Flow: GET /v3/feed/market-data-feed/authorize with the daily access token -> authorized wss URL
-> connect -> send a binary JSON subscription {"method":"sub","data":{"mode":"ltpc",...}} ->
receive protobuf `FeedResponse` messages.

The protobuf module is generated from Upstox's published MarketDataFeedV3.proto; run
`ptaa auth upstox --fetch-proto` once to download and compile it into ptaa/fast/feeds/proto/.
Instrument keys ("NSE_EQ|<ISIN>") come from Upstox's public instrument file (cached daily).
"""

from __future__ import annotations

import gzip
import json
import logging
import uuid
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

AUTHORIZE_URL = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
MODE = {"ltp": "ltpc", "quote": "full", "depth": "full_d30"}


def load_instrument_keys(cache_dir: Path, force: bool = False) -> dict[str, str]:
    """NSE equity trading symbol -> Upstox instrument key (cached daily)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "upstox_nse_instruments.json.gz"
    if force or not p.exists() or (datetime.now().timestamp() - p.stat().st_mtime) > 86400:
        r = requests.get(INSTRUMENTS_URL, timeout=30)
        r.raise_for_status()
        p.write_bytes(r.content)
    rows = json.loads(gzip.decompress(p.read_bytes()))
    return {
        row["trading_symbol"].upper(): row["instrument_key"]
        for row in rows
        if row.get("segment") == "NSE_EQ" and row.get("instrument_type") == "EQ"
    }


class UpstoxFeed:
    name = "upstox"

    def __init__(self, settings: Settings, cache_dir: Path):
        if not settings.upstox_access_token:
            raise RuntimeError(
                "Upstox access token missing; run `ptaa auth upstox` (PTAA_UPSTOX_ACCESS_TOKEN)"
            )
        self.s = settings
        self.cache_dir = cache_dir
        self._closed = False
        self._ws: Any = None

    def _authorize(self) -> str:
        r = requests.get(
            AUTHORIZE_URL,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.s.upstox_access_token.get_secret_value()}",
            },  # type: ignore[union-attr]
            timeout=15,
        )
        r.raise_for_status()
        return str(r.json()["data"]["authorized_redirect_uri"])

    @staticmethod
    def _proto() -> Any:
        try:
            from ptaa.fast.feeds.proto import MarketDataFeedV3_pb2 as pb  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "Upstox protobuf module missing. Run `ptaa auth upstox --fetch-proto` to generate "
                "ptaa/fast/feeds/proto/MarketDataFeedV3_pb2.py from Upstox's published .proto."
            ) from e
        return pb

    async def subscribe(self, symbols: list[str], mode: FeedMode = "ltp") -> AsyncIterator[Tick]:
        import asyncio

        import websockets

        pb = self._proto()
        keys = await asyncio.to_thread(load_instrument_keys, self.cache_dir)
        key_to_sym = {keys[s.upper()]: s.upper() for s in symbols if s.upper() in keys}
        missing = [s for s in symbols if s.upper() not in keys]
        if missing:
            log.warning("upstox: no instrument key for %s", missing)
        url = await asyncio.to_thread(self._authorize)

        async with websockets.connect(url, max_size=2**22) as ws:
            self._ws = ws
            sub = {
                "guid": uuid.uuid4().hex[:16],
                "method": "sub",
                "data": {"mode": MODE[mode], "instrumentKeys": list(key_to_sym)},
            }
            await ws.send(json.dumps(sub).encode())
            while not self._closed:
                raw = await ws.recv()
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                msg = pb.FeedResponse()
                msg.ParseFromString(raw)
                if msg.type != pb.Type.live_feed:  # market_info / initial_feed carry no ticks
                    continue
                for key, feed in msg.feeds.items():
                    sym = key_to_sym.get(key)
                    if sym is None:
                        continue
                    which = feed.WhichOneof("FeedUnion")
                    if which == "ltpc":
                        ltpc, vol, bid, ask = feed.ltpc, 0, None, None
                    elif which == "fullFeed":
                        mff = feed.fullFeed.marketFF
                        ltpc, vol = mff.ltpc, int(mff.vtt or 0)
                        bid = (
                            mff.marketLevel.bidAskQuote[0].bidP
                            if mff.marketLevel.bidAskQuote
                            else None
                        )
                        ask = (
                            mff.marketLevel.bidAskQuote[0].askP
                            if mff.marketLevel.bidAskQuote
                            else None
                        )
                    else:
                        continue
                    ltt_ms = int(ltpc.ltt or 0)
                    ts = (
                        datetime.fromtimestamp(ltt_ms / 1000, tz=timezone.utc).astimezone(IST)
                        if ltt_ms
                        else datetime.now(tz=IST)
                    )
                    if ltpc.ltp and ltpc.ltp > 0:
                        yield Tick(
                            symbol=sym,
                            ts=ts,
                            ltp=float(ltpc.ltp),
                            volume=vol,
                            bid=bid,
                            ask=ask,
                            source=self.name,
                        )

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:  # noqa: BLE001 - best effort
                log.debug("ignored: %s", e)
