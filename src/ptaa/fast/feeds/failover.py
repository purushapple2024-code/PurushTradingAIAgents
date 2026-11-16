"""FailoverFeed: wraps an ordered list of feeds; switches when the active one goes silent or errors.

The switch happens after `failover_after_ms` with no tick, or immediately on an exception.
It tries to return to a higher-priority feed every `recover_every_s`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from ptaa.data.adapter import FeedMode, MarketFeed
from ptaa.models import Tick

log = logging.getLogger(__name__)


class FailoverFeed:
    name = "failover"

    def __init__(
        self, feeds: list[MarketFeed], failover_after_ms: int = 1500, recover_every_s: float = 60.0
    ):
        if not feeds:
            raise ValueError("FailoverFeed needs at least one feed")
        self.feeds = feeds
        self.failover_after_ms = failover_after_ms
        self.recover_every_s = recover_every_s
        self.active_index = 0
        self.switches = 0
        self._closed = False

    @property
    def active(self) -> MarketFeed:
        return self.feeds[self.active_index]

    async def _run_one(
        self, idx: int, symbols: list[str], mode: FeedMode, out: asyncio.Queue[Tick | None]
    ) -> None:
        feed = self.feeds[idx]
        try:
            async for tick in feed.subscribe(symbols, mode):
                await out.put(tick)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - any feed error means failover
            log.warning("feed %s failed: %s", feed.name, e)
        finally:
            await out.put(None)

    async def subscribe(self, symbols: list[str], mode: FeedMode = "ltp") -> AsyncIterator[Tick]:
        timeout = self.failover_after_ms / 1000
        while not self._closed:
            q: asyncio.Queue[Tick | None] = asyncio.Queue()
            task = asyncio.create_task(self._run_one(self.active_index, symbols, mode, q))
            started = time.monotonic()
            failed = False
            try:
                while not self._closed:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        log.warning(
                            "feed %s silent for %d ms", self.active.name, self.failover_after_ms
                        )
                        failed = True
                        break
                    if item is None:
                        failed = True
                        break
                    yield item
                    # periodically try to recover to a better feed
                    if self.active_index > 0 and time.monotonic() - started > self.recover_every_s:
                        log.info("attempting recovery to primary feed")
                        self.active_index = 0
                        self.switches += 1
                        break
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception) as e:  # noqa: BLE001
                    log.debug("feed task ended: %r", e)
            if self._closed:
                return
            if failed:
                self.active_index = (self.active_index + 1) % len(self.feeds)
                self.switches += 1
                log.warning("failover -> %s", self.active.name)
                await asyncio.sleep(0.05)

    async def close(self) -> None:
        self._closed = True
        for f in self.feeds:
            try:
                await f.close()
            except Exception as e:  # noqa: BLE001 - best effort
                log.debug("ignored: %s", e)
