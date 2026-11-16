"""Telegram alerts via the Bot API (free). No-op when the token or chat id is missing.

Events: fills, rejects, halts (drawdown / daily loss / kill switch), feed loss, daily summary.
Rate limited to avoid Telegram's 30 msg/s cap and to keep the chat readable.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Literal

import requests

from ptaa.settings import Settings

log = logging.getLogger(__name__)
Kind = Literal["fill", "reject", "halt", "feed", "summary", "info"]
ICON = {"fill": "✅", "reject": "⛔", "halt": "🛑", "feed": "📡", "summary": "📊", "info": "ℹ️"}


class TelegramAlerts:
    def __init__(
        self,
        settings: Settings,
        enabled: bool = True,
        min_gap_s: float = 1.0,
        session: requests.Session | None = None,
    ):
        self.enabled = enabled and bool(settings.telegram_bot_token and settings.telegram_chat_id)
        self._token = (
            settings.telegram_bot_token.get_secret_value() if settings.telegram_bot_token else ""
        )
        self._chat = settings.telegram_chat_id or ""
        self.min_gap_s = min_gap_s
        self._s = session or requests.Session()
        self._last = 0.0
        self.sent: deque[str] = deque(maxlen=200)  # recent messages (for the terminal and tests)

    def send(self, kind: Kind, text: str) -> bool:
        msg = f"{ICON.get(kind, '')} {text}".strip()
        self.sent.append(msg)
        if not self.enabled:
            return False
        gap = self.min_gap_s - (time.time() - self._last)
        if gap > 0:
            time.sleep(gap)
        try:
            r = self._s.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={"chat_id": self._chat, "text": msg[:4000], "disable_web_page_preview": True},
                timeout=10,
            )
            self._last = time.time()
            if r.status_code != 200:
                log.warning(
                    "telegram: status %s", r.status_code
                )  # never log the body: may echo the chat
                return False
            return True
        except requests.RequestException as e:
            log.warning("telegram: %s", type(e).__name__)
            return False

    # convenience formatters -------------------------------------------------
    def fill(
        self, side: str, qty: int, symbol: str, price: float, fees: float, module: str
    ) -> bool:
        return self.send(
            "fill", f"{side} {qty} {symbol} @ ₹{price:,.2f} ({module}) · fees ₹{fees:,.2f} · PAPER"
        )

    def reject(self, symbol: str, rules: list[str]) -> bool:
        return self.send("reject", f"{symbol} rejected: {', '.join(rules) or 'rules'}")

    def halt(self, reason: str) -> bool:
        return self.send("halt", f"TRADING HALTED: {reason}")

    def feed(self, text: str) -> bool:
        return self.send("feed", text)

    def approval_request(
        self, approval_id: str, side: str, qty: int, symbol: str, notional: float
    ) -> bool:
        return self.send(
            "info",
            f"LIVE ORDER NEEDS APPROVAL · {side} {qty} {symbol} · ₹{notional:,.0f} · id {approval_id} · "
            f"approve in the terminal (Control → Live execution) within 10 minutes",
        )

    def summary(self, equity: float, day_pnl: float, positions: int, fees: float) -> bool:
        return self.send(
            "summary",
            f"Equity ₹{equity:,.0f} · day P&L ₹{day_pnl:+,.0f} · {positions} open · fees today ₹{fees:,.2f} · PAPER",
        )
