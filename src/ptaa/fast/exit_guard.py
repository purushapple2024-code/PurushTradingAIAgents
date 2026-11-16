"""Exit guard: evaluates stops, targets, trailing stops, circuit halts and square-off on every tick.

Pure Python, no I/O, no models. Designed for sub-millisecond evaluation per tick.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import time

from ptaa.models import ExitSignal, Position, Tick


class ExitGuard:
    def __init__(
        self,
        square_off_ist: time | None = time(15, 10),
        on_exit: Callable[[ExitSignal], None] | None = None,
    ):
        self._positions: dict[str, Position] = {}
        self._circuit: set[str] = set()
        self.square_off_ist = square_off_ist
        self._on_exit = on_exit

    # -- position book -------------------------------------------------------
    def add(self, pos: Position) -> None:
        self._positions[pos.symbol] = pos

    def remove(self, symbol: str) -> Position | None:
        return self._positions.pop(symbol, None)

    def positions(self) -> list[Position]:
        return list(self._positions.values())

    def set_circuit_hits(self, symbols: set[str]) -> None:
        self._circuit = {s.upper() for s in symbols}

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, tick: Tick) -> ExitSignal | None:
        pos = self._positions.get(tick.symbol)
        if pos is None:
            return None
        px = tick.ltp
        reason: str | None = None

        if pos.side == "BUY":
            pos.highest_since_entry = max(pos.highest_since_entry or px, px)
            if pos.trailing_after_r is not None and not pos.trailed:
                pos.initial_risk = pos.initial_risk or (pos.avg_price - pos.stop_loss)
            if pos.trailing_after_r is not None and pos.initial_risk and pos.initial_risk > 0:
                r = pos.initial_risk
                if pos.highest_since_entry >= pos.avg_price + pos.trailing_after_r * r:
                    new_stop = pos.highest_since_entry - r
                    if new_stop > pos.stop_loss:
                        pos.stop_loss = new_stop
                        pos.trailed = True
            if px <= pos.stop_loss:
                reason = "trailing_stop" if pos.trailed else "stop"
            elif px >= pos.target:
                reason = "target"
        else:  # SELL (short)
            pos.lowest_since_entry = min(pos.lowest_since_entry or px, px)
            if pos.trailing_after_r is not None and not pos.trailed:
                pos.initial_risk = pos.initial_risk or (pos.stop_loss - pos.avg_price)
            if pos.trailing_after_r is not None and pos.initial_risk and pos.initial_risk > 0:
                r = pos.initial_risk
                if pos.lowest_since_entry <= pos.avg_price - pos.trailing_after_r * r:
                    new_stop = pos.lowest_since_entry + r
                    if new_stop < pos.stop_loss:
                        pos.stop_loss = new_stop
                        pos.trailed = True
            if px >= pos.stop_loss:
                reason = "trailing_stop" if pos.trailed else "stop"
            elif px <= pos.target:
                reason = "target"

        if reason is None and tick.symbol in self._circuit:
            reason = "circuit"
        if (
            reason is None
            and pos.module == "intraday"
            and self.square_off_ist is not None
            and tick.ts.timetz().replace(tzinfo=None) >= self.square_off_ist
        ):
            reason = "square_off"

        if reason is None:
            return None
        sig = ExitSignal(
            symbol=tick.symbol,
            module=pos.module,
            reason=reason,  # type: ignore[arg-type]
            price=px,
            ts=tick.ts,
            quantity=pos.quantity,
        )
        self._positions.pop(tick.symbol, None)
        if self._on_exit:
            self._on_exit(sig)
        return sig
