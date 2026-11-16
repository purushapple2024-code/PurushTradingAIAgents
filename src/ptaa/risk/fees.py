"""Indian NSE cash-equity transaction costs for paper fills.

Rates are configurable (config.fees) because they change; defaults reflect late-2025 schedules:
STT 0.1% each side on delivery, 0.025% sell side on intraday; NSE transaction charge ~0.00297%;
SEBI turnover fee 0.0001%; GST 18% on brokerage + exchange + SEBI; stamp duty on buys
(0.015% delivery, 0.003% intraday). DP charges on delivery sells are broker-specific and omitted.
"""

from __future__ import annotations

from pydantic import BaseModel

from ptaa.models import Module, Side
from ptaa.settings import FeeConfig


class FeeBreakdown(BaseModel):
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    gst: float
    stamp: float

    @property
    def total(self) -> float:
        return round(
            self.brokerage + self.stt + self.exchange + self.sebi + self.gst + self.stamp, 2
        )


def compute_fees(
    cfg: FeeConfig, side: Side, module: Module, price: float, quantity: int
) -> FeeBreakdown:
    turnover = price * quantity
    pct = lambda p: turnover * p / 100.0  # noqa: E731

    if module == "swing":  # delivery
        brokerage = cfg.brokerage_per_order_inr
        stt = pct(cfg.stt_delivery_pct)
        stamp = pct(cfg.stamp_duty_buy_delivery_pct) if side == "BUY" else 0.0
    else:  # intraday
        brokerage = min(pct(cfg.brokerage_intraday_pct), cfg.brokerage_intraday_cap_inr)
        stt = pct(cfg.stt_intraday_sell_pct) if side == "SELL" else 0.0
        stamp = pct(cfg.stamp_duty_buy_intraday_pct) if side == "BUY" else 0.0

    exchange = pct(cfg.exchange_txn_pct)
    sebi = pct(cfg.sebi_fee_pct)
    gst = (brokerage + exchange + sebi) * cfg.gst_pct / 100.0
    return FeeBreakdown(
        brokerage=round(brokerage, 2),
        stt=round(stt, 2),
        exchange=round(exchange, 2),
        sebi=round(sebi, 2),
        gst=round(gst, 2),
        stamp=round(stamp, 2),
    )


def round_trip_cost(cfg: FeeConfig, module: Module, price: float, quantity: int) -> float:
    """Buy + sell fees at the same price: the hurdle a trade must clear."""
    return (
        compute_fees(cfg, "BUY", module, price, quantity).total
        + compute_fees(cfg, "SELL", module, price, quantity).total
    )
