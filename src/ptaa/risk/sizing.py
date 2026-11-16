"""Position sizing. The PPO sizer (milestone 8) proposes a fraction of the risk budget; this module
turns a fraction into a quantity and applies the hard caps. Fixed-fraction is the default."""

from __future__ import annotations

import math

from ptaa.settings import RiskLimits


def size_by_risk(
    equity: float,
    entry: float,
    stop: float,
    limits: RiskLimits,
    risk_fraction: float = 1.0,
) -> int:
    """Quantity such that (entry - stop) * qty <= risk budget, capped by max position size.

    `risk_fraction` in [0, 1] scales the per-trade risk budget (the RL sizer's lever).
    """
    if entry <= 0 or stop <= 0 or entry == stop:
        return 0
    risk_fraction = min(max(risk_fraction, 0.0), 1.0)
    risk_budget = equity * limits.max_risk_per_trade_pct / 100.0 * risk_fraction
    per_share_risk = abs(entry - stop)
    qty_by_risk = math.floor(risk_budget / per_share_risk)
    qty_by_position = math.floor(equity * limits.max_position_pct / 100.0 / entry)
    return max(0, min(qty_by_risk, qty_by_position))
